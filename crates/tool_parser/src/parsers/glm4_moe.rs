use std::collections::HashMap;

use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::Value;

use crate::{
    errors::{ParserError, ParserResult},
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// Which GLM MoE wire format to use for name extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlmFormat {
    /// Name ends at the first newline after `<tool_call>`.
    Glm45,
    /// Name is the whitespace-trimmed token before the first `<arg_key>` (or whole body if none).
    Glm47,
}

/// GLM-4 MoE format parser for tool calls
///
/// Handles both GLM-4 MoE and GLM-4.7 MoE formats:
/// - GLM-4: `<tool_call>{name}\n<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>`
/// - GLM-4.7: `<tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value></tool_call>`
///
/// Features:
/// - XML-style tags for tool calls
/// - Key-value pairs for arguments
/// - Support for multiple sequential tool calls
///
/// Close tags that appear literally inside argument values are kept by matching the
/// real closer against the next sibling open (`<arg_key>` / `<tool_call>`). An unescaped
/// open `<tool_call>` inside a value is not supported and may truncate the block.
pub struct Glm4MoeParser {
    format: GlmFormat,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Token configuration
    bot_token: &'static str,
    eot_token: &'static str,
}

impl Glm4MoeParser {
    fn new(format: GlmFormat) -> Self {
        Self {
            format,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
            bot_token: "<tool_call>",
            eot_token: "</tool_call>",
        }
    }

    /// Create a new GLM-4.5/4.6 MoE parser (with newline-based format)
    pub fn glm45() -> Self {
        Self::new(GlmFormat::Glm45)
    }

    /// Create a new GLM-4.7 MoE parser (with whitespace-based format)
    pub fn glm47() -> Self {
        Self::new(GlmFormat::Glm47)
    }

    /// Parse arguments, coercing each value by its declared schema type and
    /// falling back to [`infer_value`] when the type is unknown.
    fn parse_arguments(
        &self,
        args_text: &str,
        param_types: &HashMap<String, String>,
    ) -> serde_json::Map<String, Value> {
        let mut arguments = serde_json::Map::new();

        for (key, value_str) in parse_arg_key_value_pairs(args_text) {
            let value =
                helpers::coerce_by_schema_type(&value_str, param_types.get(&key).map(String::as_str))
                    .unwrap_or_else(|| infer_value(&value_str));

            arguments.insert(key, value);
        }

        arguments
    }

    fn split_tool_call_block<'a>(&self, block: &'a str) -> Option<(String, &'a str)> {
        let inner = block
            .strip_prefix(self.bot_token)?
            .strip_suffix(self.eot_token)?;
        match self.format {
            GlmFormat::Glm45 => {
                let nl = inner.find('\n')?;
                let name = inner[..nl].trim();
                if name.is_empty() {
                    return None;
                }
                Some((name.to_string(), inner[nl + 1..].trim_start()))
            }
            GlmFormat::Glm47 => {
                let trimmed = inner.trim();
                if trimmed.is_empty() {
                    return None;
                }
                if let Some(key_pos) = trimmed.find("<arg_key>") {
                    let name = trimmed[..key_pos].trim();
                    if name.is_empty() {
                        return None;
                    }
                    Some((name.to_string(), trimmed[key_pos..].trim_start()))
                } else {
                    // Parameterless call: whole body is the function name.
                    Some((trimmed.to_string(), ""))
                }
            }
        }
    }

    /// Parse a single tool call block
    fn parse_tool_call(&self, block: &str, tools: &[Tool]) -> ParserResult<Option<ToolCall>> {
        let Some((func_name, args_text)) = self.split_tool_call_block(block) else {
            return Ok(None);
        };

        let param_types = helpers::param_types_for_function(tools, &func_name);
        let arguments = self.parse_arguments(args_text, &param_types);

        let arguments_str = serde_json::to_string(&arguments)
            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        Ok(Some(ToolCall {
            function: FunctionCall {
                name: func_name,
                arguments: arguments_str,
            },
        }))
    }

    /// Parse all tool calls from text (shared logic for complete and incremental parsing)
    fn parse_tool_calls_from_text(&self, text: &str, tools: &[Tool]) -> Vec<ToolCall> {
        let mut parsed = Vec::new();

        for block in iter_tool_call_blocks(text) {
            match self.parse_tool_call(block, tools) {
                Ok(Some(tool)) => parsed.push(tool),
                Ok(None) => continue,
                Err(e) => {
                    tracing::debug!("Failed to parse tool call: {}", e);
                    continue;
                }
            }
        }

        parsed
    }
}

const ARG_KEY_OPEN: &str = "<arg_key>";
const ARG_KEY_CLOSE: &str = "</arg_key>";
const ARG_VAL_OPEN: &str = "<arg_value>";
const ARG_VAL_CLOSE: &str = "</arg_value>";

fn parse_arg_key_value_pairs(args_text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut cursor = args_text;

    while let Some(key_open_rel) = cursor.find(ARG_KEY_OPEN) {
        let after_key_open = &cursor[key_open_rel + ARG_KEY_OPEN.len()..];
        let Some(key_close_rel) = after_key_open.find(ARG_KEY_CLOSE) else {
            break;
        };
        let key = after_key_open[..key_close_rel].trim().to_string();
        cursor = &after_key_open[key_close_rel + ARG_KEY_CLOSE.len()..];

        let Some(val_open_rel) = cursor.find(ARG_VAL_OPEN) else {
            break;
        };
        let after_val_open = &cursor[val_open_rel + ARG_VAL_OPEN.len()..];
        let next_key = after_val_open
            .find(ARG_KEY_OPEN)
            .unwrap_or(after_val_open.len());
        let value_region = &after_val_open[..next_key];
        let Some(val_close_rel) = value_region.rfind(ARG_VAL_CLOSE) else {
            break;
        };
        let value = value_region[..val_close_rel].trim().to_string();
        pairs.push((key, value));
        cursor = &after_val_open[next_key..];
    }

    pairs
}

fn iter_tool_call_blocks<'a>(text: &'a str) -> Vec<&'a str> {
    first_tool_call_spans(text)
        .into_iter()
        .map(|(start, end)| &text[start..end])
        .collect()
}

fn first_tool_call_spans(text: &str) -> Vec<(usize, usize)> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut spans = Vec::new();
    let mut offset = 0usize;

    while let Some(open_rel) = text[offset..].find(OPEN) {
        let block_start = offset + open_rel;
        let inner_start = block_start + OPEN.len();
        let after_inner = &text[inner_start..];
        let next_open = after_inner.find(OPEN).unwrap_or(after_inner.len());
        let region = &after_inner[..next_open];
        let Some(close_rel) = region.rfind(CLOSE) else {
            break;
        };
        let block_end = inner_start + close_rel + CLOSE.len();
        spans.push((block_start, block_end));
        offset = block_end;
    }

    spans
}

impl Glm4MoeParser {
    /// Shared non-streaming parse, schema-aware when `tools` are provided.
    fn parse_complete_inner(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where tool calls begin
        // Safe: has_tool_markers() already confirmed the marker exists
        let idx = text
            .find("<tool_call>")
            .ok_or_else(|| ParserError::ParsingFailed("tool call marker not found".to_string()))?;
        let normal_text = text[..idx].to_string();

        let parsed = self.parse_tool_calls_from_text(text, tools);

        // If no tools were successfully parsed despite having markers, return entire text as fallback
        if parsed.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, parsed))
    }
}

/// Infer a JSON value from raw text when the schema type is unknown: JSON
/// (numbers/bools/null/objects/arrays), then Python-style literals, then string.
fn infer_value(value_str: &str) -> Value {
    if let Ok(json_val) = serde_json::from_str::<Value>(value_str) {
        return json_val;
    }
    match value_str {
        "true" | "True" => Value::Bool(true),
        "false" | "False" => Value::Bool(false),
        "null" | "None" => Value::Null,
        _ => {
            if let Ok(num) = value_str.parse::<i64>() {
                Value::Number(num.into())
            } else if let Ok(num) = value_str.parse::<f64>() {
                serde_json::Number::from_f64(num)
                    .map_or_else(|| Value::String(value_str.to_string()), Value::Number)
            } else {
                Value::String(value_str.to_string())
            }
        }
    }
}

impl Default for Glm4MoeParser {
    fn default() -> Self {
        Self::glm45()
    }
}

#[async_trait]
impl ToolParser for Glm4MoeParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_inner(text, &[])
    }

    async fn parse_complete_with_tools(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_inner(text, tools)
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        // Python logic: Wait for complete tool call, then parse it all at once
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if we have bot_token
        let start = current_text.find(self.bot_token);
        if start.is_none() {
            self.buffer.clear();
            // If we're in the middle of streaming (current_tool_id > 0), don't return text
            let normal_text = if self.current_tool_id > 0 {
                String::new()
            } else {
                current_text.clone()
            };
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // Wait for the first complete tool-call block (close matched against next open).
        let Some((block_start, block_end)) = first_tool_call_spans(current_text).into_iter().next()
        else {
            let Some(start_pos) = start else {
                return Ok(StreamingParseResult::default());
            };
            let normal_text = current_text[..start_pos].to_string();
            self.buffer = current_text[start_pos..].to_string();
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        };

        // Initialize state if this is the first tool call
        if self.current_tool_id == -1 {
            self.current_tool_id = 0;
            self.prev_tool_call_arr = Vec::new();
            self.streamed_args_for_tool = vec![String::new()];
        }

        // Ensure we have enough entries in our tracking arrays
        helpers::ensure_capacity(
            self.current_tool_id,
            &mut self.prev_tool_call_arr,
            &mut self.streamed_args_for_tool,
        );

        // Parse the complete block using shared helper
        let block = &current_text[block_start..block_end];
        let parsed_tools = self.parse_tool_calls_from_text(block, tools);

        // Extract normal text before tool calls
        let normal_text = current_text[..block_start].trim().to_string();

        // Build tool indices for validation
        let tool_indices = helpers::get_tool_indices(tools);

        let mut calls = Vec::new();

        if !parsed_tools.is_empty() {
            // Take the first tool and convert to ToolCallItem
            let tool_call = &parsed_tools[0];
            let tool_id = self.current_tool_id as usize;

            // Validate tool name
            if !tool_indices.contains_key(&tool_call.function.name) {
                // Invalid tool name - skip this tool, preserve indexing for next tool
                tracing::debug!("Invalid tool name '{}' - skipping", tool_call.function.name);
                helpers::reset_current_tool_state(
                    &mut self.buffer,
                    &mut false, // glm45_moe/glm47_moe doesn't track name_sent per tool
                    &mut self.streamed_args_for_tool,
                    &self.prev_tool_call_arr,
                );
                return Ok(StreamingParseResult::default());
            }

            calls.push(ToolCallItem {
                tool_index: tool_id,
                name: Some(tool_call.function.name.clone()),
                parameters: tool_call.function.arguments.clone(),
            });

            // Store in tracking arrays
            if self.prev_tool_call_arr.len() <= tool_id {
                self.prev_tool_call_arr
                    .resize_with(tool_id + 1, || Value::Null);
            }

            // Parse parameters as JSON and store
            if let Ok(args) = serde_json::from_str::<Value>(&tool_call.function.arguments) {
                self.prev_tool_call_arr[tool_id] = serde_json::json!({
                    "name": tool_call.function.name,
                    "arguments": args,
                });
            }

            if self.streamed_args_for_tool.len() <= tool_id {
                self.streamed_args_for_tool
                    .resize_with(tool_id + 1, String::new);
            }
            self.streamed_args_for_tool[tool_id].clone_from(&tool_call.function.arguments);

            self.current_tool_id += 1;
        }

        // Remove processed portion from buffer
        self.buffer = current_text[block_end..].to_string();
        Ok(StreamingParseResult { normal_text, calls })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.bot_token)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;

    use super::*;

    fn tool_with_props(props: Value) -> Vec<Tool> {
        vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": props}),
                strict: None,
            },
        }]
    }

    // String-typed params stay strings even when they look numeric/bool/array.
    #[tokio::test]
    async fn test_schema_aware_coercion_keeps_strings() {
        let tools = tool_with_props(serde_json::json!({
            "limit": {"type": "string"},
            "flag": {"type": "string"},
            "coords": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let text = "<tool_call>f\n\
            <arg_key>limit</arg_key>\n<arg_value>4</arg_value>\n\
            <arg_key>flag</arg_key>\n<arg_value>true</arg_value>\n\
            <arg_key>coords</arg_key>\n<arg_value>[60,30]</arg_value>\n\
            <arg_key>count</arg_key>\n<arg_value>5</arg_value>\n\
            </tool_call>";
        let (_, calls) = Glm4MoeParser::glm45()
            .parse_complete_with_tools(text, &tools)
            .await
            .unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["limit"], Value::String("4".to_string()));
        assert_eq!(args["flag"], Value::String("true".to_string()));
        assert_eq!(args["coords"], Value::String("[60,30]".to_string()));
        assert_eq!(args["count"], Value::Number(5.into()));
    }

    // The streaming path threads `tools` separately, so cover it too.
    #[tokio::test]
    async fn test_streaming_schema_aware_coercion() {
        let tools = tool_with_props(serde_json::json!({
            "limit": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let text = "<tool_call>f\n\
            <arg_key>limit</arg_key>\n<arg_value>4</arg_value>\n\
            <arg_key>count</arg_key>\n<arg_value>5</arg_value>\n\
            </tool_call>";
        let result = Glm4MoeParser::glm45()
            .parse_incremental(text, &tools)
            .await
            .unwrap();
        let args: Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
        assert_eq!(args["limit"], Value::String("4".to_string()));
        assert_eq!(args["count"], Value::Number(5.into()));
    }
}
