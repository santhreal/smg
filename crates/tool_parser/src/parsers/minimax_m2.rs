use std::{collections::HashMap, fmt::Write as FmtWrite};

use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::Value;

use crate::{
    errors::ParserResult,
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

const TOOL_CALL_OPEN: &str = "<minimax:tool_call>";
const TOOL_CALL_CLOSE: &str = "</minimax:tool_call>";
const INVOKE_CLOSE: &str = "</invoke>";
const PARAM_CLOSE: &str = "</parameter>";

/// MiniMax M2 format parser for tool calls
///
/// Handles the MiniMax M2 specific format:
/// `<minimax:tool_call><invoke name="func"><parameter name="key">value</parameter></invoke></minimax:tool_call>`
///
/// Features:
/// - Namespaced XML tags (`minimax:tool_call`)
/// - Function wrapped in `<invoke name="...">` tags
/// - Parameters as `<parameter name="key">value</parameter>`
/// - Incremental JSON streaming for parameters
///
/// Literal close tags inside parameter values are kept by matching each closer
/// against the next sibling open (`<parameter` / `<invoke` / wrapper end).
///
/// Reference: https://huggingface.co/MiniMaxAI/MiniMax-M2?chat_template=default
pub struct MinimaxM2Parser {
    // Streaming state
    buffer: String,
    prev_tool_call_arr: Vec<Value>,
    current_tool_id: i32,
    streamed_args_for_tool: Vec<String>,
    current_function_name: String,
    current_parameters: HashMap<String, Value>,
    in_tool_call: bool,
    function_name_sent: bool,

    // Token configuration
    tool_call_start_token: &'static str,
    tool_call_end_token: &'static str,
    invoke_end_token: &'static str,
}

impl MinimaxM2Parser {
    /// Parse a value from string with consistent logic
    #[inline]
    fn parse_value(text: &str) -> Value {
        // Try parsing as common literals first
        match text {
            "true" | "True" => return Value::Bool(true),
            "false" | "False" => return Value::Bool(false),
            "null" | "None" => return Value::Null,
            _ => {}
        }

        // Try parsing as number
        if let Ok(num) = text.parse::<i64>() {
            return Value::Number(num.into());
        }

        if let Ok(num) = text.parse::<f64>() {
            if let Some(n) = serde_json::Number::from_f64(num) {
                return Value::Number(n);
            }
        }

        // Default to string
        Value::String(text.to_string())
    }

    /// Create a new MiniMax M2 parser
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
            current_function_name: String::new(),
            current_parameters: HashMap::new(),
            in_tool_call: false,
            function_name_sent: false,
            tool_call_start_token: TOOL_CALL_OPEN,
            tool_call_end_token: TOOL_CALL_CLOSE,
            invoke_end_token: INVOKE_CLOSE,
        }
    }

    /// Parse parameter tags, coercing each value by its declared schema type when
    /// known (so a numeric-looking `string` stays a string), else inferring.
    fn parse_parameters(
        params_text: &str,
        param_types: &HashMap<String, String>,
    ) -> serde_json::Map<String, Value> {
        let mut parameters = serde_json::Map::new();

        for (key, value_str) in parse_parameter_pairs(params_text) {
            let decoded_value = Self::decode_xml_entities(&value_str);
            let value = helpers::coerce_by_schema_type(
                &decoded_value,
                param_types.get(&key).map(String::as_str),
            )
            .unwrap_or_else(|| Self::parse_value(&decoded_value));

            parameters.insert(key, value);
        }

        parameters
    }

    /// Decode common XML entities
    fn decode_xml_entities(text: &str) -> String {
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
    }

    /// Parse every `<invoke>` block in a tool-call wrapper into tool calls.
    ///
    /// MiniMax M2 emits parallel calls as multiple `<invoke>` blocks inside a single
    /// `<minimax:tool_call>` wrapper (Anthropic-style), so we iterate all matches —
    /// a single capture would drop every call after the first.
    fn parse_tool_call(block: &str, tools: &[Tool]) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        for (func_name, params_text) in iter_invoke_blocks(block) {
            let param_types = helpers::param_types_for_function(tools, &func_name);
            let parameters = Self::parse_parameters(params_text, &param_types);

            match serde_json::to_string(&parameters) {
                Ok(arguments_str) => calls.push(ToolCall {
                    function: FunctionCall {
                        name: func_name,
                        arguments: arguments_str,
                    },
                }),
                Err(e) => tracing::debug!("Failed to serialize tool call arguments: {}", e),
            }
        }
        calls
    }

    /// Parse all tool calls from text and return first valid position
    fn parse_tool_calls_from_text(text: &str, tools: &[Tool]) -> (Vec<ToolCall>, Option<usize>) {
        let mut tool_calls = Vec::new();
        let mut first_valid_pos = None;

        for (start, block) in iter_tool_call_blocks(text) {
            let calls = Self::parse_tool_call(block, tools);
            if !calls.is_empty() && first_valid_pos.is_none() {
                first_valid_pos = Some(start);
            }
            tool_calls.extend(calls);
        }

        (tool_calls, first_valid_pos)
    }

    /// Shared non-streaming parse; `tools` empty means infer types from text.
    fn parse_complete_inner(&self, text: &str, tools: &[Tool]) -> (String, Vec<ToolCall>) {
        if !self.has_tool_markers(text) {
            return (text.to_string(), vec![]);
        }
        let (tool_calls, first_valid_tool_pos) = Self::parse_tool_calls_from_text(text, tools);
        if tool_calls.is_empty() {
            return (text.to_string(), vec![]);
        }
        let normal_text = match first_valid_tool_pos {
            Some(pos) => text[..pos].to_string(),
            None => text.to_string(),
        };
        (normal_text, tool_calls)
    }

    /// Parse and stream parameters incrementally
    fn parse_and_stream_parameters(&mut self, text: &str, tools: &[Tool]) -> Vec<ToolCallItem> {
        let mut calls = Vec::new();
        let param_types = helpers::param_types_for_function(tools, &self.current_function_name);

        let param_matches: Vec<_> = parse_parameter_pairs(text)
            .into_iter()
            .map(|(name, value_str)| {
                let decoded = Self::decode_xml_entities(&value_str);

                let value = helpers::coerce_by_schema_type(
                    &decoded,
                    param_types.get(&name).map(String::as_str),
                )
                .unwrap_or_else(|| {
                    if decoded.starts_with('{') || decoded.starts_with('[') {
                        serde_json::from_str::<Value>(&decoded)
                            .unwrap_or_else(|_| Self::parse_value(&decoded))
                    } else {
                        Self::parse_value(&decoded)
                    }
                });

                (name, value)
            })
            .collect();

        // Build new parameters map
        let mut new_params = HashMap::new();
        for (name, value) in param_matches {
            new_params.insert(name, value);
        }

        // If we have new parameters that weren't in current_parameters, stream them
        if !new_params.is_empty() && new_params != self.current_parameters {
            let tool_id = self.current_tool_id as usize;

            // Ensure we have enough capacity
            while self.streamed_args_for_tool.len() <= tool_id {
                self.streamed_args_for_tool.push(String::new());
            }

            // Build incremental JSON with single allocation
            if self.current_parameters.is_empty() {
                // First parameters - start JSON object but don't close it
                let mut json_fragment = String::with_capacity(256);
                json_fragment.push('{');

                let mut first = true;
                for (key, value) in &new_params {
                    if !first {
                        json_fragment.push_str(", ");
                    }
                    // serde_json::to_string for String/Value is infallible; write! to String is infallible
                    let key_json = serde_json::to_string(key).unwrap_or_default();
                    let value_json = serde_json::to_string(value).unwrap_or_default();
                    let _ = write!(&mut json_fragment, "{key_json}: {value_json}");
                    first = false;
                }

                calls.push(ToolCallItem {
                    tool_index: tool_id,
                    name: None,
                    parameters: json_fragment.clone(),
                });

                self.streamed_args_for_tool[tool_id] = json_fragment;
            } else {
                // Additional parameters - add them incrementally
                let new_keys: Vec<_> = new_params
                    .keys()
                    .filter(|k| !self.current_parameters.contains_key(*k))
                    .collect();

                if !new_keys.is_empty() {
                    let mut json_fragment = String::with_capacity(128);

                    for key in new_keys {
                        let value = &new_params[key];
                        // serde_json::to_string for String/Value is infallible; write! to String is infallible
                        let key_json = serde_json::to_string(key).unwrap_or_default();
                        let value_json = serde_json::to_string(value).unwrap_or_default();
                        let _ = write!(&mut json_fragment, ", {key_json}: {value_json}");
                    }

                    calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: None,
                        parameters: json_fragment.clone(),
                    });

                    self.streamed_args_for_tool[tool_id].push_str(&json_fragment);
                }
            }

            // Update current parameters
            self.current_parameters = new_params;

            // Update prev_tool_call_arr
            while self.prev_tool_call_arr.len() <= tool_id {
                self.prev_tool_call_arr.push(Value::Null);
            }
            self.prev_tool_call_arr[tool_id] = serde_json::json!({
                "name": self.current_function_name,
                "arguments": self.current_parameters,
            });
        }

        calls
    }
}

/// Find `<tag ... name="...">` and return (name, absolute end index past `>`).
fn find_named_open(s: &str, tag: &str) -> Option<(String, usize)> {
    let open = format!("<{tag}");
    let mut search = 0;
    while let Some(rel) = s[search..].find(&open) {
        let abs = search + rel;
        let after_tag = &s[abs + open.len()..];
        if after_tag
            .chars()
            .next()
            .is_some_and(|c| !c.is_whitespace() && c != '>')
        {
            search = abs + 1;
            continue;
        }
        // Only inspect attributes inside this open tag (before `>`).
        let Some(gt_rel) = after_tag.find('>') else {
            search = abs + 1;
            continue;
        };
        let attrs = &after_tag[..gt_rel];
        let Some(name_rel) = attrs.find("name=\"") else {
            search = abs + 1;
            continue;
        };
        let name_start = name_rel + "name=\"".len();
        let name_rest = &attrs[name_start..];
        let Some(name_end) = name_rest.find('"') else {
            search = abs + 1;
            continue;
        };
        let name = name_rest[..name_end].to_string();
        let end = abs + open.len() + gt_rel + 1;
        return Some((name, end));
    }
    None
}

fn parse_parameter_pairs(params_text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut cursor = params_text;

    while let Some((key, value_start)) = find_named_open(cursor, "parameter") {
        let after_open = &cursor[value_start..];
        let next_param = find_parameter_open_offset(after_open).unwrap_or(after_open.len());
        let region = &after_open[..next_param];
        let Some(close_rel) = region.rfind(PARAM_CLOSE) else {
            break;
        };
        let value = region[..close_rel].to_string();
        pairs.push((key, value));
        cursor = &after_open[close_rel + PARAM_CLOSE.len()..];
    }

    pairs
}

fn find_parameter_open_offset(s: &str) -> Option<usize> {
    let open = "<parameter";
    let mut search = 0;
    while let Some(rel) = s[search..].find(open) {
        let abs = search + rel;
        if find_named_open(&s[abs..], "parameter").is_some() {
            return Some(abs);
        }
        search = abs + 1;
    }
    None
}

fn parameters_fully_closed(params_text: &str) -> bool {
    let mut cursor = params_text;
    while let Some((_key, value_start)) = find_named_open(cursor, "parameter") {
        let after_open = &cursor[value_start..];
        let next_param = find_parameter_open_offset(after_open).unwrap_or(after_open.len());
        let region = &after_open[..next_param];
        let Some(close_rel) = region.rfind(PARAM_CLOSE) else {
            return false;
        };
        cursor = &after_open[close_rel + PARAM_CLOSE.len()..];
    }
    find_parameter_open_offset(cursor).is_none()
}

fn find_structural_close(haystack: &str, close: &str, params_prefix_ok: impl Fn(&str) -> bool) -> Option<usize> {
    let mut from = 0;
    while let Some(rel) = haystack[from..].find(close) {
        let abs = from + rel;
        if params_prefix_ok(&haystack[..abs]) {
            return Some(abs);
        }
        from = abs + close.len();
    }
    None
}

fn iter_invoke_blocks(block: &str) -> Vec<(String, &str)> {
    let mut out = Vec::new();
    let mut cursor = block;

    while let Some((name, value_start)) = find_named_open(cursor, "invoke") {
        let after_open = &cursor[value_start..];
        let next_invoke = find_invoke_open_offset(after_open).unwrap_or(after_open.len());
        let region = &after_open[..next_invoke];
        let Some(close_rel) = find_structural_close(region, INVOKE_CLOSE, parameters_fully_closed)
        else {
            break;
        };
        let params_text = &region[..close_rel];
        out.push((name, params_text));
        cursor = &after_open[close_rel + INVOKE_CLOSE.len()..];
    }

    out
}

fn find_invoke_open_offset(s: &str) -> Option<usize> {
    let open = "<invoke";
    let mut search = 0;
    while let Some(rel) = s[search..].find(open) {
        let abs = search + rel;
        if find_named_open(&s[abs..], "invoke").is_some() {
            return Some(abs);
        }
        search = abs + 1;
    }
    None
}

fn iter_tool_call_blocks(text: &str) -> Vec<(usize, &str)> {
    let mut blocks = Vec::new();
    let mut abs_base = 0;
    let mut cursor = text;

    while let Some(open_rel) = cursor.find(TOOL_CALL_OPEN) {
        let block_start = abs_base + open_rel;
        let after_open = &cursor[open_rel + TOOL_CALL_OPEN.len()..];
        let next_open = after_open.find(TOOL_CALL_OPEN).unwrap_or(after_open.len());
        let region = &after_open[..next_open];
        let Some(close_rel) = region.rfind(TOOL_CALL_CLOSE) else {
            break;
        };
        let block_end_in_cursor = open_rel + TOOL_CALL_OPEN.len() + close_rel + TOOL_CALL_CLOSE.len();
        blocks.push((block_start, &cursor[open_rel..block_end_in_cursor]));
        abs_base += block_end_in_cursor;
        cursor = &cursor[block_end_in_cursor..];
    }

    blocks
}

fn try_parse_invoke_open(buffer: &str) -> Option<(String, usize)> {
    find_named_open(buffer, "invoke")
}

fn find_structural_invoke_end(buffer: &str) -> Option<usize> {
    find_structural_close(buffer, INVOKE_CLOSE, parameters_fully_closed)
}

impl Default for MinimaxM2Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for MinimaxM2Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(self.parse_complete_inner(text, &[]))
    }

    async fn parse_complete_with_tools(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(self.parse_complete_inner(text, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let mut normal_text = String::new();
        let mut calls = Vec::new();

        loop {
            // If we're not in a tool call and don't see a start token, return normal text
            if !self.in_tool_call && !self.buffer.contains(self.tool_call_start_token) {
                // Check if buffer might contain a partial start token at the end
                if let Some(partial_len) =
                    helpers::ends_with_partial_token(&self.buffer, self.tool_call_start_token)
                {
                    // Return everything except the potential partial token
                    let end = self.buffer.len() - partial_len;
                    normal_text = self.buffer[..end].to_string();
                    self.buffer = self.buffer[end..].to_string();
                } else {
                    // No partial token, return all as normal text
                    normal_text.clone_from(&self.buffer);
                    self.buffer.clear();
                }
                break;
            }

            // Look for tool call start
            if !self.in_tool_call {
                if let Some(start) = self.buffer.find(self.tool_call_start_token) {
                    normal_text = self.buffer[..start].to_string();
                    self.buffer =
                        self.buffer[start + self.tool_call_start_token.len()..].to_string();

                    self.in_tool_call = true;
                    self.function_name_sent = false;
                    self.current_function_name.clear();
                    self.current_parameters.clear();

                    continue;
                } else {
                    // No start token found
                    break;
                }
            }

            // We're in a tool call, try to parse function name if not sent yet
            if !self.function_name_sent {
                // Between invokes in a wrapper: if the wrapper-end tag arrives before
                // the next <invoke>, the wrapper is finished — consume it and exit.
                if let Some(end_pos) = self.buffer.find(self.tool_call_end_token) {
                    let next_invoke = find_invoke_open_offset(&self.buffer);
                    if next_invoke.is_none_or(|i| end_pos < i) {
                        self.buffer =
                            self.buffer[end_pos + self.tool_call_end_token.len()..].to_string();
                        self.in_tool_call = false;
                        self.current_function_name.clear();
                        self.current_parameters.clear();
                        continue;
                    }
                }

                if let Some((function_name, after_open)) = try_parse_invoke_open(&self.buffer) {
                    // Forward unknown tool names too — emit a tool_call rather than
                    // leaking the <invoke> markup into assistant text.
                    self.current_function_name.clone_from(&function_name);
                    self.function_name_sent = true;

                    // Initialize tool call tracking
                    if self.current_tool_id == -1 {
                        self.current_tool_id = 0;
                    }

                    // Ensure tracking arrays are large enough
                    helpers::ensure_capacity(
                        self.current_tool_id,
                        &mut self.prev_tool_call_arr,
                        &mut self.streamed_args_for_tool,
                    );

                    // Send tool name with empty parameters
                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: Some(function_name),
                        parameters: String::new(),
                    });

                    self.buffer = self.buffer[after_open..].to_string();
                    continue;
                }
                // No complete invoke open found yet, wait for more text
                break;
            }

            // Parse parameters incrementally
            if self.function_name_sent {
                // Process parameters and get any calls to emit
                let buffer_copy = self.buffer.clone(); // TODO: avoid cloning the buffer
                let parameter_calls = self.parse_and_stream_parameters(&buffer_copy, tools);
                calls.extend(parameter_calls);

                // Check if tool call is complete (real </invoke>, not one inside a value)
                if let Some(invoke_end) = find_structural_invoke_end(&self.buffer) {
                    // Add closing brace to complete the JSON object
                    let tool_id = self.current_tool_id as usize;
                    if tool_id < self.streamed_args_for_tool.len() {
                        let current_streamed = &self.streamed_args_for_tool[tool_id];
                        if !current_streamed.is_empty() && !current_streamed.ends_with('}') {
                            // Count opening and closing braces to check if JSON is complete
                            let open_braces = current_streamed.matches('{').count();
                            let close_braces = current_streamed.matches('}').count();
                            if open_braces > close_braces {
                                calls.push(ToolCallItem {
                                    tool_index: tool_id,
                                    name: None,
                                    parameters: "}".to_string(),
                                });
                                self.streamed_args_for_tool[tool_id].push('}');
                            }
                        }
                    }

                    // Move buffer past the </invoke>
                    self.buffer =
                        self.buffer[invoke_end + self.invoke_end_token.len()..].to_string();

                    // This invoke is done. Reset per-invoke state and advance the
                    // tool index; the next loop iteration either parses another
                    // <invoke> in the same wrapper or closes the wrapper (handled in
                    // the !function_name_sent branch). MiniMax M2 packs parallel calls
                    // as multiple <invoke> blocks in one <minimax:tool_call> wrapper.
                    self.function_name_sent = false;
                    self.current_function_name.clear();
                    self.current_parameters.clear();
                    self.current_tool_id += 1;
                    continue;
                }
                // Tool call not complete yet, wait for more text
                break;
            }
        }

        Ok(StreamingParseResult { normal_text, calls })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.tool_call_start_token)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
        self.current_function_name.clear();
        self.current_parameters.clear();
        self.in_tool_call = false;
        self.function_name_sent = false;
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn parameter_pairs_keep_literal_close_tag() {
        let pairs = parse_parameter_pairs(
            r#"<parameter name="content">use </parameter> carefully</parameter>"#,
        );
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "content");
        assert_eq!(pairs[0].1, "use </parameter> carefully");
    }

    #[test]
    fn invoke_keeps_literal_invoke_close_in_value() {
        let blocks = iter_invoke_blocks(
            r#"<invoke name="write"><parameter name="content">text with </invoke> inside</parameter></invoke>"#,
        );
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "write");
        let pairs = parse_parameter_pairs(blocks[0].1);
        assert_eq!(pairs[0].1, "text with </invoke> inside");
    }
}
