use async_trait::async_trait;
use futures::StreamExt;
use tracing::{debug, trace, warn};

use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::metrics::LlmTokenUsage;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};

use crate::function_call::{FunctionCallParams, FunctionCallRegistry};
use crate::llm::{LLMService, LLMServiceState, llm_process_frame};

use super::settings::OpenAILLMSettings;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI LLM service implementation.
///
/// Connects to the OpenAI Chat Completions API (or compatible endpoints)
/// via streaming SSE. Handles text generation and tool calls.
#[derive(Debug)]
pub struct OpenAILLMService {
    state: LLMServiceState,
    api_key: String,
    base_url: String,
    openai_settings: OpenAILLMSettings,
    client: reqwest::Client,
}

impl OpenAILLMService {
    pub fn new(api_key: impl Into<String>, settings: OpenAILLMSettings) -> Self {
        let base_settings = settings.base.clone();
        Self {
            state: LLMServiceState::new("OpenAILLMService", base_settings),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            openai_settings: settings,
            client: reqwest::Client::new(),
        }
    }

    /// Set a custom base URL (for Azure, local models, etc.).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Get a mutable reference to the function call registry.
    pub fn function_registry_mut(&mut self) -> &mut FunctionCallRegistry {
        &mut self.state.function_registry
    }

    /// Build the request body for the Chat Completions API.
    fn build_request_body(&self, context: &serde_json::Value) -> serde_json::Value {
        let settings = &self.openai_settings;
        let base = &settings.base;

        let mut body = serde_json::json!({
            "stream": true,
            "stream_options": { "include_usage": true },
        });

        if let Some(ref model) = base.model {
            body["model"] = serde_json::json!(model);
        }

        // Messages from context
        if let Some(messages) = context.get("messages") {
            let mut msgs = Vec::new();

            // Prepend system instruction if set
            if let Some(ref instruction) = base.system_instruction {
                msgs.push(serde_json::json!({
                    "role": "system",
                    "content": instruction,
                }));
            }

            if let Some(arr) = messages.as_array() {
                msgs.extend(arr.iter().cloned());
            }
            body["messages"] = serde_json::json!(msgs);
        }

        // Tools and tool_choice from context
        if let Some(tools) = context.get("tools")
            && tools.is_array()
            && !tools.as_array().unwrap().is_empty()
        {
            body["tools"] = tools.clone();
        }
        if let Some(tool_choice) = context.get("tool_choice") {
            body["tool_choice"] = tool_choice.clone();
        }

        // Optional parameters
        if let Some(temp) = base.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(max) = base.max_tokens {
            body["max_tokens"] = serde_json::json!(max);
        }
        if let Some(max) = settings.max_completion_tokens {
            body["max_completion_tokens"] = serde_json::json!(max);
        }
        if let Some(top_p) = base.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(fp) = base.frequency_penalty {
            body["frequency_penalty"] = serde_json::json!(fp);
        }
        if let Some(pp) = base.presence_penalty {
            body["presence_penalty"] = serde_json::json!(pp);
        }
        if let Some(seed) = base.seed {
            body["seed"] = serde_json::json!(seed);
        }

        // Extra settings override everything
        for (k, v) in &settings.extra {
            body[k] = v.clone();
        }

        body
    }

    /// Process an LLMContext frame: stream completions and handle tool calls.
    async fn process_context(
        &mut self,
        context_value: serde_json::Value,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        // Push LLMFullResponseStartFrame
        let skip_tts = self.state.skip_tts;
        ctx.send_downstream(Frame::LLMFullResponseStart(LLMFullResponseStartFrame {
            skip_tts,
        }))
        .await?;

        // Start metrics
        let metrics_enabled = self.state.base.metrics_enabled();
        let usage_enabled = self.state.base.usage_metrics_enabled();
        if metrics_enabled {
            self.state.base.metrics.start_ttfb();
            self.state.base.metrics.start_processing();
        }

        let body = self.build_request_body(&context_value);
        let url = format!("{}/chat/completions", self.base_url);

        debug!("OpenAI request to {}", url);
        trace!("OpenAI request body: {}", body);

        let result = self
            .stream_completions(
                &url,
                &body,
                &context_value,
                ctx,
                metrics_enabled,
                usage_enabled,
            )
            .await;

        // Always push end frame, even on error
        if let Err(ref e) = result {
            warn!("OpenAI streaming error: {}", e);
            ctx.push_error(&format!("OpenAI error: {e}"), false).await?;
        }

        // Stop processing metrics
        if metrics_enabled {
            ctx.push_processing_metrics(&mut self.state.base.metrics)
                .await?;
        }

        ctx.send_downstream(Frame::LLMFullResponseEnd(LLMFullResponseEndFrame {
            skip_tts,
        }))
        .await?;

        Ok(())
    }

    /// Stream SSE completions from the API and handle text + tool calls.
    async fn stream_completions(
        &mut self,
        url: &str,
        body: &serde_json::Value,
        context_value: &serde_json::Value,
        ctx: &ProcessorContext,
        metrics_enabled: bool,
        usage_enabled: bool,
    ) -> Result<()> {
        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| PipecatError::ProcessorError(format!("OpenAI request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(PipecatError::ProcessorError(format!(
                "OpenAI API error {status}: {text}"
            )));
        }

        // Tool call accumulation state
        let mut tool_names: Vec<String> = Vec::new();
        let mut tool_arguments: Vec<String> = Vec::new();
        let mut tool_ids: Vec<String> = Vec::new();
        let mut current_name = String::new();
        let mut current_args = String::new();
        let mut current_id = String::new();
        let mut current_idx: i64 = -1;
        let mut ttfb_stopped = false;

        let mut stream = response.bytes_stream();
        let mut line_buf = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| PipecatError::ProcessorError(format!("SSE read error: {e}")))?;

            line_buf.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines
            while let Some(newline_pos) = line_buf.find('\n') {
                let line = line_buf[..newline_pos].trim().to_string();
                line_buf = line_buf[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                if line == "data: [DONE]" {
                    break;
                }

                let Some(json_str) = line.strip_prefix("data: ") else {
                    continue;
                };

                let Ok(data) = serde_json::from_str::<serde_json::Value>(json_str) else {
                    warn!("Failed to parse SSE JSON: {}", json_str);
                    continue;
                };

                // Handle usage (comes in final chunk)
                if let Some(usage) = data.get("usage")
                    && usage.is_object()
                    && usage.get("total_tokens").is_some()
                    && usage_enabled
                {
                    let token_usage = parse_usage(usage);
                    ctx.push_llm_usage(&self.state.base.metrics, token_usage)
                        .await?;
                }

                let Some(choices) = data.get("choices").and_then(|c| c.as_array()) else {
                    continue;
                };
                if choices.is_empty() {
                    continue;
                }

                // Stop TTFB on first choice
                if metrics_enabled && !ttfb_stopped {
                    ctx.push_ttfb(&mut self.state.base.metrics).await?;
                    ttfb_stopped = true;
                }

                let Some(delta) = choices[0].get("delta") else {
                    continue;
                };

                // Handle tool calls
                if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                    for tc in tool_calls {
                        let idx = tc.get("index").and_then(|i| i.as_i64()).unwrap_or(0);

                        // New tool call index → save previous
                        if idx != current_idx && current_idx >= 0 {
                            tool_names.push(std::mem::take(&mut current_name));
                            tool_arguments.push(std::mem::take(&mut current_args));
                            tool_ids.push(std::mem::take(&mut current_id));
                        }
                        current_idx = idx;

                        if let Some(func) = tc.get("function") {
                            if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                                current_name.push_str(name);
                            }
                            if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                                current_args.push_str(args);
                            }
                        }
                        if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                            current_id = id.to_string();
                        }
                    }
                }
                // Handle text content — LLMText, not Text, so downstream
                // aggregators can distinguish LLM-generated text from other sources.
                else if let Some(content) = delta.get("content").and_then(|c| c.as_str())
                    && !content.is_empty()
                {
                    ctx.send_downstream(Frame::LLMText(TextFrame {
                        text: content.to_string(),
                        skip_tts: self.state.skip_tts,
                        includes_inter_frame_spaces: false,
                        append_to_context: true,
                    }))
                    .await?;
                }
            }
        }

        // Finalize tool calls
        if current_idx >= 0 && !current_name.is_empty() {
            tool_names.push(current_name);
            tool_arguments.push(current_args);
            tool_ids.push(current_id);
        }

        // Execute function calls if any
        if !tool_names.is_empty() {
            self.run_function_calls(&tool_names, &tool_arguments, &tool_ids, context_value, ctx)
                .await?;
        }

        Ok(())
    }

    /// Execute accumulated tool calls via the function registry.
    async fn run_function_calls(
        &self,
        names: &[String],
        arguments: &[String],
        ids: &[String],
        context_value: &serde_json::Value,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let function_calls: Vec<FunctionCallFromLLM> = names
            .iter()
            .zip(ids.iter())
            .zip(arguments.iter())
            .map(|((name, id), args)| FunctionCallFromLLM {
                function_name: name.clone(),
                tool_call_id: id.clone(),
                arguments: serde_json::from_str(args).unwrap_or(serde_json::json!({})),
            })
            .collect();

        ctx.broadcast(Frame::FunctionCallsStarted(FunctionCallsStartedFrame {
            function_calls: function_calls.clone(),
        }))
        .await?;

        // Build a temporary LLMContext from the context value for the function call handler
        let messages = context_value
            .get("messages")
            .cloned()
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let llm_context = pipecat_context::LLMContext::new(messages);
        if let Some(tools) = context_value.get("tools").cloned() {
            llm_context.set_tools(tools);
        }
        if let Some(tool_choice) = context_value.get("tool_choice").cloned() {
            llm_context.set_tool_choice(tool_choice);
        }

        for ((name, args_str), id) in names.iter().zip(arguments.iter()).zip(ids.iter()) {
            let parsed_args: serde_json::Value =
                serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));

            let registry_item = self.state.function_registry.lookup(name);

            ctx.broadcast(Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
                function_name: name.clone(),
                tool_call_id: id.clone(),
                arguments: parsed_args.clone(),
                cancel_on_interruption: registry_item
                    .map(|item| item.cancel_on_interruption)
                    .unwrap_or(false),
            }))
            .await?;

            let result_value = if let Some(item) = registry_item {
                let params = FunctionCallParams {
                    function_name: name.clone(),
                    tool_call_id: id.clone(),
                    arguments: parsed_args.clone(),
                    context: llm_context.clone(),
                };
                match tokio::time::timeout(item.timeout, (item.handler)(params)).await {
                    Ok(result) => result,
                    Err(_) => {
                        warn!(
                            "Function call [{}:{}] timed out after {:?}",
                            name, id, item.timeout
                        );
                        serde_json::Value::Null
                    }
                }
            } else {
                warn!("No handler registered for function: {}", name);
                serde_json::json!({"error": format!("No handler for {}", name)})
            };

            ctx.broadcast(Frame::FunctionCallResult(FunctionCallResultFrame {
                function_name: name.clone(),
                tool_call_id: id.clone(),
                arguments: parsed_args,
                result: result_value,
                run_llm: Some(true),
                properties: Some(FunctionCallResultProperties {
                    run_llm: Some(true),
                }),
            }))
            .await?;
        }

        Ok(())
    }
}

#[async_trait]
impl FrameProcessor for OpenAILLMService {
    fn name(&self) -> &str {
        self.state.base.processor.name()
    }

    fn id(&self) -> u64 {
        self.state.base.processor.id()
    }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        match &envelope.frame {
            Frame::LLMContext(f) => {
                let context_value = f.context.clone();
                self.process_context(context_value, ctx).await?;
            }
            _ => {
                llm_process_frame(self, envelope, direction, ctx).await?;
            }
        }
        Ok(())
    }
}

impl LLMService for OpenAILLMService {
    fn llm_service_state(&self) -> &LLMServiceState {
        &self.state
    }

    fn llm_service_state_mut(&mut self) -> &mut LLMServiceState {
        &mut self.state
    }
}

/// Parse a usage JSON object into LlmTokenUsage.
fn parse_usage(usage: &serde_json::Value) -> LlmTokenUsage {
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let total_tokens = usage
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let cache_read = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let reasoning = usage
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    LlmTokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cache_read_input_tokens: cache_read,
        cache_creation_input_tokens: None,
        reasoning_tokens: reasoning,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn parse_usage_full() {
        let usage = json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": { "cached_tokens": 20 },
            "completion_tokens_details": { "reasoning_tokens": 10 }
        });

        let result = parse_usage(&usage);
        assert_eq!(result.prompt_tokens, 100);
        assert_eq!(result.completion_tokens, 50);
        assert_eq!(result.total_tokens, 150);
        assert_eq!(result.cache_read_input_tokens, Some(20));
        assert_eq!(result.reasoning_tokens, Some(10));
    }

    #[test]
    fn parse_usage_minimal() {
        let usage = json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        });

        let result = parse_usage(&usage);
        assert_eq!(result.prompt_tokens, 10);
        assert_eq!(result.completion_tokens, 5);
        assert_eq!(result.total_tokens, 15);
        assert_eq!(result.cache_read_input_tokens, None);
        assert_eq!(result.reasoning_tokens, None);
    }

    #[test]
    fn build_request_body_basic() {
        let settings = OpenAILLMSettings::new(crate::settings::LLMSettings {
            model: Some("gpt-4".into()),
            temperature: Some(0.7),
            ..Default::default()
        });

        let svc = OpenAILLMService::new("test-key", settings);

        let context = json!({
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let body = svc.build_request_body(&context);
        assert_eq!(body["model"], "gpt-4");
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn build_request_body_with_system_instruction() {
        let settings = OpenAILLMSettings::new(crate::settings::LLMSettings {
            model: Some("gpt-4".into()),
            system_instruction: Some("You are helpful.".into()),
            ..Default::default()
        });

        let svc = OpenAILLMService::new("test-key", settings);
        let context = json!({
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let body = svc.build_request_body(&context);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn build_request_body_with_tools() {
        let settings = OpenAILLMSettings::new(crate::settings::LLMSettings {
            model: Some("gpt-4".into()),
            ..Default::default()
        });

        let svc = OpenAILLMService::new("test-key", settings);
        let context = json!({
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_choice": "auto"
        });

        let body = svc.build_request_body(&context);
        assert!(body["tools"].is_array());
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn build_request_body_extra_overrides() {
        let mut settings = OpenAILLMSettings::new(crate::settings::LLMSettings {
            model: Some("gpt-4".into()),
            temperature: Some(0.5),
            ..Default::default()
        });
        settings.extra.insert("temperature".into(), json!(0.9));
        settings.extra.insert("custom_field".into(), json!("val"));

        let svc = OpenAILLMService::new("test-key", settings);
        let body = svc.build_request_body(&json!({"messages": []}));

        assert_eq!(body["temperature"], 0.9);
        assert_eq!(body["custom_field"], "val");
    }

    #[test]
    fn tool_call_accumulation_across_chunks() {
        let chunks = vec![
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_abc",
                "function": {"name": "get_weather", "arguments": "{\"loc"}
            }]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"arguments": "ation\": \"NYC\"}"}
            }]}}]}),
        ];

        let (names, args, ids) = accumulate_tool_calls(&chunks);
        assert_eq!(names, vec!["get_weather"]);
        assert_eq!(ids, vec!["call_abc"]);
        let parsed: serde_json::Value = serde_json::from_str(&args[0]).unwrap();
        assert_eq!(parsed["location"], "NYC");
    }

    #[test]
    fn tool_call_accumulation_multiple_calls() {
        let chunks = vec![
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}
            }]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 1, "id": "call_2",
                "function": {"name": "get_time", "arguments": "{\"tz\": \"EST\"}"}
            }]}}]}),
        ];

        let (names, _args, ids) = accumulate_tool_calls(&chunks);
        assert_eq!(names, vec!["get_weather", "get_time"]);
        assert_eq!(ids, vec!["call_1", "call_2"]);
    }

    /// Test helper: run the tool call accumulation logic on canned SSE data.
    fn accumulate_tool_calls(
        chunks: &[serde_json::Value],
    ) -> (Vec<String>, Vec<String>, Vec<String>) {
        let mut tool_names = Vec::new();
        let mut tool_args = Vec::new();
        let mut tool_ids = Vec::new();
        let mut current_name = String::new();
        let mut current_args = String::new();
        let mut current_id = String::new();
        let mut current_idx: i64 = -1;

        for chunk in chunks {
            let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) else {
                continue;
            };
            let Some(delta) = choices[0].get("delta") else {
                continue;
            };
            let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) else {
                continue;
            };
            for tc in tool_calls {
                let idx = tc.get("index").and_then(|i| i.as_i64()).unwrap_or(0);
                if idx != current_idx && current_idx >= 0 {
                    tool_names.push(std::mem::take(&mut current_name));
                    tool_args.push(std::mem::take(&mut current_args));
                    tool_ids.push(std::mem::take(&mut current_id));
                }
                current_idx = idx;
                if let Some(func) = tc.get("function") {
                    if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                        current_name.push_str(name);
                    }
                    if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                        current_args.push_str(args);
                    }
                }
                if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                    current_id = id.to_string();
                }
            }
        }

        if current_idx >= 0 && !current_name.is_empty() {
            tool_names.push(current_name);
            tool_args.push(current_args);
            tool_ids.push(current_id);
        }

        (tool_names, tool_args, tool_ids)
    }

    #[tokio::test]
    async fn process_frame_forwards_non_llm_frames() {
        let settings = OpenAILLMSettings::new(crate::settings::LLMSettings {
            model: Some("gpt-4".into()),
            ..Default::default()
        });
        let mut svc = OpenAILLMService::new("test-key", settings);

        let (downstream, _upstream) = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(Frame::Text(TextFrame::new("hello")), Direction::Downstream)],
        )
        .await;

        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::Text(t) if t.text == "hello"));
    }

    #[tokio::test]
    async fn interruption_resets_metrics() {
        let settings = OpenAILLMSettings::new(crate::settings::LLMSettings {
            model: Some("gpt-4".into()),
            ..Default::default()
        });
        let mut svc = OpenAILLMService::new("test-key", settings);
        svc.state.base.metrics.start_ttfb();
        svc.state.base.metrics.start_processing();

        let (downstream, _) = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(
                Frame::Interruption(InterruptionFrame),
                Direction::Downstream,
            )],
        )
        .await;

        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::Interruption(_)));
        assert!(svc.state.base.metrics.stop_ttfb().is_none());
        assert!(svc.state.base.metrics.stop_processing().is_none());
    }

    #[test]
    fn llm_context_reconstruction_includes_tools() {
        let context_value = json!({
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_choice": "auto"
        });

        let messages = context_value
            .get("messages")
            .cloned()
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let llm_context = pipecat_context::LLMContext::new(messages);
        if let Some(tools) = context_value.get("tools").cloned() {
            llm_context.set_tools(tools);
        }
        if let Some(tool_choice) = context_value.get("tool_choice").cloned() {
            llm_context.set_tool_choice(tool_choice);
        }

        assert!(llm_context.tools().is_some());
        assert_eq!(
            llm_context.tools().unwrap()[0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(llm_context.tool_choice(), Some(json!("auto")));
        assert_eq!(llm_context.message_count(), 1);
    }

    #[tokio::test]
    async fn function_call_timeout() {
        use std::time::Duration;

        let settings = OpenAILLMSettings::new(crate::settings::LLMSettings {
            model: Some("gpt-4".into()),
            ..Default::default()
        });
        let mut svc = OpenAILLMService::new("test-key", settings);

        // Register a handler that sleeps longer than the timeout
        let slow_handler: crate::function_call::FunctionCallHandler = Arc::new(|_| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                json!({"result": "should not reach"})
            })
        });
        svc.function_registry_mut().register(
            "slow_fn",
            slow_handler,
            false,
            Duration::from_millis(100),
        );

        let item = svc.state.function_registry.lookup("slow_fn").unwrap();
        let params = crate::function_call::FunctionCallParams {
            function_name: "slow_fn".into(),
            tool_call_id: "call_1".into(),
            arguments: json!({}),
            context: pipecat_context::LLMContext::new(vec![]),
        };

        let result = match tokio::time::timeout(item.timeout, (item.handler)(params)).await {
            Ok(result) => result,
            Err(_) => serde_json::Value::Null,
        };

        assert!(result.is_null());
    }

    #[test]
    fn cancel_on_interruption_propagation() {
        use std::time::Duration;

        let mut registry = crate::function_call::FunctionCallRegistry::new();
        let handler: crate::function_call::FunctionCallHandler =
            Arc::new(|_| Box::pin(async { json!({"ok": true}) }));

        registry.register("cancellable_fn", handler, true, Duration::from_secs(10));

        let item = registry.lookup("cancellable_fn").unwrap();
        assert!(item.cancel_on_interruption);

        // Verify the frame would carry the correct flag
        let frame = FunctionCallInProgressFrame {
            function_name: "cancellable_fn".into(),
            tool_call_id: "call_1".into(),
            arguments: json!({}),
            cancel_on_interruption: item.cancel_on_interruption,
        };
        assert!(frame.cancel_on_interruption);
    }

    #[tokio::test]
    async fn settings_update_changes_model() {
        let settings = OpenAILLMSettings::new(crate::settings::LLMSettings {
            model: Some("gpt-4".into()),
            ..Default::default()
        });
        let mut svc = OpenAILLMService::new("test-key", settings);

        let mut delta = HashMap::new();
        delta.insert("model".into(), json!("gpt-4-turbo"));

        let _ = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(
                Frame::LLMUpdateSettings(ServiceUpdateSettingsFrame { settings: delta }),
                Direction::Downstream,
            )],
        )
        .await;

        assert_eq!(svc.state.settings.model.as_deref(), Some("gpt-4-turbo"));
    }
}
