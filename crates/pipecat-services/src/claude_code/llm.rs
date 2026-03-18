use std::process::Stdio;

use async_trait::async_trait;
use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{debug, trace, warn};

use super::settings::ClaudeCodeSettings;
use crate::function_call::FunctionCallParams;
use crate::llm::{LLMService, LLMServiceState, llm_process_frame};

/// Tool call instruction appended to the system prompt when tools are present.
const TOOL_CALL_INSTRUCTION: &str = r#"

When you need to call a tool, respond ONLY with a JSON object in this exact format, with no other text:
{"tool_calls":[{"id":"call_1","function":{"name":"function_name","arguments":{"arg":"value"}}}]}

If you do not need to call a tool, respond normally with text."#;

/// LLM service that uses the Claude Code CLI (`claude -p`) as its backend.
///
/// By default, Claude Code's built-in tools are disabled (`--tools ""`), and
/// its system prompt is replaced (`--system-prompt`), making it function like
/// a regular LLM. The full messages array from `LLMContext` is serialized as
/// the prompt, and tool definitions from the context are injected into the
/// system prompt.
///
/// # Tool Handling
///
/// When the `LLMContext` includes tool definitions, they are described in the
/// system prompt and Claude is instructed to respond with structured JSON for
/// tool calls. These are parsed and executed via the `FunctionCallRegistry`,
/// matching the pattern used by `OpenAILLMService`.
///
/// # Session Resumption
///
/// When `enable_session_resumption` is true (default), the service captures
/// the `session_id` from the CLI output and passes `--resume <id>` on
/// subsequent calls, allowing Claude Code to maintain conversation context.
///
/// # Agent Mode
///
/// To use Claude Code as an autonomous agent with its built-in tools, set
/// `settings.tools = Some(vec!["default".into()])` to re-enable them.
#[derive(Debug)]
pub struct ClaudeCodeLLMService {
    state: LLMServiceState,
    settings: ClaudeCodeSettings,
    session_id: Option<String>,
}

impl ClaudeCodeLLMService {
    pub fn new(settings: ClaudeCodeSettings) -> Self {
        let base_settings = settings.base.clone();
        Self {
            state: LLMServiceState::new("ClaudeCodeLLMService", base_settings),
            settings,
            session_id: None,
        }
    }

    /// Get a mutable reference to the function call registry.
    pub fn function_registry_mut(&mut self) -> &mut crate::function_call::FunctionCallRegistry {
        &mut self.state.function_registry
    }

    /// Reset the session, forcing a fresh conversation on the next call.
    pub fn reset_session(&mut self) {
        self.session_id = None;
    }

    /// Get the current session ID, if any.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Build the CLI command from settings and context.
    ///
    /// Reads `model` and `system_instruction` from `self.state.settings`
    /// (updated by `LLMUpdateSettings` frames) rather than from
    /// `self.settings.base` to ensure runtime setting changes take effect.
    fn build_command(&self, prompt: &str, system_prompt: &str) -> Command {
        let mut cmd = Command::new(&self.settings.claude_path);
        cmd.arg("-p");
        cmd.arg("--output-format").arg("stream-json");
        cmd.arg("--verbose");
        cmd.arg("--include-partial-messages");

        // Model — read from state.settings so LLMUpdateSettings takes effect
        if let Some(ref model) = self.state.settings.model {
            cmd.arg("--model").arg(model);
        }

        // System prompt — built from system_instruction + tool definitions
        if !system_prompt.is_empty() {
            cmd.arg("--system-prompt").arg(system_prompt);
        }

        // Append system prompt (additional user-specified context)
        if let Some(ref append) = self.settings.append_system_prompt {
            cmd.arg("--append-system-prompt").arg(append);
        }

        // Effort
        if let Some(effort) = self.settings.effort {
            cmd.arg("--effort").arg(effort.as_str());
        }

        // Tools — default to "" (disable all built-in tools) for LLM mode
        if let Some(ref tools) = self.settings.tools {
            cmd.arg("--tools").arg(tools.join(","));
        } else {
            cmd.arg("--tools").arg("");
        }

        // Allowed tools (each as a separate arg)
        if let Some(ref allowed) = self.settings.allowed_tools {
            for tool in allowed {
                cmd.arg("--allowedTools").arg(tool);
            }
        }

        // Disallowed tools (each as a separate arg)
        if let Some(ref disallowed) = self.settings.disallowed_tools {
            for tool in disallowed {
                cmd.arg("--disallowedTools").arg(tool);
            }
        }

        // Max turns
        if let Some(max_turns) = self.settings.max_turns {
            cmd.arg("--max-turns").arg(max_turns.to_string());
        } else {
            cmd.arg("--max-turns").arg("1");
        }

        // Permission mode
        if let Some(ref mode) = self.settings.permission_mode {
            cmd.arg("--permission-mode").arg(mode.as_str());
        }

        // Dangerously skip permissions
        if self.settings.dangerously_skip_permissions {
            cmd.arg("--dangerously-skip-permissions");
        }

        // MCP config
        if let Some(ref path) = self.settings.mcp_config {
            cmd.arg("--mcp-config").arg(path);
        }

        // JSON schema
        if let Some(ref schema) = self.settings.json_schema {
            cmd.arg("--json-schema").arg(schema.to_string());
        }

        // No session persistence
        if self.settings.no_session_persistence {
            cmd.arg("--no-session-persistence");
        }

        // Session resumption
        if self.settings.enable_session_resumption
            && let Some(ref sid) = self.session_id
        {
            cmd.arg("--resume").arg(sid);
        }

        // Extra args
        for arg in &self.settings.extra_args {
            cmd.arg(arg);
        }

        // The prompt (positional argument, must be last)
        cmd.arg(prompt);

        // Configure stdio
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        cmd
    }

    /// Build the system prompt, including tool definitions if present.
    fn build_system_prompt(&self, context: &serde_json::Value) -> String {
        let mut parts = Vec::new();

        // Base system instruction
        if let Some(ref instruction) = self.state.settings.system_instruction {
            parts.push(instruction.clone());
        }

        // Tool definitions from context
        if let Some(tools) = context.get("tools")
            && let Some(arr) = tools.as_array()
            && !arr.is_empty()
        {
            let tools_json = serde_json::to_string_pretty(tools).unwrap_or_default();
            parts.push(format!(
                "You have access to the following tools:\n\n{tools_json}{TOOL_CALL_INSTRUCTION}"
            ));
        }

        parts.join("\n\n")
    }

    /// Build the prompt from the LLMContext messages.
    ///
    /// Serializes the full messages array as JSON. Claude understands this
    /// format and maintains conversation context from it.
    fn build_prompt(context: &serde_json::Value) -> String {
        if let Some(messages) = context.get("messages") {
            serde_json::to_string(messages).unwrap_or_default()
        } else {
            String::new()
        }
    }

    /// Process an LLMContext frame: spawn CLI, stream output, emit frames.
    async fn process_context(
        &mut self,
        context_value: serde_json::Value,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let skip_tts = self.state.skip_tts;
        ctx.send_downstream(Frame::LLMFullResponseStart(LLMFullResponseStartFrame {
            skip_tts,
        }))
        .await?;

        let metrics_enabled = self.state.base.metrics_enabled();
        let usage_enabled = self.state.base.usage_metrics_enabled();
        if metrics_enabled {
            self.state.base.metrics.start_ttfb();
            self.state.base.metrics.start_processing();
        }

        let prompt = Self::build_prompt(&context_value);
        if prompt.is_empty() {
            warn!("ClaudeCode: empty prompt from LLMContext");
        }

        let system_prompt = self.build_system_prompt(&context_value);

        let result = self
            .stream_cli_response(
                &prompt,
                &system_prompt,
                &context_value,
                ctx,
                metrics_enabled,
                usage_enabled,
            )
            .await;

        if let Err(ref e) = result {
            warn!("ClaudeCode error: {}", e);
            ctx.push_error(&format!("ClaudeCode error: {e}"), false)
                .await?;
        }

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

    /// Spawn the CLI process and parse streaming JSON output.
    async fn stream_cli_response(
        &mut self,
        prompt: &str,
        system_prompt: &str,
        context_value: &serde_json::Value,
        ctx: &ProcessorContext,
        metrics_enabled: bool,
        usage_enabled: bool,
    ) -> Result<()> {
        let timeout = self.settings.timeout;

        if let Some(duration) = timeout {
            match tokio::time::timeout(
                duration,
                self.run_cli_process(
                    prompt,
                    system_prompt,
                    context_value,
                    ctx,
                    metrics_enabled,
                    usage_enabled,
                ),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Err(PipecatError::ProcessorError(format!(
                    "ClaudeCode timed out after {duration:?}",
                ))),
            }
        } else {
            self.run_cli_process(
                prompt,
                system_prompt,
                context_value,
                ctx,
                metrics_enabled,
                usage_enabled,
            )
            .await
        }
    }

    /// Inner implementation: spawn process, read stdout, reap child.
    async fn run_cli_process(
        &mut self,
        prompt: &str,
        system_prompt: &str,
        context_value: &serde_json::Value,
        ctx: &ProcessorContext,
        metrics_enabled: bool,
        usage_enabled: bool,
    ) -> Result<()> {
        let mut cmd = self.build_command(prompt, system_prompt);
        debug!("ClaudeCode: spawning process");
        trace!("ClaudeCode: command {:?}", cmd);

        let mut child = cmd.spawn().map_err(|e| {
            PipecatError::ProcessorError(format!("Failed to spawn claude CLI: {e}"))
        })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| PipecatError::ProcessorError("No stdout from claude process".into()))?;

        // Read stderr in a background task to prevent pipe buffer deadlock
        let stderr_handle = child.stderr.take().map(|stderr| {
            tokio::spawn(async move {
                let mut buf = String::new();
                let mut reader = BufReader::new(stderr);
                let _ = tokio::io::AsyncReadExt::read_to_string(&mut reader, &mut buf).await;
                buf
            })
        });

        let (stream_error, full_text) = self
            .read_stream(stdout, ctx, metrics_enabled, usage_enabled)
            .await;

        // Always reap the child process to avoid zombies
        let status = child.wait().await.map_err(|e| {
            PipecatError::ProcessorError(format!("Failed to wait for claude process: {e}"))
        })?;

        // Return stream error first (more informative than exit code)
        if let Some(err) = stream_error {
            return Err(err);
        }

        if !status.success() {
            let stderr_text = if let Some(handle) = stderr_handle {
                handle.await.unwrap_or_default()
            } else {
                String::new()
            };
            let msg = if stderr_text.is_empty() {
                format!("claude CLI exited with status: {status}")
            } else {
                format!("claude CLI exited with status {status}: {stderr_text}")
            };
            return Err(PipecatError::ProcessorError(msg));
        }

        // Check if the accumulated text is a tool call response
        if let Some(tool_calls) = Self::parse_tool_calls(&full_text) {
            self.run_function_calls(&tool_calls, context_value, ctx)
                .await?;
        }

        Ok(())
    }

    /// Read and parse the streaming JSON output from stdout.
    ///
    /// Returns `(stream_error, accumulated_text)`. The accumulated text is
    /// used to detect tool call responses after streaming completes.
    async fn read_stream(
        &mut self,
        stdout: tokio::process::ChildStdout,
        ctx: &ProcessorContext,
        metrics_enabled: bool,
        usage_enabled: bool,
    ) -> (Option<PipecatError>, String) {
        let mut full_text = String::new();

        match self
            .read_stream_inner(stdout, ctx, metrics_enabled, usage_enabled, &mut full_text)
            .await
        {
            Ok(()) => (None, full_text),
            Err(e) => (Some(e), full_text),
        }
    }

    async fn read_stream_inner(
        &mut self,
        stdout: tokio::process::ChildStdout,
        ctx: &ProcessorContext,
        metrics_enabled: bool,
        usage_enabled: bool,
        full_text: &mut String,
    ) -> Result<()> {
        let mut reader = BufReader::new(stdout).lines();
        let mut ttfb_stopped = false;

        while let Some(line) = reader.next_line().await.map_err(|e| {
            PipecatError::ProcessorError(format!("Error reading claude stdout: {e}"))
        })? {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
                trace!("ClaudeCode: skipping non-JSON line");
                continue;
            };

            let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match event_type {
                // Streaming events are wrapped: {"type":"stream_event","event":{...}}
                "stream_event" => {
                    if let Some(inner) = event.get("event") {
                        let inner_type = inner.get("type").and_then(|t| t.as_str()).unwrap_or("");

                        if inner_type == "content_block_delta" {
                            if let Some(delta) = inner.get("delta")
                                && delta.get("type").and_then(|t| t.as_str()) == Some("text_delta")
                                && let Some(text) = delta.get("text").and_then(|t| t.as_str())
                                && !text.is_empty()
                            {
                                full_text.push_str(text);

                                if metrics_enabled && !ttfb_stopped {
                                    ctx.push_ttfb(&mut self.state.base.metrics).await?;
                                    ttfb_stopped = true;
                                }

                                ctx.send_downstream(Frame::LLMText(TextFrame {
                                    text: text.to_string(),
                                    skip_tts: self.state.skip_tts,
                                    includes_inter_frame_spaces: false,
                                    append_to_context: true,
                                }))
                                .await?;
                            }
                        } else {
                            trace!("ClaudeCode: stream event {}", inner_type);
                        }
                    }
                }

                "result" => {
                    // Extract session_id for future resumption
                    if let Some(sid) = event.get("session_id").and_then(|s| s.as_str()) {
                        self.session_id = Some(sid.to_string());
                    }

                    // Extract usage from result event
                    if usage_enabled && let Some(usage) = event.get("usage") {
                        self.push_usage_metrics(usage, ctx).await?;
                    }

                    // Check for errors
                    if event.get("is_error").and_then(|e| e.as_bool()) == Some(true) {
                        let subtype = event
                            .get("subtype")
                            .and_then(|s| s.as_str())
                            .unwrap_or("unknown");
                        let result_text = event
                            .get("result")
                            .and_then(|r| r.as_str())
                            .unwrap_or("unknown error");
                        return Err(PipecatError::ProcessorError(format!(
                            "ClaudeCode error ({subtype}): {result_text}"
                        )));
                    }
                }

                "system" => {
                    trace!("ClaudeCode system event: {:?}", event);
                    // Extract session_id from init event
                    if event.get("subtype").and_then(|s| s.as_str()) == Some("init")
                        && let Some(sid) = event.get("session_id").and_then(|s| s.as_str())
                        && self.session_id.is_none()
                    {
                        self.session_id = Some(sid.to_string());
                    }
                }

                "assistant" | "rate_limit_event" => {
                    // assistant: complete message (redundant with stream_event deltas)
                    // rate_limit_event: informational
                    if let Some(sid) = event.get("session_id").and_then(|s| s.as_str()) {
                        self.session_id = Some(sid.to_string());
                    }
                }

                _ => {
                    trace!("ClaudeCode: event type={}", event_type);
                }
            }
        }

        Ok(())
    }

    /// Try to parse tool calls from the accumulated response text.
    ///
    /// Returns `Some` if the response is a JSON object with a `tool_calls` array.
    fn parse_tool_calls(text: &str) -> Option<Vec<FunctionCallFromLLM>> {
        let trimmed = text.trim();
        let parsed: serde_json::Value = serde_json::from_str(trimmed).ok()?;
        let calls = parsed.get("tool_calls")?.as_array()?;

        let mut result = Vec::new();
        for (i, call) in calls.iter().enumerate() {
            let func = call.get("function")?;
            let name = func.get("name")?.as_str()?;
            let arguments = func
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            let id = call
                .get("id")
                .and_then(|id| id.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("call_{i}"));

            result.push(FunctionCallFromLLM {
                function_name: name.to_string(),
                tool_call_id: id,
                arguments,
            });
        }

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    /// Execute tool calls via the function registry.
    async fn run_function_calls(
        &self,
        function_calls: &[FunctionCallFromLLM],
        context_value: &serde_json::Value,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        ctx.broadcast(Frame::FunctionCallsStarted(FunctionCallsStartedFrame {
            function_calls: function_calls.to_vec(),
        }))
        .await?;

        // Build LLMContext for function call handlers
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

        for call in function_calls {
            let registry_item = self.state.function_registry.lookup(&call.function_name);

            ctx.broadcast(Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
                function_name: call.function_name.clone(),
                tool_call_id: call.tool_call_id.clone(),
                arguments: call.arguments.clone(),
                cancel_on_interruption: registry_item
                    .map(|item| item.cancel_on_interruption)
                    .unwrap_or(false),
            }))
            .await?;

            let result_value = if let Some(item) = registry_item {
                let params = FunctionCallParams {
                    function_name: call.function_name.clone(),
                    tool_call_id: call.tool_call_id.clone(),
                    arguments: call.arguments.clone(),
                    context: llm_context.clone(),
                };
                match tokio::time::timeout(item.timeout, (item.handler)(params)).await {
                    Ok(result) => result,
                    Err(_) => {
                        warn!(
                            "Function call [{}:{}] timed out after {:?}",
                            call.function_name, call.tool_call_id, item.timeout
                        );
                        serde_json::Value::Null
                    }
                }
            } else {
                warn!("No handler registered for function: {}", call.function_name);
                serde_json::json!({"error": format!("No handler for {}", call.function_name)})
            };

            ctx.broadcast(Frame::FunctionCallResult(FunctionCallResultFrame {
                function_name: call.function_name.clone(),
                tool_call_id: call.tool_call_id.clone(),
                arguments: call.arguments.clone(),
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

    /// Push usage metrics from a result event.
    async fn push_usage_metrics(
        &self,
        usage: &serde_json::Value,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let input = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let output = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let cache_read = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let cache_creation = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let token_usage = pipecat_core::metrics::LlmTokenUsage {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: input + output,
            cache_read_input_tokens: cache_read,
            cache_creation_input_tokens: cache_creation,
            reasoning_tokens: None,
        };

        ctx.push_llm_usage(&self.state.base.metrics, token_usage)
            .await?;
        Ok(())
    }
}

#[async_trait]
impl FrameProcessor for ClaudeCodeLLMService {
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

impl LLMService for ClaudeCodeLLMService {
    fn llm_service_state(&self) -> &LLMServiceState {
        &self.state
    }

    fn llm_service_state_mut(&mut self) -> &mut LLMServiceState {
        &mut self.state
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;

    #[test]
    fn build_prompt_serializes_messages() {
        let context = json!({
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And times 3?"}
            ]
        });
        let prompt = ClaudeCodeLLMService::build_prompt(&context);
        let parsed: serde_json::Value = serde_json::from_str(&prompt).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 3);
    }

    #[test]
    fn build_prompt_empty_context() {
        let context = json!({});
        assert_eq!(ClaudeCodeLLMService::build_prompt(&context), "");
    }

    #[test]
    fn build_system_prompt_with_instruction_only() {
        let settings = ClaudeCodeSettings::new(crate::settings::LLMSettings {
            system_instruction: Some("You are helpful.".into()),
            ..Default::default()
        });
        let svc = ClaudeCodeLLMService::new(settings);
        let context = json!({"messages": []});
        let sp = svc.build_system_prompt(&context);
        assert_eq!(sp, "You are helpful.");
    }

    #[test]
    fn build_system_prompt_with_tools() {
        let settings = ClaudeCodeSettings::new(crate::settings::LLMSettings {
            system_instruction: Some("You are helpful.".into()),
            ..Default::default()
        });
        let svc = ClaudeCodeLLMService::new(settings);
        let context = json!({
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}]
        });
        let sp = svc.build_system_prompt(&context);
        assert!(sp.contains("You are helpful."));
        assert!(sp.contains("get_weather"));
        assert!(sp.contains("tool_calls"));
    }

    #[test]
    fn build_system_prompt_no_instruction_no_tools() {
        let settings = ClaudeCodeSettings::default();
        let svc = ClaudeCodeLLMService::new(settings);
        let context = json!({"messages": []});
        let sp = svc.build_system_prompt(&context);
        assert!(sp.is_empty());
    }

    #[test]
    fn build_command_defaults_disables_tools() {
        let settings = ClaudeCodeSettings::default();
        let svc = ClaudeCodeLLMService::new(settings);
        let cmd = svc.build_command("Hello", "");
        let args: Vec<&str> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_str().unwrap())
            .collect();

        assert!(args.contains(&"-p"));
        assert!(args.contains(&"--output-format"));
        assert!(args.contains(&"stream-json"));
        assert!(args.contains(&"--verbose"));
        assert!(args.contains(&"--include-partial-messages"));
        // Built-in tools disabled by default
        assert!(args.contains(&"--tools"));
        let tools_idx = args.iter().position(|a| *a == "--tools").unwrap();
        assert_eq!(args[tools_idx + 1], "");
        // Max turns defaults to 1 (LLM mode)
        assert!(args.contains(&"--max-turns"));
        let turns_idx = args.iter().position(|a| *a == "--max-turns").unwrap();
        assert_eq!(args[turns_idx + 1], "1");
        assert_eq!(args.last(), Some(&"Hello"));
        assert!(!args.contains(&"--resume"));
    }

    #[test]
    fn build_command_with_system_prompt() {
        let settings = ClaudeCodeSettings::default();
        let svc = ClaudeCodeLLMService::new(settings);
        let cmd = svc.build_command("Hello", "Be concise.");
        let args: Vec<&str> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_str().unwrap())
            .collect();

        assert!(args.contains(&"--system-prompt"));
        assert!(args.contains(&"Be concise."));
    }

    #[test]
    fn build_command_agent_mode() {
        let mut settings = ClaudeCodeSettings::default();
        settings.tools = Some(vec!["default".into()]);
        settings.max_turns = Some(10);

        let svc = ClaudeCodeLLMService::new(settings);
        let cmd = svc.build_command("Fix the bug", "");
        let args: Vec<&str> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_str().unwrap())
            .collect();

        let tools_idx = args.iter().position(|a| *a == "--tools").unwrap();
        assert_eq!(args[tools_idx + 1], "default");
        let turns_idx = args.iter().position(|a| *a == "--max-turns").unwrap();
        assert_eq!(args[turns_idx + 1], "10");
    }

    #[test]
    fn build_command_all_options() {
        let mut settings = ClaudeCodeSettings::new(crate::settings::LLMSettings {
            model: Some("opus".into()),
            system_instruction: Some("ignored — system_prompt arg used instead".into()),
            ..Default::default()
        });
        settings.effort = Some(super::super::settings::Effort::High);
        settings.tools = Some(vec!["Bash".into(), "Read".into()]);
        settings.allowed_tools = Some(vec!["Bash(git *)".into()]);
        settings.disallowed_tools = Some(vec!["Bash(rm *)".into()]);
        settings.append_system_prompt = Some("Extra context.".into());
        settings.max_turns = Some(5);
        settings.permission_mode = Some(super::super::settings::PermissionMode::DontAsk);
        settings.dangerously_skip_permissions = true;
        settings.extra_args = vec!["--no-chrome".into()];

        let svc = ClaudeCodeLLMService::new(settings);
        let cmd = svc.build_command("Test prompt", "Be concise.");
        let args: Vec<&str> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_str().unwrap())
            .collect();

        assert!(args.contains(&"--model"));
        assert!(args.contains(&"opus"));
        assert!(args.contains(&"--system-prompt"));
        assert!(args.contains(&"Be concise."));
        assert!(args.contains(&"--append-system-prompt"));
        assert!(args.contains(&"Extra context."));
        assert!(args.contains(&"--effort"));
        assert!(args.contains(&"high"));
        assert!(args.contains(&"--tools"));
        assert!(args.contains(&"Bash,Read"));
        assert!(args.contains(&"--allowedTools"));
        assert!(args.contains(&"Bash(git *)"));
        assert!(args.contains(&"--disallowedTools"));
        assert!(args.contains(&"Bash(rm *)"));
        assert!(args.contains(&"--max-turns"));
        assert!(args.contains(&"5"));
        assert!(args.contains(&"--permission-mode"));
        assert!(args.contains(&"dontAsk"));
        assert!(args.contains(&"--dangerously-skip-permissions"));
        assert!(args.contains(&"--no-chrome"));
        assert_eq!(args.last(), Some(&"Test prompt"));
    }

    #[test]
    fn build_command_with_session_resumption() {
        let settings = ClaudeCodeSettings::default();
        let mut svc = ClaudeCodeLLMService::new(settings);
        svc.session_id = Some("abc-123".into());

        let cmd = svc.build_command("Follow up", "");
        let args: Vec<&str> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_str().unwrap())
            .collect();

        assert!(args.contains(&"--resume"));
        assert!(args.contains(&"abc-123"));
    }

    #[test]
    fn build_command_session_resumption_disabled() {
        let mut settings = ClaudeCodeSettings::default();
        settings.enable_session_resumption = false;

        let mut svc = ClaudeCodeLLMService::new(settings);
        svc.session_id = Some("abc-123".into());

        let cmd = svc.build_command("Follow up", "");
        let args: Vec<&str> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_str().unwrap())
            .collect();

        assert!(!args.contains(&"--resume"));
    }

    #[test]
    fn reset_session_clears_id() {
        let settings = ClaudeCodeSettings::default();
        let mut svc = ClaudeCodeLLMService::new(settings);
        svc.session_id = Some("abc".into());
        assert!(svc.session_id().is_some());

        svc.reset_session();
        assert!(svc.session_id().is_none());
    }

    // -----------------------------------------------------------------------
    // Tool call parsing
    // -----------------------------------------------------------------------

    #[test]
    fn parse_tool_calls_valid() {
        let text = r#"{"tool_calls":[{"id":"call_1","function":{"name":"get_weather","arguments":{"location":"NYC"}}}]}"#;
        let calls = ClaudeCodeLLMService::parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function_name, "get_weather");
        assert_eq!(calls[0].tool_call_id, "call_1");
        assert_eq!(calls[0].arguments["location"], "NYC");
    }

    #[test]
    fn parse_tool_calls_multiple() {
        let text = r#"{"tool_calls":[
            {"id":"call_1","function":{"name":"get_weather","arguments":{"location":"NYC"}}},
            {"id":"call_2","function":{"name":"get_time","arguments":{"tz":"EST"}}}
        ]}"#;
        let calls = ClaudeCodeLLMService::parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function_name, "get_weather");
        assert_eq!(calls[1].function_name, "get_time");
    }

    #[test]
    fn parse_tool_calls_generates_ids() {
        let text = r#"{"tool_calls":[{"function":{"name":"get_weather","arguments":{}}}]}"#;
        let calls = ClaudeCodeLLMService::parse_tool_calls(text).unwrap();
        assert_eq!(calls[0].tool_call_id, "call_0");
    }

    #[test]
    fn parse_tool_calls_not_json() {
        assert!(ClaudeCodeLLMService::parse_tool_calls("Hello, world!").is_none());
    }

    #[test]
    fn parse_tool_calls_json_without_tool_calls() {
        assert!(ClaudeCodeLLMService::parse_tool_calls(r#"{"result": "done"}"#).is_none());
    }

    #[test]
    fn parse_tool_calls_with_whitespace() {
        let text = r#"
            {"tool_calls":[{"id":"c1","function":{"name":"test","arguments":{}}}]}
        "#;
        let calls = ClaudeCodeLLMService::parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Event parsing tests — use canned JSON from real `claude -p` output
    // -----------------------------------------------------------------------

    #[test]
    fn parse_system_init_extracts_session_id() {
        let event = json!({
            "type": "system",
            "subtype": "init",
            "session_id": "ed629114-58ee-43e0-b38a-512bcdd9adb2",
            "model": "claude-opus-4-6[1m]"
        });

        let mut svc = ClaudeCodeLLMService::new(ClaudeCodeSettings::default());
        assert!(svc.session_id.is_none());

        if event.get("subtype").and_then(|s| s.as_str()) == Some("init")
            && let Some(sid) = event.get("session_id").and_then(|s| s.as_str())
            && svc.session_id.is_none()
        {
            svc.session_id = Some(sid.to_string());
        }

        assert_eq!(
            svc.session_id(),
            Some("ed629114-58ee-43e0-b38a-512bcdd9adb2")
        );
    }

    #[test]
    fn parse_stream_event_text_delta() {
        let event = json!({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Ownership"}
            },
            "session_id": "fd0843c2-4276-4ac3-a208-1fcdbcc4fed5"
        });

        let inner = event.get("event").unwrap();
        let inner_type = inner.get("type").and_then(|t| t.as_str()).unwrap();
        assert_eq!(inner_type, "content_block_delta");

        let delta = inner.get("delta").unwrap();
        assert_eq!(
            delta.get("type").and_then(|t| t.as_str()),
            Some("text_delta")
        );
        assert_eq!(
            delta.get("text").and_then(|t| t.as_str()),
            Some("Ownership")
        );
    }

    #[test]
    fn parse_result_event_success() {
        let event = json!({
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "session_id": "ed629114-58ee-43e0-b38a-512bcdd9adb2",
            "usage": {
                "input_tokens": 2,
                "output_tokens": 5,
                "cache_read_input_tokens": 8706,
                "cache_creation_input_tokens": 9215
            }
        });

        assert_eq!(event.get("is_error").and_then(|e| e.as_bool()), Some(false));
        let usage = event.get("usage").unwrap();
        assert_eq!(usage.get("input_tokens").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(usage.get("output_tokens").and_then(|v| v.as_u64()), Some(5));
    }

    #[test]
    fn parse_result_event_error() {
        let event = json!({
            "type": "result",
            "subtype": "error_max_turns",
            "is_error": true,
            "result": "Max turns exceeded",
            "session_id": "abc-123"
        });

        assert_eq!(event.get("is_error").and_then(|e| e.as_bool()), Some(true));
    }

    // -----------------------------------------------------------------------
    // Frame forwarding
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn process_frame_forwards_non_llm_frames() {
        let settings = ClaudeCodeSettings::default();
        let mut svc = ClaudeCodeLLMService::new(settings);

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
        let settings = ClaudeCodeSettings::default();
        let mut svc = ClaudeCodeLLMService::new(settings);
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

    #[tokio::test]
    async fn settings_update_changes_model() {
        let settings = ClaudeCodeSettings::new(crate::settings::LLMSettings {
            model: Some("sonnet".into()),
            ..Default::default()
        });
        let mut svc = ClaudeCodeLLMService::new(settings);

        let mut delta = HashMap::new();
        delta.insert("model".into(), json!("opus"));

        let _ = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(
                Frame::LLMUpdateSettings(ServiceUpdateSettingsFrame { settings: delta }),
                Direction::Downstream,
            )],
        )
        .await;

        assert_eq!(svc.state.settings.model.as_deref(), Some("opus"));

        let cmd = svc.build_command("test", "");
        let args: Vec<&str> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_str().unwrap())
            .collect();
        assert!(args.contains(&"opus"));
        assert!(!args.contains(&"sonnet"));
    }

    #[test]
    fn build_command_with_no_session_persistence() {
        let mut settings = ClaudeCodeSettings::default();
        settings.no_session_persistence = true;

        let svc = ClaudeCodeLLMService::new(settings);
        let cmd = svc.build_command("test", "");
        let args: Vec<&str> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_str().unwrap())
            .collect();

        assert!(args.contains(&"--no-session-persistence"));
    }
}
