use std::collections::HashMap;

use async_trait::async_trait;
use tracing::{debug, trace, warn};

use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};

use crate::aggregator::LLMContextAggregatorBase;
use crate::context::LLMContext;

/// Manages assistant response lifecycle: accumulates LLM text output and
/// function calls, then pushes aggregated messages to the shared context.
///
/// Sits at the **end** of the pipeline (after TTS and output transport).
/// Consumes LLM-specific frames (text, function calls, thoughts, context
/// operations) and accumulates them into the shared `LLMContext`. Only
/// lifecycle frames (Start, End, Cancel, Interruption) and unrecognized
/// frames are forwarded downstream.
#[derive(Debug)]
pub struct LLMAssistantAggregator {
    base: ProcessorBase,
    agg: LLMContextAggregatorBase,

    /// Function calls tracked by tool_call_id → in-progress frame.
    function_calls: HashMap<String, FunctionCallInProgressFrame>,

    /// Whether we're inside a thought block.
    in_thought: bool,
    /// Accumulated thought text.
    thought_text: String,
    /// Whether thoughts should be appended to context.
    thought_append_to_context: bool,
}

impl LLMAssistantAggregator {
    pub fn new(context: LLMContext) -> Self {
        Self {
            base: ProcessorBase::new("LLMAssistantAggregator"),
            agg: LLMContextAggregatorBase::new(context, "assistant"),
            function_calls: HashMap::new(),
            in_thought: false,
            thought_text: String::new(),
            thought_append_to_context: false,
        }
    }

    /// Access the shared context.
    pub fn context(&self) -> &LLMContext {
        self.agg.context()
    }

    /// Push an LLMContextFrame in the given direction.
    async fn push_context_frame(&self, direction: Direction, ctx: &ProcessorContext) -> Result<()> {
        let context_value = self.agg.context().to_context_value();
        ctx.send(
            Frame::LLMContext(LLMContextFrame {
                context: context_value,
            }),
            direction,
        )
        .await
    }

    /// Push the current text aggregation to context and emit a context frame
    /// downstream (matching Python's `push_aggregation` behavior).
    async fn push_aggregation(&mut self, ctx: &ProcessorContext) -> Result<String> {
        let text = self.agg.push_aggregation();
        if !text.is_empty() {
            self.push_context_frame(Direction::Downstream, ctx).await?;
        }
        Ok(text)
    }

    /// Update a tool message's content by tool_call_id.
    fn update_function_call_result(&self, tool_call_id: &str, result: &str) {
        let mut messages = self.agg.context().get_messages();
        for msg in messages.iter_mut().rev() {
            if msg.get("role") == Some(&serde_json::json!("tool"))
                && msg.get("tool_call_id") == Some(&serde_json::json!(tool_call_id))
            {
                msg["content"] = serde_json::Value::String(result.to_string());
                break;
            }
        }
        self.agg.context().set_messages(messages);
    }
}

#[async_trait]
impl FrameProcessor for LLMAssistantAggregator {
    fn name(&self) -> &str {
        self.base.name()
    }

    fn id(&self) -> u64 {
        self.base.id()
    }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        match &envelope.frame {
            // --- Response lifecycle (consumed) ---
            Frame::LLMFullResponseStart(_) => {
                // Consumed — triggers internal state only
            }

            Frame::LLMFullResponseEnd(_) => {
                // Push aggregated text to context
                self.push_aggregation(ctx).await?;
            }

            // --- Text accumulation (consumed) ---
            Frame::Text(t) | Frame::LLMText(t) => {
                if t.append_to_context && !t.text.is_empty() {
                    trace!("AssistantAggregator: accumulating text: {:?}", t.text);
                    self.agg
                        .add_text_part(t.text.clone(), t.includes_inter_frame_spaces);
                }
                // Consumed — not forwarded
            }

            // --- Function call lifecycle (consumed) ---
            Frame::FunctionCallsStarted(f) => {
                self.function_calls.clear();
                for fc in &f.function_calls {
                    self.function_calls.insert(fc.tool_call_id.clone(), {
                        // We don't have the FunctionCallInProgressFrame yet,
                        // just track the IDs. Set to None equivalent by inserting
                        // a placeholder — the actual frame arrives in FunctionCallInProgress.
                        // For now, use the HashMap entry to track expected calls.
                        FunctionCallInProgressFrame {
                            function_name: fc.function_name.clone(),
                            tool_call_id: fc.tool_call_id.clone(),
                            arguments: fc.arguments.clone(),
                            cancel_on_interruption: false,
                        }
                    });
                }
                debug!("FunctionCallsStarted: {} calls", f.function_calls.len());
                // Consumed
            }

            Frame::FunctionCallInProgress(f) => {
                // Update tracking with the actual frame (has cancel_on_interruption)
                self.function_calls
                    .insert(f.tool_call_id.clone(), f.clone());

                // Add assistant message with tool_calls + tool placeholder
                // (matches Python: each in-progress call adds its own messages)
                self.agg.context().add_message(serde_json::json!({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": f.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": f.function_name,
                            "arguments": serde_json::to_string(&f.arguments).unwrap_or_default(),
                        }
                    }],
                }));
                self.agg.context().add_message(serde_json::json!({
                    "role": "tool",
                    "content": "IN_PROGRESS",
                    "tool_call_id": f.tool_call_id,
                }));
                // Consumed
            }

            Frame::FunctionCallResult(f) => {
                if !self.function_calls.contains_key(&f.tool_call_id) {
                    warn!(
                        "FunctionCallResult for unknown tool_call_id: {}",
                        f.tool_call_id
                    );
                    return Ok(());
                }

                self.function_calls.remove(&f.tool_call_id);

                // Update the tool result in context
                let result_str = if f.result.is_null() {
                    "COMPLETED".to_string()
                } else {
                    serde_json::to_string(&f.result).unwrap_or_default()
                };
                self.update_function_call_result(&f.tool_call_id, &result_str);

                // Determine whether to run LLM — only when result is non-null
                // (Python: `if frame.result` gates the entire run_llm decision)
                let run_llm = if !f.result.is_null() {
                    if let Some(ref props) = f.properties {
                        props
                            .run_llm
                            .unwrap_or_else(|| f.run_llm.unwrap_or(self.function_calls.is_empty()))
                    } else {
                        f.run_llm.unwrap_or(self.function_calls.is_empty())
                    }
                } else {
                    false
                };

                if run_llm {
                    self.push_context_frame(Direction::Upstream, ctx).await?;
                }
                // Consumed
            }

            Frame::FunctionCallCancel(f) => {
                // Only cancel if the in-progress call had cancel_on_interruption
                if let Some(in_progress) = self.function_calls.get(&f.tool_call_id)
                    && in_progress.cancel_on_interruption
                {
                    self.update_function_call_result(&f.tool_call_id, "CANCELLED");
                    self.function_calls.remove(&f.tool_call_id);
                }
                // Consumed
            }

            // --- Thought lifecycle (consumed) ---
            Frame::LLMThoughtStart(t) => {
                self.in_thought = true;
                self.thought_text.clear();
                self.thought_append_to_context = t.append_to_context;
            }

            Frame::LLMThoughtText(t) => {
                if !t.text.is_empty() {
                    self.thought_text.push_str(&t.text);
                }
            }

            Frame::LLMThoughtEnd(_) => {
                if self.thought_append_to_context && !self.thought_text.is_empty() {
                    self.agg.context().add_message(serde_json::json!({
                        "role": "assistant",
                        "content": self.thought_text.clone(),
                        "thought": true,
                    }));
                }
                self.in_thought = false;
                self.thought_text.clear();
            }

            // --- Context operations (consumed) ---
            Frame::LLMRun(_) => {
                self.push_context_frame(Direction::Upstream, ctx).await?;
            }

            Frame::LLMMessagesAppend(f) => {
                self.agg.context().add_messages(&f.messages);
                if f.run_llm == Some(true) {
                    self.push_context_frame(Direction::Upstream, ctx).await?;
                }
            }

            Frame::LLMMessagesUpdate(f) => {
                self.agg.context().set_messages(f.messages.clone());
                if f.run_llm == Some(true) {
                    self.push_context_frame(Direction::Upstream, ctx).await?;
                }
            }

            Frame::LLMSetTools(f) => {
                self.agg.context().set_tools(f.tools.clone());
            }

            Frame::LLMSetToolChoice(f) => {
                self.agg.context().set_tool_choice(f.tool_choice.clone());
            }

            Frame::LLMAssistantPushAggregation(_) => {
                self.push_aggregation(ctx).await?;
            }

            // --- Interruption (forwarded) ---
            Frame::Interruption(_) => {
                self.push_aggregation(ctx).await?;
                self.agg.reset();
                self.function_calls.clear();
                self.in_thought = false;
                self.thought_text.clear();
                ctx.push_frame(envelope, direction).await?;
            }

            // --- Lifecycle (forwarded) ---
            Frame::End(_) | Frame::Cancel(_) => {
                self.push_aggregation(ctx).await?;
                ctx.push_frame(envelope, direction).await?;
            }

            // --- Everything else: forward ---
            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipecat_core::test_utils::run_processor;
    use serde_json::json;

    fn make_aggregator() -> LLMAssistantAggregator {
        let ctx = LLMContext::new(vec![
            json!({"role": "system", "content": "You are helpful."}),
        ]);
        LLMAssistantAggregator::new(ctx)
    }

    #[tokio::test]
    async fn text_accumulation_between_start_end() {
        let mut agg = make_aggregator();
        let (downstream, _upstream) = run_processor(
            &mut agg,
            vec![
                (
                    Frame::LLMFullResponseStart(LLMFullResponseStartFrame { skip_tts: None }),
                    Direction::Downstream,
                ),
                (
                    Frame::Text(TextFrame {
                        text: "Hello".into(),
                        skip_tts: None,
                        includes_inter_frame_spaces: false,
                        append_to_context: true,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::Text(TextFrame {
                        text: "world".into(),
                        skip_tts: None,
                        includes_inter_frame_spaces: false,
                        append_to_context: true,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::LLMFullResponseEnd(LLMFullResponseEndFrame { skip_tts: None }),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        // LLM frames are consumed — only context frame pushed downstream
        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::LLMContext(_)));

        // Context should have system + assistant message
        let msgs = agg.context().get_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "Hello world");
    }

    #[tokio::test]
    async fn text_not_appended_when_flag_false() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![
                (
                    Frame::LLMFullResponseStart(LLMFullResponseStartFrame { skip_tts: None }),
                    Direction::Downstream,
                ),
                (
                    Frame::Text(TextFrame {
                        text: "skip me".into(),
                        skip_tts: None,
                        includes_inter_frame_spaces: false,
                        append_to_context: false,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::LLMFullResponseEnd(LLMFullResponseEndFrame { skip_tts: None }),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        // Only system message, no assistant message added
        assert_eq!(agg.context().message_count(), 1);
        // No context frame pushed (nothing was aggregated)
        assert!(downstream.is_empty());
    }

    #[tokio::test]
    async fn interruption_pushes_aggregation_and_forwards() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![
                (
                    Frame::LLMFullResponseStart(LLMFullResponseStartFrame { skip_tts: None }),
                    Direction::Downstream,
                ),
                (
                    Frame::Text(TextFrame {
                        text: "partial".into(),
                        skip_tts: None,
                        includes_inter_frame_spaces: false,
                        append_to_context: true,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::Interruption(InterruptionFrame),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        // Interrupted text should be pushed to context
        let msgs = agg.context().get_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["content"], "partial");

        // Context frame + Interruption forwarded downstream
        assert_eq!(downstream.len(), 2);
        assert!(matches!(&downstream[0].frame, Frame::LLMContext(_)));
        assert!(matches!(&downstream[1].frame, Frame::Interruption(_)));
    }

    #[tokio::test]
    async fn function_call_lifecycle() {
        let mut agg = make_aggregator();
        let (_downstream, upstream) = run_processor(
            &mut agg,
            vec![
                (
                    Frame::LLMFullResponseStart(LLMFullResponseStartFrame { skip_tts: None }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallsStarted(FunctionCallsStartedFrame {
                        function_calls: vec![FunctionCallFromLLM {
                            function_name: "get_weather".into(),
                            tool_call_id: "call_1".into(),
                            arguments: json!({"city": "NYC"}),
                        }],
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
                        function_name: "get_weather".into(),
                        tool_call_id: "call_1".into(),
                        arguments: json!({"city": "NYC"}),
                        cancel_on_interruption: false,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallResult(FunctionCallResultFrame {
                        function_name: "get_weather".into(),
                        tool_call_id: "call_1".into(),
                        arguments: json!({"city": "NYC"}),
                        result: json!({"temp": 72}),
                        run_llm: None,
                        properties: None,
                    }),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        // Should have pushed a context frame upstream (for LLM re-run)
        assert!(
            upstream
                .iter()
                .any(|e| matches!(&e.frame, Frame::LLMContext(_)))
        );

        // Context should have: system + assistant(tool_calls) + tool(result)
        let msgs = agg.context().get_messages();
        assert_eq!(msgs.len(), 3);
        assert!(msgs[1].get("tool_calls").is_some());

        // Check the tool result was updated from IN_PROGRESS
        let tool_msg = &msgs[2];
        assert_eq!(tool_msg["role"], "tool");
        assert!(tool_msg["content"].as_str().unwrap().contains("72"));
    }

    #[tokio::test]
    async fn function_call_result_run_llm_false() {
        let mut agg = make_aggregator();
        let (_downstream, upstream) = run_processor(
            &mut agg,
            vec![
                (
                    Frame::FunctionCallsStarted(FunctionCallsStartedFrame {
                        function_calls: vec![FunctionCallFromLLM {
                            function_name: "fn1".into(),
                            tool_call_id: "call_1".into(),
                            arguments: json!({}),
                        }],
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
                        function_name: "fn1".into(),
                        tool_call_id: "call_1".into(),
                        arguments: json!({}),
                        cancel_on_interruption: false,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallResult(FunctionCallResultFrame {
                        function_name: "fn1".into(),
                        tool_call_id: "call_1".into(),
                        arguments: json!({}),
                        result: json!("ok"),
                        run_llm: Some(false),
                        properties: None,
                    }),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        // run_llm=false: should NOT push context upstream
        assert!(upstream.is_empty());
    }

    #[tokio::test]
    async fn function_call_cancel_respects_cancel_on_interruption() {
        let mut agg = make_aggregator();

        // Call with cancel_on_interruption: false
        let _ = run_processor(
            &mut agg,
            vec![
                (
                    Frame::FunctionCallsStarted(FunctionCallsStartedFrame {
                        function_calls: vec![FunctionCallFromLLM {
                            function_name: "fn1".into(),
                            tool_call_id: "call_1".into(),
                            arguments: json!({}),
                        }],
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
                        function_name: "fn1".into(),
                        tool_call_id: "call_1".into(),
                        arguments: json!({}),
                        cancel_on_interruption: false, // NOT cancellable
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallCancel(FunctionCallCancelFrame {
                        function_name: "fn1".into(),
                        tool_call_id: "call_1".into(),
                    }),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        // Should NOT be cancelled (cancel_on_interruption was false)
        let msgs = agg.context().get_messages();
        let tool_msg = msgs.iter().find(|m| m["role"] == "tool").unwrap();
        assert_eq!(tool_msg["content"], "IN_PROGRESS"); // unchanged
    }

    #[tokio::test]
    async fn function_call_cancel_when_cancellable() {
        let mut agg = make_aggregator();
        let _ = run_processor(
            &mut agg,
            vec![
                (
                    Frame::FunctionCallsStarted(FunctionCallsStartedFrame {
                        function_calls: vec![FunctionCallFromLLM {
                            function_name: "fn1".into(),
                            tool_call_id: "call_1".into(),
                            arguments: json!({}),
                        }],
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
                        function_name: "fn1".into(),
                        tool_call_id: "call_1".into(),
                        arguments: json!({}),
                        cancel_on_interruption: true, // cancellable
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallCancel(FunctionCallCancelFrame {
                        function_name: "fn1".into(),
                        tool_call_id: "call_1".into(),
                    }),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        let msgs = agg.context().get_messages();
        let tool_msg = msgs.iter().find(|m| m["role"] == "tool").unwrap();
        assert_eq!(tool_msg["content"], "CANCELLED");
    }

    #[tokio::test]
    async fn thought_handling() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![
                (
                    Frame::LLMFullResponseStart(LLMFullResponseStartFrame { skip_tts: None }),
                    Direction::Downstream,
                ),
                (
                    Frame::LLMThoughtStart(LLMThoughtStartFrame {
                        append_to_context: true,
                        llm: None,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::LLMThoughtText(LLMThoughtTextFrame {
                        text: "thinking...".into(),
                        includes_inter_frame_spaces: true,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::LLMThoughtEnd(LLMThoughtEndFrame { signature: None }),
                    Direction::Downstream,
                ),
                (
                    Frame::LLMFullResponseEnd(LLMFullResponseEndFrame { skip_tts: None }),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        // All consumed — no frames forwarded (no text was accumulated)
        assert!(downstream.is_empty());

        let msgs = agg.context().get_messages();
        let thought_msg = msgs.iter().find(|m| m.get("thought") == Some(&json!(true)));
        assert!(thought_msg.is_some());
        assert_eq!(thought_msg.unwrap()["content"], "thinking...");
    }

    #[tokio::test]
    async fn context_operations_consumed() {
        let mut agg = make_aggregator();
        let (downstream, upstream) = run_processor(
            &mut agg,
            vec![
                (
                    Frame::LLMMessagesAppend(LLMMessagesAppendFrame {
                        messages: vec![json!({"role": "user", "content": "hello"})],
                        run_llm: None,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::LLMSetTools(LLMSetToolsFrame {
                        tools: json!([{"type": "function"}]),
                    }),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        assert_eq!(agg.context().message_count(), 2); // system + appended
        assert!(agg.context().tools().is_some());
        // Consumed — not forwarded
        assert!(downstream.is_empty());
        assert!(upstream.is_empty());
    }

    #[tokio::test]
    async fn messages_append_with_run_llm() {
        let mut agg = make_aggregator();
        let (_downstream, upstream) = run_processor(
            &mut agg,
            vec![(
                Frame::LLMMessagesAppend(LLMMessagesAppendFrame {
                    messages: vec![json!({"role": "user", "content": "hello"})],
                    run_llm: Some(true),
                }),
                Direction::Downstream,
            )],
        )
        .await;

        // Should push context upstream when run_llm is true
        assert!(
            upstream
                .iter()
                .any(|e| matches!(&e.frame, Frame::LLMContext(_)))
        );
    }

    #[tokio::test]
    async fn end_frame_pushes_aggregation_and_forwards() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![
                (
                    Frame::LLMFullResponseStart(LLMFullResponseStartFrame { skip_tts: None }),
                    Direction::Downstream,
                ),
                (
                    Frame::Text(TextFrame {
                        text: "bye".into(),
                        skip_tts: None,
                        includes_inter_frame_spaces: false,
                        append_to_context: true,
                    }),
                    Direction::Downstream,
                ),
                (Frame::End(EndFrame::default()), Direction::Downstream),
            ],
        )
        .await;

        let msgs = agg.context().get_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["content"], "bye");

        // Context frame + End forwarded
        assert_eq!(downstream.len(), 2);
        assert!(matches!(&downstream[0].frame, Frame::LLMContext(_)));
        assert!(matches!(&downstream[1].frame, Frame::End(_)));
    }

    #[tokio::test]
    async fn forwards_unknown_frames() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![(Frame::Start(StartFrame::default()), Direction::Downstream)],
        )
        .await;

        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::Start(_)));
    }

    #[tokio::test]
    async fn llm_run_pushes_context_upstream_only() {
        let mut agg = make_aggregator();
        let (downstream, upstream) = run_processor(
            &mut agg,
            vec![(Frame::LLMRun(LLMRunFrame), Direction::Downstream)],
        )
        .await;

        // Consumed — not forwarded downstream
        assert!(downstream.is_empty());
        // Context pushed upstream
        assert_eq!(upstream.len(), 1);
        assert!(matches!(&upstream[0].frame, Frame::LLMContext(_)));
    }

    /// Fix 4: Null result on FunctionCallResult defaults run_llm to false.
    #[tokio::test]
    async fn function_call_null_result_does_not_run_llm() {
        let mut agg = make_aggregator();
        let (_downstream, upstream) = run_processor(
            &mut agg,
            vec![
                (
                    Frame::FunctionCallsStarted(FunctionCallsStartedFrame {
                        function_calls: vec![FunctionCallFromLLM {
                            function_name: "notify".into(),
                            tool_call_id: "call_1".into(),
                            arguments: json!({}),
                        }],
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
                        function_name: "notify".into(),
                        tool_call_id: "call_1".into(),
                        arguments: json!({}),
                        cancel_on_interruption: false,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::FunctionCallResult(FunctionCallResultFrame {
                        function_name: "notify".into(),
                        tool_call_id: "call_1".into(),
                        arguments: json!({}),
                        result: serde_json::Value::Null, // null result
                        run_llm: None,                   // not explicitly set
                        properties: None,
                    }),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        // Null result: run_llm defaults to false, no context pushed upstream
        assert!(
            upstream.is_empty(),
            "Null result should not trigger LLM re-run"
        );
    }

    /// Fix 5: Interruption resets thought state.
    #[tokio::test]
    async fn interruption_resets_thought_state() {
        let mut agg = make_aggregator();
        let _ = run_processor(
            &mut agg,
            vec![
                (
                    Frame::LLMFullResponseStart(LLMFullResponseStartFrame { skip_tts: None }),
                    Direction::Downstream,
                ),
                (
                    Frame::LLMThoughtStart(LLMThoughtStartFrame {
                        append_to_context: true,
                        llm: None,
                    }),
                    Direction::Downstream,
                ),
                (
                    Frame::LLMThoughtText(LLMThoughtTextFrame {
                        text: "hmm let me think".into(),
                        includes_inter_frame_spaces: false,
                    }),
                    Direction::Downstream,
                ),
                // Interruption before thought ends
                (
                    Frame::Interruption(InterruptionFrame),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        // Thought state should be reset
        assert!(!agg.in_thought);
        assert!(agg.thought_text.is_empty());
    }
}
