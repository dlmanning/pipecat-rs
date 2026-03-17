use std::time::Duration;

use async_trait::async_trait;
use tracing::{debug, trace};

use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_turns::{
    TurnAction, UserTurnController, UserTurnStartedParams, UserTurnStoppedParams,
    UserTurnStrategies,
};

use crate::aggregator::LLMContextAggregatorBase;
use crate::context::LLMContext;

/// Parameters for configuring the user aggregator.
#[derive(Debug)]
pub struct LLMUserAggregatorParams {
    pub user_turn_strategies: UserTurnStrategies,
    pub user_turn_stop_timeout: Duration,
}

/// Manages user turn lifecycle with `UserTurnController`.
///
/// Accumulates transcription text during a user turn and pushes a user message
/// to the shared LLM context when the turn ends. Consumes transcription,
/// context operation, and LLMRun frames; forwards everything else downstream.
#[derive(Debug)]
pub struct LLMUserAggregator {
    base: ProcessorBase,
    agg: LLMContextAggregatorBase,
    turn_controller: UserTurnController,
    /// Whether a user turn is currently active (from our perspective).
    in_turn: bool,
}

impl LLMUserAggregator {
    pub fn new(context: LLMContext, params: LLMUserAggregatorParams) -> Self {
        let turn_controller =
            UserTurnController::new(params.user_turn_strategies, params.user_turn_stop_timeout);
        Self {
            base: ProcessorBase::new("LLMUserAggregator"),
            agg: LLMContextAggregatorBase::new(context, "user"),
            turn_controller,
            in_turn: false,
        }
    }

    /// Access the shared context.
    pub fn context(&self) -> &LLMContext {
        self.agg.context()
    }

    /// Push an LLMContextFrame downstream.
    async fn push_context_frame(&self, ctx: &ProcessorContext) -> Result<()> {
        let context_value = self.agg.context().to_context_value();
        ctx.send_downstream(Frame::LLMContext(LLMContextFrame {
            context: context_value,
        }))
        .await
    }

    /// Execute turn actions returned by the controller.
    async fn execute_turn_actions(
        &mut self,
        actions: Vec<TurnAction>,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        for action in actions {
            match action {
                TurnAction::UserTurnStarted(params) => {
                    self.handle_turn_started(params, ctx).await?;
                }
                TurnAction::UserTurnStopped(params) => {
                    self.handle_turn_stopped(params, ctx).await?;
                }
                TurnAction::UserTurnStopTimeout => {
                    debug!("User turn stop timeout");
                }
                TurnAction::PushFrame(envelope, direction) => {
                    ctx.push_frame(envelope, direction).await?;
                }
                TurnAction::BroadcastFrame(frame) => {
                    ctx.broadcast(frame).await?;
                }
            }
        }
        Ok(())
    }

    async fn handle_turn_started(
        &mut self,
        params: UserTurnStartedParams,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        self.in_turn = true;
        debug!("User turn started");

        if params.enable_interruptions {
            ctx.broadcast_interruption().await?;
        }

        if params.enable_user_speaking_frames {
            ctx.broadcast(Frame::UserStartedSpeaking(UserStartedSpeakingFrame))
                .await?;
        }

        Ok(())
    }

    async fn handle_turn_stopped(
        &mut self,
        params: UserTurnStoppedParams,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        if !self.in_turn {
            return Ok(());
        }
        self.in_turn = false;
        debug!("User turn stopped");

        // Push accumulated text as a user message + context frame downstream
        let text = self.agg.push_aggregation();
        if !text.is_empty() {
            trace!("User turn aggregated text: {}", text);
            self.push_context_frame(ctx).await?;
        }

        if params.enable_user_speaking_frames {
            ctx.broadcast(Frame::UserStoppedSpeaking(UserStoppedSpeakingFrame))
                .await?;
        }

        Ok(())
    }

    /// Emit turn stopped if currently in a turn, and always push any
    /// remaining aggregation (for End/Cancel lifecycle — matches Python's
    /// `_maybe_emit_user_turn_stopped(on_session_end=True)` behavior).
    async fn emit_turn_stopped_if_needed(&mut self, ctx: &ProcessorContext) -> Result<()> {
        if self.in_turn {
            self.handle_turn_stopped(
                UserTurnStoppedParams {
                    enable_user_speaking_frames: true,
                },
                ctx,
            )
            .await?;
        } else {
            // Even outside a turn, flush any remaining text to context
            let text = self.agg.push_aggregation();
            if !text.is_empty() {
                trace!("Flushing remaining aggregation on session end: {}", text);
                self.push_context_frame(ctx).await?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl FrameProcessor for LLMUserAggregator {
    fn name(&self) -> &str {
        self.base.name()
    }

    fn id(&self) -> u64 {
        self.base.id()
    }

    async fn setup(&mut self) {
        self.turn_controller.setup().await;
    }

    async fn cleanup(&mut self) {
        self.turn_controller.cleanup().await;
    }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        // Route the frame FIRST, then pass it through the turn controller.
        // This matches Python's ordering: accumulate transcription text before
        // any pending turn-stop fires, so no transcription is lost.
        let frame_for_turn = envelope.frame.clone();
        match &envelope.frame {
            // --- Transcription: accumulate text (consumed) ---
            Frame::Transcription(t) => {
                if !t.text.trim().is_empty() {
                    trace!("UserAggregator: transcription: {:?}", t.text);
                    self.agg.add_text_part(t.text.clone(), false);
                }
            }

            // --- Interim transcription / translation: consumed ---
            Frame::InterimTranscription(_) | Frame::Translation(_) => {}

            // --- LLMRun: push context downstream (consumed) ---
            Frame::LLMRun(_) => {
                self.push_context_frame(ctx).await?;
                // Consumed — do NOT forward the LLMRunFrame
            }

            // --- Context operations (consumed) ---
            Frame::LLMMessagesAppend(f) => {
                self.agg.context().add_messages(&f.messages);
                if f.run_llm == Some(true) {
                    self.push_context_frame(ctx).await?;
                }
                // Consumed
            }

            Frame::LLMMessagesUpdate(f) => {
                self.agg.context().set_messages(f.messages.clone());
                if f.run_llm == Some(true) {
                    self.push_context_frame(ctx).await?;
                }
                // Consumed
            }

            Frame::LLMSetTools(f) => {
                self.agg.context().set_tools(f.tools.clone());
                // Forward downstream (speech-to-speech services need tools config)
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::LLMSetToolChoice(f) => {
                self.agg.context().set_tool_choice(f.tool_choice.clone());
                // Consumed — not forwarded
            }

            // --- Lifecycle (forwarded) ---
            Frame::End(_) | Frame::Cancel(_) => {
                self.emit_turn_stopped_if_needed(ctx).await?;
                ctx.push_frame(envelope, direction).await?;
            }

            // --- Everything else: forward ---
            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }

        // Run the turn controller AFTER frame routing, so transcription text
        // is accumulated before any pending turn-stop fires.
        let actions = self.turn_controller.process_frame(&frame_for_turn).await;
        self.execute_turn_actions(actions, ctx).await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipecat_core::test_utils::run_processor;
    use pipecat_turns::{SpeechTimeoutUserTurnStopStrategy, VadUserTurnStartStrategy};
    use serde_json::json;

    fn make_params() -> LLMUserAggregatorParams {
        LLMUserAggregatorParams {
            user_turn_strategies: UserTurnStrategies {
                start: vec![Box::new(VadUserTurnStartStrategy::new())],
                stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.1))],
            },
            user_turn_stop_timeout: Duration::from_secs(5),
        }
    }

    fn make_aggregator() -> LLMUserAggregator {
        let ctx = LLMContext::new(vec![json!({"role": "system", "content": "test"})]);
        LLMUserAggregator::new(ctx, make_params())
    }

    fn vad_started() -> Frame {
        Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
            start_secs: 0.2,
            timestamp: 1.0,
        })
    }

    fn transcription(text: &str) -> Frame {
        Frame::Transcription(TranscriptionFrame {
            text: text.into(),
            user_id: "user1".into(),
            timestamp: None,
            language: None,
            finalized: true,
            result: None,
        })
    }

    #[tokio::test]
    async fn transcription_accumulation() {
        let mut agg = make_aggregator();
        agg.setup().await;

        // Start turn via VAD
        let (downstream, _) = run_processor(
            &mut agg,
            vec![
                (vad_started(), Direction::Downstream),
                (transcription("hello"), Direction::Downstream),
                (transcription("world"), Direction::Downstream),
            ],
        )
        .await;

        // Transcription frames are consumed (not forwarded)
        let forwarded_transcriptions: Vec<_> = downstream
            .iter()
            .filter(|e| matches!(&e.frame, Frame::Transcription(_)))
            .collect();
        assert!(forwarded_transcriptions.is_empty());

        agg.cleanup().await;
    }

    #[tokio::test]
    async fn interim_transcription_consumed() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![(
                Frame::InterimTranscription(InterimTranscriptionFrame {
                    text: "hel".into(),
                    user_id: "u1".into(),
                    timestamp: None,
                    language: None,
                    result: None,
                }),
                Direction::Downstream,
            )],
        )
        .await;

        assert!(downstream.is_empty());
    }

    #[tokio::test]
    async fn llm_run_pushes_context_only() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![(Frame::LLMRun(LLMRunFrame), Direction::Downstream)],
        )
        .await;

        // Should produce only LLMContext downstream (LLMRun is consumed)
        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::LLMContext(_)));
    }

    #[tokio::test]
    async fn messages_append_consumed() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![(
                Frame::LLMMessagesAppend(LLMMessagesAppendFrame {
                    messages: vec![json!({"role": "user", "content": "appended"})],
                    run_llm: None,
                }),
                Direction::Downstream,
            )],
        )
        .await;

        assert_eq!(agg.context().message_count(), 2);
        // Consumed — not forwarded
        assert!(downstream.is_empty());
    }

    #[tokio::test]
    async fn messages_append_with_run_llm() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![(
                Frame::LLMMessagesAppend(LLMMessagesAppendFrame {
                    messages: vec![json!({"role": "user", "content": "go"})],
                    run_llm: Some(true),
                }),
                Direction::Downstream,
            )],
        )
        .await;

        // Should push context downstream when run_llm is true
        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::LLMContext(_)));
    }

    #[tokio::test]
    async fn set_tools_forwarded_downstream() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![(
                Frame::LLMSetTools(LLMSetToolsFrame {
                    tools: json!([{"type": "function"}]),
                }),
                Direction::Downstream,
            )],
        )
        .await;

        assert!(agg.context().tools().is_some());
        // LLMSetTools is forwarded downstream
        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::LLMSetTools(_)));
    }

    #[tokio::test]
    async fn set_tool_choice_consumed() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![(
                Frame::LLMSetToolChoice(LLMSetToolChoiceFrame {
                    tool_choice: json!("auto"),
                }),
                Direction::Downstream,
            )],
        )
        .await;

        assert!(agg.context().tool_choice().is_some());
        // Consumed — not forwarded
        assert!(downstream.is_empty());
    }

    #[tokio::test]
    async fn end_frame_forwarded() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![(Frame::End(EndFrame::default()), Direction::Downstream)],
        )
        .await;

        let has_end = downstream.iter().any(|e| matches!(&e.frame, Frame::End(_)));
        assert!(has_end);
    }

    #[tokio::test]
    async fn unknown_frames_forwarded() {
        let mut agg = make_aggregator();
        let (downstream, _) = run_processor(
            &mut agg,
            vec![(Frame::Start(StartFrame::default()), Direction::Downstream)],
        )
        .await;

        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::Start(_)));
    }

    /// Fix 1: Turn controller runs AFTER frame routing, so a transcription
    /// arriving at the same time as a pending turn stop is accumulated first.
    #[tokio::test]
    async fn transcription_before_turn_stop_is_included() {
        let mut agg = make_aggregator();
        agg.setup().await;

        // Start a turn
        let _ = run_processor(&mut agg, vec![(vad_started(), Direction::Downstream)]).await;

        // Simulate: VAD stopped speaking (starts the speech timeout)
        let _ = run_processor(
            &mut agg,
            vec![(
                Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
                    stop_secs: 0.4,
                    timestamp: 1.2,
                }),
                Direction::Downstream,
            )],
        )
        .await;

        // Wait for the speech timeout to fire (0.1s configured)
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Now send transcription — the turn stop is pending but hasn't been
        // processed yet. With the fix, the transcription text is accumulated
        // BEFORE the turn controller processes and fires the stop.
        let (downstream, _) = run_processor(
            &mut agg,
            vec![(transcription("last words"), Direction::Downstream)],
        )
        .await;

        // The turn should have stopped and pushed a context frame containing
        // the transcription text
        let context_frames: Vec<_> = downstream
            .iter()
            .filter(|e| matches!(&e.frame, Frame::LLMContext(_)))
            .collect();
        assert!(
            !context_frames.is_empty(),
            "Expected context frame with aggregated text"
        );

        // The context should contain the transcription
        let msgs = agg.context().get_messages();
        let user_msg = msgs.iter().find(|m| m["role"] == "user");
        assert!(user_msg.is_some(), "Expected user message in context");
        assert_eq!(user_msg.unwrap()["content"], "last words");

        agg.cleanup().await;
    }

    /// Fix 2: UserStartedSpeaking/UserStoppedSpeaking are broadcast both
    /// directions (not just downstream).
    #[tokio::test]
    async fn speaking_frames_broadcast_both_directions() {
        let mut agg = make_aggregator();
        agg.setup().await;

        // Start turn via VAD (VadUserTurnStartStrategy enables speaking frames)
        let (downstream, upstream) =
            run_processor(&mut agg, vec![(vad_started(), Direction::Downstream)]).await;

        // UserStartedSpeaking should appear in BOTH directions
        let has_started_down = downstream
            .iter()
            .any(|e| matches!(&e.frame, Frame::UserStartedSpeaking(_)));
        let has_started_up = upstream
            .iter()
            .any(|e| matches!(&e.frame, Frame::UserStartedSpeaking(_)));
        assert!(has_started_down, "UserStartedSpeaking not sent downstream");
        assert!(has_started_up, "UserStartedSpeaking not sent upstream");

        agg.cleanup().await;
    }

    /// Fix 3: End frame always pushes remaining aggregation even when not
    /// in a turn.
    #[tokio::test]
    async fn end_frame_flushes_aggregation_outside_turn() {
        let mut agg = make_aggregator();

        // Manually add text without starting a turn
        agg.agg.add_text_part("orphaned text".into(), false);

        let (downstream, _) = run_processor(
            &mut agg,
            vec![(Frame::End(EndFrame::default()), Direction::Downstream)],
        )
        .await;

        // Should have flushed the text to context
        let msgs = agg.context().get_messages();
        let user_msg = msgs.iter().find(|m| m["role"] == "user");
        assert!(user_msg.is_some(), "Expected user message from flush");
        assert_eq!(user_msg.unwrap()["content"], "orphaned text");

        // Context frame + End frame should be downstream
        let has_context = downstream
            .iter()
            .any(|e| matches!(&e.frame, Frame::LLMContext(_)));
        let has_end = downstream.iter().any(|e| matches!(&e.frame, Frame::End(_)));
        assert!(has_context, "Expected context frame downstream");
        assert!(has_end, "Expected End frame downstream");
    }
}
