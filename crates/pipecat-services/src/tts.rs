use async_trait::async_trait;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tracing::trace;

use crate::service_base::ServiceBase;
use crate::settings::TTSSettings;
use crate::text_aggregator::{SimpleTextAggregator, TextAggregationMode};

/// State held by TTS service implementations.
#[derive(Debug)]
pub struct TTSServiceState {
    pub base: ServiceBase,
    pub settings: TTSSettings,
    pub sample_rate: u32,
    pub aggregator: SimpleTextAggregator,
    pub turn_context_id: Option<String>,
    pub llm_response_started: bool,
    push_text_frames: bool,
    text_agg_metrics_started: bool,
}

impl TTSServiceState {
    pub fn new(
        name: impl Into<String>,
        settings: TTSSettings,
        aggregation_mode: TextAggregationMode,
        sample_rate: u32,
    ) -> Self {
        let model = settings.model.clone();
        Self {
            base: ServiceBase::new(name, model),
            settings,
            sample_rate,
            aggregator: SimpleTextAggregator::new(aggregation_mode),
            turn_context_id: None,
            llm_response_started: false,
            push_text_frames: true,
            text_agg_metrics_started: false,
        }
    }

    /// Set whether to push TTSText frames downstream.
    pub fn set_push_text_frames(&mut self, push: bool) {
        self.push_text_frames = push;
    }
}

/// Trait for TTS service implementations.
///
/// Implementors provide `run_tts` which generates audio from text and pushes
/// audio frames via `ctx`. Frame routing is handled by the shared
/// `tts_process_frame` helper.
#[async_trait]
pub trait TTSService: FrameProcessor {
    /// Generate audio from text and push audio frames via ctx.
    async fn run_tts(&mut self, text: &str, context_id: &str, ctx: &ProcessorContext)
    -> Result<()>;

    /// Access the TTS service state.
    fn tts_service_state(&self) -> &TTSServiceState;
    fn tts_service_state_mut(&mut self) -> &mut TTSServiceState;
}

/// Common frame routing for TTS services. Call from `process_frame`.
pub async fn tts_process_frame(
    tts: &mut dyn TTSService,
    envelope: FrameEnvelope,
    direction: Direction,
    ctx: &ProcessorContext,
) -> Result<()> {
    match &envelope.frame {
        Frame::Start(f) => {
            let state = tts.tts_service_state_mut();
            state.base.handle_start_frame(f);
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::LLMFullResponseStart(_) => {
            let state = tts.tts_service_state_mut();
            state.llm_response_started = true;
            state.text_agg_metrics_started = false;
            // Generate a new context ID for this response
            let context_id = uuid_context_id();
            state.turn_context_id = Some(context_id);
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::LLMFullResponseEnd(_) => {
            // Extract data from state, then drop the borrow before calling run_tts.
            // Flush first, then stop text aggregation metrics (matching Python:
            // flush produces the final segment, so the timer should include it).
            let (flushed, context_id, push_text_frames) = {
                let state = tts.tts_service_state_mut();
                state.llm_response_started = false;
                let flushed = state.aggregator.flush();
                let context_id = state.turn_context_id.clone().unwrap_or_default();
                let push = state.push_text_frames;
                (flushed, context_id, push)
            };

            if let Some(agg_text) = flushed {
                if push_text_frames {
                    ctx.send_downstream(Frame::TTSText(AggregatedTextFrame {
                        text: agg_text.text.clone(),
                        aggregated_by: agg_text.aggregated_by,
                        context_id: Some(context_id.clone()),
                        skip_tts: None,
                        includes_inter_frame_spaces: false,
                        append_to_context: true,
                    }))
                    .await?;
                }
                tts.run_tts(&agg_text.text, &context_id, ctx).await?;
            }

            // Stop text aggregation metrics after flush (timer includes the
            // full aggregation window through the final flush).
            {
                let state = tts.tts_service_state_mut();
                if state.text_agg_metrics_started && state.base.metrics_enabled() {
                    ctx.push_text_aggregation_metrics(&mut state.base.metrics)
                        .await?;
                }
                state.text_agg_metrics_started = false;
            }

            ctx.push_frame(envelope, direction).await?;
        }

        Frame::Text(t) => {
            // Check skip_tts
            if t.skip_tts == Some(true) {
                ctx.push_frame(envelope, direction).await?;
                return Ok(());
            }

            // Extract everything we need from the borrowed frame and state
            let text = t.text.clone();
            let includes_inter_frame_spaces = t.includes_inter_frame_spaces;
            let append_to_context = t.append_to_context;

            let (segments, context_id, push_text_frames) = {
                let state = tts.tts_service_state_mut();

                // Start text aggregation metrics on first text frame (sentence
                // mode only — token mode has no aggregation delay).
                if !state.text_agg_metrics_started
                    && state.base.metrics_enabled()
                    && state.aggregator.mode() == TextAggregationMode::Sentence
                {
                    state.base.metrics.start_text_aggregation();
                    state.text_agg_metrics_started = true;
                }

                let context_id = state.turn_context_id.clone().unwrap_or_default();
                let segments = state.aggregator.process(&text);
                let push = state.push_text_frames;
                (segments, context_id, push)
            };

            for seg in segments {
                trace!("TTS aggregated text: {:?}", seg.text);

                // Stop text aggregation metrics on first produced segment —
                // this is the latency cost of sentence aggregation.
                {
                    let state = tts.tts_service_state_mut();
                    if state.text_agg_metrics_started && state.base.metrics_enabled() {
                        ctx.push_text_aggregation_metrics(&mut state.base.metrics)
                            .await?;
                        state.text_agg_metrics_started = false;
                    }
                }

                if push_text_frames {
                    ctx.send_downstream(Frame::TTSText(AggregatedTextFrame {
                        text: seg.text.clone(),
                        aggregated_by: seg.aggregated_by,
                        context_id: Some(context_id.clone()),
                        skip_tts: None,
                        includes_inter_frame_spaces,
                        append_to_context,
                    }))
                    .await?;
                }

                tts.run_tts(&seg.text, &context_id, ctx).await?;
            }
        }

        Frame::TTSUpdateSettings(f) => {
            let state = tts.tts_service_state_mut();
            let changed = state.settings.apply_delta(&f.settings);
            if changed.contains(&"model".to_string())
                && let Some(ref model) = state.settings.model
            {
                state.base.set_model(model.clone());
            }
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::End(_) => {
            // Flush remaining text and run TTS, then forward
            let (flushed, context_id, push_text_frames) = {
                let state = tts.tts_service_state_mut();
                let flushed = state.aggregator.flush();
                let context_id = state.turn_context_id.clone().unwrap_or_default();
                let push = state.push_text_frames;
                (flushed, context_id, push)
            };

            if let Some(agg_text) = flushed {
                if push_text_frames {
                    ctx.send_downstream(Frame::TTSText(AggregatedTextFrame {
                        text: agg_text.text.clone(),
                        aggregated_by: agg_text.aggregated_by,
                        context_id: Some(context_id.clone()),
                        skip_tts: None,
                        includes_inter_frame_spaces: false,
                        append_to_context: true,
                    }))
                    .await?;
                }
                tts.run_tts(&agg_text.text, &context_id, ctx).await?;
            }

            ctx.push_frame(envelope, direction).await?;
        }

        Frame::Interruption(_) => {
            let state = tts.tts_service_state_mut();
            state.aggregator.reset();
            state.base.metrics.reset();
            state.llm_response_started = false;
            state.text_agg_metrics_started = false;
            ctx.push_frame(envelope, direction).await?;
        }

        _ => {
            ctx.push_frame(envelope, direction).await?;
        }
    }

    Ok(())
}

/// Generate a simple context ID.
fn uuid_context_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    format!("ctx_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}
