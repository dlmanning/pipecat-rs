use async_trait::async_trait;
use bytes::Bytes;

use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};

use crate::service_base::ServiceBase;
use crate::settings::STTSettings;

/// State held by STT service implementations.
#[derive(Debug)]
pub struct STTServiceState {
    pub base: ServiceBase,
    pub settings: STTSettings,
    pub sample_rate: u32,
    pub muted: bool,
    pub audio_passthrough: bool,
    pub user_speaking: bool,
}

impl STTServiceState {
    pub fn new(name: impl Into<String>, settings: STTSettings) -> Self {
        let model = settings.model.clone();
        Self {
            base: ServiceBase::new(name, model),
            settings,
            sample_rate: 0,
            muted: false,
            audio_passthrough: true,
            user_speaking: false,
        }
    }
}

/// Trait for STT service implementations.
///
/// Implementors provide `run_stt` which processes audio and pushes
/// transcription frames via `ctx`. Frame routing is handled by the
/// shared `stt_process_frame` helper.
#[async_trait]
pub trait STTService: FrameProcessor {
    /// Process audio data and push transcription frames via ctx.
    async fn run_stt(&mut self, audio: Bytes, ctx: &ProcessorContext) -> Result<()>;

    /// Access the STT service state.
    fn stt_service_state(&self) -> &STTServiceState;
    fn stt_service_state_mut(&mut self) -> &mut STTServiceState;
}

/// Common frame routing for STT services. Call from `process_frame`.
pub async fn stt_process_frame(
    stt: &mut dyn STTService,
    envelope: FrameEnvelope,
    direction: Direction,
    ctx: &ProcessorContext,
) -> Result<()> {
    match &envelope.frame {
        Frame::Start(f) => {
            let state = stt.stt_service_state_mut();
            state.base.handle_start_frame(f);
            if state.sample_rate == 0 {
                state.sample_rate = f.audio_in_sample_rate;
            }
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::InputAudioRaw(audio) => {
            let state = stt.stt_service_state();
            if state.audio_passthrough {
                ctx.send_downstream(Frame::InputAudioRaw(audio.clone()))
                    .await?;
            }

            if !state.muted {
                let audio_bytes = audio.audio.clone();
                stt.run_stt(audio_bytes, ctx).await?;
            }
        }

        Frame::STTMute(f) => {
            stt.stt_service_state_mut().muted = f.mute;
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::STTUpdateSettings(f) => {
            let state = stt.stt_service_state_mut();
            let changed = state.settings.apply_delta(&f.settings);
            if changed.contains(&"model".to_string())
                && let Some(ref model) = state.settings.model
            {
                state.base.set_model(model.clone());
            }
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::UserStartedSpeaking(_) | Frame::VADUserStartedSpeaking(_) => {
            stt.stt_service_state_mut().user_speaking = true;
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::UserStoppedSpeaking(_) | Frame::VADUserStoppedSpeaking(_) => {
            stt.stt_service_state_mut().user_speaking = false;
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::Interruption(_) => {
            stt.stt_service_state_mut().base.metrics.reset();
            ctx.push_frame(envelope, direction).await?;
        }

        _ => {
            ctx.push_frame(envelope, direction).await?;
        }
    }

    Ok(())
}
