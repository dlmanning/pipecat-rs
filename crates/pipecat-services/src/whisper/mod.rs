pub mod model;

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tracing::info;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::settings::STTSettings;
use crate::stt::STTServiceState;

/// One second of pre-speech audio at 16 kHz mono 16-bit PCM.
const PRE_SPEECH_BUFFER_BYTES: usize = 16000 * 2;

/// Local Whisper STT service using whisper.cpp via whisper-rs.
///
/// Unlike cloud STT services that process streaming audio chunks,
/// Whisper needs a complete audio segment. This processor accumulates
/// audio while the user is speaking and runs transcription on
/// `VADUserStoppedSpeaking`.
#[derive(Debug)]
pub struct WhisperSTTService {
    state: STTServiceState,
    whisper_ctx: Arc<WhisperContext>,
    audio_buf: Vec<u8>,
    language: Option<String>,
}

impl WhisperSTTService {
    /// Create a new Whisper STT service from a GGML model file path.
    pub fn new(model_path: &Path, settings: STTSettings) -> Result<Self> {
        let language = settings.language.clone();
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().ok_or_else(|| {
                PipecatError::ProcessorError("invalid model path encoding".into())
            })?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| PipecatError::ProcessorError(format!("failed to load Whisper model: {e}")))?;

        Ok(Self {
            state: STTServiceState::new("WhisperSTT", settings),
            whisper_ctx: Arc::new(ctx),
            audio_buf: Vec::new(),
            language,
        })
    }

    /// Run Whisper transcription on the buffered audio and emit a TranscriptionFrame.
    async fn transcribe(&mut self, ctx: &ProcessorContext) -> Result<()> {
        if self.state.muted || self.audio_buf.is_empty() {
            return Ok(());
        }

        let audio_bytes = std::mem::take(&mut self.audio_buf);
        let whisper_ctx = self.whisper_ctx.clone();
        let language = self.language.clone();

        let text = tokio::task::spawn_blocking(move || -> std::result::Result<String, String> {
            // Convert i16 LE PCM to f32 normalized samples.
            let samples: Vec<f32> = audio_bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();

            let mut state = whisper_ctx.create_state().map_err(|e| e.to_string())?;

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            if let Some(ref lang) = language {
                params.set_language(Some(lang));
            }
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            params.set_no_timestamps(true);

            state.full(params, &samples).map_err(|e| e.to_string())?;

            let mut text = String::new();
            for segment in state.as_iter() {
                if let Ok(seg_text) = segment.to_str() {
                    text.push_str(seg_text);
                }
            }
            Ok(text.trim().to_string())
        })
        .await
        .map_err(|e| PipecatError::ProcessorError(format!("whisper task panicked: {e}")))?
        .map_err(|e| PipecatError::ProcessorError(format!("whisper inference failed: {e}")))?;

        if !text.is_empty() {
            info!(text = %text, "Whisper transcription");
            ctx.send_downstream(Frame::Transcription(TranscriptionFrame {
                text,
                user_id: "user".to_string(),
                timestamp: None,
                language: self.language.clone(),
                finalized: true,
                result: None,
            }))
            .await?;
        }

        Ok(())
    }
}

#[async_trait]
impl FrameProcessor for WhisperSTTService {
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
            Frame::Start(f) => {
                self.state.base.handle_start_frame(f);
                if self.state.sample_rate == 0 {
                    self.state.sample_rate = f.audio_in_sample_rate;
                }
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::InputAudioRaw(audio) => {
                if self.state.audio_passthrough {
                    ctx.send_downstream(Frame::InputAudioRaw(audio.clone()))
                        .await?;
                }

                if !self.state.muted {
                    self.audio_buf.extend_from_slice(&audio.audio);
                    // When not speaking, keep only a pre-speech buffer (1s).
                    if !self.state.user_speaking && self.audio_buf.len() > PRE_SPEECH_BUFFER_BYTES {
                        let excess = self.audio_buf.len() - PRE_SPEECH_BUFFER_BYTES;
                        self.audio_buf.drain(..excess);
                    }
                }
            }

            Frame::VADUserStartedSpeaking(_) | Frame::UserStartedSpeaking(_) => {
                self.state.user_speaking = true;
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::VADUserStoppedSpeaking(_) | Frame::UserStoppedSpeaking(_) => {
                self.state.user_speaking = false;
                self.transcribe(ctx).await?;
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::Interruption(_) => {
                self.audio_buf.clear();
                self.state.base.metrics.reset();
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::STTMute(f) => {
                self.state.muted = f.mute;
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::STTUpdateSettings(f) => {
                let changed = self.state.settings.apply_delta(&f.settings);
                if changed.contains(&"model".to_string())
                    && let Some(ref model) = self.state.settings.model
                {
                    self.state.base.set_model(model.clone());
                }
                ctx.push_frame(envelope, direction).await?;
            }

            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }

        Ok(())
    }
}
