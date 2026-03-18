use std::process::Stdio;

use async_trait::async_trait;
use bytes::Bytes;
use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tokio::process::Command;
use tracing::{debug, warn};

use super::settings::MacOSSaySettings;
use crate::text_aggregator::TextAggregationMode;
use crate::tts::{TTSService, TTSServiceState, tts_process_frame};

/// Default sample rate for `say` output.
const DEFAULT_SAMPLE_RATE: u32 = 16000;

/// WAV header size (standard 44-byte RIFF/WAVE header).
const WAV_HEADER_SIZE: usize = 44;

/// TTS service using the macOS `say` command.
///
/// Synthesizes speech by invoking `say -o <tmpfile> --file-format=WAVE
/// --data-format=LEI16@<rate>`, then reads the WAV file and pushes raw
/// PCM audio frames. No API key or network required.
///
/// Supports voice selection (`-v`), speech rate (`-r`), and configurable
/// sample rate. Output is always mono 16-bit little-endian PCM.
#[derive(Debug)]
pub struct MacOSSayTTSService {
    state: TTSServiceState,
    settings: MacOSSaySettings,
}

impl MacOSSayTTSService {
    pub fn new(settings: MacOSSaySettings) -> Self {
        Self::with_sample_rate(settings, DEFAULT_SAMPLE_RATE)
    }

    pub fn with_sample_rate(settings: MacOSSaySettings, sample_rate: u32) -> Self {
        Self {
            state: TTSServiceState::new(
                "MacOSSayTTSService",
                settings.base.clone(),
                TextAggregationMode::Sentence,
                sample_rate,
            ),
            settings,
        }
    }
}

#[async_trait]
impl FrameProcessor for MacOSSayTTSService {
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
                self.state.sample_rate = f.audio_out_sample_rate;
                tts_process_frame(self, envelope, direction, ctx).await?;
            }
            _ => {
                tts_process_frame(self, envelope, direction, ctx).await?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl TTSService for MacOSSayTTSService {
    async fn run_tts(
        &mut self,
        text: &str,
        context_id: &str,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let sample_rate = self.state.sample_rate;
        let data_format = format!("LEI16@{sample_rate}");

        // Create a temp file for the output
        let tmp_path = std::env::temp_dir().join(format!(
            "pipecat_say_{}.wav",
            self.state.base.processor.id()
        ));

        let mut cmd = Command::new("say");
        cmd.arg("-o").arg(&tmp_path);
        cmd.arg("--file-format=WAVE");
        cmd.arg(format!("--data-format={data_format}"));

        // Voice — read from state.settings so TTSUpdateSettings takes effect
        if let Some(ref voice) = self.state.settings.voice {
            cmd.arg("-v").arg(voice);
        }

        // Speech rate
        if let Some(rate) = self.settings.rate {
            cmd.arg("-r").arg(rate.to_string());
        }

        cmd.arg(text);
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());

        debug!("MacOSSay: synthesizing {} chars", text.len());

        // Start TTFB metrics
        let metrics_enabled = self.state.base.metrics_enabled();
        if metrics_enabled {
            self.state.base.metrics.start_ttfb();
        }

        let output = cmd
            .output()
            .await
            .map_err(|e| PipecatError::ProcessorError(format!("Failed to run say: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("say failed: {}", stderr);
            ctx.push_error(&format!("say error: {stderr}"), false)
                .await?;
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Ok(());
        }

        // Read the WAV file
        let wav_data = tokio::fs::read(&tmp_path)
            .await
            .map_err(|e| PipecatError::ProcessorError(format!("Failed to read say output: {e}")))?;

        // Clean up temp file
        let _ = tokio::fs::remove_file(&tmp_path).await;

        if wav_data.len() <= WAV_HEADER_SIZE {
            return Ok(());
        }

        // Stop TTFB on first audio
        if metrics_enabled {
            ctx.push_ttfb(&mut self.state.base.metrics).await?;
        }

        // Strip WAV header and push raw PCM
        let pcm_data = &wav_data[WAV_HEADER_SIZE..];
        ctx.send_downstream(Frame::TTSAudioRaw(TTSAudioRawFrame {
            audio: Bytes::copy_from_slice(pcm_data),
            sample_rate,
            num_channels: 1,
            context_id: Some(context_id.to_string()),
        }))
        .await?;

        // Push TTS usage metrics
        if self.state.base.usage_metrics_enabled() {
            ctx.push_tts_usage(&self.state.base.metrics, text).await?;
        }

        Ok(())
    }

    fn tts_service_state(&self) -> &TTSServiceState {
        &self.state
    }

    fn tts_service_state_mut(&mut self) -> &mut TTSServiceState {
        &mut self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::TTSSettings;

    #[test]
    fn build_command_default() {
        let settings = MacOSSaySettings::default();
        let svc = MacOSSayTTSService::new(settings);
        assert_eq!(svc.state.sample_rate, DEFAULT_SAMPLE_RATE);
    }

    #[test]
    fn build_command_with_voice() {
        let settings = MacOSSaySettings::new(TTSSettings {
            voice: Some("Samantha".into()),
            ..Default::default()
        });
        let svc = MacOSSayTTSService::new(settings);
        assert_eq!(svc.state.settings.voice.as_deref(), Some("Samantha"));
    }

    #[test]
    fn custom_sample_rate() {
        let settings = MacOSSaySettings::default();
        let svc = MacOSSayTTSService::with_sample_rate(settings, 22050);
        assert_eq!(svc.state.sample_rate, 22050);
    }

    #[tokio::test]
    async fn process_frame_forwards_non_tts_frames() {
        let settings = MacOSSaySettings::default();
        let mut svc = MacOSSayTTSService::new(settings);

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
    }

    #[tokio::test]
    async fn interruption_resets_state() {
        let settings = MacOSSaySettings::default();
        let mut svc = MacOSSayTTSService::new(settings);
        svc.state.base.metrics.start_ttfb();
        svc.state.llm_response_started = true;

        let _ = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(
                Frame::Interruption(InterruptionFrame),
                Direction::Downstream,
            )],
        )
        .await;

        assert!(!svc.state.llm_response_started);
        assert!(svc.state.base.metrics.stop_ttfb().is_none());
    }
}
