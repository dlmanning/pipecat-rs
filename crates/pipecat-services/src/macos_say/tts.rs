use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tracing::{debug, warn};

use super::settings::MacOSSaySettings;
use crate::text_aggregator::TextAggregationMode;
use crate::tts::{TTSService, TTSServiceState, tts_process_frame};

/// Default sample rate for `say` output.
const DEFAULT_SAMPLE_RATE: u32 = 16000;

/// Maximum PCM bytes to push in a single audio frame (~100ms at 16kHz mono 16-bit).
const CHUNK_SIZE: usize = 3200;

/// Counter for unique temp file names.
static INVOCATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// TTS service using the macOS `say` command.
///
/// Synthesizes speech by invoking `say -o <tmpfile> --file-format=WAVE
/// --data-format=LEI16@<rate>` with text piped via stdin (`-f -`), then
/// reads the WAV file and pushes raw PCM audio frames in chunks.
/// No API key or network required.
///
/// Supports voice selection (`-v`), speech rate (`-r`), and configurable
/// sample rate. Output is always mono 16-bit little-endian PCM.
///
/// **macOS only** — this service requires the `say` command which is only
/// available on macOS. It will fail at runtime on other platforms.
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

        // Unique temp file per invocation to avoid races
        let invocation = INVOCATION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = std::env::temp_dir().join(format!(
            "pipecat_say_{}_{invocation}.wav",
            std::process::id()
        ));

        let mut cmd = Command::new("say");
        cmd.arg("-f").arg("-"); // Read text from stdin
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

        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());

        debug!("MacOSSay: synthesizing {} chars", text.len());

        // Start TTFB metrics
        let metrics_enabled = self.state.base.metrics_enabled();
        if metrics_enabled {
            self.state.base.metrics.start_ttfb();
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| PipecatError::ProcessorError(format!("Failed to spawn say: {e}")))?;

        // Write text to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes()).await.map_err(|e| {
                PipecatError::ProcessorError(format!("Failed to write to say stdin: {e}"))
            })?;
            // Drop stdin to close the pipe and let say proceed
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| PipecatError::ProcessorError(format!("Failed to wait for say: {e}")))?;

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

        // Find the "data" chunk — don't assume a fixed 44-byte header
        let Some(pcm_offset) = find_wav_data_offset(&wav_data) else {
            warn!("MacOSSay: no data chunk found in WAV output");
            return Ok(());
        };

        if pcm_offset >= wav_data.len() {
            return Ok(());
        }

        let pcm_data = &wav_data[pcm_offset..];
        if pcm_data.is_empty() {
            return Ok(());
        }

        // Stop TTFB on first audio
        if metrics_enabled {
            ctx.push_ttfb(&mut self.state.base.metrics).await?;
        }

        // Push PCM in chunks for consistent streaming behavior
        let context_id = context_id.to_string();
        for chunk in pcm_data.chunks(CHUNK_SIZE) {
            ctx.send_downstream(Frame::TTSAudioRaw(TTSAudioRawFrame {
                audio: Bytes::copy_from_slice(chunk),
                sample_rate,
                num_channels: 1,
                context_id: Some(context_id.clone()),
            }))
            .await?;
        }

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

/// Find the byte offset where PCM data begins in a WAV file.
///
/// Scans for the `data` chunk fourcc and returns the offset past the
/// chunk header (fourcc + size = 8 bytes). Does not assume a fixed
/// 44-byte header, handling WAV files with extra metadata chunks.
fn find_wav_data_offset(wav: &[u8]) -> Option<usize> {
    // Skip the 12-byte RIFF header (RIFF + size + WAVE)
    if wav.len() < 12 {
        return None;
    }

    let mut pos = 12;
    while pos + 8 <= wav.len() {
        let fourcc = &wav[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([wav[pos + 4], wav[pos + 5], wav[pos + 6], wav[pos + 7]]) as usize;

        if fourcc == b"data" {
            return Some(pos + 8);
        }

        // Advance to next chunk (chunks are 2-byte aligned)
        let advance = 8 + chunk_size + (chunk_size & 1);
        pos += advance;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::TTSSettings;

    #[test]
    fn default_sample_rate() {
        let settings = MacOSSaySettings::default();
        let svc = MacOSSayTTSService::new(settings);
        assert_eq!(svc.state.sample_rate, DEFAULT_SAMPLE_RATE);
    }

    #[test]
    fn voice_from_settings() {
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

    #[test]
    fn find_wav_data_standard_header() {
        // Minimal WAV: RIFF header (12) + fmt chunk (24) + data chunk header (8)
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&100u32.to_le_bytes()); // file size
        wav.extend_from_slice(b"WAVE");
        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav.extend_from_slice(&[0u8; 16]); // format data
        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&4u32.to_le_bytes()); // chunk size
        wav.extend_from_slice(&[1, 2, 3, 4]); // PCM data

        let offset = find_wav_data_offset(&wav).unwrap();
        assert_eq!(offset, 44); // 12 + 8 + 16 + 8 = 44
        assert_eq!(&wav[offset..], &[1, 2, 3, 4]);
    }

    #[test]
    fn find_wav_data_with_extra_chunks() {
        // WAV with a LIST chunk before data
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&200u32.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&[0u8; 16]);
        // LIST chunk (extra metadata)
        wav.extend_from_slice(b"LIST");
        wav.extend_from_slice(&10u32.to_le_bytes());
        wav.extend_from_slice(&[0u8; 10]);
        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&2u32.to_le_bytes());
        wav.extend_from_slice(&[0xAB, 0xCD]);

        let offset = find_wav_data_offset(&wav).unwrap();
        assert_eq!(&wav[offset..], &[0xAB, 0xCD]);
    }

    #[test]
    fn find_wav_data_missing() {
        let wav = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert!(find_wav_data_offset(wav).is_none());
    }

    #[test]
    fn find_wav_data_too_short() {
        assert!(find_wav_data_offset(b"RIFF").is_none());
        assert!(find_wav_data_offset(b"").is_none());
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
