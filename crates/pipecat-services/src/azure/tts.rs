use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tracing::{debug, trace, warn};

use super::settings::AzureTTSSettings;
use crate::text_aggregator::TextAggregationMode;
use crate::tts::{TTSService, TTSServiceState, tts_process_frame};

/// Azure TTS service implementation.
///
/// Uses REST streaming (HTTP POST with streaming response) for speech synthesis.
/// Each `run_tts` call is a separate HTTP POST request with SSML content.
/// Returns raw PCM audio bytes.
#[derive(Debug)]
pub struct AzureTTSService {
    state: TTSServiceState,
    azure_settings: AzureTTSSettings,
    output_format: String,
    client: reqwest::Client,
}

impl AzureTTSService {
    pub fn new(settings: AzureTTSSettings) -> Self {
        Self {
            state: TTSServiceState::new(
                "AzureTTSService",
                settings.base.clone(),
                TextAggregationMode::Sentence,
                24000,
            ),
            output_format: AzureTTSSettings::output_format_from_sample_rate(24000).to_string(),
            azure_settings: settings,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_sample_rate(settings: AzureTTSSettings, sample_rate: u32) -> Self {
        Self {
            state: TTSServiceState::new(
                "AzureTTSService",
                settings.base.clone(),
                TextAggregationMode::Sentence,
                sample_rate,
            ),
            output_format: AzureTTSSettings::output_format_from_sample_rate(sample_rate)
                .to_string(),
            azure_settings: settings,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl FrameProcessor for AzureTTSService {
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
                self.output_format =
                    AzureTTSSettings::output_format_from_sample_rate(f.audio_out_sample_rate)
                        .to_string();
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
impl TTSService for AzureTTSService {
    async fn run_tts(
        &mut self,
        text: &str,
        context_id: &str,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let ssml = self.azure_settings.build_ssml(text);
        let url = self.azure_settings.build_url();

        debug!("Azure TTS request to {}", url);
        trace!("Azure TTS SSML: {}", ssml);

        // Start TTFB metrics
        let metrics_enabled = self.state.base.metrics_enabled();
        if metrics_enabled {
            self.state.base.metrics.start_ttfb();
        }

        let response = self
            .client
            .post(&url)
            .header("Ocp-Apim-Subscription-Key", &self.azure_settings.api_key)
            .header("Content-Type", "application/ssml+xml")
            .header("X-Microsoft-OutputFormat", &self.output_format)
            .body(ssml)
            .send()
            .await
            .map_err(|e| PipecatError::ProcessorError(format!("Azure TTS request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            warn!("Azure TTS error {}: {}", status, body);
            ctx.push_error(&format!("Azure TTS error {status}: {body}"), false)
                .await?;
            return Ok(());
        }

        // Stream response bytes as audio frames
        let sample_rate = self.state.sample_rate;
        let context_id = context_id.to_string();
        let mut ttfb_stopped = false;
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    warn!("Azure TTS stream read error: {}", e);
                    ctx.push_error(&format!("Azure TTS stream error: {e}"), false)
                        .await?;
                    break;
                }
            };

            if chunk.is_empty() {
                continue;
            }

            // Stop TTFB on first audio chunk
            if metrics_enabled && !ttfb_stopped {
                ctx.push_ttfb(&mut self.state.base.metrics).await?;
                ttfb_stopped = true;
            }

            ctx.send_downstream(Frame::TTSAudioRaw(TTSAudioRawFrame {
                audio: Bytes::from(chunk.to_vec()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn process_frame_forwards_non_tts_frames() {
        let settings = AzureTTSSettings::new("test-key", "eastus");
        let mut svc = AzureTTSService::new(settings);

        let (downstream, _upstream) = pipecat_core::test_utils::run_processor(
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
        let settings = AzureTTSSettings::new("test-key", "eastus");
        let mut svc = AzureTTSService::new(settings);
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

    #[tokio::test]
    async fn text_consumed_by_sentence_aggregator() {
        let settings = AzureTTSSettings::new("test-key", "eastus");
        let mut svc = AzureTTSService::new(settings);

        let (downstream, _upstream) = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(Frame::Text(TextFrame::new("hello")), Direction::Downstream)],
        )
        .await;

        // Text is buffered by sentence aggregator (no sentence boundary yet)
        assert!(downstream.is_empty());
    }
}
