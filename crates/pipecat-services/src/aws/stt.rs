use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tokio::task::JoinHandle;
use tracing::{debug, trace, warn};

use super::settings::AWSTranscribeSTTSettings;
use super::{event_stream, sigv4};
use crate::stt::{STTService, STTServiceState, stt_process_frame};

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type WsMessage = tokio_tungstenite::tungstenite::Message;
type WsWriter = Arc<tokio::sync::Mutex<futures::stream::SplitSink<WsStream, WsMessage>>>;

/// AWS Transcribe STT service implementation.
///
/// Uses a presigned WebSocket connection with AWS event stream binary protocol.
/// Audio is encoded in AWS event stream format and sent as binary frames.
/// Transcription results arrive as binary event stream messages containing JSON.
#[derive(Debug)]
pub struct AWSTranscribeSTTService {
    state: STTServiceState,
    aws_settings: AWSTranscribeSTTSettings,
    ws_writer: Option<WsWriter>,
    receive_task: Option<JoinHandle<()>>,
    /// Set by the receive task when a final transcription is emitted.
    processing_complete: Arc<AtomicBool>,
}

impl AWSTranscribeSTTService {
    pub fn new(settings: AWSTranscribeSTTSettings) -> Self {
        let base_settings = settings.base.clone();
        Self {
            state: STTServiceState::new("AWSTranscribeSTTService", base_settings),
            aws_settings: settings,
            ws_writer: None,
            receive_task: None,
            processing_complete: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Connect to AWS Transcribe Streaming via presigned WebSocket URL.
    async fn connect(&mut self, ctx: &ProcessorContext) -> Result<()> {
        let sample_rate = if self.state.sample_rate > 0 {
            AWSTranscribeSTTSettings::clamp_sample_rate(self.state.sample_rate)
        } else {
            16000
        };
        let language = self
            .aws_settings
            .base
            .language
            .as_deref()
            .unwrap_or("en-US");

        let url = sigv4::presign_url(
            &self.aws_settings.access_key_id,
            &self.aws_settings.secret_access_key,
            self.aws_settings.session_token.as_deref(),
            &self.aws_settings.region,
            language,
            sample_rate,
        );

        debug!(
            "Connecting to AWS Transcribe (region: {})",
            self.aws_settings.region
        );

        let host = format!(
            "transcribestreaming.{}.amazonaws.com",
            self.aws_settings.region
        );

        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&url)
            .header("Host", &host)
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .header("Sec-WebSocket-Version", "13")
            .body(())
            .map_err(|e| {
                PipecatError::ProcessorError(format!("AWS Transcribe URL build error: {e}"))
            })?;

        let (ws_stream, _response) =
            tokio_tungstenite::connect_async(request)
                .await
                .map_err(|e| {
                    PipecatError::ProcessorError(format!(
                        "AWS Transcribe WebSocket connect error: {e}"
                    ))
                })?;

        debug!("AWS Transcribe WebSocket connected");

        let (writer, reader) = ws_stream.split();
        let writer = Arc::new(tokio::sync::Mutex::new(writer));
        self.ws_writer = Some(writer);

        // Spawn receive task
        let bg_ctx = ctx.clone();
        let processing_flag = self.processing_complete.clone();
        self.receive_task = Some(tokio::spawn(receive_task(reader, bg_ctx, processing_flag)));

        Ok(())
    }

    /// Disconnect from AWS Transcribe.
    async fn disconnect(&mut self) {
        if let Some(ref writer) = self.ws_writer {
            let mut w = writer.lock().await;
            // Send end-stream message before closing (plain JSON, not event stream binary)
            let end_msg = WsMessage::Text(r#"{"message-type":"event","event":"end"}"#.into());
            let _ = w.send(end_msg).await;
            let _ = w.close().await;
        }
        self.ws_writer = None;

        if let Some(task) = self.receive_task.take() {
            task.abort();
        }
        self.processing_complete.store(false, Ordering::Release);

        debug!("AWS Transcribe disconnected");
    }
}

#[async_trait]
impl FrameProcessor for AWSTranscribeSTTService {
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
            Frame::Start(_) => {
                stt_process_frame(self, envelope, direction, ctx).await?;
                self.connect(ctx).await?;
            }
            Frame::End(_) | Frame::Cancel(_) => {
                self.disconnect().await;
                stt_process_frame(self, envelope, direction, ctx).await?;
            }
            Frame::VADUserStartedSpeaking(_) => {
                if self.state.base.metrics_enabled() {
                    self.state.base.metrics.start_processing();
                }
                stt_process_frame(self, envelope, direction, ctx).await?;
            }
            _ => {
                stt_process_frame(self, envelope, direction, ctx).await?;
            }
        }

        // Check if the background receive task signaled a final transcription.
        if self.processing_complete.swap(false, Ordering::AcqRel)
            && self.state.base.metrics_enabled()
        {
            ctx.push_processing_metrics(&mut self.state.base.metrics)
                .await?;
        }

        Ok(())
    }
}

#[async_trait]
impl STTService for AWSTranscribeSTTService {
    async fn run_stt(&mut self, audio: Bytes, _ctx: &ProcessorContext) -> Result<()> {
        if let Some(ref writer) = self.ws_writer {
            let event_msg = event_stream::encode_audio_event(&audio);
            let mut w = writer.lock().await;
            let msg = WsMessage::Binary(event_msg.into());
            w.send(msg).await.map_err(|e| {
                PipecatError::ProcessorError(format!("AWS Transcribe audio send error: {e}"))
            })?;
        }
        Ok(())
    }

    fn stt_service_state(&self) -> &STTServiceState {
        &self.state
    }

    fn stt_service_state_mut(&mut self) -> &mut STTServiceState {
        &mut self.state
    }
}

/// Generate a simple timestamp string (seconds.millis since epoch).
fn timestamp_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}.{:03}", d.as_secs(), d.subsec_millis()))
        .unwrap_or_default()
}

/// Background task that reads binary event stream messages from AWS Transcribe
/// and pushes transcription frames downstream.
async fn receive_task(
    mut reader: futures::stream::SplitStream<WsStream>,
    ctx: ProcessorContext,
    processing_complete: Arc<AtomicBool>,
) {
    while let Some(msg_result) = reader.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                warn!("AWS Transcribe WebSocket read error: {}", e);
                let _ = ctx
                    .push_error(&format!("AWS Transcribe connection error: {e}"), false)
                    .await;
                break;
            }
        };

        let data = match msg {
            WsMessage::Binary(b) => b,
            WsMessage::Close(_) => {
                debug!("AWS Transcribe WebSocket closed by server");
                break;
            }
            _ => continue,
        };

        let Some((headers, payload)) = event_stream::decode_event(&data) else {
            warn!("AWS Transcribe: failed to decode event stream message");
            continue;
        };

        let header_map: std::collections::HashMap<_, _> = headers.into_iter().collect();
        let msg_type = header_map
            .get(":message-type")
            .map(|s| s.as_str())
            .unwrap_or("");

        match msg_type {
            "event" => {
                if payload.is_empty() {
                    continue;
                }
                let Ok(json) = serde_json::from_slice::<serde_json::Value>(&payload) else {
                    warn!("AWS Transcribe: failed to parse JSON payload");
                    continue;
                };

                if let Some(frame) = parse_transcribe_result(&json) {
                    let is_final = matches!(&frame, Frame::Transcription(_));
                    if is_final {
                        processing_complete.store(true, Ordering::Release);
                    }
                    let _ = ctx.send_downstream(frame).await;
                }
            }
            "exception" => {
                let error_msg = if payload.is_empty() {
                    header_map
                        .get(":exception-type")
                        .cloned()
                        .unwrap_or_else(|| "Unknown exception".to_string())
                } else {
                    String::from_utf8_lossy(&payload).to_string()
                };
                warn!("AWS Transcribe exception: {}", error_msg);
                let _ = ctx
                    .push_error(&format!("AWS Transcribe: {error_msg}"), false)
                    .await;
            }
            _ => {
                trace!("AWS Transcribe: unknown message type: {}", msg_type);
            }
        }
    }
}

/// Parse an AWS Transcribe result message into a transcription frame.
fn parse_transcribe_result(data: &serde_json::Value) -> Option<Frame> {
    let transcript = data.get("Transcript")?;
    let results = transcript.get("Results")?.as_array()?;

    if results.is_empty() {
        return None;
    }

    let result = &results[0];
    let is_partial = result
        .get("IsPartial")
        .and_then(|p| p.as_bool())
        .unwrap_or(true);

    let alternatives = result.get("Alternatives")?.as_array()?;
    if alternatives.is_empty() {
        return None;
    }

    let text = alternatives[0]
        .get("Transcript")
        .and_then(|t| t.as_str())
        .unwrap_or("");

    if text.is_empty() {
        return None;
    }

    let timestamp = Some(timestamp_now());

    if is_partial {
        Some(Frame::InterimTranscription(InterimTranscriptionFrame {
            text: text.to_string(),
            user_id: String::new(),
            timestamp,
            language: None,
            result: Some(data.clone()),
        }))
    } else {
        Some(Frame::Transcription(TranscriptionFrame {
            text: text.to_string(),
            user_id: String::new(),
            timestamp,
            language: None,
            finalized: true,
            result: Some(data.clone()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_final_result() {
        let data = json!({
            "Transcript": {
                "Results": [{
                    "IsPartial": false,
                    "Alternatives": [{
                        "Transcript": "Hello world"
                    }]
                }]
            }
        });

        let frame = parse_transcribe_result(&data).unwrap();
        match frame {
            Frame::Transcription(t) => {
                assert_eq!(t.text, "Hello world");
                assert!(t.finalized);
                assert!(t.timestamp.is_some());
            }
            other => panic!("Expected Transcription, got {other}"),
        }
    }

    #[test]
    fn parse_partial_result() {
        let data = json!({
            "Transcript": {
                "Results": [{
                    "IsPartial": true,
                    "Alternatives": [{
                        "Transcript": "Hello"
                    }]
                }]
            }
        });

        let frame = parse_transcribe_result(&data).unwrap();
        match frame {
            Frame::InterimTranscription(t) => {
                assert_eq!(t.text, "Hello");
                assert!(t.timestamp.is_some());
            }
            other => panic!("Expected InterimTranscription, got {other}"),
        }
    }

    #[test]
    fn parse_empty_results() {
        let data = json!({
            "Transcript": {
                "Results": []
            }
        });
        assert!(parse_transcribe_result(&data).is_none());
    }

    #[test]
    fn parse_empty_transcript() {
        let data = json!({
            "Transcript": {
                "Results": [{
                    "IsPartial": false,
                    "Alternatives": [{
                        "Transcript": ""
                    }]
                }]
            }
        });
        assert!(parse_transcribe_result(&data).is_none());
    }

    #[test]
    fn parse_no_alternatives() {
        let data = json!({
            "Transcript": {
                "Results": [{
                    "IsPartial": false,
                    "Alternatives": []
                }]
            }
        });
        assert!(parse_transcribe_result(&data).is_none());
    }

    #[tokio::test]
    async fn process_frame_forwards_non_stt_frames() {
        let settings = AWSTranscribeSTTSettings::new("key", "secret");
        let mut svc = AWSTranscribeSTTService::new(settings);

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
        let settings = AWSTranscribeSTTSettings::new("key", "secret");
        let mut svc = AWSTranscribeSTTService::new(settings);
        svc.state.base.metrics.start_ttfb();

        let (downstream, _) = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(
                Frame::Interruption(InterruptionFrame),
                Direction::Downstream,
            )],
        )
        .await;

        assert_eq!(downstream.len(), 1);
        assert!(svc.state.base.metrics.stop_ttfb().is_none());
    }
}
