use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, trace, warn};

use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};

use crate::stt::{STTService, STTServiceState, stt_process_frame};

use super::settings::DeepgramSTTSettings;

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type WsMessage = tokio_tungstenite::tungstenite::Message;
type WsWriter = Arc<Mutex<futures::stream::SplitSink<WsStream, WsMessage>>>;

const DEFAULT_BASE_URL: &str = "wss://api.deepgram.com/v1/listen";
const KEEPALIVE_INTERVAL_SECS: u64 = 5;

/// Deepgram STT service implementation.
///
/// Maintains a persistent WebSocket connection. Audio is sent as binary frames,
/// transcription results arrive as JSON messages from a background receive task.
/// A keepalive task sends KeepAlive JSON every 5s to prevent timeout (Deepgram
/// closes idle connections after 10s).
#[derive(Debug)]
pub struct DeepgramSTTService {
    state: STTServiceState,
    api_key: String,
    base_url: String,
    deepgram_settings: DeepgramSTTSettings,
    ws_writer: Option<WsWriter>,
    receive_task: Option<JoinHandle<()>>,
    keepalive_task: Option<JoinHandle<()>>,
    /// Whether a finalize request has been sent and we're awaiting server confirmation.
    finalize_requested: Arc<AtomicBool>,
    /// Set by the receive task when a final transcription is emitted, signaling
    /// that processing metrics should be stopped and pushed downstream.
    processing_complete: Arc<AtomicBool>,
}

impl DeepgramSTTService {
    pub fn new(api_key: impl Into<String>, settings: DeepgramSTTSettings) -> Self {
        let base_settings = settings.base.clone();
        Self {
            state: STTServiceState::new("DeepgramSTTService", base_settings),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            deepgram_settings: settings,
            ws_writer: None,
            receive_task: None,
            keepalive_task: None,
            finalize_requested: Arc::new(AtomicBool::new(false)),
            processing_complete: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Connect to the Deepgram WebSocket.
    async fn connect(&mut self, ctx: &ProcessorContext) -> Result<()> {
        let sample_rate = if self.state.sample_rate > 0 {
            self.state.sample_rate
        } else {
            16000
        };

        let query = self.deepgram_settings.build_query_params(sample_rate);
        let url = format!("{}?{}", self.base_url, query);

        debug!("Connecting to Deepgram: {}", url);

        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&url)
            .header("Authorization", format!("Token {}", self.api_key))
            .header("Host", extract_host(&url).unwrap_or("api.deepgram.com"))
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .header("Sec-WebSocket-Version", "13")
            .body(())
            .map_err(|e| PipecatError::ProcessorError(format!("Deepgram URL build error: {e}")))?;

        let (ws_stream, _response) =
            tokio_tungstenite::connect_async(request)
                .await
                .map_err(|e| {
                    PipecatError::ProcessorError(format!("Deepgram WebSocket connect error: {e}"))
                })?;

        debug!("Deepgram WebSocket connected");

        let (writer, reader) = ws_stream.split();
        let writer = Arc::new(Mutex::new(writer));
        self.ws_writer = Some(writer.clone());

        // Spawn receive task
        let bg_ctx = ctx.clone();
        let vad_events = self.deepgram_settings.vad_events;
        let finalize_flag = self.finalize_requested.clone();
        let processing_flag = self.processing_complete.clone();
        self.receive_task = Some(tokio::spawn(receive_task(
            reader,
            bg_ctx,
            vad_events,
            finalize_flag,
            processing_flag,
        )));

        // Spawn keepalive task — sends KeepAlive JSON every 5s
        let keepalive_writer = writer;
        self.keepalive_task = Some(tokio::spawn(keepalive_task(keepalive_writer)));

        Ok(())
    }

    /// Disconnect from the Deepgram WebSocket.
    async fn disconnect(&mut self) {
        // Send close stream message
        if let Some(ref writer) = self.ws_writer {
            let mut w = writer.lock().await;
            let close_msg = WsMessage::Text(r#"{"type":"CloseStream"}"#.into());
            let _ = w.send(close_msg).await;
            let _ = w.close().await;
        }
        self.ws_writer = None;

        if let Some(task) = self.keepalive_task.take() {
            task.abort();
        }
        if let Some(task) = self.receive_task.take() {
            task.abort();
        }
        self.finalize_requested.store(false, Ordering::Release);
        self.processing_complete.store(false, Ordering::Release);

        debug!("Deepgram disconnected");
    }

    /// Send a finalize message to get the current transcription result.
    async fn send_finalize(&mut self) -> Result<()> {
        if let Some(ref writer) = self.ws_writer {
            let mut w = writer.lock().await;
            let msg = WsMessage::Text(r#"{"type":"Finalize"}"#.into());
            w.send(msg).await.map_err(|e| {
                PipecatError::ProcessorError(format!("Deepgram finalize error: {e}"))
            })?;
            self.finalize_requested.store(true, Ordering::Release);
            trace!("Deepgram: sent Finalize");
        }
        Ok(())
    }
}

#[async_trait]
impl FrameProcessor for DeepgramSTTService {
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
                // Start processing metrics when pipeline VAD detects speech
                if self.state.base.metrics_enabled() {
                    self.state.base.metrics.start_processing();
                }
                stt_process_frame(self, envelope, direction, ctx).await?;
            }
            Frame::VADUserStoppedSpeaking(_) => {
                self.send_finalize().await?;
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
impl STTService for DeepgramSTTService {
    async fn run_stt(&mut self, audio: Bytes, _ctx: &ProcessorContext) -> Result<()> {
        if let Some(ref writer) = self.ws_writer {
            let mut w = writer.lock().await;
            let msg = WsMessage::Binary(audio.to_vec().into());
            w.send(msg).await.map_err(|e| {
                PipecatError::ProcessorError(format!("Deepgram audio send error: {e}"))
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

/// Background keepalive task — sends KeepAlive JSON every 5s.
/// Deepgram closes idle connections after 10s (NET-0001 error).
async fn keepalive_task(writer: WsWriter) {
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(KEEPALIVE_INTERVAL_SECS)).await;
        let mut w = writer.lock().await;
        let msg = WsMessage::Text(r#"{"type":"KeepAlive"}"#.into());
        if w.send(msg).await.is_err() {
            trace!("Deepgram keepalive: writer closed, exiting");
            break;
        }
        trace!("Deepgram: sent KeepAlive");
    }
}

/// Background task that reads JSON messages from the Deepgram WebSocket
/// and pushes transcription frames downstream via the ProcessorContext.
async fn receive_task(
    mut reader: futures::stream::SplitStream<WsStream>,
    ctx: ProcessorContext,
    vad_events: bool,
    finalize_requested: Arc<AtomicBool>,
    processing_complete: Arc<AtomicBool>,
) {
    while let Some(msg_result) = reader.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                warn!("Deepgram WebSocket read error: {}", e);
                let _ = ctx
                    .push_error(&format!("Deepgram connection error: {e}"), false)
                    .await;
                break;
            }
        };

        let text = match msg {
            WsMessage::Text(t) => t,
            WsMessage::Close(_) => {
                debug!("Deepgram WebSocket closed by server");
                break;
            }
            _ => continue,
        };

        let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) else {
            warn!("Deepgram: failed to parse JSON: {}", text);
            continue;
        };

        let msg_type = data.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match msg_type {
            "Results" => {
                // Check if this is_final result was triggered by an explicit finalize request.
                // Only consume the flag for is_final results (interim results don't confirm finalization).
                let is_final = data
                    .get("is_final")
                    .and_then(|f| f.as_bool())
                    .unwrap_or(false);
                let from_finalize = if is_final {
                    finalize_requested.swap(false, Ordering::AcqRel)
                } else {
                    false
                };
                if let Some(frame) = parse_results(&data, from_finalize) {
                    if is_final {
                        processing_complete.store(true, Ordering::Release);
                    }
                    let _ = ctx.send_downstream(frame).await;
                }
            }
            "SpeechStarted" => {
                if vad_events {
                    let _ = ctx
                        .send_downstream(Frame::UserStartedSpeaking(UserStartedSpeakingFrame))
                        .await;
                }
            }
            "UtteranceEnd" => {
                if vad_events {
                    let _ = ctx
                        .send_downstream(Frame::UserStoppedSpeaking(UserStoppedSpeakingFrame))
                        .await;
                }
            }
            "Metadata" => {
                trace!("Deepgram metadata: {}", data);
            }
            _ => {
                trace!("Deepgram unknown message type: {}", msg_type);
            }
        }
    }
}

/// Parse a Deepgram Results message into a transcription frame.
fn parse_results(data: &serde_json::Value, from_finalize: bool) -> Option<Frame> {
    let channel = data.get("channel")?;
    let alternatives = channel.get("alternatives")?.as_array()?;
    if alternatives.is_empty() {
        return None;
    }

    let transcript = alternatives[0]
        .get("transcript")
        .and_then(|t| t.as_str())
        .unwrap_or("");

    if transcript.is_empty() {
        return None;
    }

    let is_final = data
        .get("is_final")
        .and_then(|f| f.as_bool())
        .unwrap_or(false);

    let language = alternatives[0]
        .get("languages")
        .and_then(|l| l.as_array())
        .and_then(|l| l.first())
        .and_then(|l| l.as_str())
        .map(String::from);

    let timestamp = Some(timestamp_now());

    if is_final {
        Some(Frame::Transcription(TranscriptionFrame {
            text: transcript.to_string(),
            user_id: String::new(),
            timestamp,
            language,
            // Only mark as finalized when the result was triggered by an explicit
            // finalize request (VADUserStoppedSpeaking). Regular is_final results
            // from Deepgram's endpointing are NOT finalized — they're just complete
            // utterance segments. This distinction matters for turn controllers.
            finalized: from_finalize,
            result: Some(data.clone()),
        }))
    } else {
        Some(Frame::InterimTranscription(InterimTranscriptionFrame {
            text: transcript.to_string(),
            user_id: String::new(),
            timestamp,
            language,
            result: Some(data.clone()),
        }))
    }
}

/// Extract host from a URL string.
fn extract_host(url: &str) -> Option<&str> {
    let after_scheme = url
        .strip_prefix("wss://")
        .or_else(|| url.strip_prefix("ws://"))?;
    after_scheme.split('/').next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_results_final_transcription() {
        let data = json!({
            "type": "Results",
            "channel": {
                "alternatives": [{
                    "transcript": "Hello world",
                    "languages": ["en"]
                }]
            },
            "is_final": true
        });

        let frame = parse_results(&data, false).unwrap();
        match frame {
            Frame::Transcription(t) => {
                assert_eq!(t.text, "Hello world");
                // is_final from Deepgram without finalize request → NOT finalized
                assert!(!t.finalized);
                assert_eq!(t.language.as_deref(), Some("en"));
                assert!(t.timestamp.is_some());
            }
            other => panic!("Expected Transcription, got {other}"),
        }
    }

    #[test]
    fn parse_results_finalized_from_explicit_request() {
        let data = json!({
            "type": "Results",
            "channel": {
                "alternatives": [{
                    "transcript": "Hello world",
                    "languages": ["en"]
                }]
            },
            "is_final": true
        });

        let frame = parse_results(&data, true).unwrap();
        match frame {
            Frame::Transcription(t) => {
                assert!(t.finalized); // from_finalize=true → finalized
            }
            other => panic!("Expected Transcription, got {other}"),
        }
    }

    #[test]
    fn parse_results_interim_transcription() {
        let data = json!({
            "type": "Results",
            "channel": {
                "alternatives": [{
                    "transcript": "Hello",
                    "languages": ["en"]
                }]
            },
            "is_final": false
        });

        let frame = parse_results(&data, false).unwrap();
        match frame {
            Frame::InterimTranscription(t) => {
                assert_eq!(t.text, "Hello");
                assert_eq!(t.language.as_deref(), Some("en"));
                assert!(t.timestamp.is_some());
            }
            other => panic!("Expected InterimTranscription, got {other}"),
        }
    }

    #[test]
    fn parse_results_empty_transcript() {
        let data = json!({
            "type": "Results",
            "channel": {
                "alternatives": [{"transcript": ""}]
            },
            "is_final": true
        });

        assert!(parse_results(&data, false).is_none());
    }

    #[test]
    fn parse_results_no_alternatives() {
        let data = json!({
            "type": "Results",
            "channel": { "alternatives": [] },
            "is_final": true
        });

        assert!(parse_results(&data, false).is_none());
    }

    #[test]
    fn parse_results_no_language() {
        let data = json!({
            "type": "Results",
            "channel": {
                "alternatives": [{"transcript": "test"}]
            },
            "is_final": true
        });

        let frame = parse_results(&data, false).unwrap();
        match frame {
            Frame::Transcription(t) => {
                assert_eq!(t.text, "test");
                assert!(t.language.is_none());
            }
            other => panic!("Expected Transcription, got {other}"),
        }
    }

    #[test]
    fn extract_host_wss() {
        assert_eq!(
            extract_host("wss://api.deepgram.com/v1/listen"),
            Some("api.deepgram.com")
        );
    }

    #[test]
    fn extract_host_ws() {
        assert_eq!(
            extract_host("ws://localhost:8080/listen"),
            Some("localhost:8080")
        );
    }

    #[tokio::test]
    async fn process_frame_forwards_non_stt_frames() {
        let settings = DeepgramSTTSettings::default();
        let mut svc = DeepgramSTTService::new("test-key", settings);

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
        let settings = DeepgramSTTSettings::default();
        let mut svc = DeepgramSTTService::new("test-key", settings);
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
