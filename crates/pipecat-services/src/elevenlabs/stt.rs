use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, trace, warn};

use super::stt_settings::{CommitStrategy, ElevenLabsSTTSettings, audio_format_from_sample_rate};
use crate::stt::{STTService, STTServiceState, stt_process_frame};

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type WsMessage = tokio_tungstenite::tungstenite::Message;
type WsWriter = Arc<Mutex<futures::stream::SplitSink<WsStream, WsMessage>>>;

const DEFAULT_BASE_URL: &str = "wss://api.elevenlabs.io";
const KEEPALIVE_INTERVAL_SECS: u64 = 5;

/// ElevenLabs realtime STT service implementation.
///
/// Uses a persistent WebSocket connection. Audio is sent as base64-encoded JSON,
/// transcription results arrive as JSON messages from a background receive task.
#[derive(Debug)]
pub struct ElevenLabsRealtimeSTTService {
    state: STTServiceState,
    api_key: String,
    base_url: String,
    elevenlabs_settings: ElevenLabsSTTSettings,
    audio_format: String,
    ws_writer: Option<WsWriter>,
    receive_task: Option<JoinHandle<()>>,
    keepalive_task: Option<JoinHandle<()>>,
    /// Set by the receive task when a final transcription is emitted.
    processing_complete: Arc<AtomicBool>,
}

impl ElevenLabsRealtimeSTTService {
    pub fn new(api_key: impl Into<String>, settings: ElevenLabsSTTSettings) -> Self {
        let base_settings = settings.base.clone();
        Self {
            state: STTServiceState::new("ElevenLabsRealtimeSTTService", base_settings),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            elevenlabs_settings: settings,
            audio_format: "pcm_16000".to_string(),
            ws_writer: None,
            receive_task: None,
            keepalive_task: None,
            processing_complete: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build the WebSocket URL.
    fn build_url(&self) -> String {
        let model = self
            .elevenlabs_settings
            .base
            .model
            .as_deref()
            .unwrap_or("scribe_v2_realtime");
        let strategy = self.elevenlabs_settings.commit_strategy.as_str();

        let mut url = format!(
            "{}/v1/speech-to-text/realtime?model_id={}&audio_format={}&commit_strategy={}",
            self.base_url, model, self.audio_format, strategy
        );

        if let Some(ref lang) = self.elevenlabs_settings.base.language {
            url.push_str(&format!("&language_code={lang}"));
        }

        url
    }

    /// Connect to the ElevenLabs STT WebSocket.
    async fn connect(&mut self, ctx: &ProcessorContext) -> Result<()> {
        let url = self.build_url();
        debug!("Connecting to ElevenLabs STT: {}", url);

        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&url)
            .header("xi-api-key", &self.api_key)
            .header("Host", extract_host(&url).unwrap_or("api.elevenlabs.io"))
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .header("Sec-WebSocket-Version", "13")
            .body(())
            .map_err(|e| {
                PipecatError::ProcessorError(format!("ElevenLabs STT URL build error: {e}"))
            })?;

        let (ws_stream, _response) =
            tokio_tungstenite::connect_async(request)
                .await
                .map_err(|e| {
                    PipecatError::ProcessorError(format!(
                        "ElevenLabs STT WebSocket connect error: {e}"
                    ))
                })?;

        debug!("ElevenLabs STT WebSocket connected");

        let (writer, reader) = ws_stream.split();
        let writer = Arc::new(Mutex::new(writer));
        self.ws_writer = Some(writer.clone());

        // Spawn receive task
        let bg_ctx = ctx.clone();
        let commit_strategy = self.elevenlabs_settings.commit_strategy;
        let processing_flag = self.processing_complete.clone();
        self.receive_task = Some(tokio::spawn(receive_task(
            reader,
            bg_ctx,
            commit_strategy,
            processing_flag,
        )));

        // Spawn keepalive task — sends silent audio every 5s
        let keepalive_writer = writer;
        let sample_rate = self.state.sample_rate;
        self.keepalive_task = Some(tokio::spawn(keepalive_task(keepalive_writer, sample_rate)));

        Ok(())
    }

    /// Disconnect from the ElevenLabs STT WebSocket.
    async fn disconnect(&mut self) {
        if let Some(ref writer) = self.ws_writer {
            let mut w = writer.lock().await;
            let _ = w.close().await;
        }
        self.ws_writer = None;

        if let Some(task) = self.receive_task.take() {
            task.abort();
        }
        if let Some(task) = self.keepalive_task.take() {
            task.abort();
        }
        self.processing_complete.store(false, Ordering::Release);

        debug!("ElevenLabs STT disconnected");
    }

    /// Send a manual commit (empty audio with commit: true).
    async fn send_commit(&mut self) -> Result<()> {
        if let Some(ref writer) = self.ws_writer {
            let sample_rate = if self.state.sample_rate > 0 {
                self.state.sample_rate
            } else {
                16000
            };
            let msg = serde_json::json!({
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": true,
                "sample_rate": sample_rate,
            });
            let ws_msg = WsMessage::Text(msg.to_string().into());
            writer.lock().await.send(ws_msg).await.map_err(|e| {
                PipecatError::ProcessorError(format!("ElevenLabs STT commit send error: {e}"))
            })?;
            trace!("ElevenLabs STT: sent commit");
        }
        Ok(())
    }
}

#[async_trait]
impl FrameProcessor for ElevenLabsRealtimeSTTService {
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
                self.audio_format = audio_format_from_sample_rate(f.audio_in_sample_rate);
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
            Frame::VADUserStoppedSpeaking(_) => {
                // With manual commit strategy, send commit on VAD stop
                if self.elevenlabs_settings.commit_strategy == CommitStrategy::Manual {
                    self.send_commit().await?;
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
impl STTService for ElevenLabsRealtimeSTTService {
    async fn run_stt(&mut self, audio: Bytes, _ctx: &ProcessorContext) -> Result<()> {
        if let Some(ref writer) = self.ws_writer {
            let audio_b64 =
                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &audio);
            let sample_rate = if self.state.sample_rate > 0 {
                self.state.sample_rate
            } else {
                16000
            };
            let msg = serde_json::json!({
                "message_type": "input_audio_chunk",
                "audio_base_64": audio_b64,
                "commit": false,
                "sample_rate": sample_rate,
            });
            let ws_msg = WsMessage::Text(msg.to_string().into());
            writer.lock().await.send(ws_msg).await.map_err(|e| {
                PipecatError::ProcessorError(format!("ElevenLabs STT audio send error: {e}"))
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

/// Background task that reads JSON messages from the ElevenLabs STT WebSocket
/// and pushes transcription frames downstream.
async fn receive_task(
    mut reader: futures::stream::SplitStream<WsStream>,
    ctx: ProcessorContext,
    commit_strategy: CommitStrategy,
    processing_complete: Arc<AtomicBool>,
) {
    while let Some(msg_result) = reader.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                warn!("ElevenLabs STT WebSocket read error: {}", e);
                let _ = ctx
                    .push_error(&format!("ElevenLabs STT connection error: {e}"), false)
                    .await;
                break;
            }
        };

        let text = match msg {
            WsMessage::Text(t) => t,
            WsMessage::Close(_) => {
                debug!("ElevenLabs STT WebSocket closed by server");
                break;
            }
            _ => continue,
        };

        let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) else {
            warn!("ElevenLabs STT: failed to parse JSON: {}", text);
            continue;
        };

        let msg_type = data
            .get("message_type")
            .and_then(|t| t.as_str())
            .unwrap_or("");

        match msg_type {
            "session_started" => {
                debug!("ElevenLabs STT: session started");
            }
            "partial_transcript" => {
                let transcript = data.get("text").and_then(|t| t.as_str()).unwrap_or("");
                if !transcript.is_empty() {
                    let language = data
                        .get("language_code")
                        .and_then(|l| l.as_str())
                        .map(String::from);
                    let _ = ctx
                        .send_downstream(Frame::InterimTranscription(InterimTranscriptionFrame {
                            text: transcript.to_string(),
                            user_id: String::new(),
                            timestamp: Some(timestamp_now()),
                            language,
                            result: Some(data.clone()),
                        }))
                        .await;
                }
            }
            "committed_transcript" | "committed_transcript_with_timestamps" => {
                let transcript = data.get("text").and_then(|t| t.as_str()).unwrap_or("");
                if !transcript.is_empty() {
                    let language = data
                        .get("language_code")
                        .and_then(|l| l.as_str())
                        .map(String::from);
                    // With manual commit strategy, the transcription is finalized
                    // because pipecat controls when to commit.
                    let finalized = commit_strategy == CommitStrategy::Manual;
                    processing_complete.store(true, Ordering::Release);
                    let _ = ctx
                        .send_downstream(Frame::Transcription(TranscriptionFrame {
                            text: transcript.to_string(),
                            user_id: String::new(),
                            timestamp: Some(timestamp_now()),
                            language,
                            finalized,
                            result: Some(data.clone()),
                        }))
                        .await;
                }
            }
            "error"
            | "auth_error"
            | "quota_exceeded_error"
            | "transcriber_error"
            | "input_error"
            | "commit_throttled"
            | "unaccepted_terms_error"
            | "rate_limited"
            | "queue_overflow"
            | "resource_exhausted"
            | "session_time_limit_exceeded"
            | "chunk_size_exceeded"
            | "insufficient_audio_activity" => {
                let error_msg = data
                    .get("error")
                    .and_then(|e| e.as_str())
                    .unwrap_or(msg_type);
                warn!("ElevenLabs STT error ({}): {}", msg_type, error_msg);
                let _ = ctx
                    .push_error(
                        &format!("ElevenLabs STT {}: {}", msg_type, error_msg),
                        false,
                    )
                    .await;
            }
            _ => {
                trace!("ElevenLabs STT: unknown message type: {}", msg_type);
            }
        }
    }
}

/// Keepalive task — sends silent audio every 5s to prevent timeout.
async fn keepalive_task(writer: WsWriter, sample_rate: u32) {
    let rate = if sample_rate > 0 { sample_rate } else { 16000 };
    // 100ms of silent PCM (16-bit mono)
    let silent_samples = (rate / 10) as usize;
    let silent_bytes = vec![0u8; silent_samples * 2];
    let silent_b64 =
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &silent_bytes);

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(KEEPALIVE_INTERVAL_SECS)).await;

        let msg = serde_json::json!({
            "message_type": "input_audio_chunk",
            "audio_base_64": &silent_b64,
            "commit": false,
            "sample_rate": rate,
        });
        let ws_msg = WsMessage::Text(msg.to_string().into());
        if writer.lock().await.send(ws_msg).await.is_err() {
            trace!("ElevenLabs STT keepalive: writer closed, exiting");
            break;
        }
        trace!("ElevenLabs STT: keepalive sent");
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
    use serde_json::json;

    use super::*;

    #[test]
    fn build_url_default() {
        let settings = ElevenLabsSTTSettings::default();
        let svc = ElevenLabsRealtimeSTTService::new("test-key", settings);
        let url = svc.build_url();
        assert!(url.contains("api.elevenlabs.io"));
        assert!(url.contains("speech-to-text/realtime"));
        assert!(url.contains("model_id=scribe_v2_realtime"));
        assert!(url.contains("commit_strategy=manual"));
        assert!(url.contains("audio_format=pcm_16000"));
        assert!(!url.contains("language_code"));
    }

    #[test]
    fn build_url_with_language() {
        let mut settings = ElevenLabsSTTSettings::default();
        settings.base.language = Some("eng".into());
        let svc = ElevenLabsRealtimeSTTService::new("test-key", settings);
        let url = svc.build_url();
        assert!(url.contains("language_code=eng"));
    }

    #[test]
    fn build_url_vad_strategy() {
        let mut settings = ElevenLabsSTTSettings::default();
        settings.commit_strategy = CommitStrategy::Vad;
        let svc = ElevenLabsRealtimeSTTService::new("test-key", settings);
        let url = svc.build_url();
        assert!(url.contains("commit_strategy=vad"));
    }

    #[test]
    fn parse_partial_transcript() {
        let data = json!({
            "message_type": "partial_transcript",
            "text": "Hello",
            "language_code": "eng"
        });

        let msg_type = data.get("message_type").and_then(|t| t.as_str()).unwrap();
        assert_eq!(msg_type, "partial_transcript");
        let text = data.get("text").and_then(|t| t.as_str()).unwrap();
        assert_eq!(text, "Hello");
    }

    #[test]
    fn parse_committed_transcript() {
        let data = json!({
            "message_type": "committed_transcript",
            "text": "Hello world",
            "language_code": "eng"
        });

        let msg_type = data.get("message_type").and_then(|t| t.as_str()).unwrap();
        assert_eq!(msg_type, "committed_transcript");
        let text = data.get("text").and_then(|t| t.as_str()).unwrap();
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn commit_message_construction() {
        let msg = serde_json::json!({
            "message_type": "input_audio_chunk",
            "audio_base_64": "",
            "commit": true,
            "sample_rate": 16000,
        });

        assert_eq!(msg["message_type"], "input_audio_chunk");
        assert_eq!(msg["audio_base_64"], "");
        assert_eq!(msg["commit"], true);
        assert_eq!(msg["sample_rate"], 16000);
    }

    #[test]
    fn audio_chunk_message_construction() {
        use base64::Engine;
        let audio = vec![0u8; 320]; // 10ms at 16kHz, 16-bit mono
        let audio_b64 = base64::engine::general_purpose::STANDARD.encode(&audio);
        let msg = serde_json::json!({
            "message_type": "input_audio_chunk",
            "audio_base_64": audio_b64,
            "commit": false,
            "sample_rate": 16000,
        });

        assert_eq!(msg["message_type"], "input_audio_chunk");
        assert_eq!(msg["commit"], false);
        // Verify base64 round-trip
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(msg["audio_base_64"].as_str().unwrap())
            .unwrap();
        assert_eq!(decoded.len(), 320);
    }

    #[tokio::test]
    async fn process_frame_forwards_non_stt_frames() {
        let settings = ElevenLabsSTTSettings::default();
        let mut svc = ElevenLabsRealtimeSTTService::new("test-key", settings);

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
        let settings = ElevenLabsSTTSettings::default();
        let mut svc = ElevenLabsRealtimeSTTService::new("test-key", settings);
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
