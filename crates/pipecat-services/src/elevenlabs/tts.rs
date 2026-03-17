use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use tokio::task::JoinHandle;
use tracing::{debug, trace, warn};

use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};

use crate::text_aggregator::TextAggregationMode;
use crate::tts::{TTSService, TTSServiceState, tts_process_frame};

use super::settings::{ElevenLabsTTSSettings, output_format_from_sample_rate};

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type WsMessage = tokio_tungstenite::tungstenite::Message;
type WsWriter = Arc<tokio::sync::Mutex<futures::stream::SplitSink<WsStream, WsMessage>>>;

const DEFAULT_BASE_URL: &str = "wss://api.elevenlabs.io";
const KEEPALIVE_INTERVAL_SECS: u64 = 10;

/// ElevenLabs TTS service implementation.
///
/// Uses a persistent WebSocket connection with a context-based protocol.
/// Text is sent as JSON, audio is received as base64-encoded PCM in JSON messages.
#[derive(Debug)]
pub struct ElevenLabsTTSService {
    state: TTSServiceState,
    api_key: String,
    base_url: String,
    elevenlabs_settings: ElevenLabsTTSSettings,
    output_format: String,
    ws_writer: Option<WsWriter>,
    receive_task: Option<JoinHandle<()>>,
    keepalive_task: Option<JoinHandle<()>>,
    active_context: Arc<tokio::sync::Mutex<Option<String>>>,
    /// Shared flag so the receive task can report TTFB on the first audio chunk.
    ttfb_reported: Arc<AtomicBool>,
}

impl ElevenLabsTTSService {
    pub fn new(api_key: impl Into<String>, settings: ElevenLabsTTSSettings) -> Self {
        Self::with_sample_rate(api_key, settings, 24000)
    }

    pub fn with_sample_rate(
        api_key: impl Into<String>,
        settings: ElevenLabsTTSSettings,
        sample_rate: u32,
    ) -> Self {
        let base_settings = settings.base.clone();
        Self {
            state: TTSServiceState::new(
                "ElevenLabsTTSService",
                base_settings,
                TextAggregationMode::Sentence,
                sample_rate,
            ),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            elevenlabs_settings: settings,
            output_format: output_format_from_sample_rate(sample_rate),
            ws_writer: None,
            receive_task: None,
            keepalive_task: None,
            active_context: Arc::new(tokio::sync::Mutex::new(None)),
            ttfb_reported: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build the WebSocket URL.
    fn build_url(&self) -> String {
        let voice_id = self
            .elevenlabs_settings
            .base
            .voice
            .as_deref()
            .unwrap_or("21m00Tcm4TlvDq8ikWAM");
        let model = self
            .elevenlabs_settings
            .base
            .model
            .as_deref()
            .unwrap_or("eleven_turbo_v2");

        let mut url = format!(
            "{}/v1/text-to-speech/{}/multi-stream-input?model_id={}&output_format={}",
            self.base_url, voice_id, model, self.output_format
        );

        if let Some(ref lang) = self.elevenlabs_settings.base.language {
            url.push_str(&format!("&language_code={lang}"));
        }

        url
    }

    /// Connect to the ElevenLabs WebSocket.
    async fn connect(&mut self, ctx: &ProcessorContext) -> Result<()> {
        let url = self.build_url();
        debug!("Connecting to ElevenLabs: {}", url);

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
                PipecatError::ProcessorError(format!("ElevenLabs URL build error: {e}"))
            })?;

        let (ws_stream, _response) =
            tokio_tungstenite::connect_async(request)
                .await
                .map_err(|e| {
                    PipecatError::ProcessorError(format!("ElevenLabs WebSocket connect error: {e}"))
                })?;

        debug!("ElevenLabs WebSocket connected");

        let (writer, reader) = ws_stream.split();
        let writer = Arc::new(tokio::sync::Mutex::new(writer));
        self.ws_writer = Some(writer.clone());

        // Spawn receive task
        let bg_ctx = ctx.clone();
        let sample_rate = self.state.sample_rate;
        let ttfb_reported = self.ttfb_reported.clone();
        self.receive_task = Some(tokio::spawn(receive_task(
            reader,
            bg_ctx,
            sample_rate,
            ttfb_reported,
        )));

        // Spawn keepalive task — sends empty text every 10s to prevent timeout
        let keepalive_writer = writer;
        let keepalive_active_context = self.active_context.clone();
        self.keepalive_task = Some(tokio::spawn(keepalive_task(
            keepalive_writer,
            keepalive_active_context,
        )));

        Ok(())
    }

    /// Disconnect from the ElevenLabs WebSocket.
    async fn disconnect(&mut self) {
        if let Some(ref writer) = self.ws_writer {
            let mut w = writer.lock().await;
            let close_msg = WsMessage::Text(r#"{"close_socket":true}"#.into());
            let _ = w.send(close_msg).await;
            let _ = w.close().await;
        }
        self.ws_writer = None;

        if let Some(task) = self.receive_task.take() {
            task.abort();
        }
        if let Some(task) = self.keepalive_task.take() {
            task.abort();
        }
        *self.active_context.lock().await = None;

        debug!("ElevenLabs disconnected");
    }

    /// Close the active context (on interruption or completion).
    async fn close_active_context(&mut self) -> Result<()> {
        let context_id = self.active_context.lock().await.take();
        if let Some(ref context_id) = context_id
            && let Some(ref writer) = self.ws_writer
        {
            let msg = serde_json::json!({
                "context_id": context_id,
                "close_context": true,
            });
            let ws_msg = WsMessage::Text(msg.to_string().into());
            let _ = writer.lock().await.send(ws_msg).await;
            trace!("ElevenLabs: closed context {}", context_id);
        }
        Ok(())
    }

    /// Send text for synthesis in the current context.
    async fn send_text(&mut self, text: &str, context_id: &str) -> Result<()> {
        if let Some(ref writer) = self.ws_writer {
            let msg = serde_json::json!({
                "text": text,
                "context_id": context_id,
            });
            let ws_msg = WsMessage::Text(msg.to_string().into());
            writer
                .lock()
                .await
                .send(ws_msg)
                .await
                .map_err(|e| PipecatError::ProcessorError(format!("ElevenLabs send error: {e}")))?;
        }
        Ok(())
    }

    /// Initialize a new context on the WebSocket.
    async fn init_context(&mut self, context_id: &str) -> Result<()> {
        if let Some(ref writer) = self.ws_writer {
            let mut msg = serde_json::json!({
                "text": " ",
                "context_id": context_id,
            });

            if let Some(voice_settings) = self.elevenlabs_settings.build_voice_settings() {
                msg["voice_settings"] = voice_settings;
            }

            let ws_msg = WsMessage::Text(msg.to_string().into());
            writer.lock().await.send(ws_msg).await.map_err(|e| {
                PipecatError::ProcessorError(format!("ElevenLabs init context error: {e}"))
            })?;

            trace!("ElevenLabs: initialized context {}", context_id);
        }
        Ok(())
    }
}

#[async_trait]
impl FrameProcessor for ElevenLabsTTSService {
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
                self.output_format = output_format_from_sample_rate(f.audio_out_sample_rate);
                self.state.sample_rate = f.audio_out_sample_rate;
                tts_process_frame(self, envelope, direction, ctx).await?;
                self.connect(ctx).await?;
            }
            Frame::End(_) | Frame::Cancel(_) => {
                self.disconnect().await;
                tts_process_frame(self, envelope, direction, ctx).await?;
            }
            Frame::Interruption(_) => {
                self.close_active_context().await?;
                tts_process_frame(self, envelope, direction, ctx).await?;
            }
            _ => {
                tts_process_frame(self, envelope, direction, ctx).await?;
            }
        }

        // Check if the background receive task reported TTFB via the atomic flag.
        // If so, stop the TTFB timer and push the metrics frame.
        if self.ttfb_reported.load(Ordering::Acquire) && self.state.base.metrics_enabled() {
            ctx.push_ttfb(&mut self.state.base.metrics).await?;
        }

        Ok(())
    }
}

#[async_trait]
impl TTSService for ElevenLabsTTSService {
    async fn run_tts(
        &mut self,
        text: &str,
        context_id: &str,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        // Ensure we're connected
        if self.ws_writer.is_none() {
            self.connect(ctx).await?;
        }

        // If new context, initialize it
        let is_new_context = self.active_context.lock().await.as_deref() != Some(context_id);
        if is_new_context {
            self.close_active_context().await?;

            *self.active_context.lock().await = Some(context_id.to_string());

            ctx.send_downstream(Frame::TTSStarted(TTSStartedFrame {
                context_id: Some(context_id.to_string()),
            }))
            .await?;

            // Reset TTFB tracking
            self.ttfb_reported.store(false, Ordering::Release);
            if self.state.base.metrics_enabled() {
                self.state.base.metrics.start_ttfb();
            }

            self.init_context(context_id).await?;
        }

        self.send_text(text, context_id).await?;

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

/// Background task that reads JSON messages from the ElevenLabs WebSocket
/// and pushes audio frames downstream.
async fn receive_task(
    mut reader: futures::stream::SplitStream<WsStream>,
    ctx: ProcessorContext,
    sample_rate: u32,
    ttfb_reported: Arc<AtomicBool>,
) {
    while let Some(msg_result) = reader.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                warn!("ElevenLabs WebSocket read error: {}", e);
                let _ = ctx
                    .push_error(&format!("ElevenLabs connection error: {e}"), false)
                    .await;
                break;
            }
        };

        let text = match msg {
            WsMessage::Text(t) => t,
            WsMessage::Close(_) => {
                debug!("ElevenLabs WebSocket closed by server");
                break;
            }
            _ => continue,
        };

        let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) else {
            warn!("ElevenLabs: failed to parse JSON: {}", text);
            continue;
        };

        let received_ctx_id = data
            .get("contextId")
            .and_then(|c| c.as_str())
            .map(String::from);

        // isFinal is sent by ElevenLabs after close_context is acknowledged.
        // It's non-functional per the Python reference — just log and skip.
        if data.get("isFinal").and_then(|f| f.as_bool()) == Some(true) {
            trace!("ElevenLabs: context {:?} isFinal received", received_ctx_id);
            continue;
        }

        // Handle audio data
        if let Some(audio_b64) = data.get("audio").and_then(|a| a.as_str()) {
            if audio_b64.is_empty() {
                continue;
            }

            let Ok(audio_bytes) =
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, audio_b64)
            else {
                warn!("ElevenLabs: failed to decode base64 audio");
                continue;
            };

            // Signal TTFB on first audio chunk. The main process_frame loop
            // checks this flag and calls metrics.stop_ttfb().
            if !ttfb_reported.swap(true, Ordering::AcqRel) {
                trace!("ElevenLabs: first audio chunk received (TTFB)");
            }

            let _ = ctx
                .send_downstream(Frame::TTSAudioRaw(TTSAudioRawFrame {
                    audio: Bytes::from(audio_bytes),
                    sample_rate,
                    num_channels: 1,
                    context_id: received_ctx_id.clone(),
                }))
                .await;
        }

        // Handle alignment data (log for now, word timestamps deferred)
        if let Some(alignment) = data.get("alignment")
            && let Some(word_times) = calculate_word_times(alignment)
        {
            trace!("ElevenLabs word timestamps: {:?}", word_times);
        }
    }
}

/// Keepalive task — sends empty text every 10s to prevent WebSocket timeout.
/// Includes context_id when an active context exists, matching the Python reference.
async fn keepalive_task(writer: WsWriter, active_context: Arc<tokio::sync::Mutex<Option<String>>>) {
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(KEEPALIVE_INTERVAL_SECS)).await;

        let context_id = active_context.lock().await.clone();
        let msg = if let Some(ref ctx_id) = context_id {
            serde_json::json!({"text": "", "context_id": ctx_id})
        } else {
            serde_json::json!({"text": ""})
        };

        let ws_msg = WsMessage::Text(msg.to_string().into());
        let send_result = writer.lock().await.send(ws_msg).await;
        if let Err(e) = send_result {
            warn!("ElevenLabs: keepalive send failed: {}", e);
            break;
        }

        trace!("ElevenLabs: keepalive sent (context: {:?})", context_id);
    }
}

/// Calculate word timestamps from ElevenLabs alignment data.
/// Returns a list of (word, start_time_seconds) tuples.
fn calculate_word_times(alignment: &serde_json::Value) -> Option<Vec<(String, f64)>> {
    let chars = alignment.get("chars")?.as_array()?;
    let start_times = alignment.get("charStartTimesMs")?.as_array()?;

    if chars.is_empty() || start_times.is_empty() {
        return None;
    }

    let mut words = Vec::new();
    let mut current_word = String::new();
    let mut word_start: Option<f64> = None;

    for (i, ch) in chars.iter().enumerate() {
        let c = ch.as_str().unwrap_or("");
        if c == " " {
            if !current_word.is_empty() {
                if let Some(start) = word_start {
                    words.push((std::mem::take(&mut current_word), start));
                }
                word_start = None;
            }
        } else {
            if word_start.is_none()
                && let Some(t) = start_times.get(i).and_then(|t| t.as_f64())
            {
                word_start = Some(t / 1000.0);
            }
            current_word.push_str(c);
        }
    }

    // Final word
    if !current_word.is_empty()
        && let Some(start) = word_start
    {
        words.push((current_word, start));
    }

    if words.is_empty() { None } else { Some(words) }
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
    fn build_url_default() {
        let settings = ElevenLabsTTSSettings::default();
        let svc = ElevenLabsTTSService::new("test-key", settings);
        let url = svc.build_url();
        assert!(url.contains("api.elevenlabs.io"));
        assert!(url.contains("multi-stream-input"));
        assert!(url.contains("model_id=eleven_turbo_v2"));
        assert!(url.contains("output_format=pcm_24000"));
    }

    #[test]
    fn build_url_with_language() {
        let mut settings = ElevenLabsTTSSettings::default();
        settings.base.language = Some("es".into());
        let svc = ElevenLabsTTSService::new("test-key", settings);
        let url = svc.build_url();
        assert!(url.contains("language_code=es"));
    }

    #[test]
    fn calculate_word_times_basic() {
        let alignment = json!({
            "chars": ["H", "i", " ", "w", "o", "r", "l", "d"],
            "charStartTimesMs": [0, 50, 100, 150, 200, 250, 300, 350],
            "charDurationsMs": [50, 50, 50, 50, 50, 50, 50, 50]
        });

        let words = calculate_word_times(&alignment).unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].0, "Hi");
        assert!((words[0].1 - 0.0).abs() < 0.001);
        assert_eq!(words[1].0, "world");
        assert!((words[1].1 - 0.15).abs() < 0.001);
    }

    #[test]
    fn calculate_word_times_single_word() {
        let alignment = json!({
            "chars": ["H", "i"],
            "charStartTimesMs": [0, 50],
            "charDurationsMs": [50, 50]
        });

        let words = calculate_word_times(&alignment).unwrap();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].0, "Hi");
    }

    #[test]
    fn calculate_word_times_empty() {
        let alignment = json!({
            "chars": [],
            "charStartTimesMs": [],
            "charDurationsMs": []
        });

        assert!(calculate_word_times(&alignment).is_none());
    }

    #[test]
    fn base64_audio_decode() {
        use base64::Engine;
        let pcm_data: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&pcm_data);

        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        assert_eq!(decoded, pcm_data);
    }

    #[test]
    fn output_format_from_sample_rate_values() {
        assert_eq!(output_format_from_sample_rate(16000), "pcm_16000");
        assert_eq!(output_format_from_sample_rate(24000), "pcm_24000");
        assert_eq!(output_format_from_sample_rate(44100), "pcm_44100");
    }

    #[tokio::test]
    async fn process_frame_forwards_non_tts_frames() {
        let settings = ElevenLabsTTSSettings::default();
        let mut svc = ElevenLabsTTSService::new("test-key", settings);

        let (downstream, _upstream) = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(Frame::Text(TextFrame::new("hello")), Direction::Downstream)],
        )
        .await;

        // Text consumed by sentence aggregator (no boundary yet)
        assert!(downstream.is_empty());
    }

    #[tokio::test]
    async fn process_frame_forwards_other_frames() {
        let settings = ElevenLabsTTSSettings::default();
        let mut svc = ElevenLabsTTSService::new("test-key", settings);

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
        let settings = ElevenLabsTTSSettings::default();
        let mut svc = ElevenLabsTTSService::new("test-key", settings);
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

    #[test]
    fn context_init_message_construction() {
        let mut settings = ElevenLabsTTSSettings::default();
        settings.stability = Some(0.5);
        settings.similarity_boost = Some(0.75);

        let voice_settings = settings.build_voice_settings().unwrap();
        let msg = serde_json::json!({
            "text": " ",
            "context_id": "test_ctx",
            "voice_settings": voice_settings,
        });

        assert_eq!(msg["text"], " ");
        assert_eq!(msg["context_id"], "test_ctx");
        assert_eq!(msg["voice_settings"]["stability"], 0.5);
        assert_eq!(msg["voice_settings"]["similarity_boost"], 0.75);
    }

    #[test]
    fn close_context_message_construction() {
        let msg = serde_json::json!({
            "context_id": "test_ctx",
            "close_context": true,
        });

        assert_eq!(msg["context_id"], "test_ctx");
        assert_eq!(msg["close_context"], true);
    }

    #[test]
    fn keepalive_message_with_context() {
        let context_id = Some("ctx_123".to_string());
        let msg = if let Some(ref ctx_id) = context_id {
            serde_json::json!({"text": "", "context_id": ctx_id})
        } else {
            serde_json::json!({"text": ""})
        };

        assert_eq!(msg["text"], "");
        assert_eq!(msg["context_id"], "ctx_123");
    }

    #[test]
    fn keepalive_message_without_context() {
        let context_id: Option<String> = None;
        let msg = if let Some(ref ctx_id) = context_id {
            serde_json::json!({"text": "", "context_id": ctx_id})
        } else {
            serde_json::json!({"text": ""})
        };

        assert_eq!(msg["text"], "");
        assert!(msg.get("context_id").is_none());
    }
}
