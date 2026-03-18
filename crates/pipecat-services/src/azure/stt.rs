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

use super::settings::AzureSTTSettings;
use crate::stt::{STTService, STTServiceState, stt_process_frame};

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type WsMessage = tokio_tungstenite::tungstenite::Message;
type WsWriter = Arc<Mutex<futures::stream::SplitSink<WsStream, WsMessage>>>;

/// Azure Speech STT service implementation.
///
/// Uses the Azure Speech Service WebSocket protocol. After connecting, sends
/// a `speech.config` text message, then streams audio as binary messages with
/// a 2-byte header-length prefix. Transcription results arrive as text messages
/// with `Path:speech.phrase` (final) and `Path:speech.hypothesis` (interim).
#[derive(Debug)]
pub struct AzureSTTService {
    state: STTServiceState,
    azure_settings: AzureSTTSettings,
    ws_writer: Option<WsWriter>,
    receive_task: Option<JoinHandle<()>>,
    request_id: String,
    /// Set by the receive task when a final transcription is emitted.
    processing_complete: Arc<AtomicBool>,
}

impl AzureSTTService {
    pub fn new(settings: AzureSTTSettings) -> Self {
        let base_settings = settings.base.clone();
        Self {
            state: STTServiceState::new("AzureSTTService", base_settings),
            azure_settings: settings,
            ws_writer: None,
            receive_task: None,
            request_id: new_request_id(),
            processing_complete: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Connect to the Azure Speech STT WebSocket.
    async fn connect(&mut self, ctx: &ProcessorContext) -> Result<()> {
        self.request_id = new_request_id();
        let url = self.azure_settings.build_url();
        debug!("Connecting to Azure STT: {}", url);

        let host = format!("{}.stt.speech.microsoft.com", self.azure_settings.region);
        let connection_id = new_request_id();

        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&url)
            .header("Ocp-Apim-Subscription-Key", &self.azure_settings.api_key)
            .header("X-ConnectionId", &connection_id)
            .header("Host", &host)
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .header("Sec-WebSocket-Version", "13")
            .body(())
            .map_err(|e| PipecatError::ProcessorError(format!("Azure STT URL build error: {e}")))?;

        let (ws_stream, _response) =
            tokio_tungstenite::connect_async(request)
                .await
                .map_err(|e| {
                    PipecatError::ProcessorError(format!("Azure STT WebSocket connect error: {e}"))
                })?;

        debug!("Azure STT WebSocket connected");

        let (writer, reader) = ws_stream.split();
        let writer = Arc::new(Mutex::new(writer));
        self.ws_writer = Some(writer.clone());

        // Send speech.config message
        let config_msg = build_speech_config_message(&self.request_id);
        writer
            .lock()
            .await
            .send(WsMessage::Text(config_msg.into()))
            .await
            .map_err(|e| {
                PipecatError::ProcessorError(format!("Azure STT speech.config send error: {e}"))
            })?;
        debug!("Azure STT: sent speech.config");

        // Send speech.context message
        let context_msg = build_speech_context_message(&self.request_id);
        writer
            .lock()
            .await
            .send(WsMessage::Text(context_msg.into()))
            .await
            .map_err(|e| {
                PipecatError::ProcessorError(format!("Azure STT speech.context send error: {e}"))
            })?;
        debug!("Azure STT: sent speech.context");

        // Send initial audio header (binary, with content-type for raw PCM)
        let sample_rate = if self.state.sample_rate > 0 {
            self.state.sample_rate
        } else {
            16000
        };
        let audio_header_msg = build_audio_header_message(&self.request_id, sample_rate);
        writer
            .lock()
            .await
            .send(WsMessage::Binary(audio_header_msg.into()))
            .await
            .map_err(|e| {
                PipecatError::ProcessorError(format!("Azure STT audio header send error: {e}"))
            })?;
        debug!("Azure STT: sent audio header");

        // Spawn receive task
        let bg_ctx = ctx.clone();
        let processing_flag = self.processing_complete.clone();
        self.receive_task = Some(tokio::spawn(receive_task(reader, bg_ctx, processing_flag)));

        Ok(())
    }

    /// Disconnect from the Azure STT WebSocket.
    async fn disconnect(&mut self) {
        if let Some(ref writer) = self.ws_writer {
            // Send empty audio message to signal end of stream
            let end_msg = build_audio_message(&self.request_id, None);
            let mut w = writer.lock().await;
            let _ = w.send(WsMessage::Binary(end_msg.into())).await;
            let _ = w.close().await;
        }
        self.ws_writer = None;

        if let Some(task) = self.receive_task.take() {
            task.abort();
        }
        self.processing_complete.store(false, Ordering::Release);

        debug!("Azure STT disconnected");
    }
}

#[async_trait]
impl FrameProcessor for AzureSTTService {
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
impl STTService for AzureSTTService {
    async fn run_stt(&mut self, audio: Bytes, _ctx: &ProcessorContext) -> Result<()> {
        if let Some(ref writer) = self.ws_writer {
            let msg = build_audio_message(&self.request_id, Some(&audio));
            let mut w = writer.lock().await;
            w.send(WsMessage::Binary(msg.into())).await.map_err(|e| {
                PipecatError::ProcessorError(format!("Azure STT audio send error: {e}"))
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

// ---------------------------------------------------------------------------
// Azure Speech WebSocket Protocol helpers
// ---------------------------------------------------------------------------

/// Generate a new request ID (UUID without dashes, uppercase hex).
fn new_request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::SystemTime;
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let ts = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:016X}{:016X}", ts, count)
}

/// Get current timestamp as milliseconds since epoch (string).
fn timestamp_ms() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis().to_string())
        .unwrap_or_default()
}

/// Generate a simple timestamp string (seconds.millis since epoch).
fn timestamp_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}.{:03}", d.as_secs(), d.subsec_millis()))
        .unwrap_or_default()
}

/// Build a text-frame payload with headers separated from body by \r\n\r\n.
///
/// Azure Speech text messages use the format:
/// ```text
/// Header1:value1\r\n
/// Header2:value2\r\n
/// \r\n
/// {json body}
/// ```
fn build_text_payload(headers: &[(&str, &str)], body: Option<&str>) -> String {
    let mut payload = String::new();
    for (k, v) in headers {
        payload.push_str(k);
        payload.push(':');
        payload.push_str(v);
        payload.push_str("\r\n");
    }
    payload.push_str("\r\n");
    if let Some(body) = body {
        payload.push_str(body);
    }
    payload
}

/// Build a binary-frame payload with 2-byte big-endian header length prefix.
///
/// Azure Speech binary messages use the format:
/// ```text
/// [2 bytes: header_length (big-endian u16)]
/// [header_length bytes: headers as text]
/// [remaining bytes: payload data]
/// ```
fn build_binary_payload(headers: &[(&str, &str)], data: Option<&[u8]>) -> Vec<u8> {
    let mut header_text = String::new();
    for (k, v) in headers {
        header_text.push_str(k);
        header_text.push(':');
        header_text.push_str(v);
        header_text.push_str("\r\n");
    }

    let header_bytes = header_text.as_bytes();
    let header_len = header_bytes.len();
    let data_len = data.map_or(0, |d| d.len());

    let mut payload = Vec::with_capacity(2 + header_len + data_len);
    payload.push(((header_len >> 8) & 0xff) as u8);
    payload.push((header_len & 0xff) as u8);
    payload.extend_from_slice(header_bytes);
    if let Some(d) = data {
        payload.extend_from_slice(d);
    }

    payload
}

/// Build the speech.config text message sent after connecting.
fn build_speech_config_message(request_id: &str) -> String {
    let body = serde_json::json!({
        "context": {
            "system": {
                "name": "pipecat-rs",
                "version": "0.1.0",
                "build": "rust"
            },
            "os": {
                "platform": std::env::consts::OS,
                "name": std::env::consts::OS,
                "version": ""
            },
            "audio": {
                "source": {
                    "connectivity": "Unknown",
                    "manufacturer": "Unknown",
                    "model": "Unknown",
                    "type": "Stream"
                }
            }
        },
        "recognition": "conversation"
    });

    build_text_payload(
        &[
            ("X-RequestId", request_id),
            ("Path", "speech.config"),
            ("Content-Type", "application/json"),
            ("X-Timestamp", &timestamp_ms()),
        ],
        Some(&body.to_string()),
    )
}

/// Build the speech.context text message.
fn build_speech_context_message(request_id: &str) -> String {
    build_text_payload(
        &[
            ("X-RequestId", request_id),
            ("Path", "speech.context"),
            ("Content-Type", "application/json"),
            ("X-Timestamp", &timestamp_ms()),
        ],
        Some("{}"),
    )
}

/// Build the initial audio header binary message (with content-type).
fn build_audio_header_message(request_id: &str, sample_rate: u32) -> Vec<u8> {
    // Build a minimal RIFF/WAV header for raw PCM
    let wav_header = build_wav_header(sample_rate, 16, 1);
    let ts = timestamp_ms();
    build_binary_payload(
        &[
            ("Path", "audio"),
            ("X-RequestId", request_id),
            ("X-Timestamp", &ts),
            ("Content-Type", "audio/wav"),
        ],
        Some(&wav_header),
    )
}

/// Build an audio data binary message.
fn build_audio_message(request_id: &str, data: Option<&[u8]>) -> Vec<u8> {
    let ts = timestamp_ms();
    build_binary_payload(
        &[
            ("Path", "audio"),
            ("X-RequestId", request_id),
            ("X-Timestamp", &ts),
        ],
        data,
    )
}

/// Build a minimal RIFF/WAV header for PCM audio.
/// The data chunk size is set to 0 (streaming mode).
fn build_wav_header(sample_rate: u32, bits_per_sample: u16, channels: u16) -> Vec<u8> {
    let byte_rate = sample_rate * (channels as u32) * (bits_per_sample as u32) / 8;
    let block_align = channels * bits_per_sample / 8;

    let mut header = Vec::with_capacity(44);
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&0u32.to_le_bytes()); // file size (0 for streaming)
    header.extend_from_slice(b"WAVE");
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    header.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    header.extend_from_slice(&channels.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(&byte_rate.to_le_bytes());
    header.extend_from_slice(&block_align.to_le_bytes());
    header.extend_from_slice(&bits_per_sample.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&0u32.to_le_bytes()); // data chunk size (0 for streaming)
    header
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

/// Extract the Path header and JSON body from an Azure Speech text message.
fn parse_text_message(text: &str) -> Option<(String, String)> {
    let (headers_part, body) = text.split_once("\r\n\r\n")?;
    let mut path = None;
    for line in headers_part.split("\r\n") {
        if let Some((key, value)) = line.split_once(':')
            && key.eq_ignore_ascii_case("Path")
        {
            path = Some(value.to_string());
        }
    }
    Some((path?, body.to_string()))
}

/// Background task that reads messages from the Azure STT WebSocket
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
                warn!("Azure STT WebSocket read error: {}", e);
                let _ = ctx
                    .push_error(&format!("Azure STT connection error: {e}"), false)
                    .await;
                break;
            }
        };

        let text = match msg {
            WsMessage::Text(t) => t.to_string(),
            WsMessage::Close(_) => {
                debug!("Azure STT WebSocket closed by server");
                break;
            }
            _ => continue,
        };

        let Some((path, body)) = parse_text_message(&text) else {
            trace!("Azure STT: received message without parseable path");
            continue;
        };

        match path.as_str() {
            "turn.start" => {
                debug!("Azure STT: turn started");
            }
            "speech.startDetected" => {
                trace!("Azure STT: speech start detected");
            }
            "speech.hypothesis" | "speech.fragment" => {
                if let Some(frame) = parse_hypothesis(&body) {
                    let _ = ctx.send_downstream(frame).await;
                }
            }
            "speech.phrase" => {
                if let Some(frame) = parse_phrase(&body) {
                    let is_final = matches!(&frame, Frame::Transcription(_));
                    if is_final {
                        processing_complete.store(true, Ordering::Release);
                    }
                    let _ = ctx.send_downstream(frame).await;
                }
            }
            "speech.endDetected" => {
                trace!("Azure STT: speech end detected");
            }
            "turn.end" => {
                debug!("Azure STT: turn ended");
            }
            _ => {
                trace!("Azure STT: unknown path: {}", path);
            }
        }
    }
}

/// Parse a speech.hypothesis message into an InterimTranscription frame.
fn parse_hypothesis(body: &str) -> Option<Frame> {
    let data: serde_json::Value = serde_json::from_str(body).ok()?;
    let text = data.get("Text").and_then(|t| t.as_str()).unwrap_or("");
    if text.is_empty() {
        return None;
    }

    Some(Frame::InterimTranscription(InterimTranscriptionFrame {
        text: text.to_string(),
        user_id: String::new(),
        timestamp: Some(timestamp_now()),
        language: None,
        result: Some(data),
    }))
}

/// Parse a speech.phrase message into a Transcription frame.
fn parse_phrase(body: &str) -> Option<Frame> {
    let data: serde_json::Value = serde_json::from_str(body).ok()?;
    let status = data
        .get("RecognitionStatus")
        .and_then(|s| s.as_str())
        .unwrap_or("");

    match status {
        "Success" => {
            // Get text from DisplayText (simple format) or NBest[0].Display (detailed format)
            let text = data
                .get("DisplayText")
                .and_then(|d| d.as_str())
                .or_else(|| {
                    data.get("NBest")
                        .and_then(|n| n.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|best| best.get("Display"))
                        .and_then(|d| d.as_str())
                })
                .unwrap_or("");

            if text.is_empty() {
                return None;
            }

            Some(Frame::Transcription(TranscriptionFrame {
                text: text.to_string(),
                user_id: String::new(),
                timestamp: Some(timestamp_now()),
                language: None,
                finalized: true,
                result: Some(data),
            }))
        }
        "NoMatch" | "InitialSilenceTimeout" | "EndOfDictation" => {
            trace!("Azure STT phrase: {}", status);
            None
        }
        "Error" => {
            let error_msg = data
                .get("DisplayText")
                .and_then(|d| d.as_str())
                .unwrap_or("Unknown error");
            warn!("Azure STT error: {}", error_msg);
            None
        }
        _ => {
            trace!("Azure STT: unknown recognition status: {}", status);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn build_text_payload_with_body() {
        let payload = build_text_payload(
            &[("Path", "speech.config"), ("X-RequestId", "abc123")],
            Some("{\"key\":\"value\"}"),
        );
        assert!(payload.contains("Path:speech.config\r\n"));
        assert!(payload.contains("X-RequestId:abc123\r\n"));
        assert!(payload.contains("\r\n\r\n{\"key\":\"value\"}"));
    }

    #[test]
    fn build_text_payload_without_body() {
        let payload = build_text_payload(&[("Path", "audio")], None);
        assert!(payload.ends_with("\r\n\r\n"));
    }

    #[test]
    fn build_binary_payload_structure() {
        let payload = build_binary_payload(
            &[("Path", "audio"), ("X-RequestId", "test123")],
            Some(b"audio_data"),
        );

        // First 2 bytes are big-endian header length
        let header_len = ((payload[0] as usize) << 8) | (payload[1] as usize);
        assert!(header_len > 0);
        assert_eq!(payload.len(), 2 + header_len + b"audio_data".len());

        // Header text should contain our headers
        let header_text = std::str::from_utf8(&payload[2..2 + header_len]).unwrap();
        assert!(header_text.contains("Path:audio\r\n"));
        assert!(header_text.contains("X-RequestId:test123\r\n"));

        // Payload data follows headers
        assert_eq!(&payload[2 + header_len..], b"audio_data");
    }

    #[test]
    fn build_binary_payload_no_data() {
        let payload = build_binary_payload(&[("Path", "audio")], None);
        let header_len = ((payload[0] as usize) << 8) | (payload[1] as usize);
        assert_eq!(payload.len(), 2 + header_len); // no data after headers
    }

    #[test]
    fn parse_text_message_basic() {
        let msg = "X-RequestId:abc123\r\nPath:speech.phrase\r\nContent-Type:application/json\r\n\r\n{\"RecognitionStatus\":\"Success\",\"DisplayText\":\"hello\"}";
        let (path, body) = parse_text_message(msg).unwrap();
        assert_eq!(path, "speech.phrase");
        assert!(body.contains("RecognitionStatus"));
    }

    #[test]
    fn parse_text_message_no_path() {
        let msg = "X-RequestId:abc123\r\n\r\n{}";
        assert!(parse_text_message(msg).is_none());
    }

    #[test]
    fn parse_hypothesis_basic() {
        let body = json!({"Text": "hello world", "Offset": 100, "Duration": 50}).to_string();
        let frame = parse_hypothesis(&body).unwrap();
        match frame {
            Frame::InterimTranscription(t) => {
                assert_eq!(t.text, "hello world");
                assert!(t.timestamp.is_some());
            }
            other => panic!("Expected InterimTranscription, got {other}"),
        }
    }

    #[test]
    fn parse_hypothesis_empty() {
        let body = json!({"Text": ""}).to_string();
        assert!(parse_hypothesis(&body).is_none());
    }

    #[test]
    fn parse_phrase_success_simple() {
        let body =
            json!({"RecognitionStatus": "Success", "DisplayText": "Hello world"}).to_string();
        let frame = parse_phrase(&body).unwrap();
        match frame {
            Frame::Transcription(t) => {
                assert_eq!(t.text, "Hello world");
                assert!(t.finalized);
            }
            other => panic!("Expected Transcription, got {other}"),
        }
    }

    #[test]
    fn parse_phrase_success_detailed() {
        let body = json!({
            "RecognitionStatus": "Success",
            "NBest": [{"Display": "Hello world", "Confidence": 0.95}]
        })
        .to_string();
        let frame = parse_phrase(&body).unwrap();
        match frame {
            Frame::Transcription(t) => {
                assert_eq!(t.text, "Hello world");
            }
            other => panic!("Expected Transcription, got {other}"),
        }
    }

    #[test]
    fn parse_phrase_no_match() {
        let body = json!({"RecognitionStatus": "NoMatch"}).to_string();
        assert!(parse_phrase(&body).is_none());
    }

    #[test]
    fn parse_phrase_empty_text() {
        let body = json!({"RecognitionStatus": "Success", "DisplayText": ""}).to_string();
        assert!(parse_phrase(&body).is_none());
    }

    #[test]
    fn parse_phrase_end_of_dictation() {
        let body = json!({"RecognitionStatus": "EndOfDictation"}).to_string();
        assert!(parse_phrase(&body).is_none());
    }

    #[test]
    fn wav_header_structure() {
        let header = build_wav_header(16000, 16, 1);
        assert_eq!(header.len(), 44);
        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(&header[8..12], b"WAVE");
        assert_eq!(&header[12..16], b"fmt ");
        assert_eq!(&header[36..40], b"data");
        // Sample rate at offset 24 (little-endian u32)
        let sr = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
        assert_eq!(sr, 16000);
    }

    #[test]
    fn speech_config_message_structure() {
        let msg = build_speech_config_message("TESTREQID");
        assert!(msg.contains("Path:speech.config\r\n"));
        assert!(msg.contains("X-RequestId:TESTREQID\r\n"));
        assert!(msg.contains("Content-Type:application/json\r\n"));
        assert!(msg.contains("pipecat-rs"));
        assert!(msg.contains("\"recognition\""));
    }

    #[test]
    fn speech_context_message_structure() {
        let msg = build_speech_context_message("TESTREQID");
        assert!(msg.contains("Path:speech.context\r\n"));
        assert!(msg.contains("X-RequestId:TESTREQID\r\n"));
        assert!(msg.contains("\r\n\r\n{}"));
    }

    #[test]
    fn binary_payload_round_trip() {
        // Encode a binary payload and verify we can decode it
        let payload = build_binary_payload(
            &[
                ("Path", "audio"),
                ("X-RequestId", "ABC123"),
                ("Content-Type", "audio/wav"),
            ],
            Some(b"\x00\x01\x02\x03"),
        );

        // Decode: first 2 bytes = big-endian header length
        let header_len = ((payload[0] as usize) << 8) | (payload[1] as usize);
        let header_text = std::str::from_utf8(&payload[2..2 + header_len]).unwrap();
        let data = &payload[2 + header_len..];

        // Parse headers back
        let mut path = None;
        let mut request_id = None;
        for line in header_text.split("\r\n") {
            if let Some((k, v)) = line.split_once(':') {
                match k {
                    "Path" => path = Some(v),
                    "X-RequestId" => request_id = Some(v),
                    _ => {}
                }
            }
        }

        assert_eq!(path, Some("audio"));
        assert_eq!(request_id, Some("ABC123"));
        assert_eq!(data, b"\x00\x01\x02\x03");
    }

    #[test]
    fn text_payload_round_trip() {
        // Encode a text payload and verify parse_text_message can extract it
        let payload = build_text_payload(
            &[
                ("X-RequestId", "REQ1"),
                ("Path", "speech.phrase"),
                ("Content-Type", "application/json"),
            ],
            Some("{\"RecognitionStatus\":\"Success\",\"DisplayText\":\"test\"}"),
        );

        let (path, body) = parse_text_message(&payload).unwrap();
        assert_eq!(path, "speech.phrase");

        let data: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(data["DisplayText"], "test");
    }

    #[tokio::test]
    async fn process_frame_forwards_non_stt_frames() {
        let settings = AzureSTTSettings::new("test-key", "eastus");
        let mut svc = AzureSTTService::new(settings);

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
        let settings = AzureSTTSettings::new("test-key", "eastus");
        let mut svc = AzureSTTService::new(settings);
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
