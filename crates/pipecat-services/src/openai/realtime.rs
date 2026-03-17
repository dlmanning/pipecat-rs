//! OpenAI Realtime API service.
//!
//! Implements `FrameProcessor` directly — replaces the STT→LLM→TTS cascade.
//! `InputAudioRawFrame` is consumed (not forwarded); `TTSAudioRawFrame` is
//! emitted from the background receive task.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use tokio::task::JoinHandle;
use tracing::{debug, trace, warn};

use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};

use crate::function_call::{FunctionCallParams, FunctionCallRegistry};
use crate::service_base::ServiceBase;

use super::realtime_events::{self, ServerEvent, calculate_audio_duration_ms};
use super::settings::{OpenAIRealtimeSettings, ResponseProperties, SessionProperties};

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type WsMessage = tokio_tungstenite::tungstenite::Message;
type WsWriter = Arc<tokio::sync::Mutex<futures::stream::SplitSink<WsStream, WsMessage>>>;

const DEFAULT_BASE_URL: &str = "wss://api.openai.com/v1/realtime";

/// Shared mutable state accessed by both `process_frame` and the background
/// receive task. Protected by `tokio::sync::Mutex`.
#[derive(Debug)]
struct SharedRealtimeState {
    /// Whether the API session has been configured (session.updated received).
    api_session_ready: bool,
    /// If true, a response.create will be sent once the session is ready.
    run_llm_when_api_session_ready: bool,
    /// Tracks the current audio response being streamed.
    current_audio_response: Option<CurrentAudioResponse>,
    /// Pending function calls by call_id.
    pending_function_calls: HashMap<String, realtime_events::ConversationItem>,
    /// Tool call IDs that have already been sent back to the API.
    completed_tool_calls: HashSet<String>,
    /// Items we created via conversation.item.create (to avoid duplicating).
    messages_added_manually: HashSet<String>,
    /// Whether audio input is paused.
    audio_input_paused: bool,
    /// Processing metrics start time — shared so the receive task can compute
    /// processing duration on response.done.
    processing_start: Option<std::time::Instant>,
    /// Current LLM context — shared so the receive task can pass it to
    /// function call handlers.
    context: Option<serde_json::Value>,
}

impl SharedRealtimeState {
    fn new(audio_input_paused: bool) -> Self {
        Self {
            api_session_ready: false,
            run_llm_when_api_session_ready: false,
            current_audio_response: None,
            pending_function_calls: HashMap::new(),
            completed_tool_calls: HashSet::new(),
            messages_added_manually: HashSet::new(),
            audio_input_paused,
            processing_start: None,
            context: None,
        }
    }
}

/// Tracks an in-flight audio response for truncation math.
#[derive(Debug, Clone)]
struct CurrentAudioResponse {
    item_id: String,
    content_index: u32,
    start_time_ms: u64,
    total_size: usize,
}

type SharedState = Arc<tokio::sync::Mutex<SharedRealtimeState>>;

/// OpenAI Realtime API service.
///
/// Replaces the STT→LLM→TTS cascade with a single bidirectional WebSocket
/// connection. Consumes `InputAudioRawFrame`, emits `TTSAudioRawFrame`.
pub struct OpenAIRealtimeService {
    base: ServiceBase,
    api_key: String,
    base_url: String,
    settings: OpenAIRealtimeSettings,

    // WebSocket
    ws_writer: Option<WsWriter>,
    receive_task: Option<JoinHandle<()>>,
    disconnecting: bool,

    // Shared state
    shared: SharedState,

    // Function calling — Arc so the receive task always sees current registrations
    function_registry: Arc<FunctionCallRegistry>,

    // Metrics coordination — the receive task signals TTFB via this flag
    ttfb_reported: Arc<AtomicBool>,

    // Context tracking
    context: Option<serde_json::Value>,
    llm_needs_conversation_setup: bool,

    // Video input
    video_input_paused: bool,
    video_frame_detail: String,
    last_video_sent: Option<Instant>,
}

impl std::fmt::Debug for OpenAIRealtimeService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIRealtimeService")
            .field("base", &self.base)
            .field("base_url", &self.base_url)
            .field("disconnecting", &self.disconnecting)
            .finish()
    }
}

impl OpenAIRealtimeService {
    pub fn new(api_key: impl Into<String>, settings: OpenAIRealtimeSettings) -> Self {
        let model = settings.base.model.clone();
        Self {
            base: ServiceBase::new("OpenAIRealtimeService", model),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            settings,
            ws_writer: None,
            receive_task: None,
            disconnecting: false,
            shared: Arc::new(tokio::sync::Mutex::new(SharedRealtimeState::new(false))),
            function_registry: Arc::new(FunctionCallRegistry::new()),
            ttfb_reported: Arc::new(AtomicBool::new(false)),
            context: None,
            llm_needs_conversation_setup: true,
            video_input_paused: false,
            video_frame_detail: "auto".to_string(),
            last_video_sent: None,
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Start with audio input paused.
    pub fn with_audio_paused(mut self, paused: bool) -> Self {
        let shared = Arc::get_mut(&mut self.shared).unwrap();
        shared.get_mut().audio_input_paused = paused;
        self
    }

    /// Start with video input paused.
    pub fn with_video_paused(mut self, paused: bool) -> Self {
        self.video_input_paused = paused;
        self
    }

    /// Set the detail level for video frames sent to the API.
    /// Valid values: "auto", "low", "high".
    pub fn with_video_frame_detail(mut self, detail: impl Into<String>) -> Self {
        self.video_frame_detail = detail.into();
        self
    }

    /// Access the function call registry for registration.
    ///
    /// Panics if called after `connect()` (i.e., after the receive task holds
    /// a reference). Register all functions before the pipeline starts.
    pub fn function_registry_mut(&mut self) -> &mut FunctionCallRegistry {
        Arc::get_mut(&mut self.function_registry).expect(
            "cannot modify function registry after connect (receive task holds a reference)",
        )
    }

    // -----------------------------------------------------------------------
    // WebSocket lifecycle
    // -----------------------------------------------------------------------

    async fn connect(&mut self, ctx: &ProcessorContext) -> Result<()> {
        let model = self
            .settings
            .base
            .model
            .as_deref()
            .unwrap_or("gpt-4o-realtime-preview");
        let url = format!("{}?model={}", self.base_url, model);
        debug!("Connecting to OpenAI Realtime: {}", url);

        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("OpenAI-Beta", "realtime=v1")
            .header("Host", extract_host(&url).unwrap_or("api.openai.com"))
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .header("Sec-WebSocket-Version", "13")
            .body(())
            .map_err(|e| {
                PipecatError::ProcessorError(format!("OpenAI Realtime URL build error: {e}"))
            })?;

        let (ws_stream, _response) =
            tokio_tungstenite::connect_async(request)
                .await
                .map_err(|e| {
                    PipecatError::ProcessorError(format!(
                        "OpenAI Realtime WebSocket connect error: {e}"
                    ))
                })?;

        debug!("OpenAI Realtime WebSocket connected");

        let (writer, reader) = ws_stream.split();
        let writer = Arc::new(tokio::sync::Mutex::new(writer));
        self.ws_writer = Some(writer.clone());

        // Clone everything the receive task needs
        let bg_ctx = ctx.clone();
        let shared = self.shared.clone();
        let ttfb_reported = self.ttfb_reported.clone();
        let function_registry = self.function_registry.clone();
        let settings = self.settings.clone();

        self.receive_task = Some(tokio::spawn(receive_task(
            reader,
            writer,
            bg_ctx,
            shared,
            ttfb_reported,
            function_registry,
            settings,
        )));

        Ok(())
    }

    async fn disconnect(&mut self) {
        self.disconnecting = true;
        {
            let mut shared = self.shared.lock().await;
            shared.api_session_ready = false;
        }
        self.base.metrics.reset();

        if let Some(ref writer) = self.ws_writer {
            let mut w = writer.lock().await;
            let _ = w.close().await;
        }
        self.ws_writer = None;

        if let Some(task) = self.receive_task.take() {
            task.abort();
        }

        {
            let mut shared = self.shared.lock().await;
            shared.completed_tool_calls.clear();
            shared.pending_function_calls.clear();
            shared.current_audio_response = None;
        }

        self.disconnecting = false;
        debug!("OpenAI Realtime disconnected");
    }

    // -----------------------------------------------------------------------
    // WebSocket send helpers
    // -----------------------------------------------------------------------

    async fn send_event(&self, event: &impl serde::Serialize) -> Result<()> {
        if self.disconnecting {
            return Ok(());
        }
        if let Some(ref writer) = self.ws_writer {
            let json = serde_json::to_string(event)
                .map_err(|e| PipecatError::ProcessorError(format!("JSON serialize error: {e}")))?;
            trace!("OpenAI Realtime send: {}", json);
            writer
                .lock()
                .await
                .send(WsMessage::Text(json.into()))
                .await
                .map_err(|e| {
                    PipecatError::ProcessorError(format!("OpenAI Realtime send error: {e}"))
                })?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Session management
    // -----------------------------------------------------------------------

    async fn send_session_update(&self) -> Result<()> {
        let mut props = self.settings.session_properties.clone();

        // Merge context instructions/tools into session properties
        if let Some(ref ctx_value) = self.context {
            if let Some(instruction) = ctx_value.get("system_instruction").and_then(|v| v.as_str())
            {
                props.instructions = Some(instruction.to_string());
            }
            if let Some(tools) = ctx_value.get("tools")
                && tools.is_array()
                && !tools.as_array().unwrap().is_empty()
            {
                props.tools = Some(tools.clone());
            }
        }

        // Also pull system_instruction from base settings
        if props.instructions.is_none()
            && let Some(ref instruction) = self.settings.base.system_instruction
        {
            props.instructions = Some(instruction.clone());
        }

        let event = realtime_events::SessionUpdateEvent::new(props);
        self.send_event(&event).await
    }

    // -----------------------------------------------------------------------
    // Context and response creation
    // -----------------------------------------------------------------------

    async fn create_response(&mut self, ctx: &ProcessorContext) -> Result<()> {
        let api_ready = {
            let shared = self.shared.lock().await;
            shared.api_session_ready
        };

        if !api_ready {
            let mut shared = self.shared.lock().await;
            shared.run_llm_when_api_session_ready = true;
            return Ok(());
        }

        // If this is the first context, send initial conversation items
        if self.llm_needs_conversation_setup {
            if let Some(ref context_value) = self.context
                && let Some(messages) = context_value.get("messages").and_then(|m| m.as_array())
            {
                for msg in messages {
                    let item = context_message_to_conversation_item(msg);
                    let event_item_id = item.id.clone();
                    let event = realtime_events::ConversationItemCreateEvent::new(item);
                    {
                        let mut shared = self.shared.lock().await;
                        shared.messages_added_manually.insert(event_item_id);
                    }
                    self.send_event(&event).await?;
                }
            }

            self.send_session_update().await?;
            self.llm_needs_conversation_setup = false;
        }

        // Push LLMFullResponseStart
        ctx.send_downstream(Frame::LLMFullResponseStart(LLMFullResponseStartFrame {
            skip_tts: None,
        }))
        .await?;

        // Start metrics
        if self.base.metrics_enabled() {
            let now = std::time::Instant::now();
            self.base.metrics.start_processing();
            self.base.metrics.start_ttfb();
            self.ttfb_reported.store(false, Ordering::Release);
            // Share processing start time with receive task for response.done
            let mut shared = self.shared.lock().await;
            shared.processing_start = Some(now);
        }

        // Determine output modalities
        let modalities = self
            .settings
            .session_properties
            .output_modalities
            .clone()
            .unwrap_or_else(|| vec!["text".into(), "audio".into()]);

        let event = realtime_events::ResponseCreateEvent::new(Some(ResponseProperties {
            output_modalities: Some(modalities),
            ..Default::default()
        }));
        self.send_event(&event).await
    }

    async fn handle_context(
        &mut self,
        context_value: serde_json::Value,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let first_context = self.context.is_none();
        self.context = Some(context_value.clone());
        {
            let mut shared = self.shared.lock().await;
            shared.context = Some(context_value);
        }

        if first_context {
            // Process any existing tool results without sending them
            self.process_completed_function_calls(false, ctx).await?;
            self.create_response(ctx).await?;
        } else {
            // Updated context — send any new tool results
            self.process_completed_function_calls(true, ctx).await?;
        }

        Ok(())
    }

    /// Scan context messages for tool results and send new ones to the API.
    async fn process_completed_function_calls(
        &mut self,
        send_new_results: bool,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let context_value = match &self.context {
            Some(c) => c.clone(),
            None => return Ok(()),
        };

        let messages = context_value
            .get("messages")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();

        let mut sent_new = false;

        for msg in &messages {
            if let Some(tool_call_id) = msg.get("tool_call_id").and_then(|v| v.as_str()) {
                let already_completed = {
                    let shared = self.shared.lock().await;
                    shared.completed_tool_calls.contains(tool_call_id)
                };

                if !already_completed {
                    if send_new_results {
                        let content = msg
                            .get("content")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        let content_str = match &content {
                            serde_json::Value::String(s) => s.clone(),
                            _ => serde_json::to_string(&content).unwrap_or_default(),
                        };

                        let item = realtime_events::ConversationItem {
                            id: realtime_events::make_item_id(),
                            item_type: "function_call_output".into(),
                            role: None,
                            content: None,
                            call_id: Some(tool_call_id.to_string()),
                            name: None,
                            arguments: None,
                            output: Some(content_str),
                            status: None,
                            object: None,
                        };
                        let event = realtime_events::ConversationItemCreateEvent::new(item);
                        self.send_event(&event).await?;
                        sent_new = true;
                    }

                    let mut shared = self.shared.lock().await;
                    shared.completed_tool_calls.insert(tool_call_id.to_string());
                }
            }
        }

        // If we sent new results, trigger a new response (matching Python's
        // _create_response call which includes LLMFullResponseStart + metrics)
        if sent_new {
            self.create_response(ctx).await?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Interruption handling
    // -----------------------------------------------------------------------

    async fn handle_interruption(&mut self, ctx: &ProcessorContext) -> Result<()> {
        // Check if turn detection is explicitly disabled
        let turn_detection_disabled = self
            .settings
            .session_properties
            .audio
            .as_ref()
            .and_then(|a| a.input.as_ref())
            .and_then(|i| i.turn_detection.as_ref())
            .is_some_and(|td| td.is_none());

        if turn_detection_disabled {
            self.send_event(&realtime_events::InputAudioBufferClearEvent::new())
                .await?;
            self.send_event(&realtime_events::ResponseCancelEvent::new())
                .await?;
        }

        // Truncate current audio response
        self.truncate_current_audio_response().await?;

        // Reset metrics
        self.base.metrics.reset();

        // Push end frames
        ctx.send_downstream(Frame::LLMFullResponseEnd(LLMFullResponseEndFrame {
            skip_tts: None,
        }))
        .await?;

        let has_audio_modality = self
            .settings
            .session_properties
            .output_modalities
            .as_ref()
            .is_none_or(|m| m.iter().any(|s| s == "audio"));

        if has_audio_modality {
            ctx.send_downstream(Frame::TTSStopped(TTSStoppedFrame::default()))
                .await?;
        }

        Ok(())
    }

    async fn send_user_video(&mut self, frame: &ImageRawFrame) -> Result<()> {
        // Rate limit: 1 frame per second
        if let Some(last) = self.last_video_sent
            && last.elapsed() < Duration::from_secs(1)
        {
            return Ok(());
        }
        self.last_video_sent = Some(Instant::now());

        trace!(
            "Sending video frame to OpenAI Realtime ({}x{})",
            frame.size.0, frame.size.1
        );

        // Raw pixels → JPEG (supports RGB and RGBA)
        let (w, h) = frame.size;
        let format = frame.format.as_deref().unwrap_or("RGB");
        let img: image::DynamicImage = match format {
            "RGBA" => image::RgbaImage::from_raw(w, h, frame.image.to_vec())
                .map(image::DynamicImage::ImageRgba8)
                .ok_or_else(|| {
                    PipecatError::ProcessorError("invalid RGBA image dimensions".into())
                })?,
            _ => image::RgbImage::from_raw(w, h, frame.image.to_vec())
                .map(image::DynamicImage::ImageRgb8)
                .ok_or_else(|| {
                    PipecatError::ProcessorError("invalid RGB image dimensions".into())
                })?,
        };
        let mut jpeg_buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut jpeg_buf);
        img.write_to(&mut cursor, image::ImageFormat::Jpeg)
            .map_err(|e| PipecatError::ProcessorError(format!("JPEG encode error: {e}")))?;

        // base64 → data URI
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &jpeg_buf);
        let data_uri = format!("data:image/jpeg;base64,{}", b64);

        // Send as conversation item
        let item = realtime_events::ConversationItem {
            id: realtime_events::make_item_id(),
            item_type: "message".into(),
            role: Some("user".into()),
            content: Some(vec![realtime_events::ItemContent {
                content_type: "input_image".into(),
                text: None,
                audio: None,
                transcript: None,
                image_url: Some(data_uri),
                detail: Some(self.video_frame_detail.clone()),
            }]),
            call_id: None,
            name: None,
            arguments: None,
            output: None,
            status: None,
            object: None,
        };
        let event = realtime_events::ConversationItemCreateEvent::new(item);
        self.send_event(&event).await
    }

    async fn truncate_current_audio_response(&mut self) -> Result<()> {
        let current = {
            let mut shared = self.shared.lock().await;
            shared.current_audio_response.take()
        };

        let Some(current) = current else {
            return Ok(());
        };

        let audio_duration_ms = calculate_audio_duration_ms(current.total_size);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let elapsed_ms = now_ms.saturating_sub(current.start_time_ms);
        let truncate_ms = std::cmp::min(elapsed_ms, audio_duration_ms);

        let event = realtime_events::ConversationItemTruncateEvent::new(
            current.item_id,
            current.content_index,
            truncate_ms,
        );
        self.send_event(&event).await
    }
}

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for OpenAIRealtimeService {
    fn name(&self) -> &str {
        self.base.processor.name()
    }

    fn id(&self) -> u64 {
        self.base.processor.id()
    }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        match &envelope.frame {
            Frame::Start(f) => {
                self.base.handle_start_frame(f);
                ctx.push_frame(envelope, direction).await?;
                self.connect(ctx).await?;
            }

            Frame::End(_) | Frame::Cancel(_) => {
                self.disconnect().await;
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::InputAudioRaw(f) => {
                let paused = {
                    let shared = self.shared.lock().await;
                    shared.audio_input_paused
                };
                if !paused {
                    let audio_b64 = base64::Engine::encode(
                        &base64::engine::general_purpose::STANDARD,
                        &f.audio,
                    );
                    let event = realtime_events::InputAudioBufferAppendEvent::new(audio_b64);
                    self.send_event(&event).await?;
                }
                // Consumed — not forwarded
            }

            Frame::InputImageRaw(f) => {
                if !self.video_input_paused {
                    self.send_user_video(f).await?;
                }
                // Consumed — not forwarded
            }

            Frame::LLMContext(f) => {
                let context_value = f.context.clone();
                self.handle_context(context_value, ctx).await?;
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::Interruption(_) => {
                self.handle_interruption(ctx).await?;
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::UserStoppedSpeaking(_) => {
                // If turn detection is disabled, commit buffer manually
                let turn_detection_disabled = self
                    .settings
                    .session_properties
                    .audio
                    .as_ref()
                    .and_then(|a| a.input.as_ref())
                    .and_then(|i| i.turn_detection.as_ref())
                    .is_some_and(|td| td.is_none());

                if turn_detection_disabled {
                    self.send_event(&realtime_events::InputAudioBufferCommitEvent::new())
                        .await?;
                }
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::BotStoppedSpeaking(_) => {
                let mut shared = self.shared.lock().await;
                shared.current_audio_response = None;
                drop(shared);
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::LLMUpdateSettings(f) => {
                let changed = self.settings.base.apply_delta(&f.settings);
                if changed.contains(&"model".to_string())
                    && let Some(ref model) = self.settings.base.model
                {
                    self.base.set_model(model.clone());
                }
                // If session properties changed, send session.update
                if let Some(session_props) = f.settings.get("session_properties")
                    && let Ok(props) =
                        serde_json::from_value::<SessionProperties>(session_props.clone())
                {
                    self.settings.session_properties = props;
                    self.send_session_update().await?;
                }
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::LLMSetTools(_) => {
                // Send updated session to API
                self.send_session_update().await?;
                ctx.push_frame(envelope, direction).await?;
            }

            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }

        // Check TTFB flag from receive task
        if self.ttfb_reported.load(Ordering::Acquire) && self.base.metrics_enabled() {
            ctx.push_ttfb(&mut self.base.metrics).await?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Background receive task
// ---------------------------------------------------------------------------

async fn receive_task(
    mut reader: futures::stream::SplitStream<WsStream>,
    writer: WsWriter,
    ctx: ProcessorContext,
    shared: SharedState,
    ttfb_reported: Arc<AtomicBool>,
    function_registry: Arc<FunctionCallRegistry>,
    settings: OpenAIRealtimeSettings,
) {
    while let Some(msg_result) = reader.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                warn!("OpenAI Realtime WebSocket read error: {}", e);
                let _ = ctx
                    .push_error(&format!("OpenAI Realtime connection error: {e}"), false)
                    .await;
                break;
            }
        };

        let text = match msg {
            WsMessage::Text(t) => t,
            WsMessage::Close(_) => {
                debug!("OpenAI Realtime WebSocket closed by server");
                break;
            }
            _ => continue,
        };

        let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) else {
            warn!("OpenAI Realtime: failed to parse JSON");
            continue;
        };

        let event = realtime_events::parse_server_event(&data);
        trace!("OpenAI Realtime event: {:?}", event);

        match event {
            ServerEvent::SessionCreated => {
                // Send session.update with current properties
                let props = build_session_properties(&settings);
                let update = realtime_events::SessionUpdateEvent::new(props);
                if let Err(e) = send_ws_event(&writer, &update).await {
                    warn!("Failed to send session.update: {}", e);
                }
            }

            ServerEvent::SessionUpdated => {
                let mut state = shared.lock().await;
                state.api_session_ready = true;
                let should_create_response = state.run_llm_when_api_session_ready;
                if should_create_response {
                    state.run_llm_when_api_session_ready = false;
                    drop(state);
                    // Trigger response creation
                    let event = realtime_events::ResponseCreateEvent::new(None);
                    if let Err(e) = send_ws_event(&writer, &event).await {
                        warn!("Failed to send response.create: {}", e);
                    }
                }
            }

            ServerEvent::ResponseAudioDelta {
                item_id,
                content_index,
                delta,
            } => {
                // Decode base64 audio
                let Ok(audio_bytes) =
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &delta)
                else {
                    warn!("OpenAI Realtime: failed to decode base64 audio");
                    continue;
                };

                let audio_len = audio_bytes.len();

                {
                    let mut state = shared.lock().await;
                    if state.current_audio_response.is_none() {
                        // First audio chunk — signal TTSStarted
                        state.current_audio_response = Some(CurrentAudioResponse {
                            item_id: item_id.clone(),
                            content_index,
                            start_time_ms: now_ms(),
                            total_size: 0,
                        });
                        let _ = ctx
                            .send_downstream(Frame::TTSStarted(TTSStartedFrame::default()))
                            .await;
                    }
                    if let Some(ref mut current) = state.current_audio_response {
                        current.total_size += audio_len;
                    }
                }

                // Signal TTFB on first audio chunk
                if !ttfb_reported.swap(true, Ordering::AcqRel) {
                    trace!("OpenAI Realtime: first audio chunk (TTFB)");
                }

                // Push audio downstream
                let _ = ctx
                    .send_downstream(Frame::TTSAudioRaw(TTSAudioRawFrame {
                        audio: Bytes::from(audio_bytes),
                        sample_rate: 24000,
                        num_channels: 1,
                        context_id: None,
                    }))
                    .await;
            }

            ServerEvent::ResponseAudioDone { .. } => {
                // Audio stream complete — TTSStopped (but don't clear tracking
                // until BotStoppedSpeaking)
                let _ = ctx
                    .send_downstream(Frame::TTSStopped(TTSStoppedFrame::default()))
                    .await;
            }

            ServerEvent::ResponseTextDelta { delta } => {
                let _ = ctx
                    .send_downstream(Frame::LLMText(TextFrame {
                        text: delta,
                        skip_tts: None,
                        includes_inter_frame_spaces: false,
                        append_to_context: false,
                    }))
                    .await;
            }

            ServerEvent::ResponseAudioTranscriptDelta { delta } => {
                // Send as both LLMText and TTSText for context/logging
                let _ = ctx
                    .send_downstream(Frame::LLMText(TextFrame {
                        text: delta.clone(),
                        skip_tts: None,
                        includes_inter_frame_spaces: false,
                        append_to_context: false,
                    }))
                    .await;
                let _ = ctx
                    .send_downstream(Frame::TTSText(AggregatedTextFrame {
                        text: delta,
                        aggregated_by: "realtime".into(),
                        context_id: None,
                        skip_tts: None,
                        includes_inter_frame_spaces: false,
                        append_to_context: false,
                    }))
                    .await;
            }

            ServerEvent::SpeechStarted { item_id, .. } => {
                // Server VAD detected speech start — broadcast interruption
                debug!("OpenAI Realtime: speech started (item: {})", item_id);

                // Truncate current audio
                {
                    let current = {
                        let mut state = shared.lock().await;
                        state.current_audio_response.take()
                    };
                    if let Some(current) = current {
                        let audio_duration_ms = calculate_audio_duration_ms(current.total_size);
                        let elapsed_ms = now_ms().saturating_sub(current.start_time_ms);
                        let truncate_ms = std::cmp::min(elapsed_ms, audio_duration_ms);
                        let event = realtime_events::ConversationItemTruncateEvent::new(
                            current.item_id,
                            current.content_index,
                            truncate_ms,
                        );
                        let _ = send_ws_event(&writer, &event).await;
                    }
                }

                let _ = ctx
                    .broadcast(Frame::UserStartedSpeaking(UserStartedSpeakingFrame))
                    .await;
                let _ = ctx.broadcast(Frame::Interruption(InterruptionFrame)).await;
            }

            ServerEvent::SpeechStopped { .. } => {
                let _ = ctx
                    .broadcast(Frame::UserStoppedSpeaking(UserStoppedSpeakingFrame))
                    .await;
            }

            ServerEvent::TranscriptionCompleted { transcript } => {
                let _ = ctx
                    .send_upstream(Frame::Transcription(TranscriptionFrame {
                        text: transcript,
                        user_id: String::new(),
                        timestamp: None,
                        language: None,
                        finalized: true,
                        result: None,
                    }))
                    .await;
            }

            ServerEvent::TranscriptionDelta { delta } => {
                let _ = ctx
                    .send_upstream(Frame::InterimTranscription(InterimTranscriptionFrame {
                        text: delta,
                        user_id: String::new(),
                        timestamp: None,
                        language: None,
                        result: None,
                    }))
                    .await;
            }

            ServerEvent::ConversationItemAdded { item } => {
                if item.item_type == "function_call"
                    && let Some(ref call_id) = item.call_id
                {
                    let mut state = shared.lock().await;
                    if !state.pending_function_calls.contains_key(call_id) {
                        state
                            .pending_function_calls
                            .insert(call_id.clone(), item.clone());
                    }
                }

                // Skip items we created manually (prevents duplicate LLMFullResponseStart)
                let was_manual = {
                    let mut state = shared.lock().await;
                    state.messages_added_manually.remove(&item.id)
                };
                if was_manual {
                    continue;
                }

                // Track assistant response — push LLMFullResponseStart for
                // server-originated assistant items
                if item.role.as_deref() == Some("assistant") {
                    let _ = ctx
                        .send_downstream(Frame::LLMFullResponseStart(LLMFullResponseStartFrame {
                            skip_tts: None,
                        }))
                        .await;
                }
            }

            ServerEvent::FunctionCallArgumentsDone { call_id, arguments } => {
                handle_function_call(
                    &call_id,
                    &arguments,
                    &shared,
                    &function_registry,
                    &writer,
                    &ctx,
                )
                .await;
            }

            ServerEvent::ResponseDone { response } => {
                let mut metrics = Vec::new();

                // Push token usage metrics
                if let Some(usage) = &response.usage {
                    let cache_read = usage
                        .input_token_details
                        .as_ref()
                        .and_then(|d| d.cached_tokens);

                    metrics.push(MetricsData::LlmUsage {
                        processor: "OpenAIRealtimeService".into(),
                        model: settings.base.model.clone(),
                        prompt_tokens: usage.input_tokens,
                        completion_tokens: usage.output_tokens,
                        total_tokens: usage.total_tokens,
                        cache_read_input_tokens: cache_read,
                        cache_creation_input_tokens: None,
                    });
                }

                // Stop processing metrics (started in create_response)
                {
                    let mut state = shared.lock().await;
                    if let Some(start) = state.processing_start.take() {
                        metrics.push(MetricsData::Processing {
                            processor: "OpenAIRealtimeService".into(),
                            model: settings.base.model.clone(),
                            value_secs: start.elapsed().as_secs_f64(),
                        });
                    }
                }

                if !metrics.is_empty() {
                    let _ = ctx
                        .send_downstream(Frame::Metrics(MetricsFrame { data: metrics }))
                        .await;
                }

                let _ = ctx
                    .send_downstream(Frame::LLMFullResponseEnd(LLMFullResponseEndFrame {
                        skip_tts: None,
                    }))
                    .await;
            }

            ServerEvent::Error { error } => {
                let code = error.code.as_deref().unwrap_or("");
                // Some errors are expected and non-fatal
                if code == "response_cancel_not_active"
                    || code == "conversation_already_has_active_response"
                {
                    debug!("OpenAI Realtime (non-fatal): {}", error.message);
                } else {
                    warn!("OpenAI Realtime error: {} ({})", error.message, code);
                    let _ = ctx
                        .push_error(
                            &format!("OpenAI Realtime error: {} ({})", error.message, code),
                            false,
                        )
                        .await;
                    break;
                }
            }

            ServerEvent::Unknown { event_type } => {
                trace!("OpenAI Realtime: unhandled event type: {}", event_type);
            }
        }
    }

    debug!("OpenAI Realtime receive task ended");
}

// ---------------------------------------------------------------------------
// Function call handling (in receive task)
// ---------------------------------------------------------------------------

async fn handle_function_call(
    call_id: &str,
    arguments: &str,
    shared: &SharedState,
    function_registry: &Arc<FunctionCallRegistry>,
    writer: &WsWriter,
    ctx: &ProcessorContext,
) {
    let function_call_item = {
        let mut state = shared.lock().await;
        state.pending_function_calls.remove(call_id)
    };

    let Some(item) = function_call_item else {
        warn!(
            "OpenAI Realtime: function call arguments done for unknown call_id: {}",
            call_id
        );
        return;
    };

    let function_name = item.name.unwrap_or_default();
    let parsed_args: serde_json::Value =
        serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));

    // Broadcast FunctionCallsStarted
    let function_call = FunctionCallFromLLM {
        function_name: function_name.clone(),
        tool_call_id: call_id.to_string(),
        arguments: parsed_args.clone(),
    };
    let _ = ctx
        .broadcast(Frame::FunctionCallsStarted(FunctionCallsStartedFrame {
            function_calls: vec![function_call],
        }))
        .await;

    let registry_item = function_registry.lookup(&function_name);

    // Broadcast FunctionCallInProgress
    let _ = ctx
        .broadcast(Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
            function_name: function_name.clone(),
            tool_call_id: call_id.to_string(),
            arguments: parsed_args.clone(),
            cancel_on_interruption: registry_item
                .map(|item| item.cancel_on_interruption)
                .unwrap_or(false),
        }))
        .await;

    // Build LLMContext from shared state
    let llm_context = {
        let state = shared.lock().await;
        if let Some(ref ctx_value) = state.context {
            let messages = ctx_value
                .get("messages")
                .cloned()
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default();
            let context = pipecat_context::LLMContext::new(messages);
            if let Some(tools) = ctx_value.get("tools").cloned() {
                context.set_tools(tools);
            }
            if let Some(tool_choice) = ctx_value.get("tool_choice").cloned() {
                context.set_tool_choice(tool_choice);
            }
            context
        } else {
            pipecat_context::LLMContext::new(vec![])
        }
    };

    // Execute function
    let result_value = if let Some(reg_item) = registry_item {
        let params = FunctionCallParams {
            function_name: function_name.clone(),
            tool_call_id: call_id.to_string(),
            arguments: parsed_args.clone(),
            context: llm_context,
        };
        match tokio::time::timeout(reg_item.timeout, (reg_item.handler)(params)).await {
            Ok(result) => result,
            Err(_) => {
                warn!(
                    "Function call [{}:{}] timed out after {:?}",
                    function_name, call_id, reg_item.timeout
                );
                serde_json::Value::Null
            }
        }
    } else {
        warn!("No handler registered for function: {}", function_name);
        serde_json::json!({"error": format!("No handler for {}", function_name)})
    };

    // Broadcast FunctionCallResult
    let _ = ctx
        .broadcast(Frame::FunctionCallResult(FunctionCallResultFrame {
            function_name: function_name.clone(),
            tool_call_id: call_id.to_string(),
            arguments: parsed_args,
            result: result_value.clone(),
            run_llm: Some(true),
            properties: Some(FunctionCallResultProperties {
                run_llm: Some(true),
            }),
        }))
        .await;

    // Send function result back to API
    let result_str = match &result_value {
        serde_json::Value::String(s) => s.clone(),
        _ => serde_json::to_string(&result_value).unwrap_or_default(),
    };

    let result_item = realtime_events::ConversationItem {
        id: realtime_events::make_item_id(),
        item_type: "function_call_output".into(),
        role: None,
        content: None,
        call_id: Some(call_id.to_string()),
        name: None,
        arguments: None,
        output: Some(result_str),
        status: None,
        object: None,
    };
    let create_event = realtime_events::ConversationItemCreateEvent::new(result_item);
    let _ = send_ws_event(writer, &create_event).await;

    // Mark as completed
    {
        let mut state = shared.lock().await;
        state.completed_tool_calls.insert(call_id.to_string());
    }

    // Trigger new response for continuation
    let response_event = realtime_events::ResponseCreateEvent::new(None);
    let _ = send_ws_event(writer, &response_event).await;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Send a serializable event via the shared WebSocket writer.
async fn send_ws_event(writer: &WsWriter, event: &impl serde::Serialize) -> Result<()> {
    let json = serde_json::to_string(event)
        .map_err(|e| PipecatError::ProcessorError(format!("JSON serialize error: {e}")))?;
    writer
        .lock()
        .await
        .send(WsMessage::Text(json.into()))
        .await
        .map_err(|e| PipecatError::ProcessorError(format!("OpenAI Realtime send error: {e}")))
}

/// Build session properties for the initial session.update from settings.
fn build_session_properties(settings: &OpenAIRealtimeSettings) -> SessionProperties {
    let mut props = settings.session_properties.clone();
    if props.instructions.is_none()
        && let Some(ref instruction) = settings.base.system_instruction
    {
        props.instructions = Some(instruction.clone());
    }
    props
}

/// Convert an LLM context message (JSON) into a ConversationItem for the
/// Realtime API's conversation.item.create.
fn context_message_to_conversation_item(
    msg: &serde_json::Value,
) -> realtime_events::ConversationItem {
    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");

    let content_text = msg.get("content").and_then(|v| v.as_str());

    let content = content_text.map(|text| {
        let content_type = if role == "user" { "input_text" } else { "text" };
        vec![realtime_events::ItemContent {
            content_type: content_type.to_string(),
            text: Some(text.to_string()),
            audio: None,
            transcript: None,
            image_url: None,
            detail: None,
        }]
    });

    realtime_events::ConversationItem {
        id: realtime_events::make_item_id(),
        item_type: "message".into(),
        role: Some(role.to_string()),
        content,
        call_id: None,
        name: None,
        arguments: None,
        output: None,
        status: None,
        object: None,
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn extract_host(url: &str) -> Option<&str> {
    let after_scheme = url
        .strip_prefix("wss://")
        .or_else(|| url.strip_prefix("ws://"))?;
    after_scheme.split('/').next()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn context_message_to_item_user() {
        let msg = json!({"role": "user", "content": "Hello"});
        let item = context_message_to_conversation_item(&msg);
        assert_eq!(item.item_type, "message");
        assert_eq!(item.role.as_deref(), Some("user"));
        let content = item.content.unwrap();
        assert_eq!(content[0].content_type, "input_text");
        assert_eq!(content[0].text.as_deref(), Some("Hello"));
    }

    #[test]
    fn context_message_to_item_assistant() {
        let msg = json!({"role": "assistant", "content": "Hi there!"});
        let item = context_message_to_conversation_item(&msg);
        assert_eq!(item.role.as_deref(), Some("assistant"));
        let content = item.content.unwrap();
        assert_eq!(content[0].content_type, "text");
    }

    #[test]
    fn extract_host_basic() {
        assert_eq!(
            extract_host("wss://api.openai.com/v1/realtime"),
            Some("api.openai.com")
        );
        assert_eq!(
            extract_host("ws://localhost:8080/ws"),
            Some("localhost:8080")
        );
    }

    #[test]
    fn truncation_math() {
        // 48000 bytes at 24kHz 16-bit mono = 1000ms
        let audio_duration_ms = calculate_audio_duration_ms(48000);
        assert_eq!(audio_duration_ms, 1000);

        // If elapsed is 500ms and audio is 1000ms, truncate at 500
        let truncate = std::cmp::min(500u64, audio_duration_ms);
        assert_eq!(truncate, 500);

        // If elapsed is 2000ms and audio is 1000ms, truncate at 1000
        let truncate = std::cmp::min(2000u64, audio_duration_ms);
        assert_eq!(truncate, 1000);
    }

    #[test]
    fn turn_detection_disabled_check() {
        let settings = OpenAIRealtimeSettings {
            base: crate::settings::LLMSettings::default(),
            session_properties: SessionProperties {
                audio: Some(super::super::settings::AudioConfiguration {
                    input: Some(super::super::settings::AudioInput {
                        turn_detection: Some(None), // explicitly disabled
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
        };

        let disabled = settings
            .session_properties
            .audio
            .as_ref()
            .and_then(|a| a.input.as_ref())
            .and_then(|i| i.turn_detection.as_ref())
            .is_some_and(|td| td.is_none());

        assert!(disabled);
    }

    #[test]
    fn turn_detection_not_disabled_when_absent() {
        let settings = OpenAIRealtimeSettings::default();

        let disabled = settings
            .session_properties
            .audio
            .as_ref()
            .and_then(|a| a.input.as_ref())
            .and_then(|i| i.turn_detection.as_ref())
            .is_some_and(|td| td.is_none());

        assert!(!disabled);
    }

    #[tokio::test]
    async fn process_frame_forwards_text() {
        let settings = OpenAIRealtimeSettings::default();
        let mut svc = OpenAIRealtimeService::new("test-key", settings);

        let (downstream, _upstream) = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(Frame::Text(TextFrame::new("hello")), Direction::Downstream)],
        )
        .await;

        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::Text(t) if t.text == "hello"));
    }

    #[tokio::test]
    async fn process_frame_consumes_input_audio() {
        let settings = OpenAIRealtimeSettings::default();
        let mut svc = OpenAIRealtimeService::new("test-key", settings);

        let (downstream, _upstream) = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(
                Frame::InputAudioRaw(AudioRawFrame {
                    audio: Bytes::from(vec![0u8; 100]),
                    sample_rate: 24000,
                    num_channels: 1,
                }),
                Direction::Downstream,
            )],
        )
        .await;

        // InputAudioRaw is consumed, not forwarded
        assert!(downstream.is_empty());
    }

    #[tokio::test]
    async fn process_frame_forwards_interruption() {
        let settings = OpenAIRealtimeSettings::default();
        let mut svc = OpenAIRealtimeService::new("test-key", settings);

        let (downstream, _upstream) = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(
                Frame::Interruption(InterruptionFrame),
                Direction::Downstream,
            )],
        )
        .await;

        // Interruption is forwarded (plus LLMFullResponseEnd and TTSStopped)
        let frame_names: Vec<String> = downstream.iter().map(|e| format!("{}", e.frame)).collect();
        assert!(frame_names.contains(&"LLMFullResponseEnd".to_string()));
        assert!(frame_names.contains(&"TTSStopped".to_string()));
        assert!(frame_names.contains(&"Interruption".to_string()));
    }

    #[tokio::test]
    async fn settings_update_applies_model() {
        let settings = OpenAIRealtimeSettings {
            base: crate::settings::LLMSettings {
                model: Some("gpt-4o-realtime-preview".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut svc = OpenAIRealtimeService::new("test-key", settings);

        let mut delta = std::collections::HashMap::new();
        delta.insert("model".into(), json!("gpt-4o-mini-realtime"));

        let _ = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(
                Frame::LLMUpdateSettings(ServiceUpdateSettingsFrame { settings: delta }),
                Direction::Downstream,
            )],
        )
        .await;

        assert_eq!(
            svc.settings.base.model.as_deref(),
            Some("gpt-4o-mini-realtime")
        );
    }

    #[tokio::test]
    async fn process_frame_consumes_input_image() {
        let settings = OpenAIRealtimeSettings::default();
        let mut svc = OpenAIRealtimeService::new("test-key", settings);

        let (downstream, _upstream) = pipecat_core::test_utils::run_processor(
            &mut svc,
            vec![(
                Frame::InputImageRaw(ImageRawFrame {
                    image: Bytes::from(vec![255u8; 3 * 3 * 2]), // 3x2 RGB
                    size: (3, 2),
                    format: Some("RGB".into()),
                }),
                Direction::Downstream,
            )],
        )
        .await;

        // InputImageRaw is consumed, not forwarded
        assert!(downstream.is_empty());
    }

    #[tokio::test]
    async fn send_user_video_rate_limits() {
        let settings = OpenAIRealtimeSettings::default();
        let mut svc = OpenAIRealtimeService::new("test-key", settings);

        let frame = ImageRawFrame {
            image: Bytes::from(vec![128u8; 3 * 3 * 2]), // 3x2 RGB
            size: (3, 2),
            format: Some("RGB".into()),
        };

        // First call should succeed (last_video_sent starts as None)
        assert!(svc.last_video_sent.is_none());
        assert!(svc.send_user_video(&frame).await.is_ok());
        assert!(svc.last_video_sent.is_some());
        let first_sent = svc.last_video_sent.unwrap();

        // Second immediate call should be skipped (rate limited)
        assert!(svc.send_user_video(&frame).await.is_ok());
        // last_video_sent should not have been updated
        assert_eq!(svc.last_video_sent.unwrap(), first_sent);
    }

    #[test]
    fn jpeg_encode_small_rgb_image() {
        // 3x2 RGB image — verify JPEG encoding produces valid output
        let pixels = vec![
            255, 0, 0, 0, 255, 0, 0, 0, 255, // row 1: red, green, blue
            255, 255, 0, 0, 255, 255, 255, 0, 255, // row 2: yellow, cyan, magenta
        ];
        let rgb = image::RgbImage::from_raw(3, 2, pixels).unwrap();
        let mut jpeg_buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut jpeg_buf);
        rgb.write_to(&mut cursor, image::ImageFormat::Jpeg).unwrap();

        // JPEG files start with 0xFF 0xD8
        assert!(jpeg_buf.len() > 2);
        assert_eq!(jpeg_buf[0], 0xFF);
        assert_eq!(jpeg_buf[1], 0xD8);
    }

    #[test]
    fn jpeg_encode_rgba_image() {
        // 3x2 RGBA image — verify RGBA→JPEG conversion works
        let pixels = vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 128, // row 1
            255, 255, 0, 255, 0, 255, 255, 255, 255, 0, 255, 0, // row 2
        ];
        let rgba = image::RgbaImage::from_raw(3, 2, pixels).unwrap();
        let img = image::DynamicImage::ImageRgba8(rgba);
        let mut jpeg_buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut jpeg_buf);
        img.write_to(&mut cursor, image::ImageFormat::Jpeg).unwrap();

        assert!(jpeg_buf.len() > 2);
        assert_eq!(jpeg_buf[0], 0xFF);
        assert_eq!(jpeg_buf[1], 0xD8);
    }
}
