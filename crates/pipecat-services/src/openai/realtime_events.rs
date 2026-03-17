//! WebSocket protocol types for the OpenAI Realtime API.
//!
//! Client events are serialized and sent to the API. Server events are
//! deserialized from incoming WebSocket messages via two-step dispatch:
//! peek the `"type"` field from a `serde_json::Value`, then deserialize
//! into the specific struct.

use serde::{Deserialize, Serialize};

use super::settings::{ResponseProperties, SessionProperties};

// ---------------------------------------------------------------------------
// Client events (serialize and send)
// ---------------------------------------------------------------------------

/// Base fields for all client events.
fn make_event_id() -> String {
    format!("evt_{:032x}", rand_u128())
}

/// Simple pseudo-random u128 using std (no extra crate needed).
fn rand_u128() -> u128 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let s = RandomState::new();
    let mut h = s.build_hasher();
    h.write_u64(0);
    let a = h.finish() as u128;
    let mut h = s.build_hasher();
    h.write_u64(1);
    let b = h.finish() as u128;
    a | (b << 64)
}

#[derive(Debug, Serialize)]
pub struct SessionUpdateEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub event_id: String,
    pub session: SessionProperties,
}

impl SessionUpdateEvent {
    pub fn new(session: SessionProperties) -> Self {
        Self {
            event_type: "session.update",
            event_id: make_event_id(),
            session,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InputAudioBufferAppendEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub event_id: String,
    pub audio: String,
}

impl InputAudioBufferAppendEvent {
    pub fn new(audio: String) -> Self {
        Self {
            event_type: "input_audio_buffer.append",
            event_id: make_event_id(),
            audio,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InputAudioBufferCommitEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub event_id: String,
}

impl Default for InputAudioBufferCommitEvent {
    fn default() -> Self {
        Self::new()
    }
}

impl InputAudioBufferCommitEvent {
    pub fn new() -> Self {
        Self {
            event_type: "input_audio_buffer.commit",
            event_id: make_event_id(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InputAudioBufferClearEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub event_id: String,
}

impl Default for InputAudioBufferClearEvent {
    fn default() -> Self {
        Self::new()
    }
}

impl InputAudioBufferClearEvent {
    pub fn new() -> Self {
        Self {
            event_type: "input_audio_buffer.clear",
            event_id: make_event_id(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ResponseCreateEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub event_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<ResponseProperties>,
}

impl ResponseCreateEvent {
    pub fn new(response: Option<ResponseProperties>) -> Self {
        Self {
            event_type: "response.create",
            event_id: make_event_id(),
            response,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ResponseCancelEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub event_id: String,
}

impl Default for ResponseCancelEvent {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseCancelEvent {
    pub fn new() -> Self {
        Self {
            event_type: "response.cancel",
            event_id: make_event_id(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItem {
    #[serde(default = "default_item_id")]
    pub id: String,
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ItemContent>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
}

fn default_item_id() -> String {
    make_item_id()
}

/// Generate a unique item ID.
pub fn make_item_id() -> String {
    format!("{:032x}", rand_u128())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemContent {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ConversationItemCreateEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub event_id: String,
    pub item: ConversationItem,
}

impl ConversationItemCreateEvent {
    pub fn new(item: ConversationItem) -> Self {
        Self {
            event_type: "conversation.item.create",
            event_id: make_event_id(),
            item,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ConversationItemTruncateEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub event_id: String,
    pub item_id: String,
    pub content_index: u32,
    pub audio_end_ms: u64,
}

impl ConversationItemTruncateEvent {
    pub fn new(item_id: String, content_index: u32, audio_end_ms: u64) -> Self {
        Self {
            event_type: "conversation.item.truncate",
            event_id: make_event_id(),
            item_id,
            content_index,
            audio_end_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Server events (deserialize from incoming messages)
// ---------------------------------------------------------------------------

/// Usage statistics from a completed response.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub input_token_details: Option<TokenDetails>,
    #[serde(default)]
    pub output_token_details: Option<TokenDetails>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TokenDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
    #[serde(default)]
    pub text_tokens: Option<u32>,
    #[serde(default)]
    pub audio_tokens: Option<u32>,
}

/// Response object from `response.done`.
#[derive(Debug, Clone, Deserialize)]
pub struct Response {
    pub id: String,
    pub status: String,
    #[serde(default)]
    pub output: Vec<ConversationItem>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Error from the Realtime API.
#[derive(Debug, Clone, Deserialize)]
pub struct RealtimeError {
    #[serde(rename = "type", default)]
    pub error_type: String,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub message: String,
}

// ---------------------------------------------------------------------------
// Server event dispatch
// ---------------------------------------------------------------------------

/// Parsed server event. Each variant holds only the fields we actually use.
#[derive(Debug)]
pub enum ServerEvent {
    SessionCreated,
    SessionUpdated,

    /// Audio data from the assistant's response.
    ResponseAudioDelta {
        item_id: String,
        content_index: u32,
        delta: String,
    },
    ResponseAudioDone {
        item_id: String,
        content_index: u32,
    },

    /// Text output delta.
    ResponseTextDelta {
        delta: String,
    },

    /// Audio transcript delta (for logging/display).
    ResponseAudioTranscriptDelta {
        delta: String,
    },

    /// A new conversation item was added.
    ConversationItemAdded {
        item: ConversationItem,
    },

    /// Input audio transcription completed.
    TranscriptionCompleted {
        transcript: String,
    },

    /// Input audio transcription delta (interim).
    TranscriptionDelta {
        delta: String,
    },

    /// Function call arguments are complete.
    FunctionCallArgumentsDone {
        call_id: String,
        arguments: String,
    },

    /// Server detected speech start (server VAD).
    SpeechStarted {
        audio_start_ms: u64,
        item_id: String,
    },

    /// Server detected speech stop (server VAD).
    SpeechStopped {
        audio_end_ms: u64,
    },

    /// Response is complete with usage stats.
    ResponseDone {
        response: Response,
    },

    /// Error from the API.
    Error {
        error: RealtimeError,
    },

    /// An event type we don't handle — logged and ignored.
    Unknown {
        event_type: String,
    },
}

/// Parse a raw JSON server event into a typed `ServerEvent`.
pub fn parse_server_event(value: &serde_json::Value) -> ServerEvent {
    let event_type = value
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("unknown");

    match event_type {
        "session.created" => ServerEvent::SessionCreated,

        "session.updated" => ServerEvent::SessionUpdated,

        "response.output_audio.delta" => ServerEvent::ResponseAudioDelta {
            item_id: str_field(value, "item_id"),
            content_index: u32_field(value, "content_index"),
            delta: str_field(value, "delta"),
        },

        "response.output_audio.done" => ServerEvent::ResponseAudioDone {
            item_id: str_field(value, "item_id"),
            content_index: u32_field(value, "content_index"),
        },

        "response.output_text.delta" => ServerEvent::ResponseTextDelta {
            delta: str_field(value, "delta"),
        },

        "response.output_audio_transcript.delta" => ServerEvent::ResponseAudioTranscriptDelta {
            delta: str_field(value, "delta"),
        },

        "conversation.item.added" => {
            let item = value
                .get("item")
                .and_then(|v| serde_json::from_value::<ConversationItem>(v.clone()).ok())
                .unwrap_or_else(|| ConversationItem {
                    id: String::new(),
                    item_type: String::new(),
                    role: None,
                    content: None,
                    call_id: None,
                    name: None,
                    arguments: None,
                    output: None,
                    status: None,
                    object: None,
                });
            ServerEvent::ConversationItemAdded { item }
        }

        "conversation.item.input_audio_transcription.completed" => {
            ServerEvent::TranscriptionCompleted {
                transcript: str_field(value, "transcript"),
            }
        }

        "conversation.item.input_audio_transcription.delta" => ServerEvent::TranscriptionDelta {
            delta: str_field(value, "delta"),
        },

        "response.function_call_arguments.done" => ServerEvent::FunctionCallArgumentsDone {
            call_id: str_field(value, "call_id"),
            arguments: str_field(value, "arguments"),
        },

        "input_audio_buffer.speech_started" => ServerEvent::SpeechStarted {
            audio_start_ms: u64_field(value, "audio_start_ms"),
            item_id: str_field(value, "item_id"),
        },

        "input_audio_buffer.speech_stopped" => ServerEvent::SpeechStopped {
            audio_end_ms: u64_field(value, "audio_end_ms"),
        },

        "response.done" => {
            let response = value
                .get("response")
                .and_then(|v| serde_json::from_value::<Response>(v.clone()).ok())
                .unwrap_or_else(|| Response {
                    id: String::new(),
                    status: "unknown".into(),
                    output: vec![],
                    usage: None,
                });
            ServerEvent::ResponseDone { response }
        }

        "error" => {
            let error = value
                .get("error")
                .and_then(|v| serde_json::from_value::<RealtimeError>(v.clone()).ok())
                .unwrap_or_else(|| RealtimeError {
                    error_type: String::new(),
                    code: None,
                    message: "unknown error".into(),
                });
            ServerEvent::Error { error }
        }

        _ => ServerEvent::Unknown {
            event_type: event_type.to_string(),
        },
    }
}

fn str_field(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

fn u32_field(value: &serde_json::Value, key: &str) -> u32 {
    value.get(key).and_then(|v| v.as_u64()).unwrap_or(0) as u32
}

fn u64_field(value: &serde_json::Value, key: &str) -> u64 {
    value.get(key).and_then(|v| v.as_u64()).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn session_update_event_serialization() {
        let event = SessionUpdateEvent::new(SessionProperties {
            instructions: Some("Be helpful.".into()),
            ..Default::default()
        });
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "session.update");
        assert!(json["event_id"].as_str().unwrap().starts_with("evt_"));
        assert_eq!(json["session"]["instructions"], "Be helpful.");
    }

    #[test]
    fn input_audio_buffer_append_serialization() {
        let event = InputAudioBufferAppendEvent::new("AQID".into());
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "input_audio_buffer.append");
        assert_eq!(json["audio"], "AQID");
    }

    #[test]
    fn response_create_event_serialization() {
        let event = ResponseCreateEvent::new(Some(ResponseProperties {
            output_modalities: Some(vec!["audio".into()]),
            ..Default::default()
        }));
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "response.create");
        assert_eq!(json["response"]["output_modalities"], json!(["audio"]));
    }

    #[test]
    fn response_create_event_no_response() {
        let event = ResponseCreateEvent::new(None);
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "response.create");
        assert!(json.get("response").is_none());
    }

    #[test]
    fn conversation_item_create_event() {
        let item = ConversationItem {
            id: "item_123".into(),
            item_type: "function_call_output".into(),
            role: None,
            content: None,
            call_id: Some("call_abc".into()),
            name: None,
            arguments: None,
            output: Some(r#"{"result":"ok"}"#.into()),
            status: None,
            object: None,
        };
        let event = ConversationItemCreateEvent::new(item);
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "conversation.item.create");
        assert_eq!(json["item"]["type"], "function_call_output");
        assert_eq!(json["item"]["call_id"], "call_abc");
        assert_eq!(json["item"]["output"], r#"{"result":"ok"}"#);
    }

    #[test]
    fn conversation_item_truncate_event() {
        let event = ConversationItemTruncateEvent::new("item_123".into(), 0, 5000);
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "conversation.item.truncate");
        assert_eq!(json["item_id"], "item_123");
        assert_eq!(json["content_index"], 0);
        assert_eq!(json["audio_end_ms"], 5000);
    }

    #[test]
    fn parse_session_created() {
        let raw = json!({ "type": "session.created", "event_id": "e1", "session": {} });
        match parse_server_event(&raw) {
            ServerEvent::SessionCreated => {}
            other => panic!("expected SessionCreated, got {:?}", other),
        }
    }

    #[test]
    fn parse_audio_delta() {
        let raw = json!({
            "type": "response.output_audio.delta",
            "item_id": "item_1",
            "content_index": 0,
            "delta": "AQIDBA=="
        });
        match parse_server_event(&raw) {
            ServerEvent::ResponseAudioDelta {
                item_id,
                content_index,
                delta,
            } => {
                assert_eq!(item_id, "item_1");
                assert_eq!(content_index, 0);
                assert_eq!(delta, "AQIDBA==");
            }
            other => panic!("expected ResponseAudioDelta, got {:?}", other),
        }
    }

    #[test]
    fn parse_speech_started() {
        let raw = json!({
            "type": "input_audio_buffer.speech_started",
            "audio_start_ms": 1000,
            "item_id": "item_2"
        });
        match parse_server_event(&raw) {
            ServerEvent::SpeechStarted {
                audio_start_ms,
                item_id,
            } => {
                assert_eq!(audio_start_ms, 1000);
                assert_eq!(item_id, "item_2");
            }
            other => panic!("expected SpeechStarted, got {:?}", other),
        }
    }

    #[test]
    fn parse_function_call_arguments_done() {
        let raw = json!({
            "type": "response.function_call_arguments.done",
            "call_id": "call_abc",
            "arguments": r#"{"city":"NYC"}"#
        });
        match parse_server_event(&raw) {
            ServerEvent::FunctionCallArgumentsDone { call_id, arguments } => {
                assert_eq!(call_id, "call_abc");
                let args: serde_json::Value = serde_json::from_str(&arguments).unwrap();
                assert_eq!(args["city"], "NYC");
            }
            other => panic!("expected FunctionCallArgumentsDone, got {:?}", other),
        }
    }

    #[test]
    fn parse_response_done_with_usage() {
        let raw = json!({
            "type": "response.done",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [],
                "usage": {
                    "total_tokens": 150,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "input_token_details": { "cached_tokens": 20 }
                }
            }
        });
        match parse_server_event(&raw) {
            ServerEvent::ResponseDone { response } => {
                assert_eq!(response.id, "resp_1");
                assert_eq!(response.status, "completed");
                let usage = response.usage.unwrap();
                assert_eq!(usage.total_tokens, 150);
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.output_tokens, 50);
                assert_eq!(
                    usage
                        .input_token_details
                        .as_ref()
                        .and_then(|d| d.cached_tokens),
                    Some(20)
                );
            }
            other => panic!("expected ResponseDone, got {:?}", other),
        }
    }

    #[test]
    fn parse_error_event() {
        let raw = json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "response_cancel_not_active",
                "message": "No active response to cancel."
            }
        });
        match parse_server_event(&raw) {
            ServerEvent::Error { error } => {
                assert_eq!(error.code.as_deref(), Some("response_cancel_not_active"));
                assert_eq!(error.message, "No active response to cancel.");
            }
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn parse_unknown_event() {
        let raw = json!({ "type": "rate_limits.updated", "rate_limits": [] });
        match parse_server_event(&raw) {
            ServerEvent::Unknown { event_type } => {
                assert_eq!(event_type, "rate_limits.updated");
            }
            other => panic!("expected Unknown, got {:?}", other),
        }
    }

    #[test]
    fn parse_conversation_item_added_function_call() {
        let raw = json!({
            "type": "conversation.item.added",
            "item": {
                "id": "item_func",
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "status": "in_progress"
            }
        });
        match parse_server_event(&raw) {
            ServerEvent::ConversationItemAdded { item } => {
                assert_eq!(item.id, "item_func");
                assert_eq!(item.item_type, "function_call");
                assert_eq!(item.call_id.as_deref(), Some("call_123"));
                assert_eq!(item.name.as_deref(), Some("get_weather"));
            }
            other => panic!("expected ConversationItemAdded, got {:?}", other),
        }
    }

    #[test]
    fn audio_duration_calculation() {
        // 24000 Hz, 16-bit mono: 2 bytes per sample
        // 48000 bytes = 24000 samples = 1000ms
        let total_bytes: usize = 48000;
        let duration_ms = calculate_audio_duration_ms(total_bytes);
        assert_eq!(duration_ms, 1000);

        // 96000 bytes = 48000 samples = 2000ms
        assert_eq!(calculate_audio_duration_ms(96000), 2000);

        // 0 bytes = 0ms
        assert_eq!(calculate_audio_duration_ms(0), 0);
    }
}

/// Calculate audio duration in milliseconds from byte count.
/// Assumes 24kHz, 16-bit (2 bytes/sample), mono.
pub fn calculate_audio_duration_ms(total_bytes: usize) -> u64 {
    if total_bytes == 0 {
        return 0;
    }
    let samples = total_bytes / 2;
    (samples as u64 * 1000) / 24000
}
