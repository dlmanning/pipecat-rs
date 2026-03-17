use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Downstream,
    Upstream,
}

// ---------------------------------------------------------------------------
// FrameHeader / FrameEnvelope
// ---------------------------------------------------------------------------

static NEXT_FRAME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub id: u64,
    pub pts: Option<u64>,
    /// Links two envelopes that were created by a single `broadcast()` call.
    /// Each sibling's `broadcast_sibling_id` is set to the other's `id`.
    pub broadcast_sibling_id: Option<u64>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub transport_source: Option<String>,
    pub transport_destination: Option<String>,
}

impl Default for FrameHeader {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameHeader {
    pub fn new() -> Self {
        Self {
            id: NEXT_FRAME_ID.fetch_add(1, Ordering::Relaxed),
            pts: None,
            broadcast_sibling_id: None,
            metadata: HashMap::new(),
            transport_source: None,
            transport_destination: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameEnvelope {
    pub header: FrameHeader,
    pub frame: super::Frame,
}

impl FrameEnvelope {
    pub fn new(frame: super::Frame) -> Self {
        Self {
            header: FrameHeader::new(),
            frame,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared data types reused across frame structs
// ---------------------------------------------------------------------------

/// Raw audio payload. Used by InputAudioRaw, OutputAudioRaw, SpeechOutputAudioRaw.
#[derive(Debug, Clone)]
pub struct AudioRawFrame {
    pub audio: Bytes,
    pub sample_rate: u32,
    pub num_channels: u16,
}

/// Raw image payload.
#[derive(Debug, Clone)]
pub struct ImageRawFrame {
    pub image: Bytes,
    pub size: (u32, u32),
    pub format: Option<String>,
}

/// VAD parameters. Shared between VadParamsUpdateFrame and pipecat-audio's VAD analyzer.
#[derive(Debug, Clone)]
pub struct VadParams {
    pub confidence: f64,
    pub start_secs: f64,
    pub stop_secs: f64,
    pub min_volume: f64,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            confidence: 0.7,
            start_secs: 0.2,
            stop_secs: 0.2,
            min_volume: 0.6,
        }
    }
}

/// Settings map used by all service update frames.
pub type Settings = HashMap<String, serde_json::Value>;

/// Metrics data emitted by processors. Each variant carries the processor name
/// and an optional model name for attribution.
#[derive(Debug, Clone)]
pub enum MetricsData {
    /// Time-to-first-byte: time from request start to first response data.
    Ttfb {
        processor: String,
        model: Option<String>,
        value_secs: f64,
    },
    /// General processing duration.
    Processing {
        processor: String,
        model: Option<String>,
        value_secs: f64,
    },
    /// Text aggregation latency: time from first LLM token to first complete sentence.
    TextAggregation {
        processor: String,
        model: Option<String>,
        value_secs: f64,
    },
    /// LLM token usage counts.
    LlmUsage {
        processor: String,
        model: Option<String>,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        cache_read_input_tokens: Option<u32>,
        cache_creation_input_tokens: Option<u32>,
    },
    /// TTS character count.
    TtsUsage {
        processor: String,
        model: Option<String>,
        characters: u32,
    },
    /// Turn detection prediction result.
    Turn {
        processor: String,
        model: Option<String>,
        is_complete: bool,
        probability: f64,
        /// End-to-end processing time from VAD speech-to-silence to turn completion.
        e2e_processing_time_ms: f64,
    },
}

/// Function call request from an LLM.
#[derive(Debug, Clone)]
pub struct FunctionCallFromLLM {
    pub function_name: String,
    pub tool_call_id: String,
    pub arguments: serde_json::Value,
}
