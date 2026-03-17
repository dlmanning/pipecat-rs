use super::common::{Settings, VadParams};

// ---------------------------------------------------------------------------
// Lifecycle (uninterruptible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct EndFrame {
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StopFrame;

// ---------------------------------------------------------------------------
// Function call in progress (uninterruptible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FunctionCallInProgressFrame {
    pub function_name: String,
    pub tool_call_id: String,
    pub arguments: serde_json::Value,
    pub cancel_on_interruption: bool,
}

// ---------------------------------------------------------------------------
// Settings updates (uninterruptible)
//
// All settings update variants reuse the same ServiceUpdateSettingsFrame struct.
// The Frame enum variant name provides the type dispatch (STT vs TTS vs LLM).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ServiceUpdateSettingsFrame {
    pub settings: Settings,
}

// ---------------------------------------------------------------------------
// TTS/LLM lifecycle
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct TTSStartedFrame {
    pub context_id: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TTSStoppedFrame {
    pub context_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LLMFullResponseStartFrame {
    /// When set, overrides the default TTS behavior for this response.
    pub skip_tts: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct LLMFullResponseEndFrame {
    /// When set, overrides the default TTS behavior for this response.
    pub skip_tts: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct LLMThoughtStartFrame {
    pub append_to_context: bool,
    pub llm: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LLMThoughtEndFrame {
    pub signature: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct LLMAssistantPushAggregationFrame;

// ---------------------------------------------------------------------------
// LLM context summarization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LLMSummarizeContextFrame {
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct LLMContextSummaryRequestFrame {
    pub request_id: String,
    pub context: serde_json::Value,
    pub min_messages_to_keep: u32,
    pub target_context_tokens: u32,
    pub summarization_prompt: String,
    pub summarization_timeout: Option<f64>,
}

/// Uninterruptible — result of a context summarization.
#[derive(Debug, Clone)]
pub struct LLMContextSummaryResultFrame {
    pub request_id: String,
    pub summary: String,
    pub last_summarized_index: u32,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// VAD / speech params
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VadParamsUpdateFrame {
    pub params: VadParams,
}

// ---------------------------------------------------------------------------
// Transport control
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OutputTransportReadyFrame;

#[derive(Debug, Clone)]
pub struct HeartbeatFrame {
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// Processor pause/resume (normal priority)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ProcessorPauseFrame {
    pub processor_name: String,
}

#[derive(Debug, Clone)]
pub struct ProcessorResumeFrame {
    pub processor_name: String,
}

// ---------------------------------------------------------------------------
// Filter & Mixer control
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FilterUpdateSettingsFrame {
    pub settings: Settings,
}

#[derive(Debug, Clone)]
pub struct FilterEnableFrame {
    pub enable: bool,
}

#[derive(Debug, Clone)]
pub struct MixerUpdateSettingsFrame {
    pub settings: Settings,
}

#[derive(Debug, Clone)]
pub struct MixerEnableFrame {
    pub enable: bool,
}

// ---------------------------------------------------------------------------
// Service switcher
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ServiceSwitcherRequestMetadataFrame {
    pub service_name: String,
}

// ---------------------------------------------------------------------------
// Vision (aliases for LLM response frames)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VisionFullResponseStartFrame {
    /// When set, overrides the default TTS behavior for this response.
    pub skip_tts: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct VisionFullResponseEndFrame {
    /// When set, overrides the default TTS behavior for this response.
    pub skip_tts: Option<bool>,
}
