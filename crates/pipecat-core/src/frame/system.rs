use bytes::Bytes;

use super::common::{FunctionCallFromLLM, MetricsData, VadParams};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct StartFrame {
    pub audio_in_sample_rate: u32,
    pub audio_out_sample_rate: u32,
    pub allow_interruptions: bool,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
    /// When true, only the first TTFB measurement per processor is reported.
    pub report_only_initial_ttfb: bool,
}

impl Default for StartFrame {
    fn default() -> Self {
        Self {
            audio_in_sample_rate: 16000,
            audio_out_sample_rate: 24000,
            allow_interruptions: false,
            enable_metrics: false,
            enable_usage_metrics: false,
            report_only_initial_ttfb: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CancelFrame {
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ErrorFrame {
    pub error: String,
    pub fatal: bool,
    pub source_processor: String,
}

#[derive(Debug, Clone)]
pub struct InterruptionFrame;

// ---------------------------------------------------------------------------
// Speaking events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct UserStartedSpeakingFrame;

#[derive(Debug, Clone, Default)]
pub struct UserStoppedSpeakingFrame;

#[derive(Debug, Clone)]
pub struct UserMuteStartedFrame;

#[derive(Debug, Clone)]
pub struct UserMuteStoppedFrame;

/// Fires continuously while user is speaking (per-chunk signal).
#[derive(Debug, Clone)]
pub struct UserSpeakingFrame;

#[derive(Debug, Clone)]
pub struct VADUserStartedSpeakingFrame {
    pub start_secs: f64,
    pub timestamp: f64,
}

#[derive(Debug, Clone)]
pub struct VADUserStoppedSpeakingFrame {
    pub stop_secs: f64,
    pub timestamp: f64,
}

#[derive(Debug, Clone)]
pub struct BotStartedSpeakingFrame;

#[derive(Debug, Clone)]
pub struct BotStoppedSpeakingFrame;

/// Fires continuously while bot is speaking (per-chunk signal).
#[derive(Debug, Clone)]
pub struct BotSpeakingFrame;

// ---------------------------------------------------------------------------
// Input (system priority)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UserAudioRawFrame {
    pub audio: Bytes,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub user_id: String,
}

#[derive(Debug, Clone)]
pub struct UserImageRawFrame {
    pub image: Bytes,
    pub size: (u32, u32),
    pub format: Option<String>,
    pub user_id: String,
    pub text: Option<String>,
    pub append_to_context: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct UserImageRequestFrame {
    pub user_id: String,
    pub text: Option<String>,
    pub append_to_context: Option<bool>,
    pub video_source: Option<String>,
    pub function_name: Option<String>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InputTextRawFrame {
    pub text: String,
}

// ---------------------------------------------------------------------------
// System-priority control
// ---------------------------------------------------------------------------

/// Urgent pause (system priority).
#[derive(Debug, Clone)]
pub struct ProcessorPauseUrgentFrame {
    pub processor_name: String,
}

/// Urgent resume (system priority).
#[derive(Debug, Clone)]
pub struct ProcessorResumeUrgentFrame {
    pub processor_name: String,
}

#[derive(Debug, Clone)]
pub struct STTMuteFrame {
    pub mute: bool,
}

#[derive(Debug, Clone)]
pub struct SpeechControlParamsFrame {
    pub vad_params: Option<VadParams>,
    /// Turn analyzer parameters. Stored as JSON since BaseTurnParams is an
    /// extensible base class in Python — concrete types vary by analyzer.
    pub turn_params: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct BotConnectedFrame;

#[derive(Debug, Clone)]
pub struct ClientConnectedFrame;

#[derive(Debug, Clone)]
pub struct UserIdleTimeoutUpdateFrame {
    pub timeout: f64,
}

// ---------------------------------------------------------------------------
// Transport (system priority)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct InputTransportMessageFrame {
    pub message: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct OutputTransportMessageUrgentFrame {
    pub message: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Function call events (system priority)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FunctionCallsStartedFrame {
    pub function_calls: Vec<FunctionCallFromLLM>,
}

#[derive(Debug, Clone)]
pub struct FunctionCallCancelFrame {
    pub function_name: String,
    pub tool_call_id: String,
}

// ---------------------------------------------------------------------------
// Service metadata (system priority)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ServiceMetadataFrame {
    pub service_name: String,
}

#[derive(Debug, Clone)]
pub struct STTMetadataFrame {
    pub service_name: String,
    pub ttfs_p99_latency: f64,
}

// ---------------------------------------------------------------------------
// Metrics (system priority)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MetricsFrame {
    pub data: Vec<MetricsData>,
}

// ---------------------------------------------------------------------------
// Task control (system priority)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EndTaskFrame {
    pub task_id: String,
    pub handler_id: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CancelTaskFrame {
    pub task_id: String,
    pub handler_id: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StopTaskFrame {
    pub task_id: String,
    pub handler_id: String,
}

#[derive(Debug, Clone)]
pub struct InterruptionTaskFrame {
    pub task_id: String,
    pub handler_id: String,
}
