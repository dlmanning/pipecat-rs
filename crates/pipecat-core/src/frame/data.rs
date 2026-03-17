use std::fmt::Write as _;

use bytes::Bytes;

use super::common::ImageRawFrame;

// ---------------------------------------------------------------------------
// Audio output
// ---------------------------------------------------------------------------

/// TTS audio carries an optional context_id for word-timestamp routing.
#[derive(Debug, Clone)]
pub struct TTSAudioRawFrame {
    pub audio: Bytes,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub context_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Image output
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct URLImageRawFrame {
    pub image: Bytes,
    pub size: (u32, u32),
    pub format: Option<String>,
    pub url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AssistantImageRawFrame {
    pub image: Bytes,
    pub size: (u32, u32),
    pub format: Option<String>,
    pub original_data: Option<Bytes>,
    pub original_mime_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SpriteFrame {
    pub images: Vec<ImageRawFrame>,
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TextFrame {
    pub text: String,
    /// If true, TTS should skip this text (e.g., function call output).
    pub skip_tts: Option<bool>,
    /// Whether spaces between streamed tokens are included in text.
    /// LLM aggregators use this to avoid double-spacing.
    pub includes_inter_frame_spaces: bool,
    /// Whether this text should be appended to LLM context.
    pub append_to_context: bool,
}

impl TextFrame {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            skip_tts: None,
            includes_inter_frame_spaces: false,
            append_to_context: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TranscriptionFrame {
    pub text: String,
    pub user_id: String,
    pub timestamp: Option<String>,
    pub language: Option<String>,
    pub finalized: bool,
    /// Raw STT service output (provider-specific).
    pub result: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct InterimTranscriptionFrame {
    pub text: String,
    pub user_id: String,
    pub timestamp: Option<String>,
    pub language: Option<String>,
    /// Raw STT service output (provider-specific).
    pub result: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct TranslationFrame {
    pub text: String,
    pub user_id: String,
    pub timestamp: Option<String>,
    pub language: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TTSSpeakFrame {
    pub text: String,
    pub append_to_context: Option<bool>,
}

// ---------------------------------------------------------------------------
// LLM data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LLMThoughtTextFrame {
    pub text: String,
    /// Whether spaces between streamed tokens are included in text.
    /// Defaults to true (unlike TextFrame which defaults to false).
    pub includes_inter_frame_spaces: bool,
}

#[derive(Debug, Clone)]
pub struct LLMContextAssistantTimestampFrame {
    pub timestamp: String,
}

#[derive(Debug, Clone)]
pub struct LLMRunFrame;

#[derive(Debug, Clone)]
pub struct LLMMessagesAppendFrame {
    pub messages: Vec<serde_json::Value>,
    pub run_llm: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct LLMMessagesUpdateFrame {
    pub messages: Vec<serde_json::Value>,
    pub run_llm: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct LLMSetToolsFrame {
    pub tools: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct LLMSetToolChoiceFrame {
    pub tool_choice: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct LLMEnablePromptCachingFrame {
    pub enable: bool,
}

#[derive(Debug, Clone)]
pub struct LLMConfigureOutputFrame {
    pub skip_tts: bool,
}

#[derive(Debug, Clone)]
pub struct LLMContextFrame {
    pub context: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Function call result (data + uninterruptible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FunctionCallResultFrame {
    pub function_name: String,
    pub tool_call_id: String,
    pub arguments: serde_json::Value,
    pub result: serde_json::Value,
    pub run_llm: Option<bool>,
}

// ---------------------------------------------------------------------------
// Transport data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OutputTransportMessageFrame {
    pub message: serde_json::Value,
}

// ---------------------------------------------------------------------------
// DTMF
// ---------------------------------------------------------------------------

/// Keypad button for DTMF tone generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeypadEntry {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Pound,
    Star,
}

impl std::fmt::Display for KeypadEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = match self {
            Self::Zero => '0',
            Self::One => '1',
            Self::Two => '2',
            Self::Three => '3',
            Self::Four => '4',
            Self::Five => '5',
            Self::Six => '6',
            Self::Seven => '7',
            Self::Eight => '8',
            Self::Nine => '9',
            Self::Pound => '#',
            Self::Star => '*',
        };
        f.write_char(c)
    }
}

/// DTMF tone output (queued through audio pipeline for ordering).
#[derive(Debug, Clone)]
pub struct OutputDTMFFrame {
    pub button: KeypadEntry,
}
