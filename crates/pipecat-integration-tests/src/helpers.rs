use bytes::Bytes;
use pipecat_core::frame::*;

/// Create an InputAudioRawFrame from i16 samples.
pub fn make_input_audio_frame(samples: &[i16], sample_rate: u32) -> FrameEnvelope {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    FrameEnvelope::new(Frame::InputAudioRaw(AudioRawFrame {
        audio: Bytes::from(bytes),
        sample_rate,
        num_channels: 1,
    }))
}

/// Create a VADUserStartedSpeaking frame.
pub fn make_vad_started() -> FrameEnvelope {
    FrameEnvelope::new(Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
        start_secs: 0.0,
        timestamp: 0.0,
    }))
}

/// Create a VADUserStoppedSpeaking frame.
pub fn make_vad_stopped() -> FrameEnvelope {
    FrameEnvelope::new(Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
        stop_secs: 0.0,
        timestamp: 0.0,
    }))
}

/// Create an LLMContextFrame with the given messages.
pub fn make_llm_context_frame(messages: Vec<serde_json::Value>) -> FrameEnvelope {
    FrameEnvelope::new(Frame::LLMContext(LLMContextFrame {
        context: serde_json::json!({ "messages": messages }),
    }))
}

/// Create an LLMContextFrame with messages, tools, and tool_choice.
pub fn make_llm_context_frame_with_tools(
    messages: Vec<serde_json::Value>,
    tools: serde_json::Value,
    tool_choice: serde_json::Value,
) -> FrameEnvelope {
    FrameEnvelope::new(Frame::LLMContext(LLMContextFrame {
        context: serde_json::json!({
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
        }),
    }))
}
