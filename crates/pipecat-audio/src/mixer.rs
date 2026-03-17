use async_trait::async_trait;
use bytes::Bytes;
use pipecat_core::frame::control::{MixerEnableFrame, MixerUpdateSettingsFrame};

/// Control frame for audio mixer operations.
///
/// Wraps the flat `Frame` enum variants into a typed enum for dispatch
/// within the mixer implementation.
#[derive(Debug, Clone)]
pub enum MixerControlFrame {
    UpdateSettings(MixerUpdateSettingsFrame),
    Enable(MixerEnableFrame),
}

/// Trait for output transport audio mixers.
///
/// If an audio mixer is provided to the output transport, it mixes incoming
/// audio frames with mixer-generated audio (e.g. background sounds). Control
/// frames update settings or enable/disable the mixer at runtime.
#[async_trait]
pub trait AudioMixer: Send + Sync {
    /// Initialize the mixer when the output transport starts.
    async fn start(&mut self, sample_rate: u32);

    /// Clean up the mixer when the output transport stops.
    async fn stop(&mut self);

    /// Process a control frame (settings update or enable/disable).
    async fn process_frame(&mut self, frame: MixerControlFrame);

    /// Mix transport audio with mixer-generated audio.
    async fn mix(&mut self, audio: Bytes) -> Bytes;
}
