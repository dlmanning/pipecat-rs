use async_trait::async_trait;
use bytes::Bytes;
use pipecat_core::frame::control::{FilterEnableFrame, FilterUpdateSettingsFrame};

/// Control frame for audio filter operations.
///
/// Wraps the flat `Frame` enum variants into a typed enum for dispatch
/// within the filter implementation.
#[derive(Debug, Clone)]
pub enum FilterControlFrame {
    UpdateSettings(FilterUpdateSettingsFrame),
    Enable(FilterEnableFrame),
}

/// Trait for input transport audio filters.
///
/// If an audio filter is provided to the input transport, it processes audio
/// before VAD and before pushing it downstream. Control frames update settings
/// or enable/disable the filter at runtime.
#[async_trait]
pub trait AudioFilter: Send + Sync {
    /// Initialize the filter when the input transport starts.
    async fn start(&mut self, sample_rate: u32);

    /// Clean up the filter when the input transport stops.
    async fn stop(&mut self);

    /// Process a control frame (settings update or enable/disable).
    async fn process_frame(&mut self, frame: FilterControlFrame);

    /// Apply the filter to raw audio data. Returns filtered audio.
    async fn filter(&mut self, audio: Bytes) -> Bytes;
}
