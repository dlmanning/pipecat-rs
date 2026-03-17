use async_trait::async_trait;
use pipecat_core::MetricsData;

// ---------------------------------------------------------------------------
// EndOfTurnState
// ---------------------------------------------------------------------------

/// State of the end-of-turn analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndOfTurnState {
    /// The user has finished their turn and stopped speaking.
    Complete,
    /// The user is still speaking or may continue speaking.
    Incomplete,
}

// ---------------------------------------------------------------------------
// TurnAnalyzer trait
// ---------------------------------------------------------------------------

/// Trait for end-of-turn detection analyzers.
///
/// Concrete implementations (e.g. smart turn analyzers) analyze accumulated
/// audio and speech state to determine when a user has finished speaking.
/// The base trait provides the interface that the input transport uses.
#[async_trait]
pub trait TurnAnalyzer: Send + Sync + std::fmt::Debug {
    /// The current sample rate.
    fn sample_rate(&self) -> u32;

    /// Set the sample rate for audio processing.
    fn set_sample_rate(&mut self, sample_rate: u32);

    /// Whether speech has been detected by the analyzer.
    fn speech_triggered(&self) -> bool;

    /// Get the current turn analyzer parameters as JSON.
    ///
    /// Returns `serde_json::Value` because concrete parameter types vary
    /// by analyzer implementation (matching Python's extensible `BaseTurnParams`).
    fn params(&self) -> serde_json::Value;

    /// Append audio data for analysis.
    ///
    /// Called on each audio chunk with the VAD's speech detection result.
    fn append_audio(&mut self, buffer: &[u8], is_speech: bool) -> EndOfTurnState;

    /// Analyze whether the end of turn has occurred.
    ///
    /// Returns the turn state and optional metrics data for reporting.
    async fn analyze_end_of_turn(&mut self) -> (EndOfTurnState, Option<MetricsData>);

    /// Update the VAD start trigger time.
    ///
    /// Default implementation does nothing — override if the analyzer needs
    /// to adjust its behavior based on VAD timing.
    fn update_vad_start_secs(&mut self, _vad_start_secs: f64) {}

    /// Reset the turn analyzer to its initial state.
    fn clear(&mut self);

    /// Async cleanup when the analyzer is being shut down.
    ///
    /// Default implementation does nothing.
    async fn cleanup(&mut self) {}
}
