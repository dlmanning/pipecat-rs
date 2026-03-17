mod external;
mod min_words;
mod transcription;
mod vad;

use std::fmt::Debug;

use async_trait::async_trait;
pub use external::ExternalUserTurnStartStrategy;
pub use min_words::MinWordsUserTurnStartStrategy;
use pipecat_core::frame::Frame;
pub use transcription::TranscriptionUserTurnStartStrategy;
pub use vad::VadUserTurnStartStrategy;

use crate::action::TurnAction;

/// Trait for strategies that detect when a user starts a speaking turn.
///
/// Strategies return `Vec<TurnAction>` from `process_frame`. To signal that a
/// turn has started, return `TurnAction::UserTurnStarted(params)`. The
/// controller handles deduplication (double-start prevention).
#[async_trait]
pub trait StartStrategy: Send + Sync + Debug {
    /// Process an incoming frame. Returns actions to execute.
    async fn process_frame(&mut self, frame: &Frame) -> Vec<TurnAction>;

    /// Reset the strategy to its initial state.
    /// Called when a user turn stops (preparing to detect the next start).
    async fn reset(&mut self) {}

    /// One-time initialization.
    async fn setup(&mut self) {}

    /// Shutdown cleanup.
    async fn cleanup(&mut self) {}

    /// Whether this strategy enables interruptions when it triggers.
    fn enable_interruptions(&self) -> bool;

    /// Whether this strategy enables user speaking frames when it triggers.
    fn enable_user_speaking_frames(&self) -> bool;
}
