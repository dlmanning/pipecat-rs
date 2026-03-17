mod external;
mod speech_timeout;
mod turn_analyzer;

pub use external::ExternalUserTurnStopStrategy;
pub use speech_timeout::SpeechTimeoutUserTurnStopStrategy;
pub use turn_analyzer::TurnAnalyzerUserTurnStopStrategy;

use std::fmt::Debug;

use async_trait::async_trait;
use pipecat_core::frame::Frame;
use pipecat_core::node::ProcessorNodeHandle;

use crate::action::TurnAction;

/// Trait for strategies that detect when a user finishes a speaking turn.
///
/// Stop strategies may spawn background timeout tasks. These produce actions
/// asynchronously via an internal channel. The controller calls
/// `drain_pending_actions()` on each `process_frame` cycle to collect them.
///
/// Strategies that spawn background timeout tasks should implement
/// `set_node_handle` to receive a handle for sending wake-up frames.
#[async_trait]
pub trait StopStrategy: Send + Sync + Debug {
    /// Process an incoming frame. Returns immediate actions.
    async fn process_frame(&mut self, frame: &Frame) -> Vec<TurnAction>;

    /// Drain any pending actions produced by background tasks (e.g. timeouts).
    /// Uses non-blocking `try_recv` internally.
    fn drain_pending_actions(&mut self) -> Vec<TurnAction>;

    /// Reset the strategy to its initial state.
    /// Called when a user turn starts (preparing to detect the stop).
    async fn reset(&mut self) {}

    /// One-time initialization.
    async fn setup(&mut self) {}

    /// Shutdown cleanup.
    async fn cleanup(&mut self) {}

    /// Whether this strategy enables user speaking frames when it triggers.
    fn enable_user_speaking_frames(&self) -> bool;

    /// Provide a node handle for self-notification. Background timeout tasks
    /// use this to send a Wakeup frame that triggers drain_pending_actions.
    fn set_node_handle(&mut self, _handle: ProcessorNodeHandle) {}
}
