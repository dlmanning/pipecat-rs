use pipecat_core::frame::{Direction, Frame, FrameEnvelope};

use crate::params::{UserTurnStartedParams, UserTurnStoppedParams};

/// Actions produced by strategies and the controller.
///
/// The caller (typically the input transport) is responsible for executing
/// these actions against its `ProcessorContext`.
#[derive(Debug)]
pub enum TurnAction {
    /// Push a frame in the given direction through the pipeline.
    PushFrame(FrameEnvelope, Direction),
    /// Broadcast a frame in both directions.
    BroadcastFrame(Frame),
    /// A user turn has started.
    UserTurnStarted(UserTurnStartedParams),
    /// A user turn has stopped.
    UserTurnStopped(UserTurnStoppedParams),
    /// The global user-turn stop timeout fired (informational).
    UserTurnStopTimeout,
}
