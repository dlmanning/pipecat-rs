/// Parameters emitted when a user turn starts.
#[derive(Debug, Clone, Copy)]
pub struct UserTurnStartedParams {
    /// Whether the user aggregator should emit an interruption frame.
    pub enable_interruptions: bool,
    /// Whether to emit frames indicating user speaking state changes.
    pub enable_user_speaking_frames: bool,
}

/// Parameters emitted when a user turn stops.
#[derive(Debug, Clone, Copy)]
pub struct UserTurnStoppedParams {
    /// Whether to emit frames indicating user speaking state changes.
    pub enable_user_speaking_frames: bool,
}
