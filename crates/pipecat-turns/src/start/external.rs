use async_trait::async_trait;
use pipecat_core::frame::Frame;

use crate::action::TurnAction;
use crate::params::UserTurnStartedParams;
use crate::start::StartStrategy;

/// Triggers a user turn start from an external `UserStartedSpeakingFrame`.
///
/// Used when an external service (e.g. Deepgram Flux) provides speaking
/// detection instead of VAD. Hard-codes `enable_interruptions` and
/// `enable_user_speaking_frames` to false since the external service handles
/// those concerns.
#[derive(Debug)]
pub struct ExternalUserTurnStartStrategy;

impl ExternalUserTurnStartStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ExternalUserTurnStartStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StartStrategy for ExternalUserTurnStartStrategy {
    async fn process_frame(&mut self, frame: &Frame) -> Vec<TurnAction> {
        if matches!(frame, Frame::UserStartedSpeaking(_)) {
            vec![TurnAction::UserTurnStarted(UserTurnStartedParams {
                enable_interruptions: false,
                enable_user_speaking_frames: false,
            })]
        } else {
            vec![]
        }
    }

    fn enable_interruptions(&self) -> bool {
        false
    }

    fn enable_user_speaking_frames(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use pipecat_core::frame::{UserStartedSpeakingFrame, VADUserStartedSpeakingFrame};

    use super::*;

    #[tokio::test]
    async fn triggers_on_user_started_speaking() {
        let mut strategy = ExternalUserTurnStartStrategy::new();
        let frame = Frame::UserStartedSpeaking(UserStartedSpeakingFrame);
        let actions = strategy.process_frame(&frame).await;
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            TurnAction::UserTurnStarted(p) if !p.enable_interruptions && !p.enable_user_speaking_frames
        ));
    }

    #[tokio::test]
    async fn ignores_vad_started() {
        let mut strategy = ExternalUserTurnStartStrategy::new();
        let frame = Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
            start_secs: 0.2,
            timestamp: 1.0,
        });
        assert!(strategy.process_frame(&frame).await.is_empty());
    }
}
