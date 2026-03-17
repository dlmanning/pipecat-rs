use async_trait::async_trait;
use pipecat_core::frame::Frame;

use crate::action::TurnAction;
use crate::params::UserTurnStartedParams;
use crate::start::StartStrategy;

/// Triggers a user turn start immediately when VAD detects speech.
#[derive(Debug)]
pub struct VadUserTurnStartStrategy {
    enable_interruptions: bool,
    enable_user_speaking_frames: bool,
}

impl VadUserTurnStartStrategy {
    pub fn new() -> Self {
        Self {
            enable_interruptions: true,
            enable_user_speaking_frames: true,
        }
    }

    pub fn with_options(enable_interruptions: bool, enable_user_speaking_frames: bool) -> Self {
        Self {
            enable_interruptions,
            enable_user_speaking_frames,
        }
    }
}

impl Default for VadUserTurnStartStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StartStrategy for VadUserTurnStartStrategy {
    async fn process_frame(&mut self, frame: &Frame) -> Vec<TurnAction> {
        if matches!(frame, Frame::VADUserStartedSpeaking(_)) {
            vec![TurnAction::UserTurnStarted(UserTurnStartedParams {
                enable_interruptions: self.enable_interruptions,
                enable_user_speaking_frames: self.enable_user_speaking_frames,
            })]
        } else {
            vec![]
        }
    }

    fn enable_interruptions(&self) -> bool {
        self.enable_interruptions
    }

    fn enable_user_speaking_frames(&self) -> bool {
        self.enable_user_speaking_frames
    }
}

#[cfg(test)]
mod tests {
    use pipecat_core::frame::{
        InterimTranscriptionFrame, TranscriptionFrame, VADUserStartedSpeakingFrame,
    };

    use super::*;

    #[tokio::test]
    async fn triggers_on_vad_started() {
        let mut strategy = VadUserTurnStartStrategy::new();
        let frame = Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
            start_secs: 0.2,
            timestamp: 1.0,
        });
        let actions = strategy.process_frame(&frame).await;
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            TurnAction::UserTurnStarted(p) if p.enable_interruptions && p.enable_user_speaking_frames
        ));
    }

    #[tokio::test]
    async fn ignores_other_frames() {
        let mut strategy = VadUserTurnStartStrategy::new();

        let transcription = Frame::Transcription(TranscriptionFrame {
            text: "hello".into(),
            user_id: String::new(),
            timestamp: None,
            language: None,
            finalized: true,
            result: None,
        });
        assert!(strategy.process_frame(&transcription).await.is_empty());

        let interim = Frame::InterimTranscription(InterimTranscriptionFrame {
            text: "hel".into(),
            user_id: String::new(),
            timestamp: None,
            language: None,
            result: None,
        });
        assert!(strategy.process_frame(&interim).await.is_empty());
    }

    #[tokio::test]
    async fn custom_options() {
        let mut strategy = VadUserTurnStartStrategy::with_options(false, false);
        let frame = Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
            start_secs: 0.2,
            timestamp: 1.0,
        });
        let actions = strategy.process_frame(&frame).await;
        assert!(matches!(
            &actions[0],
            TurnAction::UserTurnStarted(p) if !p.enable_interruptions && !p.enable_user_speaking_frames
        ));
    }
}
