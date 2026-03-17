use async_trait::async_trait;
use pipecat_core::frame::Frame;

use crate::action::TurnAction;
use crate::params::UserTurnStartedParams;
use crate::start::StartStrategy;

/// Triggers a user turn start when transcription (final or interim) is received.
///
/// Acts as a fallback when VAD doesn't detect speech but STT produces results.
#[derive(Debug)]
pub struct TranscriptionUserTurnStartStrategy {
    enable_interruptions: bool,
    enable_user_speaking_frames: bool,
    use_interim: bool,
}

impl TranscriptionUserTurnStartStrategy {
    pub fn new() -> Self {
        Self {
            enable_interruptions: true,
            enable_user_speaking_frames: true,
            use_interim: true,
        }
    }

    pub fn with_options(
        enable_interruptions: bool,
        enable_user_speaking_frames: bool,
        use_interim: bool,
    ) -> Self {
        Self {
            enable_interruptions,
            enable_user_speaking_frames,
            use_interim,
        }
    }
}

impl Default for TranscriptionUserTurnStartStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StartStrategy for TranscriptionUserTurnStartStrategy {
    async fn process_frame(&mut self, frame: &Frame) -> Vec<TurnAction> {
        let should_trigger = match frame {
            Frame::InterimTranscription(_) if self.use_interim => true,
            Frame::Transcription(_) => true,
            _ => false,
        };

        if should_trigger {
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
    use pipecat_core::frame::{InterimTranscriptionFrame, TranscriptionFrame};

    use super::*;

    fn transcription_frame(text: &str) -> Frame {
        Frame::Transcription(TranscriptionFrame {
            text: text.into(),
            user_id: String::new(),
            timestamp: None,
            language: None,
            finalized: true,
            result: None,
        })
    }

    fn interim_frame(text: &str) -> Frame {
        Frame::InterimTranscription(InterimTranscriptionFrame {
            text: text.into(),
            user_id: String::new(),
            timestamp: None,
            language: None,
            result: None,
        })
    }

    #[tokio::test]
    async fn triggers_on_final_transcription() {
        let mut strategy = TranscriptionUserTurnStartStrategy::new();
        let actions = strategy.process_frame(&transcription_frame("hello")).await;
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], TurnAction::UserTurnStarted(_)));
    }

    #[tokio::test]
    async fn triggers_on_interim_when_enabled() {
        let mut strategy = TranscriptionUserTurnStartStrategy::new();
        let actions = strategy.process_frame(&interim_frame("hel")).await;
        assert_eq!(actions.len(), 1);
    }

    #[tokio::test]
    async fn ignores_interim_when_disabled() {
        let mut strategy = TranscriptionUserTurnStartStrategy::with_options(true, true, false);
        let actions = strategy.process_frame(&interim_frame("hel")).await;
        assert!(actions.is_empty());
    }

    #[tokio::test]
    async fn still_triggers_on_final_when_interim_disabled() {
        let mut strategy = TranscriptionUserTurnStartStrategy::with_options(true, true, false);
        let actions = strategy.process_frame(&transcription_frame("hello")).await;
        assert_eq!(actions.len(), 1);
    }
}
