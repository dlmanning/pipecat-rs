use async_trait::async_trait;
use pipecat_core::frame::Frame;

use crate::action::TurnAction;
use crate::params::UserTurnStartedParams;
use crate::start::StartStrategy;

/// Requires a minimum word count in transcription before triggering a turn start.
///
/// When the bot is speaking, requires `min_words` to interrupt. When the bot
/// is not speaking, requires only 1 word. This allows tuning the interruption
/// threshold separately from the normal turn-start threshold.
#[derive(Debug)]
pub struct MinWordsUserTurnStartStrategy {
    enable_interruptions: bool,
    enable_user_speaking_frames: bool,
    min_words: usize,
    use_interim: bool,
    bot_speaking: bool,
}

impl MinWordsUserTurnStartStrategy {
    pub fn new(min_words: usize) -> Self {
        Self {
            enable_interruptions: true,
            enable_user_speaking_frames: true,
            min_words,
            use_interim: true,
            bot_speaking: false,
        }
    }

    pub fn with_options(
        min_words: usize,
        use_interim: bool,
        enable_interruptions: bool,
        enable_user_speaking_frames: bool,
    ) -> Self {
        Self {
            enable_interruptions,
            enable_user_speaking_frames,
            min_words,
            use_interim,
            bot_speaking: false,
        }
    }

    fn check_words(&self, text: &str) -> Vec<TurnAction> {
        let threshold = if self.bot_speaking { self.min_words } else { 1 };

        if text.split_whitespace().count() >= threshold {
            vec![TurnAction::UserTurnStarted(UserTurnStartedParams {
                enable_interruptions: self.enable_interruptions,
                enable_user_speaking_frames: self.enable_user_speaking_frames,
            })]
        } else {
            vec![]
        }
    }
}

#[async_trait]
impl StartStrategy for MinWordsUserTurnStartStrategy {
    async fn process_frame(&mut self, frame: &Frame) -> Vec<TurnAction> {
        match frame {
            Frame::BotStartedSpeaking(_) => {
                self.bot_speaking = true;
                vec![]
            }
            Frame::BotStoppedSpeaking(_) => {
                self.bot_speaking = false;
                vec![]
            }
            Frame::Transcription(t) => self.check_words(&t.text),
            Frame::InterimTranscription(t) if self.use_interim => self.check_words(&t.text),
            _ => vec![],
        }
    }

    async fn reset(&mut self) {
        self.bot_speaking = false;
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
        BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterimTranscriptionFrame,
        TranscriptionFrame,
    };

    use super::*;

    fn transcription(text: &str) -> Frame {
        Frame::Transcription(TranscriptionFrame {
            text: text.into(),
            user_id: String::new(),
            timestamp: None,
            language: None,
            finalized: true,
            result: None,
        })
    }

    fn interim(text: &str) -> Frame {
        Frame::InterimTranscription(InterimTranscriptionFrame {
            text: text.into(),
            user_id: String::new(),
            timestamp: None,
            language: None,
            result: None,
        })
    }

    #[tokio::test]
    async fn triggers_on_one_word_when_bot_not_speaking() {
        let mut strategy = MinWordsUserTurnStartStrategy::new(3);
        let actions = strategy.process_frame(&transcription("hello")).await;
        assert_eq!(actions.len(), 1);
    }

    #[tokio::test]
    async fn requires_min_words_when_bot_speaking() {
        let mut strategy = MinWordsUserTurnStartStrategy::new(3);
        strategy
            .process_frame(&Frame::BotStartedSpeaking(BotStartedSpeakingFrame))
            .await;

        // 2 words — not enough
        let actions = strategy.process_frame(&transcription("hello world")).await;
        assert!(actions.is_empty());

        // 3 words — triggers
        let actions = strategy
            .process_frame(&transcription("hello world goodbye"))
            .await;
        assert_eq!(actions.len(), 1);
    }

    #[tokio::test]
    async fn bot_stopped_resets_threshold() {
        let mut strategy = MinWordsUserTurnStartStrategy::new(3);
        strategy
            .process_frame(&Frame::BotStartedSpeaking(BotStartedSpeakingFrame))
            .await;
        strategy
            .process_frame(&Frame::BotStoppedSpeaking(BotStoppedSpeakingFrame))
            .await;

        // Only 1 word needed now
        let actions = strategy.process_frame(&transcription("hi")).await;
        assert_eq!(actions.len(), 1);
    }

    #[tokio::test]
    async fn interim_triggers_when_enabled() {
        let mut strategy = MinWordsUserTurnStartStrategy::new(3);
        let actions = strategy.process_frame(&interim("hello")).await;
        assert_eq!(actions.len(), 1);
    }

    #[tokio::test]
    async fn interim_ignored_when_disabled() {
        let mut strategy = MinWordsUserTurnStartStrategy::with_options(3, false, true, true);
        let actions = strategy.process_frame(&interim("hello")).await;
        assert!(actions.is_empty());
    }

    #[tokio::test]
    async fn reset_clears_bot_speaking() {
        let mut strategy = MinWordsUserTurnStartStrategy::new(3);
        strategy
            .process_frame(&Frame::BotStartedSpeaking(BotStartedSpeakingFrame))
            .await;
        assert!(strategy.bot_speaking);

        strategy.reset().await;
        assert!(!strategy.bot_speaking);
    }
}
