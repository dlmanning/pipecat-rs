use crate::start::{
    ExternalUserTurnStartStrategy, StartStrategy, TranscriptionUserTurnStartStrategy,
    VadUserTurnStartStrategy,
};
use crate::stop::{ExternalUserTurnStopStrategy, SpeechTimeoutUserTurnStopStrategy, StopStrategy};

/// Configuration holding the start and stop strategies for turn detection.
#[derive(Debug)]
pub struct UserTurnStrategies {
    pub start: Vec<Box<dyn StartStrategy>>,
    pub stop: Vec<Box<dyn StopStrategy>>,
}

impl UserTurnStrategies {
    /// Creates strategies for VAD + transcription start detection with a
    /// speech-timeout stop strategy. This is a common default when no ML
    /// turn analyzer is available.
    pub fn vad_with_speech_timeout(user_speech_timeout: f64) -> Self {
        Self {
            start: vec![
                Box::new(VadUserTurnStartStrategy::new()),
                Box::new(TranscriptionUserTurnStartStrategy::new()),
            ],
            stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(
                user_speech_timeout,
            ))],
        }
    }

    /// Creates strategies for external turn detection (e.g. Deepgram Flux).
    pub fn external() -> Self {
        Self {
            start: vec![Box::new(ExternalUserTurnStartStrategy::new())],
            stop: vec![Box::new(ExternalUserTurnStopStrategy::default())],
        }
    }
}
