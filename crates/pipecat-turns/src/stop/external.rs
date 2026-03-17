use std::time::Duration;

use async_trait::async_trait;
use pipecat_core::frame::Frame;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::trace;

use crate::action::TurnAction;
use crate::params::UserTurnStoppedParams;
use crate::stop::StopStrategy;

/// Internal signal from the aggregation timer.
#[derive(Debug)]
enum TimeoutSignal {
    Elapsed,
}

/// Stops a user turn when an external service emits `UserStoppedSpeakingFrame`.
///
/// Uses an aggregation timer to handle delayed or consecutive transcriptions:
/// each transcription restarts the timer. The turn stops when the user is not
/// speaking, no interim results are pending, and transcription text exists.
#[derive(Debug)]
pub struct ExternalUserTurnStopStrategy {
    enable_user_speaking_frames: bool,
    timeout: f64,
    text: String,
    user_speaking: bool,
    seen_interim_results: bool,

    // Aggregation timer communication
    signal_tx: mpsc::UnboundedSender<TimeoutSignal>,
    signal_rx: mpsc::UnboundedReceiver<TimeoutSignal>,
    timer_handle: Option<JoinHandle<()>>,
}

impl ExternalUserTurnStopStrategy {
    pub fn new(timeout: f64) -> Self {
        let (signal_tx, signal_rx) = mpsc::unbounded_channel();
        Self {
            enable_user_speaking_frames: false,
            timeout,
            text: String::new(),
            user_speaking: false,
            seen_interim_results: false,
            signal_tx,
            signal_rx,
            timer_handle: None,
        }
    }

    fn start_timer(&mut self) {
        self.cancel_timer();
        let tx = self.signal_tx.clone();
        let timeout = self.timeout;
        self.timer_handle = Some(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs_f64(timeout)).await;
            let _ = tx.send(TimeoutSignal::Elapsed);
        }));
    }

    fn cancel_timer(&mut self) {
        if let Some(handle) = self.timer_handle.take() {
            handle.abort();
        }
    }

    fn maybe_trigger(&self) -> Vec<TurnAction> {
        if !self.user_speaking && !self.seen_interim_results && !self.text.is_empty() {
            vec![TurnAction::UserTurnStopped(UserTurnStoppedParams {
                enable_user_speaking_frames: self.enable_user_speaking_frames,
            })]
        } else {
            vec![]
        }
    }
}

impl Default for ExternalUserTurnStopStrategy {
    fn default() -> Self {
        Self::new(0.5)
    }
}

#[async_trait]
impl StopStrategy for ExternalUserTurnStopStrategy {
    async fn process_frame(&mut self, frame: &Frame) -> Vec<TurnAction> {
        match frame {
            Frame::UserStartedSpeaking(_) => {
                self.user_speaking = true;
            }
            Frame::UserStoppedSpeaking(_) => {
                self.user_speaking = false;
                trace!("ExternalStop: user stopped speaking, checking trigger");
                return self.maybe_trigger();
            }
            Frame::InterimTranscription(_) => {
                self.seen_interim_results = true;
            }
            Frame::Transcription(t) => {
                self.text.push_str(&t.text);
                self.seen_interim_results = false;
                // Restart aggregation timer on each final transcription
                self.start_timer();
            }
            _ => {}
        }
        vec![]
    }

    fn drain_pending_actions(&mut self) -> Vec<TurnAction> {
        let mut actions = Vec::new();
        while let Ok(TimeoutSignal::Elapsed) = self.signal_rx.try_recv() {
            actions.extend(self.maybe_trigger());
        }
        actions
    }

    async fn reset(&mut self) {
        self.cancel_timer();
        self.text.clear();
        self.user_speaking = false;
        self.seen_interim_results = false;
        while self.signal_rx.try_recv().is_ok() {}
    }

    async fn cleanup(&mut self) {
        self.cancel_timer();
    }

    fn enable_user_speaking_frames(&self) -> bool {
        self.enable_user_speaking_frames
    }
}

#[cfg(test)]
mod tests {
    use pipecat_core::frame::{
        InterimTranscriptionFrame, TranscriptionFrame, UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    };

    use super::*;

    fn user_started() -> Frame {
        Frame::UserStartedSpeaking(UserStartedSpeakingFrame)
    }

    fn user_stopped() -> Frame {
        Frame::UserStoppedSpeaking(UserStoppedSpeakingFrame)
    }

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
    async fn triggers_after_stop_speaking_with_text() {
        let mut strategy = ExternalUserTurnStopStrategy::new(0.1);
        strategy.process_frame(&user_started()).await;
        strategy.process_frame(&transcription("hello")).await;
        let actions = strategy.process_frame(&user_stopped()).await;
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], TurnAction::UserTurnStopped(_)));
    }

    #[tokio::test]
    async fn no_trigger_without_text() {
        let mut strategy = ExternalUserTurnStopStrategy::new(0.1);
        strategy.process_frame(&user_started()).await;
        let actions = strategy.process_frame(&user_stopped()).await;
        assert!(actions.is_empty());
    }

    #[tokio::test]
    async fn no_trigger_with_pending_interim() {
        let mut strategy = ExternalUserTurnStopStrategy::new(0.1);
        strategy.process_frame(&user_started()).await;
        strategy.process_frame(&transcription("hello")).await;
        strategy.process_frame(&interim("world")).await;
        let actions = strategy.process_frame(&user_stopped()).await;
        // Interim results pending, so no trigger
        assert!(actions.is_empty());
    }

    #[tokio::test]
    async fn aggregation_timer_triggers() {
        let mut strategy = ExternalUserTurnStopStrategy::new(0.1);
        // Not speaking, has text from transcription
        strategy.process_frame(&transcription("hello")).await;

        tokio::time::sleep(Duration::from_millis(200)).await;
        let actions = strategy.drain_pending_actions();
        assert_eq!(actions.len(), 1);
    }

    #[tokio::test]
    async fn timer_restarted_on_new_transcription() {
        let mut strategy = ExternalUserTurnStopStrategy::new(0.2);
        strategy.process_frame(&transcription("hello")).await;

        // New transcription before timeout restarts the timer
        tokio::time::sleep(Duration::from_millis(100)).await;
        strategy.process_frame(&transcription(" world")).await;

        // Original timer would have fired, but was restarted
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(strategy.drain_pending_actions().is_empty());

        // New timer fires
        tokio::time::sleep(Duration::from_millis(100)).await;
        let actions = strategy.drain_pending_actions();
        assert_eq!(actions.len(), 1);
    }

    #[tokio::test]
    async fn reset_clears_state() {
        let mut strategy = ExternalUserTurnStopStrategy::new(0.1);
        strategy.process_frame(&transcription("hello")).await;
        strategy.reset().await;

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(strategy.drain_pending_actions().is_empty());
    }
}
