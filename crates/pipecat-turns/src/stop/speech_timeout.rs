use std::time::{Duration, Instant};

use async_trait::async_trait;
use pipecat_core::frame::Frame;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::trace;

use crate::action::TurnAction;
use crate::params::UserTurnStoppedParams;
use crate::stop::StopStrategy;

/// Internal signal from background timeout tasks.
#[derive(Debug)]
enum TimeoutSignal {
    Elapsed,
}

/// Stops a user turn based on a timeout after VAD detects silence.
///
/// Coordinates three timing signals:
/// 1. **VAD stop** — user paused speaking
/// 2. **STT P99 latency** — expected transcription delay
/// 3. **User speech timeout** — grace period for the user to continue
///
/// The effective wait after VAD stop is:
/// - If transcript is finalized: `user_speech_timeout`
/// - Otherwise: `max(stt_timeout - stop_secs, user_speech_timeout)`
#[derive(Debug)]
pub struct SpeechTimeoutUserTurnStopStrategy {
    enable_user_speaking_frames: bool,
    user_speech_timeout: f64,
    stt_timeout: f64,
    stop_secs: f64,
    text: String,
    vad_user_speaking: bool,
    transcript_finalized: bool,
    /// Wall-clock instant when VAD stop was received. Used for elapsed-time
    /// checks when finalized transcripts arrive.
    vad_stopped_time: Option<Instant>,

    // Background timeout communication
    signal_tx: mpsc::UnboundedSender<TimeoutSignal>,
    signal_rx: mpsc::UnboundedReceiver<TimeoutSignal>,
    timeout_handle: Option<JoinHandle<()>>,
}

impl SpeechTimeoutUserTurnStopStrategy {
    pub fn new(user_speech_timeout: f64) -> Self {
        let (signal_tx, signal_rx) = mpsc::unbounded_channel();
        Self {
            enable_user_speaking_frames: true,
            user_speech_timeout,
            stt_timeout: 0.0,
            stop_secs: 0.0,
            text: String::new(),
            vad_user_speaking: false,
            transcript_finalized: false,
            vad_stopped_time: None,
            signal_tx,
            signal_rx,
            timeout_handle: None,
        }
    }

    pub fn with_options(user_speech_timeout: f64, enable_user_speaking_frames: bool) -> Self {
        let mut s = Self::new(user_speech_timeout);
        s.enable_user_speaking_frames = enable_user_speaking_frames;
        s
    }

    fn calculate_timeout(&self) -> f64 {
        let effective_stt_wait = (self.stt_timeout - self.stop_secs).max(0.0);
        if self.transcript_finalized {
            self.user_speech_timeout
        } else {
            effective_stt_wait.max(self.user_speech_timeout)
        }
    }

    fn start_timeout(&mut self, timeout_secs: f64) {
        self.cancel_timeout();
        let tx = self.signal_tx.clone();
        self.timeout_handle = Some(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs_f64(timeout_secs)).await;
            let _ = tx.send(TimeoutSignal::Elapsed);
        }));
    }

    fn cancel_timeout(&mut self) {
        if let Some(handle) = self.timeout_handle.take() {
            handle.abort();
        }
    }

    fn make_stopped_action(&self) -> TurnAction {
        TurnAction::UserTurnStopped(UserTurnStoppedParams {
            enable_user_speaking_frames: self.enable_user_speaking_frames,
        })
    }

    /// Check trigger conditions matching Python's `_maybe_trigger_user_turn_stopped`.
    ///
    /// For finalized transcripts: trigger immediately if enough wall-clock time
    /// has elapsed since VAD stop. For non-finalized: trigger only if the
    /// timeout task has already completed (signaled via channel).
    fn maybe_trigger(&mut self) -> Vec<TurnAction> {
        if self.vad_user_speaking || self.text.is_empty() {
            return vec![];
        }

        // For finalized transcripts with a known VAD stop time, check elapsed time.
        if self.transcript_finalized
            && let Some(stopped_at) = self.vad_stopped_time
        {
            let elapsed = stopped_at.elapsed().as_secs_f64();
            if elapsed >= self.user_speech_timeout {
                self.cancel_timeout();
                return vec![self.make_stopped_action()];
            }
        }

        // For non-finalized or not-enough-time: only trigger if timeout has completed
        // (which is indicated by being called from drain_pending_actions).
        vec![]
    }

    /// Called from `drain_pending_actions` when a timeout signal fires.
    /// The timeout has elapsed, so we just check basic conditions.
    fn maybe_trigger_from_timeout(&self) -> Vec<TurnAction> {
        if !self.vad_user_speaking && !self.text.is_empty() {
            vec![self.make_stopped_action()]
        } else {
            vec![]
        }
    }
}

#[async_trait]
impl StopStrategy for SpeechTimeoutUserTurnStopStrategy {
    async fn process_frame(&mut self, frame: &Frame) -> Vec<TurnAction> {
        match frame {
            Frame::STTMetadata(m) => {
                self.stt_timeout = m.ttfs_p99_latency;
                trace!(
                    stt_timeout = self.stt_timeout,
                    "SpeechTimeout: updated STT latency"
                );
            }
            Frame::VADUserStartedSpeaking(_) => {
                self.vad_user_speaking = true;
                self.transcript_finalized = false;
                self.vad_stopped_time = None;
                self.cancel_timeout();
            }
            Frame::VADUserStoppedSpeaking(f) => {
                self.vad_user_speaking = false;
                self.stop_secs = f.stop_secs;
                self.vad_stopped_time = Some(Instant::now());
                let timeout = self.calculate_timeout();
                trace!(timeout, "SpeechTimeout: VAD stopped, starting timeout");
                self.start_timeout(timeout);
            }
            Frame::Transcription(t) => {
                self.text.push_str(&t.text);
                if t.finalized {
                    self.transcript_finalized = true;
                    // Check if we can trigger immediately (elapsed time check)
                    let immediate = self.maybe_trigger();
                    if !immediate.is_empty() {
                        return immediate;
                    }
                    // Otherwise, if VAD already stopped, recalculate timeout
                    if self.vad_stopped_time.is_some() {
                        let timeout = self.calculate_timeout();
                        self.start_timeout(timeout);
                    }
                }
                // Fallback: transcription without VAD stop
                if self.vad_stopped_time.is_none() && !self.vad_user_speaking {
                    let timeout = self.stt_timeout.max(self.user_speech_timeout);
                    self.start_timeout(timeout);
                }
            }
            _ => {}
        }
        vec![]
    }

    fn drain_pending_actions(&mut self) -> Vec<TurnAction> {
        let mut actions = Vec::new();
        while let Ok(TimeoutSignal::Elapsed) = self.signal_rx.try_recv() {
            actions.extend(self.maybe_trigger_from_timeout());
        }
        actions
    }

    async fn reset(&mut self) {
        self.cancel_timeout();
        self.text.clear();
        self.vad_user_speaking = false;
        self.transcript_finalized = false;
        self.vad_stopped_time = None;
        self.stop_secs = 0.0;
        // Drain any leftover signals
        while self.signal_rx.try_recv().is_ok() {}
    }

    async fn cleanup(&mut self) {
        self.cancel_timeout();
    }

    fn enable_user_speaking_frames(&self) -> bool {
        self.enable_user_speaking_frames
    }
}

#[cfg(test)]
mod tests {
    use pipecat_core::frame::{
        STTMetadataFrame, TranscriptionFrame, VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    };

    use super::*;

    fn vad_started() -> Frame {
        Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
            start_secs: 0.2,
            timestamp: 1.0,
        })
    }

    fn vad_stopped(stop_secs: f64) -> Frame {
        Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs,
            timestamp: 2.0,
        })
    }

    fn transcription(text: &str, finalized: bool) -> Frame {
        Frame::Transcription(TranscriptionFrame {
            text: text.into(),
            user_id: String::new(),
            timestamp: None,
            language: None,
            finalized,
            result: None,
        })
    }

    fn stt_metadata(latency: f64) -> Frame {
        Frame::STTMetadata(STTMetadataFrame {
            service_name: "test".into(),
            ttfs_p99_latency: latency,
        })
    }

    #[tokio::test]
    async fn basic_timeout_flow() {
        let mut strategy = SpeechTimeoutUserTurnStopStrategy::new(0.1);
        strategy.process_frame(&vad_started()).await;
        strategy.process_frame(&transcription("hello", false)).await;
        strategy.process_frame(&vad_stopped(0.2)).await;

        // No actions yet
        assert!(strategy.drain_pending_actions().is_empty());

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(200)).await;

        let actions = strategy.drain_pending_actions();
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], TurnAction::UserTurnStopped(_)));
    }

    #[tokio::test]
    async fn cancel_on_re_speak() {
        let mut strategy = SpeechTimeoutUserTurnStopStrategy::new(0.2);
        strategy.process_frame(&vad_started()).await;
        strategy.process_frame(&transcription("hello", false)).await;
        strategy.process_frame(&vad_stopped(0.2)).await;

        // Re-start speaking before timeout
        tokio::time::sleep(Duration::from_millis(50)).await;
        strategy.process_frame(&vad_started()).await;

        // Wait past original timeout
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Should not have fired
        assert!(strategy.drain_pending_actions().is_empty());
    }

    #[tokio::test]
    async fn no_text_no_trigger() {
        let mut strategy = SpeechTimeoutUserTurnStopStrategy::new(0.1);
        strategy.process_frame(&vad_started()).await;
        strategy.process_frame(&vad_stopped(0.2)).await;

        tokio::time::sleep(Duration::from_millis(200)).await;

        // No transcription text → no trigger
        assert!(strategy.drain_pending_actions().is_empty());
    }

    #[tokio::test]
    async fn stt_metadata_affects_timeout() {
        let mut strategy = SpeechTimeoutUserTurnStopStrategy::new(0.05);
        strategy.process_frame(&stt_metadata(0.5)).await;
        strategy.process_frame(&vad_started()).await;
        strategy.process_frame(&transcription("hello", false)).await;
        strategy.process_frame(&vad_stopped(0.1)).await;

        // effective_stt_wait = max(0, 0.5 - 0.1) = 0.4
        // timeout = max(0.4, 0.05) = 0.4
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(strategy.drain_pending_actions().is_empty());

        tokio::time::sleep(Duration::from_millis(400)).await;
        let actions = strategy.drain_pending_actions();
        assert_eq!(actions.len(), 1);
    }

    #[tokio::test]
    async fn finalized_transcript_shortens_timeout() {
        let mut strategy = SpeechTimeoutUserTurnStopStrategy::new(0.1);
        strategy.process_frame(&stt_metadata(1.0)).await;
        strategy.process_frame(&vad_started()).await;
        strategy.process_frame(&vad_stopped(0.2)).await;
        // Finalized transcript → timeout = user_speech_timeout (0.1) not stt_timeout
        strategy.process_frame(&transcription("hello", true)).await;

        tokio::time::sleep(Duration::from_millis(200)).await;
        let actions = strategy.drain_pending_actions();
        assert_eq!(actions.len(), 1);
    }

    #[tokio::test]
    async fn reset_clears_state() {
        let mut strategy = SpeechTimeoutUserTurnStopStrategy::new(0.1);
        strategy.process_frame(&vad_started()).await;
        strategy.process_frame(&transcription("hello", false)).await;
        strategy.process_frame(&vad_stopped(0.2)).await;

        strategy.reset().await;

        // Wait past timeout — should not fire because state was reset
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(strategy.drain_pending_actions().is_empty());
    }

    #[tokio::test]
    async fn fallback_transcription_without_vad() {
        let mut strategy = SpeechTimeoutUserTurnStopStrategy::new(0.1);
        // Transcription arrives without any VAD events
        strategy.process_frame(&transcription("hello", true)).await;

        tokio::time::sleep(Duration::from_millis(200)).await;
        let actions = strategy.drain_pending_actions();
        assert_eq!(actions.len(), 1);
    }
}
