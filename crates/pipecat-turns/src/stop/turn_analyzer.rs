use std::time::Duration;

use async_trait::async_trait;
use pipecat_audio::turn::{EndOfTurnState, TurnAnalyzer};
use pipecat_core::frame::{
    self, Direction, Frame, FrameEnvelope, MetricsFrame, SpeechControlParamsFrame,
};
use pipecat_core::node::ProcessorNodeHandle;
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

/// Stops a user turn using an ML-based turn analyzer.
///
/// Feeds audio to a [`TurnAnalyzer`] which predicts whether the user has
/// finished their turn. Coordinates the prediction with VAD state and STT
/// transcription timing, similar to [`SpeechTimeoutUserTurnStopStrategy`] but
/// with the additional ML signal.
#[derive(Debug)]
pub struct TurnAnalyzerUserTurnStopStrategy {
    enable_user_speaking_frames: bool,
    turn_analyzer: Box<dyn TurnAnalyzer>,
    stt_timeout: f64,
    stop_secs: f64,
    text: String,
    turn_complete: bool,
    vad_user_speaking: bool,
    vad_stopped: bool,
    transcript_finalized: bool,

    // Background timeout communication
    signal_tx: mpsc::UnboundedSender<TimeoutSignal>,
    signal_rx: mpsc::UnboundedReceiver<TimeoutSignal>,
    timeout_handle: Option<JoinHandle<()>>,
    node_handle: Option<ProcessorNodeHandle>,
}

impl TurnAnalyzerUserTurnStopStrategy {
    pub fn new(turn_analyzer: Box<dyn TurnAnalyzer>) -> Self {
        let (signal_tx, signal_rx) = mpsc::unbounded_channel();
        Self {
            enable_user_speaking_frames: true,
            turn_analyzer,
            stt_timeout: 0.0,
            stop_secs: 0.0,
            text: String::new(),
            turn_complete: false,
            vad_user_speaking: false,
            vad_stopped: false,
            transcript_finalized: false,
            signal_tx,
            signal_rx,
            timeout_handle: None,
            node_handle: None,
        }
    }

    pub fn with_options(
        turn_analyzer: Box<dyn TurnAnalyzer>,
        enable_user_speaking_frames: bool,
    ) -> Self {
        let mut s = Self::new(turn_analyzer);
        s.enable_user_speaking_frames = enable_user_speaking_frames;
        s
    }

    fn start_timeout(&mut self, timeout_secs: f64) {
        self.cancel_timeout();
        let tx = self.signal_tx.clone();
        let node_handle = self.node_handle.clone();
        self.timeout_handle = Some(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs_f64(timeout_secs)).await;
            let _ = tx.send(TimeoutSignal::Elapsed);
            if let Some(handle) = node_handle {
                handle.try_send_normal(
                    frame::FrameEnvelope::new(frame::Frame::Wakeup(frame::WakeupFrame)),
                    frame::Direction::Downstream,
                );
            }
        }));
    }

    fn cancel_timeout(&mut self) {
        if let Some(handle) = self.timeout_handle.take() {
            handle.abort();
        }
    }

    fn maybe_trigger(&self) -> Vec<TurnAction> {
        // Need text, turn_complete from analyzer, and not currently speaking
        if !self.text.is_empty() && self.turn_complete && !self.vad_user_speaking {
            vec![TurnAction::UserTurnStopped(UserTurnStoppedParams {
                enable_user_speaking_frames: self.enable_user_speaking_frames,
            })]
        } else {
            vec![]
        }
    }
}

#[async_trait]
impl StopStrategy for TurnAnalyzerUserTurnStopStrategy {
    async fn process_frame(&mut self, frame: &Frame) -> Vec<TurnAction> {
        let mut actions = Vec::new();

        match frame {
            Frame::Start(start) => {
                self.turn_analyzer
                    .set_sample_rate(start.audio_in_sample_rate);
                // Broadcast speech control params so the transport knows
                // about the turn analyzer's configuration.
                actions.push(TurnAction::BroadcastFrame(Frame::SpeechControlParams(
                    SpeechControlParamsFrame {
                        vad_params: None,
                        turn_params: Some(self.turn_analyzer.params()),
                    },
                )));
            }
            Frame::STTMetadata(m) => {
                self.stt_timeout = m.ttfs_p99_latency;
            }
            Frame::InputAudioRaw(audio) => {
                let state = self
                    .turn_analyzer
                    .append_audio(&audio.audio, self.vad_user_speaking);
                if state == EndOfTurnState::Complete {
                    let (_, metrics) = self.turn_analyzer.analyze_end_of_turn().await;
                    self.turn_complete = true;
                    if let Some(metrics_data) = metrics {
                        actions.push(TurnAction::PushFrame(
                            FrameEnvelope::new(Frame::Metrics(MetricsFrame {
                                data: vec![metrics_data],
                            })),
                            Direction::Downstream,
                        ));
                    }
                    actions.extend(self.maybe_trigger());
                }
            }
            Frame::VADUserStartedSpeaking(f) => {
                self.turn_analyzer.update_vad_start_secs(f.start_secs);
                self.turn_complete = false;
                self.vad_user_speaking = true;
                self.vad_stopped = false;
                self.transcript_finalized = false;
                self.cancel_timeout();
            }
            Frame::VADUserStoppedSpeaking(f) => {
                self.vad_user_speaking = false;
                self.stop_secs = f.stop_secs;
                self.vad_stopped = true;

                // Analyze end-of-turn on VAD stop
                let (state, metrics) = self.turn_analyzer.analyze_end_of_turn().await;
                self.turn_complete = state == EndOfTurnState::Complete;
                if let Some(metrics_data) = metrics {
                    actions.push(TurnAction::PushFrame(
                        FrameEnvelope::new(Frame::Metrics(MetricsFrame {
                            data: vec![metrics_data],
                        })),
                        Direction::Downstream,
                    ));
                }

                let timeout = (self.stt_timeout - self.stop_secs).max(0.0);
                trace!(
                    timeout,
                    turn_complete = self.turn_complete,
                    "TurnAnalyzer: VAD stopped, starting timeout"
                );
                self.start_timeout(timeout);
            }
            Frame::Transcription(t) => {
                // Replace (not accumulate) — track only the latest transcription.
                self.text.clone_from(&t.text);
                if t.finalized {
                    self.transcript_finalized = true;
                    actions.extend(self.maybe_trigger());
                }
                // Fallback: transcription without VAD stop
                if !self.vad_stopped {
                    let timeout = self.stt_timeout.max(0.1);
                    self.start_timeout(timeout);
                }
            }
            _ => {}
        }

        actions
    }

    fn drain_pending_actions(&mut self) -> Vec<TurnAction> {
        let mut actions = Vec::new();
        while let Ok(TimeoutSignal::Elapsed) = self.signal_rx.try_recv() {
            actions.extend(self.maybe_trigger());
        }
        actions
    }

    async fn reset(&mut self) {
        self.cancel_timeout();
        self.turn_analyzer.clear();
        self.text.clear();
        self.turn_complete = false;
        self.vad_user_speaking = false;
        self.vad_stopped = false;
        self.transcript_finalized = false;
        self.stop_secs = 0.0;
        while self.signal_rx.try_recv().is_ok() {}
    }

    async fn cleanup(&mut self) {
        self.cancel_timeout();
        self.turn_analyzer.cleanup().await;
    }

    fn enable_user_speaking_frames(&self) -> bool {
        self.enable_user_speaking_frames
    }

    fn set_node_handle(&mut self, handle: ProcessorNodeHandle) {
        self.node_handle = Some(handle);
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use pipecat_core::MetricsData;
    use pipecat_core::frame::{
        AudioRawFrame, StartFrame, TranscriptionFrame, VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    };

    use super::*;

    /// Mock turn analyzer for testing.
    #[derive(Debug)]
    struct MockTurnAnalyzer {
        sample_rate: u32,
        should_complete: bool,
        append_count: usize,
    }

    impl MockTurnAnalyzer {
        fn new(should_complete: bool) -> Self {
            Self {
                sample_rate: 16000,
                should_complete,
                append_count: 0,
            }
        }
    }

    #[async_trait]
    impl TurnAnalyzer for MockTurnAnalyzer {
        fn sample_rate(&self) -> u32 {
            self.sample_rate
        }

        fn set_sample_rate(&mut self, sample_rate: u32) {
            self.sample_rate = sample_rate;
        }

        fn speech_triggered(&self) -> bool {
            self.append_count > 0
        }

        fn params(&self) -> serde_json::Value {
            serde_json::json!({"mock": true})
        }

        fn append_audio(&mut self, _buffer: &[u8], _is_speech: bool) -> EndOfTurnState {
            self.append_count += 1;
            if self.should_complete {
                EndOfTurnState::Complete
            } else {
                EndOfTurnState::Incomplete
            }
        }

        async fn analyze_end_of_turn(&mut self) -> (EndOfTurnState, Option<MetricsData>) {
            let state = if self.should_complete {
                EndOfTurnState::Complete
            } else {
                EndOfTurnState::Incomplete
            };
            (state, None)
        }

        fn clear(&mut self) {
            self.append_count = 0;
        }
    }

    fn vad_started() -> Frame {
        Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
            start_secs: 0.2,
            timestamp: 1.0,
        })
    }

    fn vad_stopped() -> Frame {
        Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.2,
            timestamp: 2.0,
        })
    }

    fn audio_frame() -> Frame {
        Frame::InputAudioRaw(AudioRawFrame {
            audio: Bytes::from(vec![0u8; 320]),
            sample_rate: 16000,
            num_channels: 1,
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

    #[tokio::test]
    async fn broadcasts_speech_control_params_on_start() {
        let analyzer = MockTurnAnalyzer::new(false);
        let mut strategy = TurnAnalyzerUserTurnStopStrategy::new(Box::new(analyzer));
        let frame = Frame::Start(StartFrame::default());
        let actions = strategy.process_frame(&frame).await;
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            TurnAction::BroadcastFrame(Frame::SpeechControlParams(_))
        ));
    }

    #[tokio::test]
    async fn triggers_on_complete_with_text() {
        let analyzer = MockTurnAnalyzer::new(true);
        let mut strategy = TurnAnalyzerUserTurnStopStrategy::new(Box::new(analyzer));

        strategy.process_frame(&vad_started()).await;
        strategy.process_frame(&transcription("hello", false)).await;
        // VAD stop triggers analyze_end_of_turn → turn_complete = true
        strategy.process_frame(&vad_stopped()).await;
        // Finalized transcription with turn_complete + not speaking → triggers
        let actions = strategy.process_frame(&transcription(" world", true)).await;
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, TurnAction::UserTurnStopped(_)))
        );
    }

    #[tokio::test]
    async fn no_trigger_without_text() {
        let analyzer = MockTurnAnalyzer::new(true);
        let mut strategy = TurnAnalyzerUserTurnStopStrategy::new(Box::new(analyzer));

        strategy.process_frame(&vad_started()).await;
        // No transcription — audio completion alone shouldn't trigger
        let actions = strategy.process_frame(&audio_frame()).await;
        assert!(
            actions
                .iter()
                .all(|a| !matches!(a, TurnAction::UserTurnStopped(_)))
        );
    }

    #[tokio::test]
    async fn vad_stop_analyzes_turn() {
        let analyzer = MockTurnAnalyzer::new(true);
        let mut strategy = TurnAnalyzerUserTurnStopStrategy::new(Box::new(analyzer));

        strategy.process_frame(&vad_started()).await;
        strategy.process_frame(&transcription("hello", false)).await;
        strategy.process_frame(&vad_stopped()).await;

        // VAD stop → analyze_end_of_turn → turn_complete=true, starts timeout.
        // Finalized transcription should trigger immediately via maybe_trigger.
        let actions = strategy.process_frame(&transcription(" world", true)).await;
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, TurnAction::UserTurnStopped(_)))
        );
    }

    #[tokio::test]
    async fn reset_clears_analyzer() {
        let analyzer = MockTurnAnalyzer::new(true);
        let mut strategy = TurnAnalyzerUserTurnStopStrategy::new(Box::new(analyzer));

        strategy.process_frame(&vad_started()).await;
        strategy.process_frame(&audio_frame()).await;
        strategy.process_frame(&transcription("hello", true)).await;

        strategy.reset().await;
        assert!(strategy.text.is_empty());
        assert!(!strategy.turn_complete);
    }
}
