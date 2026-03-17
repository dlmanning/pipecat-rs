use pipecat_core::VadParams;

use super::analyzer::{VadAnalyzer, VadAnalyzerBase, VadEvent, VadState};

// ---------------------------------------------------------------------------
// VadControllerEvent
// ---------------------------------------------------------------------------

/// Events emitted by the VAD controller.
///
/// The caller (typically the input transport) receives these and translates
/// them into frame pushes/broadcasts as appropriate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadControllerEvent {
    /// Speech confirmed (transitioned to Speaking).
    SpeechStarted,
    /// Silence confirmed (transitioned to Quiet).
    SpeechStopped,
    /// Activity signal emitted on every audio chunk while user is speaking.
    /// The caller is responsible for throttling if needed (matching Python,
    /// where the transport throttles `UserSpeakingFrame` independently).
    SpeechActivity,
}

// ---------------------------------------------------------------------------
// VadController
// ---------------------------------------------------------------------------

/// Wraps a [`VadAnalyzerBase`] and tracks externally-visible speech state.
///
/// The controller filters out intermediate states (`Starting`/`Stopping`),
/// only exposing `Quiet` and `Speaking` transitions. It emits
/// [`VadControllerEvent`]s that the owning transport can handle.
///
/// This is NOT a `FrameProcessor` — it's owned by a transport or processor
/// that feeds it audio and dispatches the returned events.
#[derive(Debug)]
pub struct VadController<A: VadAnalyzer> {
    analyzer: VadAnalyzerBase<A>,
    /// Externally-visible state: only Quiet or Speaking.
    vad_state: VadState,
}

impl<A: VadAnalyzer> VadController<A> {
    pub fn new(analyzer: VadAnalyzerBase<A>) -> Self {
        Self {
            analyzer,
            vad_state: VadState::Quiet,
        }
    }

    pub fn analyzer(&self) -> &VadAnalyzerBase<A> {
        &self.analyzer
    }

    pub fn analyzer_mut(&mut self) -> &mut VadAnalyzerBase<A> {
        &mut self.analyzer
    }

    pub fn vad_state(&self) -> &VadState {
        &self.vad_state
    }

    /// Handle a `StartFrame`. Sets the sample rate on the analyzer and
    /// returns the current VAD params (for broadcasting to other processors).
    pub fn handle_start(&mut self, sample_rate: u32) -> VadParams {
        self.analyzer.set_sample_rate(sample_rate);
        self.analyzer.params().clone()
    }

    /// Handle an `InputAudioRawFrame`. Runs VAD analysis and returns any events.
    ///
    /// Matches Python `_handle_audio` + `_handle_vad` behavior:
    /// - Only emits `SpeechStarted`/`SpeechStopped` on actual state transitions
    ///   (filters out `Starting`/`Stopping`)
    /// - Emits `SpeechActivity` on every call while in Speaking state
    ///   (caller is responsible for throttling if needed)
    pub fn handle_audio(&mut self, audio: &[u8]) -> Vec<VadControllerEvent> {
        let mut events = Vec::new();

        let (_new_state, event) = self.analyzer.analyze_audio(audio);

        // Map analyzer events to controller state transitions.
        // Only update visible state on Speaking/Quiet transitions.
        if let Some(vad_event) = event {
            match vad_event {
                VadEvent::SpeechStarted if self.vad_state != VadState::Speaking => {
                    self.vad_state = VadState::Speaking;
                    events.push(VadControllerEvent::SpeechStarted);
                }
                VadEvent::SpeechStopped if self.vad_state != VadState::Quiet => {
                    self.vad_state = VadState::Quiet;
                    events.push(VadControllerEvent::SpeechStopped);
                }
                _ => {}
            }
        }

        // Emit SpeechActivity on every call while speaking (Python line 123-124).
        if self.vad_state == VadState::Speaking {
            events.push(VadControllerEvent::SpeechActivity);
        }

        events
    }

    /// Handle a `VadParamsUpdateFrame`. Updates analyzer params and returns
    /// the new params (for broadcasting).
    pub fn handle_params_update(&mut self, params: VadParams) -> VadParams {
        self.analyzer.set_params(params);
        self.analyzer.params().clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockVadAnalyzer {
        confidence: std::sync::atomic::AtomicU64,
        frames_required: usize,
    }

    impl MockVadAnalyzer {
        fn new(confidence: f64, frames_required: usize) -> Self {
            Self {
                confidence: std::sync::atomic::AtomicU64::new(confidence.to_bits()),
                frames_required,
            }
        }

        fn set_confidence(&self, confidence: f64) {
            self.confidence
                .store(confidence.to_bits(), std::sync::atomic::Ordering::Relaxed);
        }
    }

    impl VadAnalyzer for MockVadAnalyzer {
        fn num_frames_required(&self) -> usize {
            self.frames_required
        }

        fn voice_confidence(&mut self, _buffer: &[u8]) -> f64 {
            f64::from_bits(self.confidence.load(std::sync::atomic::Ordering::Relaxed))
        }
    }

    fn make_audio(num_samples: usize, amplitude: i16) -> Vec<u8> {
        let mut audio = Vec::with_capacity(num_samples * 2);
        for _ in 0..num_samples {
            audio.extend_from_slice(&amplitude.to_le_bytes());
        }
        audio
    }

    fn make_controller(confidence: f64) -> VadController<MockVadAnalyzer> {
        let mock = MockVadAnalyzer::new(confidence, 160);
        let params = VadParams {
            confidence: 0.7,
            start_secs: 0.02,
            stop_secs: 0.02,
            min_volume: 0.0, // disable volume gating for tests
        };
        let analyzer = VadAnalyzerBase::new(mock, None, Some(params));
        let mut controller = VadController::new(analyzer);
        controller.handle_start(16000);
        controller
    }

    #[test]
    fn test_handle_start_returns_params() {
        let mock = MockVadAnalyzer::new(0.9, 160);
        let params = VadParams {
            confidence: 0.5,
            ..Default::default()
        };
        let analyzer = VadAnalyzerBase::new(mock, None, Some(params));
        let mut controller = VadController::new(analyzer);

        let returned_params = controller.handle_start(16000);
        assert_eq!(returned_params.confidence, 0.5);
    }

    #[test]
    fn test_speech_started_event() {
        let mut controller = make_controller(0.9);
        let audio = make_audio(160 * 10, 5000);
        let events = controller.handle_audio(&audio);
        assert!(
            events.contains(&VadControllerEvent::SpeechStarted),
            "expected SpeechStarted in {events:?}"
        );
    }

    #[test]
    fn test_speech_activity_while_speaking() {
        let mut controller = make_controller(0.9);

        // Get into speaking state
        let audio = make_audio(160 * 10, 5000);
        let events = controller.handle_audio(&audio);
        assert!(events.contains(&VadControllerEvent::SpeechStarted));
        assert!(events.contains(&VadControllerEvent::SpeechActivity));

        // More audio while speaking — should fire SpeechActivity every time
        let audio = make_audio(160 * 3, 5000);
        let events = controller.handle_audio(&audio);
        assert!(
            events.contains(&VadControllerEvent::SpeechActivity),
            "expected SpeechActivity in {events:?}"
        );
        // Should NOT have SpeechStarted again
        assert!(
            !events.contains(&VadControllerEvent::SpeechStarted),
            "unexpected duplicate SpeechStarted in {events:?}"
        );
    }

    #[test]
    fn test_speech_stopped_event() {
        let mut controller = make_controller(0.9);
        // Get into speaking state
        let audio = make_audio(160 * 10, 5000);
        controller.handle_audio(&audio);
        assert_eq!(*controller.vad_state(), VadState::Speaking);

        // Now switch to low confidence (simulating silence)
        controller.analyzer_mut().analyzer().set_confidence(0.1);
        let audio = make_audio(160 * 10, 0);
        let events = controller.handle_audio(&audio);
        assert!(
            events.contains(&VadControllerEvent::SpeechStopped),
            "expected SpeechStopped in {events:?}"
        );
    }

    #[test]
    fn test_params_update() {
        let mut controller = make_controller(0.9);
        let new_params = VadParams {
            confidence: 0.3,
            start_secs: 0.5,
            stop_secs: 0.5,
            min_volume: 0.1,
        };
        let returned = controller.handle_params_update(new_params);
        assert_eq!(returned.confidence, 0.3);
        assert_eq!(returned.start_secs, 0.5);
    }

    #[test]
    fn test_no_events_on_silence() {
        let mut controller = make_controller(0.1); // low confidence
        let audio = make_audio(160 * 10, 0);
        let events = controller.handle_audio(&audio);
        assert!(
            events.is_empty(),
            "expected no events for silence, got {events:?}"
        );
    }
}
