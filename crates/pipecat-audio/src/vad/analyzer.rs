use pipecat_core::VadParams;
use tracing::debug;

use crate::utils::{calculate_audio_volume, exp_smoothing};

// ---------------------------------------------------------------------------
// VadState — enum with data in variants (principles.md #9)
// ---------------------------------------------------------------------------

/// Voice Activity Detection states.
///
/// `Starting` and `Stopping` carry a frame counter to track how many
/// consecutive frames have been observed in that direction. This makes
/// invalid states unrepresentable — there is no counter when in `Quiet`
/// or `Speaking`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    Quiet,
    Starting { consecutive_frames: u32 },
    Speaking,
    Stopping { consecutive_frames: u32 },
}

// ---------------------------------------------------------------------------
// VadEvent — state transition events
// ---------------------------------------------------------------------------

/// Events emitted by the VAD state machine on confirmed transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadEvent {
    /// Transitioned from Starting → Speaking (speech confirmed).
    SpeechStarted,
    /// Transitioned from Stopping → Quiet (silence confirmed).
    SpeechStopped,
}

// ---------------------------------------------------------------------------
// VadStateMachine — pure state machine, independently testable
// ---------------------------------------------------------------------------

/// Core VAD state machine. Processes boolean "speaking" signals and tracks
/// state transitions with hysteresis.
///
/// Designed to match the Python `_run_analyzer` behavior exactly:
/// - [`update`] is called once per audio chunk inside a processing loop
///   (handles count increments and basic transitions).
/// - [`finalize`] is called once after the loop (promotes `Starting→Speaking`
///   or `Stopping→Quiet` when count thresholds are met).
///
/// This two-phase design matches Python lines 198-243 of `vad_analyzer.py`,
/// where threshold promotions happen *after* the while loop, not inside it.
#[derive(Debug)]
pub struct VadStateMachine {
    state: VadState,
    start_frames_required: u32,
    stop_frames_required: u32,
}

impl VadStateMachine {
    pub fn new(start_frames: u32, stop_frames: u32) -> Self {
        Self {
            state: VadState::Quiet,
            start_frames_required: start_frames,
            stop_frames_required: stop_frames,
        }
    }

    pub fn state(&self) -> &VadState {
        &self.state
    }

    /// Per-chunk update. Called inside the audio processing loop.
    ///
    /// Handles basic transitions and count increments:
    /// - `(Quiet, speaking)` → `Starting { 1 }`
    /// - `(Starting, speaking)` → increment count
    /// - `(Starting, !speaking)` → `Quiet`
    /// - `(Speaking, !speaking)` → `Stopping { 1 }`
    /// - `(Stopping, !speaking)` → increment count
    /// - `(Stopping, speaking)` → `Speaking`
    ///
    /// Does NOT do threshold promotions — that's [`finalize`].
    pub fn update(&mut self, speaking: bool) {
        if speaking {
            match &self.state {
                VadState::Quiet => {
                    self.state = VadState::Starting {
                        consecutive_frames: 1,
                    };
                }
                VadState::Starting {
                    consecutive_frames: n,
                } => {
                    self.state = VadState::Starting {
                        consecutive_frames: n + 1,
                    };
                }
                VadState::Stopping { .. } => {
                    self.state = VadState::Speaking;
                }
                VadState::Speaking => {}
            }
        } else {
            match &self.state {
                VadState::Starting { .. } => {
                    self.state = VadState::Quiet;
                }
                VadState::Speaking => {
                    self.state = VadState::Stopping {
                        consecutive_frames: 1,
                    };
                }
                VadState::Stopping {
                    consecutive_frames: n,
                } => {
                    self.state = VadState::Stopping {
                        consecutive_frames: n + 1,
                    };
                }
                VadState::Quiet => {}
            }
        }
    }

    /// Post-loop threshold check. Called once after all chunks are processed.
    ///
    /// Promotes `Starting→Speaking` or `Stopping→Quiet` when the accumulated
    /// frame count meets the required threshold, emitting the corresponding event.
    pub fn finalize(&mut self) -> Option<VadEvent> {
        match &self.state {
            VadState::Starting {
                consecutive_frames: n,
            } if *n >= self.start_frames_required => {
                self.state = VadState::Speaking;
                Some(VadEvent::SpeechStarted)
            }
            VadState::Stopping {
                consecutive_frames: n,
            } if *n >= self.stop_frames_required => {
                self.state = VadState::Quiet;
                Some(VadEvent::SpeechStopped)
            }
            _ => None,
        }
    }

    /// Update the required frame counts (called when VAD params change).
    pub fn set_frame_counts(&mut self, start_frames: u32, stop_frames: u32) {
        self.start_frames_required = start_frames;
        self.stop_frames_required = stop_frames;
    }

    /// Reset to Quiet state.
    pub fn reset(&mut self) {
        self.state = VadState::Quiet;
    }
}

// ---------------------------------------------------------------------------
// VadAnalyzer — trait for VAD backends
// ---------------------------------------------------------------------------

/// Trait for Voice Activity Detection backends.
///
/// Concrete implementations (e.g. Silero ONNX) provide model-specific voice
/// confidence calculation. The common buffering, volume smoothing, and state
/// machine logic lives in [`VadAnalyzerBase`].
///
/// `voice_confidence` takes `&mut self` because real backends (e.g. Silero)
/// maintain internal model state (LSTM hidden states, reset timers) that
/// gets updated on every inference call.
pub trait VadAnalyzer: Send + Sync + std::fmt::Debug {
    /// Number of audio frames (samples) required for one analysis chunk.
    /// For example, Silero at 16kHz requires 512 frames.
    fn num_frames_required(&self) -> usize;

    /// Calculate voice activity confidence for the given audio buffer.
    /// Returns a value between 0.0 and 1.0.
    ///
    /// This may be a blocking operation (ML model inference). Callers should
    /// use `tokio::task::spawn_blocking` when using real ML backends.
    fn voice_confidence(&mut self, buffer: &[u8]) -> f64;
}

// ---------------------------------------------------------------------------
// VadAnalyzerBase — wraps a backend with buffering + state machine
// ---------------------------------------------------------------------------

/// Wraps a [`VadAnalyzer`] backend with audio buffering, volume smoothing,
/// and the VAD state machine. This is the main entry point for VAD analysis.
///
/// Generic over the backend to avoid dynamic dispatch on the hot path
/// (`voice_confidence` is called per chunk).
#[derive(Debug)]
pub struct VadAnalyzerBase<A: VadAnalyzer> {
    analyzer: A,
    params: VadParams,
    sample_rate: u32,
    init_sample_rate: Option<u32>,
    num_channels: u16,

    // Audio buffering
    buffer: Vec<u8>,
    vad_frames_num_bytes: usize,

    // State machine
    state_machine: VadStateMachine,

    // Volume smoothing
    smoothing_factor: f64,
    prev_volume: f64,
}

impl<A: VadAnalyzer> VadAnalyzerBase<A> {
    /// Create a new analyzer base wrapping the given backend.
    ///
    /// If `sample_rate` is provided, it overrides whatever rate is set later
    /// via [`set_sample_rate`]. If `params` is `None`, defaults are used.
    pub fn new(analyzer: A, sample_rate: Option<u32>, params: Option<VadParams>) -> Self {
        let params = params.unwrap_or_default();
        Self {
            analyzer,
            params,
            sample_rate: 0,
            init_sample_rate: sample_rate,
            num_channels: 1,
            buffer: Vec::new(),
            vad_frames_num_bytes: 0,
            state_machine: VadStateMachine::new(0, 0),
            smoothing_factor: 0.2,
            prev_volume: 0.0,
        }
    }

    /// Access the underlying VAD analyzer backend.
    pub fn analyzer(&self) -> &A {
        &self.analyzer
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn params(&self) -> &VadParams {
        &self.params
    }

    pub fn state(&self) -> &VadState {
        self.state_machine.state()
    }

    /// Set the sample rate and recalculate internal values.
    ///
    /// If an `init_sample_rate` was provided at construction, it takes
    /// precedence (matching Python behavior).
    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = self.init_sample_rate.unwrap_or(sample_rate);
        self.recalculate();
    }

    /// Update VAD parameters and recalculate thresholds.
    pub fn set_params(&mut self, params: VadParams) {
        debug!(?params, "setting VAD params");
        self.params = params;
        self.recalculate();
    }

    /// Bytes required for one analysis chunk.
    pub fn chunk_size(&self) -> usize {
        self.vad_frames_num_bytes
    }

    /// Process exactly one chunk-sized buffer. No internal buffering —
    /// the caller is responsible for feeding exactly `chunk_size()` bytes.
    ///
    /// Returns the current state and any transition event. Unlike
    /// `analyze_audio`, this calls `finalize()` after each chunk, so it
    /// can detect transitions on every call.
    pub fn process_chunk(&mut self, chunk: &[u8]) -> (VadState, Option<VadEvent>) {
        debug_assert_eq!(
            chunk.len(),
            self.vad_frames_num_bytes,
            "process_chunk requires exactly chunk_size() bytes"
        );

        let confidence = self.analyzer.voice_confidence(chunk);
        let volume = self.get_smoothed_volume(chunk);
        self.prev_volume = volume;

        let speaking = confidence >= self.params.confidence && volume >= self.params.min_volume;
        self.state_machine.update(speaking);
        let event = self.state_machine.finalize();

        (*self.state_machine.state(), event)
    }

    /// Analyze an audio buffer. Accumulates audio, processes in chunks,
    /// runs the state machine, and returns the current state plus any
    /// transition event.
    ///
    /// This is synchronous — the caller decides whether to use
    /// `tokio::task::spawn_blocking` for backends with blocking inference.
    pub fn analyze_audio(&mut self, audio: &[u8]) -> (VadState, Option<VadEvent>) {
        self.buffer.extend_from_slice(audio);

        if self.vad_frames_num_bytes == 0 || self.buffer.len() < self.vad_frames_num_bytes {
            return (*self.state_machine.state(), None);
        }

        // Process all complete chunks (Python while loop, lines 198-228)
        while self.buffer.len() >= self.vad_frames_num_bytes {
            let confidence = self.analyzer.voice_confidence(&self.buffer[..self.vad_frames_num_bytes]);
            let volume = self.get_smoothed_volume(&self.buffer[..self.vad_frames_num_bytes]);
            self.buffer.drain(..self.vad_frames_num_bytes);
            self.prev_volume = volume;

            let speaking = confidence >= self.params.confidence && volume >= self.params.min_volume;

            self.state_machine.update(speaking);
        }

        // Post-loop threshold check (Python lines 230-243)
        let event = self.state_machine.finalize();

        (*self.state_machine.state(), event)
    }

    /// Recalculate internal values from current params and sample rate.
    fn recalculate(&mut self) {
        let vad_frames = self.analyzer.num_frames_required();
        self.vad_frames_num_bytes = vad_frames * self.num_channels as usize * 2;

        if self.sample_rate > 0 && vad_frames > 0 {
            let vad_frames_per_sec = vad_frames as f64 / self.sample_rate as f64;
            let start_frames = (self.params.start_secs / vad_frames_per_sec).round() as u32;
            let stop_frames = (self.params.stop_secs / vad_frames_per_sec).round() as u32;
            self.state_machine
                .set_frame_counts(start_frames, stop_frames);
        }

        // Reset state on recalculation (matches Python set_params)
        self.state_machine.reset();
        self.buffer.clear();
    }

    fn get_smoothed_volume(&self, audio: &[u8]) -> f64 {
        let volume = calculate_audio_volume(audio, self.sample_rate);
        exp_smoothing(volume, self.prev_volume, self.smoothing_factor)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- VadStateMachine tests (pure state transitions) --

    #[test]
    fn test_quiet_to_starting_on_speech() {
        let mut sm = VadStateMachine::new(3, 3);
        sm.update(true);
        assert_eq!(
            *sm.state(),
            VadState::Starting {
                consecutive_frames: 1
            }
        );
    }

    #[test]
    fn test_starting_increments_on_speech() {
        let mut sm = VadStateMachine::new(3, 3);
        sm.update(true);
        sm.update(true);
        assert_eq!(
            *sm.state(),
            VadState::Starting {
                consecutive_frames: 2
            }
        );
    }

    #[test]
    fn test_starting_to_quiet_on_silence() {
        let mut sm = VadStateMachine::new(3, 3);
        sm.update(true);
        sm.update(false);
        assert_eq!(*sm.state(), VadState::Quiet);
    }

    #[test]
    fn test_finalize_starting_to_speaking() {
        let mut sm = VadStateMachine::new(2, 2);
        sm.update(true);
        sm.update(true);
        let event = sm.finalize();
        assert_eq!(event, Some(VadEvent::SpeechStarted));
        assert_eq!(*sm.state(), VadState::Speaking);
    }

    #[test]
    fn test_finalize_no_event_below_threshold() {
        let mut sm = VadStateMachine::new(3, 3);
        sm.update(true);
        sm.update(true);
        let event = sm.finalize();
        assert_eq!(event, None);
        assert_eq!(
            *sm.state(),
            VadState::Starting {
                consecutive_frames: 2
            }
        );
    }

    #[test]
    fn test_speaking_to_stopping_on_silence() {
        let mut sm = VadStateMachine::new(1, 3);
        sm.update(true);
        sm.finalize(); // → Speaking
        sm.update(false);
        assert_eq!(
            *sm.state(),
            VadState::Stopping {
                consecutive_frames: 1
            }
        );
    }

    #[test]
    fn test_finalize_stopping_to_quiet() {
        let mut sm = VadStateMachine::new(1, 2);
        sm.update(true);
        sm.finalize(); // → Speaking
        sm.update(false);
        sm.update(false);
        let event = sm.finalize();
        assert_eq!(event, Some(VadEvent::SpeechStopped));
        assert_eq!(*sm.state(), VadState::Quiet);
    }

    #[test]
    fn test_stopping_to_speaking_on_speech() {
        let mut sm = VadStateMachine::new(1, 3);
        sm.update(true);
        sm.finalize(); // → Speaking
        sm.update(false);
        assert_eq!(
            *sm.state(),
            VadState::Stopping {
                consecutive_frames: 1
            }
        );
        sm.update(true);
        assert_eq!(*sm.state(), VadState::Speaking);
    }

    #[test]
    fn test_quiet_stays_quiet_on_silence() {
        let mut sm = VadStateMachine::new(3, 3);
        sm.update(false);
        assert_eq!(*sm.state(), VadState::Quiet);
    }

    #[test]
    fn test_speaking_stays_speaking_on_speech() {
        let mut sm = VadStateMachine::new(1, 3);
        sm.update(true);
        sm.finalize(); // → Speaking
        sm.update(true);
        assert_eq!(*sm.state(), VadState::Speaking);
    }

    #[test]
    fn test_reset() {
        let mut sm = VadStateMachine::new(1, 1);
        sm.update(true);
        sm.finalize(); // → Speaking
        sm.reset();
        assert_eq!(*sm.state(), VadState::Quiet);
    }

    #[test]
    fn test_post_loop_semantics() {
        // Verify the critical behavioral difference: if speech is interrupted
        // mid-loop, the promotion should NOT happen even if the count was met.
        let mut sm = VadStateMachine::new(2, 2);
        // Simulate 3 chunks: true, true, false
        sm.update(true); // Starting{1}
        sm.update(true); // Starting{2} — count meets threshold...
        sm.update(false); // ...but this reverts to Quiet
        let event = sm.finalize();
        // No promotion because state is now Quiet, not Starting
        assert_eq!(event, None);
        assert_eq!(*sm.state(), VadState::Quiet);
    }

    // -- VadAnalyzerBase tests with mock backend --

    #[derive(Debug)]
    struct MockVadAnalyzer {
        confidence: f64,
        frames_required: usize,
    }

    impl VadAnalyzer for MockVadAnalyzer {
        fn num_frames_required(&self) -> usize {
            self.frames_required
        }

        fn voice_confidence(&mut self, _buffer: &[u8]) -> f64 {
            self.confidence
        }
    }

    fn make_loud_audio(num_samples: usize) -> Vec<u8> {
        let mut audio = Vec::with_capacity(num_samples * 2);
        for _ in 0..num_samples {
            audio.extend_from_slice(&5000i16.to_le_bytes());
        }
        audio
    }

    fn make_silent_audio(num_samples: usize) -> Vec<u8> {
        vec![0u8; num_samples * 2]
    }

    #[test]
    fn test_analyzer_base_short_buffer_no_analysis() {
        let mock = MockVadAnalyzer {
            confidence: 0.9,
            frames_required: 512,
        };
        let mut base = VadAnalyzerBase::new(mock, None, None);
        base.set_sample_rate(16000);

        // Send less than one chunk
        let audio = make_loud_audio(100);
        let (state, event) = base.analyze_audio(&audio);
        assert_eq!(state, VadState::Quiet);
        assert_eq!(event, None);
    }

    #[test]
    fn test_analyzer_base_speech_detection() {
        let mock = MockVadAnalyzer {
            confidence: 0.9,
            frames_required: 160, // small for testing
        };
        let params = VadParams {
            confidence: 0.7,
            start_secs: 0.02, // ~1 frame at 160/16000
            stop_secs: 0.02,
            min_volume: 0.0, // disable volume check for this test
        };
        let mut base = VadAnalyzerBase::new(mock, None, Some(params));
        base.set_sample_rate(16000);

        // Send enough loud audio for start_frames to be met
        let audio = make_loud_audio(160 * 5);
        let (state, event) = base.analyze_audio(&audio);
        assert_eq!(state, VadState::Speaking);
        assert_eq!(event, Some(VadEvent::SpeechStarted));
    }

    #[test]
    fn test_analyzer_base_silence_after_speech() {
        // Use a mock whose confidence can change mid-test
        let mock = MockVadAnalyzer {
            confidence: 0.9,
            frames_required: 160,
        };
        let params = VadParams {
            confidence: 0.7,
            start_secs: 0.01,
            stop_secs: 0.01,
            min_volume: 0.0,
        };
        let mut base = VadAnalyzerBase::new(mock, None, Some(params));
        base.set_sample_rate(16000);

        // Get into Speaking state with high confidence
        let audio = make_loud_audio(160 * 5);
        let (state, _) = base.analyze_audio(&audio);
        assert_eq!(state, VadState::Speaking);

        // Switch to low confidence and send silence
        base.analyzer.confidence = 0.1;
        let audio = make_silent_audio(160 * 5);
        let (state, event) = base.analyze_audio(&audio);
        assert_eq!(state, VadState::Quiet);
        assert_eq!(event, Some(VadEvent::SpeechStopped));
    }

    #[test]
    fn test_analyzer_base_set_params_recalculates() {
        let mock = MockVadAnalyzer {
            confidence: 0.9,
            frames_required: 160,
        };
        let mut base = VadAnalyzerBase::new(mock, None, None);
        base.set_sample_rate(16000);

        // Get into a non-quiet state
        let audio = make_loud_audio(160);
        base.analyze_audio(&audio);

        // Update params — should reset state
        let new_params = VadParams {
            confidence: 0.5,
            start_secs: 0.1,
            stop_secs: 0.1,
            min_volume: 0.3,
        };
        base.set_params(new_params.clone());
        assert_eq!(*base.state(), VadState::Quiet);
        assert_eq!(base.params().confidence, 0.5);
    }

    #[test]
    fn test_analyzer_base_init_sample_rate_override() {
        let mock = MockVadAnalyzer {
            confidence: 0.9,
            frames_required: 512,
        };
        // Provide init_sample_rate of 8000
        let mut base = VadAnalyzerBase::new(mock, Some(8000), None);

        // set_sample_rate with 16000 — should use init_sample_rate (8000)
        base.set_sample_rate(16000);
        assert_eq!(base.sample_rate(), 8000);
    }
}
