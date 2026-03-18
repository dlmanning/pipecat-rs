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
// SpeechSegment
// ---------------------------------------------------------------------------

/// A contiguous region of speech found by scanning PCM audio with VAD.
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// Byte offset into the PCM data where speech starts.
    pub start_byte: usize,
    /// Byte offset into the PCM data where speech ends.
    pub end_byte: usize,
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
/// Audio is buffered internally so callers can pass arbitrary-length buffers
/// without needing to know the VAD chunk size.
///
/// This is NOT a `FrameProcessor` — it's owned by a transport or processor
/// that feeds it audio and dispatches the returned events. For a ready-made
/// `FrameProcessor`, see [`super::processor::VadProcessor`].
#[derive(Debug)]
pub struct VadController<A: VadAnalyzer> {
    analyzer: VadAnalyzerBase<A>,
    /// Externally-visible state: only Quiet or Speaking.
    vad_state: VadState,
    /// Internal audio buffer for accumulating sub-chunk audio.
    buffer: Vec<u8>,
}

impl<A: VadAnalyzer> VadController<A> {
    /// Create a controller with default VAD params and call `handle_start`
    /// immediately. Ready to use after construction.
    pub fn new(analyzer: A, sample_rate: u32) -> Self {
        Self::with_params(analyzer, sample_rate, VadParams::default())
    }

    /// Create a controller with custom VAD params and call `handle_start`
    /// immediately. Ready to use after construction.
    pub fn with_params(analyzer: A, sample_rate: u32, params: VadParams) -> Self {
        let base = VadAnalyzerBase::new(analyzer, Some(sample_rate), Some(params));
        let mut controller = Self {
            analyzer: base,
            vad_state: VadState::Quiet,
            buffer: Vec::new(),
        };
        controller.handle_start(sample_rate);
        controller
    }

    /// Create a controller from a pre-configured `VadAnalyzerBase`.
    ///
    /// Unlike `new` and `with_params`, this does NOT call `handle_start` —
    /// the caller must do so before feeding audio. Use this when you need
    /// direct control over the `VadAnalyzerBase` (e.g., deferred sample
    /// rate configuration).
    pub fn from_base(analyzer: VadAnalyzerBase<A>) -> Self {
        Self {
            analyzer,
            vad_state: VadState::Quiet,
            buffer: Vec::new(),
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

    /// Handle a `StartFrame`. Sets the sample rate on the analyzer, clears
    /// the internal buffer, and returns the current VAD params (for
    /// broadcasting to other processors).
    pub fn handle_start(&mut self, sample_rate: u32) -> VadParams {
        self.analyzer.set_sample_rate(sample_rate);
        self.buffer.clear();
        self.analyzer.params().clone()
    }

    /// Handle an `InputAudioRawFrame`. Buffers audio internally and
    /// processes all complete chunks, returning events from each.
    ///
    /// Callers can pass arbitrary-length buffers — the controller handles
    /// chunking internally using `VadAnalyzerBase::process_chunk`.
    ///
    /// Matches Python `_handle_audio` + `_handle_vad` behavior:
    /// - Only emits `SpeechStarted`/`SpeechStopped` on actual state transitions
    ///   (filters out `Starting`/`Stopping`)
    /// - Emits `SpeechActivity` on every chunk while in Speaking state
    ///   (caller is responsible for throttling if needed)
    pub fn handle_audio(&mut self, audio: &[u8]) -> Vec<VadControllerEvent> {
        self.buffer.extend_from_slice(audio);
        let chunk_size = self.analyzer.chunk_size();

        if chunk_size == 0 {
            return Vec::new();
        }

        let mut events = Vec::new();

        while self.buffer.len() >= chunk_size {
            let (_state, event) = self.analyzer.process_chunk(&self.buffer[..chunk_size]);
            self.buffer.drain(..chunk_size);

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

            if self.vad_state == VadState::Speaking {
                events.push(VadControllerEvent::SpeechActivity);
            }
        }

        events
    }

    /// Handle a `VadParamsUpdateFrame`. Updates analyzer params and returns
    /// the new params (for broadcasting).
    pub fn handle_params_update(&mut self, params: VadParams) -> VadParams {
        self.analyzer.set_params(params);
        self.buffer.clear();
        self.analyzer.params().clone()
    }

    /// Scan raw PCM audio and return byte ranges of speech segments.
    ///
    /// Runs VAD at full speed over the buffer, collecting start/stop byte
    /// offsets. If speech is active at the end of the buffer, the final
    /// segment extends to `pcm.len()`.
    pub fn scan_segments(&mut self, pcm: &[u8]) -> Vec<SpeechSegment> {
        let chunk_size = self.analyzer.chunk_size();
        if chunk_size == 0 {
            return Vec::new();
        }

        let mut segments = Vec::new();
        let mut current_start: Option<usize> = None;

        for (i, chunk) in pcm.chunks_exact(chunk_size).enumerate() {
            let byte_offset = i * chunk_size;
            let events = self.handle_audio(chunk);
            for event in &events {
                match event {
                    VadControllerEvent::SpeechStarted => {
                        current_start = Some(byte_offset);
                    }
                    VadControllerEvent::SpeechStopped => {
                        if let Some(start) = current_start.take() {
                            segments.push(SpeechSegment {
                                start_byte: start,
                                end_byte: byte_offset + chunk_size,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        if let Some(start) = current_start {
            segments.push(SpeechSegment {
                start_byte: start,
                end_byte: pcm.len(),
            });
        }
        segments
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
        VadController::with_params(mock, 16000, params)
    }

    #[test]
    fn test_handle_start_returns_params() {
        let mock = MockVadAnalyzer::new(0.9, 160);
        let params = VadParams {
            confidence: 0.5,
            ..Default::default()
        };
        let base = VadAnalyzerBase::new(mock, None, Some(params));
        let mut controller = VadController::from_base(base);

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

    #[test]
    fn test_internal_buffering_arbitrary_sizes() {
        // Feed audio in odd-sized chunks that don't align with VAD chunk size.
        // The controller should buffer internally and produce the same result.
        let mut controller_aligned = make_controller(0.9);
        let mut controller_unaligned = make_controller(0.9);

        let audio = make_audio(160 * 10, 5000); // 3200 bytes total

        // Aligned: one shot
        let events_aligned = controller_aligned.handle_audio(&audio);

        // Unaligned: feed in 100-byte chunks (not a multiple of 320)
        let mut events_unaligned = Vec::new();
        for chunk in audio.chunks(100) {
            events_unaligned.extend(controller_unaligned.handle_audio(chunk));
        }

        // Both should detect speech
        assert!(events_aligned.contains(&VadControllerEvent::SpeechStarted));
        assert!(events_unaligned.contains(&VadControllerEvent::SpeechStarted));
    }

    #[test]
    fn test_scan_segments() {
        let mock = MockVadAnalyzer::new(0.9, 160);
        let params = VadParams {
            confidence: 0.7,
            start_secs: 0.02,
            stop_secs: 0.02,
            min_volume: 0.0,
        };
        let mut controller = VadController::with_params(mock, 16000, params);

        // Build audio: speech then silence
        let speech = make_audio(160 * 10, 5000);
        let silence = make_audio(160 * 10, 0);
        controller.analyzer_mut().analyzer().set_confidence(0.9);
        let mut pcm = speech;
        controller.analyzer_mut().analyzer().set_confidence(0.1);
        pcm.extend(silence);

        // Reset and scan — we need a fresh controller for scanning
        let mock = MockVadAnalyzer::new(0.9, 160);
        let params = VadParams {
            confidence: 0.7,
            start_secs: 0.02,
            stop_secs: 0.02,
            min_volume: 0.0,
        };
        let mut controller = VadController::with_params(mock, 16000, params);

        // For this mock, confidence is constant, so we'll get speech detection
        // for the loud part. Since mock returns 0.9 for everything, the silence
        // won't trigger a stop (confidence is still high). This test just
        // verifies the API works without panicking and returns segments.
        let segments = controller.scan_segments(&pcm);
        assert!(
            !segments.is_empty(),
            "should find at least one segment in loud audio"
        );
        assert!(segments[0].start_byte < segments[0].end_byte);
    }

    #[test]
    fn test_from_base_compat() {
        // Verify from_base preserves the old constructor behavior
        let mock = MockVadAnalyzer::new(0.9, 160);
        let params = VadParams {
            confidence: 0.5,
            ..Default::default()
        };
        let base = VadAnalyzerBase::new(mock, None, Some(params));
        let mut controller = VadController::from_base(base);
        controller.handle_start(16000);

        assert_eq!(controller.analyzer().params().confidence, 0.5);
    }
}
