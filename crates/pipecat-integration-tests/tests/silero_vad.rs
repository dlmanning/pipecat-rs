//! Integration tests for the Silero VAD backend.
//!
//! These tests exercise the full VAD stack (SileroVadAnalyzer → VadAnalyzerBase →
//! VadController) with real speech audio from the Silero VAD test suite.
//!
//! Run with: `cargo test -p pipecat-integration-tests --features silero`

#![cfg(feature = "silero")]

use pipecat_audio::vad::{
    SileroVadAnalyzer, VadAnalyzerBase, VadController, VadControllerEvent, VadState,
};
use pipecat_core::VadParams;

/// 0.5s of real speech at 16 kHz mono 16-bit PCM (8000 samples, 16000 bytes).
/// Extracted from the Silero VAD test suite (MIT licensed):
/// https://github.com/snakers4/silero-vad/blob/master/tests/data/test.wav
const TEST_SPEECH_PCM: &[u8] = include_bytes!("../fixtures/test_speech_16khz.pcm");

/// Silero at 16 kHz requires 512 samples per VAD chunk = 1024 bytes of int16 PCM.
const VAD_CHUNK_BYTES: usize = 512 * 2;

fn make_silent_pcm(num_samples: usize) -> Vec<u8> {
    vec![0u8; num_samples * 2]
}

/// Build a VadController wired to a real Silero backend.
///
/// Uses defaults for confidence (0.7) and timing (start: 0.2s, stop: 0.2s).
/// Volume gating is disabled so we test the ML model in isolation without
/// the volume threshold filtering out low-amplitude speech.
fn make_controller() -> VadController<SileroVadAnalyzer> {
    let analyzer = SileroVadAnalyzer::new(16000).unwrap();
    let params = VadParams {
        min_volume: 0.0,
        ..Default::default()
    };
    let base = VadAnalyzerBase::new(analyzer, Some(16000), Some(params));
    let mut controller = VadController::new(base);
    controller.handle_start(16000);
    controller
}

/// Feed PCM data through the controller in VAD-sized chunks, collecting all events.
fn feed_audio(
    controller: &mut VadController<SileroVadAnalyzer>,
    pcm: &[u8],
) -> Vec<VadControllerEvent> {
    let mut all_events = Vec::new();
    for chunk in pcm.chunks_exact(VAD_CHUNK_BYTES) {
        all_events.extend(controller.handle_audio(chunk));
    }
    all_events
}

// ---------------------------------------------------------------------------
// Full-stack VadController tests with real speech audio
// ---------------------------------------------------------------------------

#[test]
fn controller_detects_speech_start_from_real_audio() {
    let mut controller = make_controller();
    let events = feed_audio(&mut controller, TEST_SPEECH_PCM);

    assert!(
        events.contains(&VadControllerEvent::SpeechStarted),
        "VadController should emit SpeechStarted for real speech audio, got: {events:?}"
    );
}

#[test]
fn controller_detects_full_speech_cycle() {
    let mut controller = make_controller();

    // Phase 1: Feed real speech → expect SpeechStarted + SpeechActivity
    let speech_events = feed_audio(&mut controller, TEST_SPEECH_PCM);

    assert!(
        speech_events.contains(&VadControllerEvent::SpeechStarted),
        "should detect speech start"
    );
    assert!(
        speech_events.contains(&VadControllerEvent::SpeechActivity),
        "should emit speech activity while speaking"
    );

    // Phase 2: Feed silence until SpeechStopped.
    // With default stop_secs=0.2 at 16kHz/512 samples per chunk (~32ms each),
    // we need roughly 7 consecutive silent chunks. Use 2 seconds (62 chunks)
    // of silence as generous headroom.
    let silence = make_silent_pcm(512 * 62);
    let silence_events = feed_audio(&mut controller, &silence);

    assert!(
        silence_events.contains(&VadControllerEvent::SpeechStopped),
        "should emit SpeechStopped after silence follows speech, got: {silence_events:?}"
    );
    assert_eq!(*controller.vad_state(), VadState::Quiet);
}

#[test]
fn controller_detects_speech_resumption_after_stop() {
    let mut controller = make_controller();

    // Turn 1: speech → silence → stopped
    feed_audio(&mut controller, TEST_SPEECH_PCM);
    let silence = make_silent_pcm(512 * 62);
    let silence_events = feed_audio(&mut controller, &silence);
    assert!(
        silence_events.contains(&VadControllerEvent::SpeechStopped),
        "first turn should complete"
    );
    assert_eq!(*controller.vad_state(), VadState::Quiet);

    // Turn 2: speech again → should get a new SpeechStarted
    let events = feed_audio(&mut controller, TEST_SPEECH_PCM);
    assert!(
        events.contains(&VadControllerEvent::SpeechStarted),
        "should detect speech start on second turn, got: {events:?}"
    );
    assert_eq!(*controller.vad_state(), VadState::Speaking);
}

#[test]
fn controller_stays_quiet_on_silence() {
    let mut controller = make_controller();

    // Feed 1 second of silence (31 chunks). No events should be emitted at all.
    let silence = make_silent_pcm(512);
    for _ in 0..31 {
        let events = controller.handle_audio(&silence);
        assert!(
            events.is_empty(),
            "no events should be emitted for silence, got: {events:?}"
        );
    }

    assert_eq!(*controller.vad_state(), VadState::Quiet);
}
