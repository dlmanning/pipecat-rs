//! Integration tests for the Silero VAD backend.
//!
//! These tests exercise the full VAD stack (SileroVadAnalyzer → VadAnalyzerBase →
//! VadController) with real speech audio from the Silero VAD test suite.
//!
//! Run with: `cargo test -p pipecat-integration-tests --features silero`

#![cfg(feature = "silero")]

use pipecat_audio::vad::{SileroVadAnalyzer, VadController, VadControllerEvent, VadState};
use pipecat_core::VadParams;

/// 0.5s of real speech at 16 kHz mono 16-bit PCM (8000 samples, 16000 bytes).
/// Extracted from the Silero VAD test suite (MIT licensed):
/// https://github.com/snakers4/silero-vad/blob/master/tests/data/test.wav
const TEST_SPEECH_PCM: &[u8] = include_bytes!("../fixtures/test_speech_16khz.pcm");

/// 60s of real speech at 16 kHz mono 16-bit PCM WAV (from Silero VAD test suite).
const TEST_SPEECH_WAV: &[u8] = include_bytes!("../fixtures/test.wav");

fn make_silent_pcm(num_samples: usize) -> Vec<u8> {
    vec![0u8; num_samples * 2]
}

/// Build a VadController wired to a real Silero backend.
///
/// Uses defaults for confidence (0.7) and timing (start: 0.2s, stop: 0.2s).
/// Volume gating is disabled so we test the ML model in isolation without
/// the volume threshold filtering out low-amplitude speech.
fn make_controller() -> VadController<SileroVadAnalyzer> {
    VadController::with_params(
        SileroVadAnalyzer::new(16000).unwrap(),
        16000,
        VadParams {
            min_volume: 0.0,
            ..Default::default()
        },
    )
}

/// Extract raw PCM data from a WAV file by finding the "data" chunk.
fn wav_to_pcm(wav: &[u8]) -> Vec<u8> {
    let mut i = 12; // skip RIFF header
    while i + 8 <= wav.len() {
        let chunk_size = u32::from_le_bytes(wav[i + 4..i + 8].try_into().unwrap()) as usize;
        if &wav[i..i + 4] == b"data" {
            return wav[i + 8..i + 8 + chunk_size].to_vec();
        }
        i += 8 + chunk_size;
    }
    panic!("no data chunk found in WAV");
}

// ---------------------------------------------------------------------------
// Full-stack VadController tests with real speech audio
// ---------------------------------------------------------------------------

#[test]
fn controller_detects_speech_start_from_real_audio() {
    let mut controller = make_controller();
    let events = controller.handle_audio(TEST_SPEECH_PCM);

    assert!(
        events.contains(&VadControllerEvent::SpeechStarted),
        "VadController should emit SpeechStarted for real speech audio, got: {events:?}"
    );
}

#[test]
fn controller_detects_full_speech_cycle() {
    let mut controller = make_controller();

    // Phase 1: Feed real speech → expect SpeechStarted + SpeechActivity
    let speech_events = controller.handle_audio(TEST_SPEECH_PCM);

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
    let silence_events = controller.handle_audio(&silence);

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
    controller.handle_audio(TEST_SPEECH_PCM);
    let silence = make_silent_pcm(512 * 62);
    let silence_events = controller.handle_audio(&silence);
    assert!(
        silence_events.contains(&VadControllerEvent::SpeechStopped),
        "first turn should complete"
    );
    assert_eq!(*controller.vad_state(), VadState::Quiet);

    // Turn 2: speech again → should get a new SpeechStarted
    let events = controller.handle_audio(TEST_SPEECH_PCM);
    assert!(
        events.contains(&VadControllerEvent::SpeechStarted),
        "should detect speech start on second turn, got: {events:?}"
    );
    assert_eq!(*controller.vad_state(), VadState::Speaking);
}

#[test]
fn controller_stays_quiet_on_silence() {
    let mut controller = make_controller();

    // Feed 1 second of silence in one call. No events should be emitted.
    let silence = make_silent_pcm(512 * 31);
    let events = controller.handle_audio(&silence);
    assert!(
        events.is_empty(),
        "no events should be emitted for silence, got: {events:?}"
    );

    assert_eq!(*controller.vad_state(), VadState::Quiet);
}

#[test]
fn full_60s_wav_speech_segments() {
    let pcm_bytes = wav_to_pcm(TEST_SPEECH_WAV);

    let mut controller = make_controller();
    let chunk_size = controller.analyzer().chunk_size();

    let mut starts = 0usize;
    let mut stops = 0usize;
    let chunk_duration_ms = (chunk_size as f64 / 2.0) / 16.0; // ms per chunk

    for (i, chunk) in pcm_bytes.chunks_exact(chunk_size).enumerate() {
        let time_ms = i as f64 * chunk_duration_ms;
        let events = controller.handle_audio(chunk);
        for event in &events {
            match event {
                VadControllerEvent::SpeechStarted => {
                    starts += 1;
                    println!("  speech start #{starts} at {:.1}s", time_ms / 1000.0);
                }
                VadControllerEvent::SpeechStopped => {
                    stops += 1;
                    println!("  speech stop  #{stops} at {:.1}s", time_ms / 1000.0);
                }
                _ => {}
            }
        }
    }

    println!("60s WAV as-fast-as-possible: {starts} speech starts, {stops} speech stops");

    // The 60s recording contains continuous speech with natural pauses.
    // We expect at least a few speech segments.
    assert!(
        starts >= 2,
        "expected at least 2 speech starts in 60s of audio, got {starts}"
    );
    assert!(
        stops >= 1,
        "expected at least 1 speech stop in 60s of audio, got {stops}"
    );
    // Every start should eventually stop (except possibly the last if audio ends mid-speech).
    assert!(
        stops >= starts - 1,
        "stops ({stops}) should be at least starts-1 ({starts}-1)"
    );
}

// ---------------------------------------------------------------------------
// Whisper STT: standalone transcription of VAD segments (no pipeline)
// ---------------------------------------------------------------------------

/// Pre-scans the 60s WAV with VAD at full speed, then feeds each speech
/// segment directly to Whisper for transcription. No pipeline involved —
/// validates that Whisper works with our audio and VAD segment boundaries.
#[cfg(feature = "whisper")]
#[test]
fn whisper_transcribe_vad_segments() {
    use std::path::PathBuf;

    use pipecat_services::whisper::model::ensure_model;
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

    let pcm = wav_to_pcm(TEST_SPEECH_WAV);

    let mut controller = make_controller();
    let segments = controller.scan_segments(&pcm);
    assert!(
        segments.len() >= 2,
        "expected at least 2 speech segments, got {}",
        segments.len()
    );

    // Download/cache Whisper model.
    let home = std::env::var("HOME").expect("HOME not set");
    let cache_dir = PathBuf::from(home).join(".cache/pipecat-rs/whisper");
    let model_path = ensure_model("tiny.en", &cache_dir).expect("failed to get Whisper model");

    let ctx = WhisperContext::new_with_params(
        model_path.to_str().unwrap(),
        WhisperContextParameters::default(),
    )
    .expect("failed to create Whisper context");

    println!(
        "Found {} speech segments, transcribing with Whisper...",
        segments.len()
    );

    for (i, segment) in segments.iter().enumerate() {
        let audio = &pcm[segment.start_byte..segment.end_byte];
        let samples: Vec<f32> = audio
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();

        let mut state = ctx.create_state().unwrap();
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_no_timestamps(true);

        state.full(params, &samples).unwrap();

        let mut text = String::new();
        for segment_result in state.as_iter() {
            if let Ok(seg_text) = segment_result.to_str() {
                text.push_str(seg_text);
            }
        }
        let text = text.trim();

        let start_s = segment.start_byte as f64 / (16000.0 * 2.0);
        let end_s = segment.end_byte as f64 / (16000.0 * 2.0);
        let duration_s = end_s - start_s;

        println!("  Segment {i}: {start_s:.1}s - {end_s:.1}s ({duration_s:.1}s): \"{text}\"");

        assert!(
            !text.is_empty(),
            "segment {i} should produce non-empty transcription"
        );
    }
}

/// Same as above but feeds audio at real-time pace (20ms per chunk).
/// Verifies the VAD produces identical results regardless of pacing.
#[tokio::test]
async fn full_60s_wav_speech_segments_realtime_pacing() {
    use bytes::Bytes;
    use pipecat_audio::vad::VadProcessor;
    use pipecat_transport::TransportParams;
    use pipecat_transport::local::*;

    let pcm_bytes = wav_to_pcm(TEST_SPEECH_WAV);

    // Run the same VAD at max speed to get the ground truth count.
    let mut fast_controller = make_controller();
    let fast_events = fast_controller.handle_audio(&pcm_bytes);
    let expected_starts = fast_events
        .iter()
        .filter(|e| matches!(e, VadControllerEvent::SpeechStarted))
        .count();
    let expected_stops = fast_events
        .iter()
        .filter(|e| matches!(e, VadControllerEvent::SpeechStopped))
        .count();

    // Now run through the local transport at real-time pacing with a
    // VadCounter processor that counts VAD frames from VadProcessor.
    let in_params = TransportParams {
        audio_in_enabled: true,
        audio_in_passthrough: true,
        ..Default::default()
    };

    let input_transport = LocalAudioInputTransport::new(
        in_params,
        AudioInputSource::Buffer(Bytes::from_static(TEST_SPEECH_WAV)),
    )
    .with_format(AudioFormat::Encoded)
    .with_pacing(AudioPacing::RealTime);

    // VadProcessor handles all the VAD logic — no manual chunking needed.
    let vad_processor = VadProcessor::new(make_controller());

    // Collector that counts VAD events coming out of VadProcessor.
    let vad_starts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let vad_stops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    #[derive(Debug)]
    struct VadCounter {
        base: pipecat_core::processor::ProcessorBase,
        starts: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        stops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl pipecat_core::processor::FrameProcessor for VadCounter {
        fn name(&self) -> &str {
            self.base.name()
        }
        fn id(&self) -> u64 {
            self.base.id()
        }
        async fn process_frame(
            &mut self,
            envelope: pipecat_core::frame::FrameEnvelope,
            direction: pipecat_core::frame::Direction,
            ctx: &pipecat_core::processor::ProcessorContext,
        ) -> pipecat_core::error::Result<()> {
            use pipecat_core::frame::Frame;
            match &envelope.frame {
                Frame::VADUserStartedSpeaking(_) => {
                    let n = self
                        .starts
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        + 1;
                    println!("  [rt] speech start #{n}");
                }
                Frame::VADUserStoppedSpeaking(_) => {
                    let n = self
                        .stops
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        + 1;
                    println!("  [rt] speech stop  #{n}");
                }
                _ => {}
            }
            ctx.push_frame(envelope, direction).await?;
            Ok(())
        }
    }

    let counter = VadCounter {
        base: pipecat_core::processor::ProcessorBase::new("VadCounter"),
        starts: vad_starts.clone(),
        stops: vad_stops.clone(),
    };

    let pipeline = pipecat_pipeline::Pipeline::new(vec![
        Box::new(input_transport),
        Box::new(vad_processor),
        Box::new(counter),
    ]);

    let (node, handle, down_rx, _up_rx) = pipecat_core::test_utils::make_node(Box::new(pipeline));
    let _down = pipecat_core::test_utils::FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    pipecat_core::test_utils::send_frame(
        &handle,
        pipecat_core::frame::Frame::Start(pipecat_core::frame::StartFrame {
            audio_in_sample_rate: 16000,
            ..Default::default()
        }),
        pipecat_core::frame::Direction::Downstream,
    )
    .await;

    // Wait for all audio to flow through (60s at real-time = ~60s).
    tokio::time::sleep(std::time::Duration::from_secs(65)).await;

    pipecat_core::test_utils::send_frame(
        &handle,
        pipecat_core::frame::Frame::Cancel(pipecat_core::frame::CancelFrame::default()),
        pipecat_core::frame::Direction::Downstream,
    )
    .await;

    tokio::time::timeout(std::time::Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap();

    let rt_starts = vad_starts.load(std::sync::atomic::Ordering::Relaxed);
    let rt_stops = vad_stops.load(std::sync::atomic::Ordering::Relaxed);

    println!("60s WAV real-time pacing: {rt_starts} speech starts, {rt_stops} speech stops");
    println!(
        "60s WAV as-fast-as-possible: {expected_starts} speech starts, {expected_stops} speech stops"
    );

    // The transport chunks at 20ms (640 bytes) while VAD needs 1024-byte chunks.
    // VadProcessor buffers across frames internally, but the final partial buffer
    // may not form a complete VAD chunk, so we can lose one segment at the boundary.
    assert!(
        rt_starts.abs_diff(expected_starts) <= 1,
        "real-time ({rt_starts}) and max-speed ({expected_starts}) speech starts should match within 1"
    );
    assert!(
        rt_stops.abs_diff(expected_stops) <= 1,
        "real-time ({rt_stops}) and max-speed ({expected_stops}) speech stops should match within 1"
    );
}
