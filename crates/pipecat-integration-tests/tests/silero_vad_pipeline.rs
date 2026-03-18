//! End-to-end integration test: real audio → real Silero VAD → full pipeline.
//!
//! Real speech audio drives a real ML-based VAD, which triggers turn detection,
//! fake STT/LLM/TTS, and context accumulation — exercising the full conversational
//! lifecycle with real VAD state transitions.
//!
//! Run with: `cargo test -p pipecat-integration-tests --features silero --test silero_vad_pipeline`

#![cfg(feature = "silero")]

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use pipecat_audio::vad::{SileroVadAnalyzer, VadAnalyzerBase, VadController, VadControllerEvent};
use pipecat_context::{LLMContext, LLMContextAggregatorPair, LLMUserAggregatorParams};
use pipecat_core::VadParams;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_core::test_utils::*;
use pipecat_integration_tests::mock_services::*;
use pipecat_pipeline::Pipeline;
use pipecat_services::settings::STTSettings;
use pipecat_services::stt::{STTService, STTServiceState, stt_process_frame};
use pipecat_transport::TransportParams;
use pipecat_transport::local::*;
use pipecat_turns::{
    SpeechTimeoutUserTurnStopStrategy, UserTurnStrategies, VadUserTurnStartStrategy,
};
use serde_json::json;
use tokio::time::timeout;

/// 60s of real speech at 16 kHz mono 16-bit PCM WAV (from Silero VAD test suite).
const TEST_SPEECH_WAV: &[u8] = include_bytes!("../fixtures/test.wav");

/// Silero at 16 kHz requires 512 samples per VAD chunk = 1024 bytes of int16 PCM.
const VAD_CHUNK_BYTES: usize = 512 * 2;

// ---------------------------------------------------------------------------
// VadBridgeProcessor: wraps real Silero VAD and emits VAD frames
// ---------------------------------------------------------------------------

/// Test-only processor that wraps a real `VadController<SileroVadAnalyzer>`.
///
/// Converts `InputAudioRaw` frames into `VADUserStartedSpeaking`,
/// `VADUserStoppedSpeaking`, and `UserSpeaking` frames — mimicking what
/// the input transport does in production.
///
/// Buffers incoming audio to produce VAD-chunk-sized blocks, since the
/// transport may chunk at a different size (e.g. 20ms = 640 bytes) than
/// what Silero requires (512 samples = 1024 bytes).
#[derive(Debug)]
struct VadBridgeProcessor {
    base: ProcessorBase,
    controller: Option<VadController<SileroVadAnalyzer>>,
    audio_buf: Vec<u8>,
}

impl VadBridgeProcessor {
    fn new() -> Self {
        Self {
            base: ProcessorBase::new("VadBridge"),
            controller: None,
            audio_buf: Vec::new(),
        }
    }

    async fn drain_vad_chunks(&mut self, ctx: &ProcessorContext) -> Result<()> {
        let controller = self.controller.as_mut().unwrap();
        while self.audio_buf.len() >= VAD_CHUNK_BYTES {
            let chunk: Vec<u8> = self.audio_buf.drain(..VAD_CHUNK_BYTES).collect();
            let events = controller.handle_audio(&chunk);
            for event in &events {
                match event {
                    VadControllerEvent::SpeechStarted => {
                        let params = controller.analyzer().params();
                        ctx.send_downstream(Frame::VADUserStartedSpeaking(
                            VADUserStartedSpeakingFrame {
                                start_secs: params.start_secs,
                                timestamp: 0.0,
                            },
                        ))
                        .await?;
                    }
                    VadControllerEvent::SpeechStopped => {
                        let params = controller.analyzer().params();
                        ctx.send_downstream(Frame::VADUserStoppedSpeaking(
                            VADUserStoppedSpeakingFrame {
                                stop_secs: params.stop_secs,
                                timestamp: 0.0,
                            },
                        ))
                        .await?;
                    }
                    VadControllerEvent::SpeechActivity => {
                        ctx.send_downstream(Frame::UserSpeaking(UserSpeakingFrame))
                            .await?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl FrameProcessor for VadBridgeProcessor {
    fn name(&self) -> &str {
        self.base.name()
    }
    fn id(&self) -> u64 {
        self.base.id()
    }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        match &envelope.frame {
            Frame::Start(_) => {
                let analyzer =
                    SileroVadAnalyzer::new(16000).expect("failed to create Silero VAD analyzer");
                let params = VadParams {
                    min_volume: 0.0, // disable volume gating for test audio
                    ..Default::default()
                };
                let base = VadAnalyzerBase::new(analyzer, Some(16000), Some(params));
                let mut controller = VadController::new(base);
                controller.handle_start(16000);
                self.controller = Some(controller);
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::InputAudioRaw(audio_frame) => {
                self.audio_buf.extend_from_slice(&audio_frame.audio);
                self.drain_vad_chunks(ctx).await?;
                // Forward all audio downstream — matching real transport behavior.
                ctx.push_frame(envelope, direction).await?;
            }

            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VadGatedSTT: only transcribes while user is speaking
// ---------------------------------------------------------------------------

/// STT mock that emits one transcription per speech segment.
///
/// Models real STT behavior for VAD-integrated tests: audio is received
/// continuously, but transcriptions are only produced from speech. The
/// `stt_process_frame` base tracks `user_speaking` from VAD events. This
/// mock uses that state plus a per-segment flag to emit exactly one
/// transcription per speech segment after accumulating enough audio chunks.
#[derive(Debug)]
struct VadGatedSTT {
    state: STTServiceState,
    responses: Vec<String>,
    response_index: usize,
    chunks_before_emit: usize,
    audio_chunk_count: usize,
    /// Prevents multiple transcriptions within a single speech segment.
    emitted_for_segment: bool,
}

impl VadGatedSTT {
    fn new(responses: Vec<&str>, chunks_before_emit: usize) -> Self {
        Self {
            state: STTServiceState::new("VadGatedSTT", STTSettings::default()),
            responses: responses.into_iter().map(String::from).collect(),
            response_index: 0,
            chunks_before_emit,
            audio_chunk_count: 0,
            emitted_for_segment: false,
        }
    }
}

#[async_trait]
impl STTService for VadGatedSTT {
    async fn run_stt(&mut self, _audio: Bytes, ctx: &ProcessorContext) -> Result<()> {
        if !self.state.user_speaking || self.emitted_for_segment {
            return Ok(());
        }
        self.audio_chunk_count += 1;
        if self.audio_chunk_count >= self.chunks_before_emit {
            self.emitted_for_segment = true;
            let text = self.responses[self.response_index % self.responses.len()].clone();
            self.response_index += 1;

            ctx.send_downstream(Frame::Transcription(TranscriptionFrame {
                text,
                user_id: "user".to_string(),
                timestamp: None,
                language: None,
                finalized: true,
                result: None,
            }))
            .await?;
        }
        Ok(())
    }

    fn stt_service_state(&self) -> &STTServiceState {
        &self.state
    }
    fn stt_service_state_mut(&mut self) -> &mut STTServiceState {
        &mut self.state
    }
}

#[async_trait]
impl FrameProcessor for VadGatedSTT {
    fn name(&self) -> &str {
        self.state.base.processor.name()
    }
    fn id(&self) -> u64 {
        self.state.base.processor.id()
    }
    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        // Reset per-segment state when a new speech segment starts.
        // Only reset on the original VAD event going downstream (from VadBridge),
        // NOT on UserStartedSpeaking which the UserAggregator broadcasts back
        // upstream — that's a derived event from the same speech start.
        if direction == Direction::Downstream
            && matches!(&envelope.frame, Frame::VADUserStartedSpeaking(_))
        {
            self.audio_chunk_count = 0;
            self.emitted_for_segment = false;
        }
        stt_process_frame(self, envelope, direction, ctx).await
    }
}

// ---------------------------------------------------------------------------
// Ordering assertion helpers
// ---------------------------------------------------------------------------

/// Find the first index of a frame name in the list, panicking with context if absent.
fn first_index(names: &[String], target: &str) -> usize {
    names
        .iter()
        .position(|n| n == target)
        .unwrap_or_else(|| panic!("expected {target} in frame list: {names:?}"))
}

/// Assert that `earlier` appears before `later` in the frame name list.
fn assert_order(names: &[String], earlier: &str, later: &str) {
    let ei = first_index(names, earlier);
    let li = first_index(names, later);
    assert!(
        ei < li,
        "{earlier} (idx {ei}) should appear before {later} (idx {li}): {names:?}"
    );
}

// ===========================================================================
// Test: Real WAV file → LocalTransport → VAD → STT → LLM → TTS pipeline
// ===========================================================================

/// Full-stack test feeding a real 60s WAV file through `LocalAudioInputTransport`
/// with real Silero VAD. The VAD detects natural speech/silence transitions in
/// the recording, triggering the conversational pipeline.
#[tokio::test]
async fn single_turn_real_vad_full_pipeline() {
    let context = LLMContext::new(vec![
        json!({"role": "system", "content": "You are helpful."}),
    ]);
    let context_ref = context.clone();

    let params = LLMUserAggregatorParams {
        user_turn_strategies: UserTurnStrategies {
            start: vec![Box::new(VadUserTurnStartStrategy::new())],
            stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.1))],
        },
        user_turn_stop_timeout: Duration::from_secs(5),
    };
    let pair = LLMContextAggregatorPair::new(context, params);
    let (user_agg, assistant_agg) = pair.into_processors();

    let in_params = TransportParams {
        audio_in_enabled: true,
        audio_in_passthrough: true,
        ..Default::default()
    };

    let input_transport = LocalAudioInputTransport::new(
        in_params,
        AudioInputSource::Buffer(Bytes::from_static(TEST_SPEECH_WAV)),
    )
    .with_format(AudioFormat::Wav)
    .with_pacing(AudioPacing::RealTime);

    let pipeline = Pipeline::new(vec![
        Box::new(input_transport),
        Box::new(VadBridgeProcessor::new()),
        Box::new(VadGatedSTT::new(vec!["hello from real vad"], 1)),
        user_agg,
        Box::new(FakeLLMService::new(vec!["I heard you!".to_string()])),
        Box::new(FakeTTSService::new()),
        assistant_agg,
    ]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame {
            audio_in_sample_rate: 16000,
            audio_out_sample_rate: 16000,
            ..Default::default()
        }),
        Direction::Downstream,
    )
    .await;

    // Wait for the first complete turn: VAD detects speech → silence transition
    // in the real audio, triggering STT → LLM → TTS cascade.
    down.wait_for_frame_timeout("LLMContext", Duration::from_secs(30))
        .await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;
    timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap();

    // --- Frame ordering: VAD start → VAD stop → turn completion ---
    let names = down.frame_names();

    assert_order(&names, "VADUserStartedSpeaking", "VADUserStoppedSpeaking");

    // UserSpeaking should fire while speaking
    assert!(
        names.contains(&"UserSpeaking".to_string()),
        "should emit UserSpeaking activity frames: {names:?}"
    );

    // --- Context accumulated correctly ---
    let messages = context_ref.get_messages();
    assert!(
        messages.len() >= 3,
        "expected system + user + assistant >= 3 messages, got {}: {messages:?}",
        messages.len()
    );
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"], "hello from real vad");
    assert_eq!(messages[2]["role"], "assistant");
}

// ===========================================================================
// Test 2: Two turns — real VAD drives both, context accumulates across turns
// ===========================================================================

/// Same setup as test 1, but waits for two complete turns from the real WAV.
/// The 60s recording contains multiple speech/silence segments, so the VAD
/// naturally detects multiple turns. Verifies context accumulates across turns.
#[tokio::test]
async fn two_turns_real_vad_context_accumulation() {
    let context = LLMContext::new(vec![
        json!({"role": "system", "content": "You are helpful."}),
    ]);
    let context_ref = context.clone();

    let params = LLMUserAggregatorParams {
        user_turn_strategies: UserTurnStrategies {
            start: vec![Box::new(VadUserTurnStartStrategy::new())],
            stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.1))],
        },
        user_turn_stop_timeout: Duration::from_secs(5),
    };
    let pair = LLMContextAggregatorPair::new(context, params);
    let (user_agg, assistant_agg) = pair.into_processors();

    let in_params = TransportParams {
        audio_in_enabled: true,
        audio_in_passthrough: true,
        ..Default::default()
    };

    let input_transport = LocalAudioInputTransport::new(
        in_params,
        AudioInputSource::Buffer(Bytes::from_static(TEST_SPEECH_WAV)),
    )
    .with_format(AudioFormat::Wav)
    .with_pacing(AudioPacing::RealTime);

    let pipeline = Pipeline::new(vec![
        Box::new(input_transport),
        Box::new(VadBridgeProcessor::new()),
        Box::new(VadGatedSTT::new(vec!["hello", "goodbye"], 1)),
        user_agg,
        Box::new(FakeLLMService::new(vec!["Got it.".to_string()])),
        Box::new(FakeTTSService::new()),
        assistant_agg,
    ]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame {
            audio_in_sample_rate: 16000,
            audio_out_sample_rate: 16000,
            ..Default::default()
        }),
        Direction::Downstream,
    )
    .await;

    // Wait for first turn.
    down.wait_for_frame_timeout("LLMContext", Duration::from_secs(30))
        .await;

    let msg_count = context_ref.message_count();
    assert_eq!(
        msg_count, 3,
        "after turn 1: expected system + user + assistant = 3, got {msg_count}"
    );
    assert_eq!(context_ref.get_messages()[1]["content"], "hello");

    // Clear and wait for second turn.
    down.take_frames();
    down.wait_for_frame_timeout("LLMContext", Duration::from_secs(30))
        .await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;
    timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap();

    // --- At least 5 messages: system + 2*(user + assistant) ---
    // May have more if additional turns started before cancel.
    let messages = context_ref.get_messages();
    assert!(
        messages.len() >= 5,
        "expected at least system + 2*(user + assistant) = 5, got {}: {messages:?}",
        messages.len()
    );

    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[3]["role"], "user");
    assert_eq!(messages[4]["role"], "assistant");

    assert_eq!(messages[1]["content"], "hello", "turn 1 user message");
    assert_eq!(messages[3]["content"], "goodbye", "turn 2 user message");
}

// ===========================================================================
// VAD pre-scan helpers
// ===========================================================================

/// A speech segment found by running the VAD at full speed.
struct SpeechSegment {
    /// Byte offset into the PCM data where speech starts.
    start_byte: usize,
    /// Byte offset into the PCM data where speech ends.
    end_byte: usize,
}

/// Run the VAD at max speed over raw PCM, returning speech segment byte ranges.
fn prescan_vad(pcm: &[u8]) -> Vec<SpeechSegment> {
    let analyzer = SileroVadAnalyzer::new(16000).expect("failed to create Silero VAD analyzer");
    let params = VadParams {
        min_volume: 0.0,
        ..Default::default()
    };
    let base = VadAnalyzerBase::new(analyzer, Some(16000), Some(params));
    let mut controller = VadController::new(base);
    controller.handle_start(16000);

    let mut segments = Vec::new();
    let mut current_start: Option<usize> = None;

    for (i, chunk) in pcm.chunks_exact(VAD_CHUNK_BYTES).enumerate() {
        let byte_offset = i * VAD_CHUNK_BYTES;
        let events = controller.handle_audio(chunk);
        for event in &events {
            match event {
                VadControllerEvent::SpeechStarted => {
                    current_start = Some(byte_offset);
                }
                VadControllerEvent::SpeechStopped => {
                    if let Some(start) = current_start.take() {
                        segments.push(SpeechSegment {
                            start_byte: start,
                            end_byte: byte_offset + VAD_CHUNK_BYTES,
                        });
                    }
                }
                _ => {}
            }
        }
    }
    // Close any open segment at end of file.
    if let Some(start) = current_start {
        segments.push(SpeechSegment {
            start_byte: start,
            end_byte: pcm.len(),
        });
    }
    segments
}

/// Decode WAV bytes to raw PCM.
fn wav_to_pcm(wav: &[u8]) -> Vec<u8> {
    let reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
    reader
        .into_samples::<i16>()
        .map(|s| s.unwrap())
        .flat_map(|s| s.to_le_bytes())
        .collect()
}

// ===========================================================================
// Test 3: Pre-scanned VAD → fast pipeline test (no real-time pacing)
// ===========================================================================

/// Pre-scans the 60s WAV with the VAD at full speed, then replays each speech
/// segment into the pipeline as pre-determined VAD events + audio chunks.
/// No VadBridge processor needed — the test injects VAD frames directly.
/// Runs as fast as the pipeline can process, not at real-time.
#[tokio::test]
async fn prescanned_vad_fast_pipeline() {
    let pcm = wav_to_pcm(TEST_SPEECH_WAV);
    let segments = prescan_vad(&pcm);

    assert!(
        segments.len() >= 2,
        "expected at least 2 speech segments, got {}",
        segments.len()
    );

    let context = LLMContext::new(vec![
        json!({"role": "system", "content": "You are helpful."}),
    ]);
    let context_ref = context.clone();

    let params = LLMUserAggregatorParams {
        user_turn_strategies: UserTurnStrategies {
            start: vec![Box::new(VadUserTurnStartStrategy::new())],
            stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.05))],
        },
        user_turn_stop_timeout: Duration::from_secs(5),
    };
    let pair = LLMContextAggregatorPair::new(context, params);
    let (user_agg, assistant_agg) = pair.into_processors();

    // Pipeline without VadBridge — we inject VAD events directly.
    let stt = VadGatedSTT::new(vec!["turn one", "turn two"], 1);
    let pipeline = Pipeline::new(vec![
        Box::new(stt),
        user_agg,
        Box::new(FakeLLMService::new(vec!["Got it.".to_string()])),
        Box::new(FakeTTSService::new()),
        assistant_agg,
    ]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Replay two speech segments from the pre-scan.
    for segment in segments.iter().take(2) {
        let audio_data = &pcm[segment.start_byte..segment.end_byte];

        // Signal speech start.
        send_frame(
            &handle,
            Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
                start_secs: 0.2,
                timestamp: 0.0,
            }),
            Direction::Downstream,
        )
        .await;

        // Send the speech audio in VAD-chunk-sized frames.
        for chunk in audio_data.chunks(VAD_CHUNK_BYTES) {
            send_frame(
                &handle,
                Frame::InputAudioRaw(AudioRawFrame {
                    audio: Bytes::from(chunk.to_vec()),
                    sample_rate: 16000,
                    num_channels: 1,
                }),
                Direction::Downstream,
            )
            .await;
        }

        // Signal speech stop.
        send_frame(
            &handle,
            Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
                stop_secs: 0.2,
                timestamp: 0.0,
            }),
            Direction::Downstream,
        )
        .await;

        // Wait for this turn to complete.
        down.wait_for_frame("LLMContext").await;
        down.take_frames();
    }

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;
    timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap();

    // Verify: system + 2*(user + assistant) = 5 messages.
    let messages = context_ref.get_messages();
    assert!(
        messages.len() >= 5,
        "expected at least 5 messages, got {}: {messages:?}",
        messages.len()
    );
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["content"], "turn one");
    assert_eq!(messages[3]["content"], "turn two");
}

// ===========================================================================
// Test 4: Pre-scanned VAD → Whisper STT pipeline (real transcription)
// ===========================================================================

/// Pre-scans the 60s WAV with VAD, then replays speech segments through a
/// pipeline using real Whisper STT instead of fake/canned responses.
/// Verifies that real transcriptions are produced from the audio.
#[cfg(feature = "whisper")]
#[tokio::test]
async fn prescanned_vad_whisper_pipeline() {
    use std::path::PathBuf;

    use pipecat_services::whisper::WhisperSTTService;
    use pipecat_services::whisper::model::ensure_model;

    let pcm = wav_to_pcm(TEST_SPEECH_WAV);
    let segments = prescan_vad(&pcm);

    assert!(
        segments.len() >= 2,
        "expected at least 2 speech segments, got {}",
        segments.len()
    );

    // Download/cache Whisper model.
    let home = std::env::var("HOME").expect("HOME not set");
    let cache_dir = PathBuf::from(home).join(".cache/pipecat-rs/whisper");
    let model_path = ensure_model("tiny.en", &cache_dir).expect("failed to get Whisper model");

    let context = LLMContext::new(vec![
        json!({"role": "system", "content": "You are helpful."}),
    ]);
    let context_ref = context.clone();

    let params = LLMUserAggregatorParams {
        user_turn_strategies: UserTurnStrategies {
            start: vec![Box::new(VadUserTurnStartStrategy::new())],
            stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.05))],
        },
        user_turn_stop_timeout: Duration::from_secs(5),
    };
    let pair = LLMContextAggregatorPair::new(context, params);
    let (user_agg, assistant_agg) = pair.into_processors();

    let whisper_stt = WhisperSTTService::new(
        &model_path,
        pipecat_services::settings::STTSettings {
            language: Some("en".to_string()),
            ..Default::default()
        },
    )
    .expect("failed to create WhisperSTTService");

    // Pipeline: WhisperSTT → user aggregator → fake LLM → fake TTS → assistant aggregator
    let pipeline = Pipeline::new(vec![
        Box::new(whisper_stt),
        user_agg,
        Box::new(FakeLLMService::new(vec!["Got it.".to_string()])),
        Box::new(FakeTTSService::new()),
        assistant_agg,
    ]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Replay first two speech segments.
    for segment in segments.iter().take(2) {
        let audio_data = &pcm[segment.start_byte..segment.end_byte];

        send_frame(
            &handle,
            Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
                start_secs: 0.2,
                timestamp: 0.0,
            }),
            Direction::Downstream,
        )
        .await;

        for chunk in audio_data.chunks(VAD_CHUNK_BYTES) {
            send_frame(
                &handle,
                Frame::InputAudioRaw(AudioRawFrame {
                    audio: Bytes::from(chunk.to_vec()),
                    sample_rate: 16000,
                    num_channels: 1,
                }),
                Direction::Downstream,
            )
            .await;
        }

        send_frame(
            &handle,
            Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
                stop_secs: 0.2,
                timestamp: 0.0,
            }),
            Direction::Downstream,
        )
        .await;

        // Wait for this turn to complete (Whisper transcription → LLM → TTS → context).
        down.wait_for_frame_timeout("LLMContext", Duration::from_secs(30))
            .await;
        down.take_frames();
    }

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;
    timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap();

    // Verify: system + 2*(user + assistant) = 5 messages.
    let messages = context_ref.get_messages();
    assert!(
        messages.len() >= 5,
        "expected at least 5 messages, got {}: {messages:?}",
        messages.len()
    );
    assert_eq!(messages[0]["role"], "system");

    // Real Whisper transcriptions — verify they're non-empty strings.
    let turn1_text = messages[1]["content"].as_str().unwrap();
    let turn2_text = messages[3]["content"].as_str().unwrap();
    println!("Turn 1 transcription: \"{turn1_text}\"");
    println!("Turn 2 transcription: \"{turn2_text}\"");
    assert!(
        !turn1_text.is_empty(),
        "turn 1 should have non-empty transcription"
    );
    assert!(
        !turn2_text.is_empty(),
        "turn 2 should have non-empty transcription"
    );
}

// ===========================================================================
// Test 5: Fast-as-possible full transcript — all VAD segments through Whisper
// ===========================================================================

/// Pre-scans the 60s WAV with VAD, then replays ALL speech segments through
/// a pipeline with WhisperSTT. Prints the full transcript with timestamps.
/// No turn management — just WhisperSTT collecting Transcription frames.
#[cfg(feature = "whisper")]
#[tokio::test]
async fn whisper_full_transcript_fast() {
    use std::path::PathBuf;

    use pipecat_services::whisper::WhisperSTTService;
    use pipecat_services::whisper::model::ensure_model;

    let pcm = wav_to_pcm(TEST_SPEECH_WAV);
    let segments = prescan_vad(&pcm);

    let home = std::env::var("HOME").expect("HOME not set");
    let cache_dir = PathBuf::from(home).join(".cache/pipecat-rs/whisper");
    let model_path = ensure_model("tiny.en", &cache_dir).expect("failed to get Whisper model");

    let whisper_stt = WhisperSTTService::new(
        &model_path,
        pipecat_services::settings::STTSettings {
            language: Some("en".to_string()),
            ..Default::default()
        },
    )
    .expect("failed to create WhisperSTTService");

    let pipeline = Pipeline::new(vec![Box::new(whisper_stt)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame {
            audio_in_sample_rate: 16000,
            ..Default::default()
        }),
        Direction::Downstream,
    )
    .await;

    // Process segments sequentially: send one segment's audio, wait for its
    // transcription, collect it, then proceed to the next.
    let mut transcriptions: Vec<String> = Vec::new();

    for segment in &segments {
        let audio_data = &pcm[segment.start_byte..segment.end_byte];

        send_frame(
            &handle,
            Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
                start_secs: 0.2,
                timestamp: 0.0,
            }),
            Direction::Downstream,
        )
        .await;

        for chunk in audio_data.chunks(VAD_CHUNK_BYTES) {
            send_frame(
                &handle,
                Frame::InputAudioRaw(AudioRawFrame {
                    audio: Bytes::from(chunk.to_vec()),
                    sample_rate: 16000,
                    num_channels: 1,
                }),
                Direction::Downstream,
            )
            .await;
        }

        send_frame(
            &handle,
            Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
                stop_secs: 0.2,
                timestamp: 0.0,
            }),
            Direction::Downstream,
        )
        .await;

        // Wait for the Transcription frame from this segment, then drain.
        down.wait_for_frame_timeout("Transcription", Duration::from_secs(30))
            .await;
        let frames = down.take_frames();
        for f in &frames {
            if let Frame::Transcription(t) = &f.frame {
                transcriptions.push(t.text.clone());
            }
        }
    }

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;
    timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap();

    println!(
        "\n=== Fast-as-possible pipeline: {} segments, {} transcriptions ===",
        segments.len(),
        transcriptions.len()
    );
    for (i, (seg, text)) in segments.iter().zip(transcriptions.iter()).enumerate() {
        let start_s = seg.start_byte as f64 / (16000.0 * 2.0);
        let end_s = seg.end_byte as f64 / (16000.0 * 2.0);
        println!("  [{i:2}] {start_s:5.1}s - {end_s:5.1}s: \"{text}\"");
    }

    assert_eq!(
        transcriptions.len(),
        segments.len(),
        "should produce one transcription per segment"
    );
    assert!(
        transcriptions.iter().all(|t| !t.is_empty()),
        "all transcriptions should be non-empty"
    );
}

// ===========================================================================
// Test 6: Real-time pacing full transcript — LocalTransport → VAD → Whisper
// ===========================================================================

/// Feeds the 60s WAV through the LocalTransport at real-time pace with a
/// real VadBridge and Whisper STT. Prints the full transcript as it would
/// be produced in a live session.
#[cfg(feature = "whisper")]
#[tokio::test]
async fn whisper_full_transcript_realtime() {
    use std::path::PathBuf;

    use pipecat_services::whisper::WhisperSTTService;
    use pipecat_services::whisper::model::ensure_model;

    let home = std::env::var("HOME").expect("HOME not set");
    let cache_dir = PathBuf::from(home).join(".cache/pipecat-rs/whisper");
    let model_path = ensure_model("tiny.en", &cache_dir).expect("failed to get Whisper model");

    let whisper_stt = WhisperSTTService::new(
        &model_path,
        pipecat_services::settings::STTSettings {
            language: Some("en".to_string()),
            ..Default::default()
        },
    )
    .expect("failed to create WhisperSTTService");

    let in_params = TransportParams {
        audio_in_enabled: true,
        audio_in_passthrough: true,
        ..Default::default()
    };

    let input_transport = LocalAudioInputTransport::new(
        in_params,
        AudioInputSource::Buffer(Bytes::from_static(TEST_SPEECH_WAV)),
    )
    .with_format(AudioFormat::Wav)
    .with_pacing(AudioPacing::RealTime);

    let pipeline = Pipeline::new(vec![
        Box::new(input_transport),
        Box::new(VadBridgeProcessor::new()),
        Box::new(whisper_stt),
    ]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame {
            audio_in_sample_rate: 16000,
            audio_out_sample_rate: 16000,
            ..Default::default()
        }),
        Direction::Downstream,
    )
    .await;

    // Wait for the full 60s of real-time audio to be consumed.
    tokio::time::sleep(Duration::from_secs(65)).await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;
    timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap();

    let frames = down.take_frames();
    let transcriptions: Vec<String> = frames
        .iter()
        .filter_map(|f| match &f.frame {
            Frame::Transcription(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect();

    println!(
        "\n=== Real-time pipeline: {} transcriptions ===",
        transcriptions.len()
    );
    for (i, text) in transcriptions.iter().enumerate() {
        println!("  [{i:2}] \"{text}\"");
    }

    assert!(
        !transcriptions.is_empty(),
        "should produce at least one transcription"
    );
    assert!(
        transcriptions.iter().all(|t| !t.is_empty()),
        "all transcriptions should be non-empty"
    );
}
