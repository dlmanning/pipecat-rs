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
use pipecat_core::node::ProcessorNodeHandle;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_core::test_utils::*;
use pipecat_integration_tests::mock_services::*;
use pipecat_pipeline::Pipeline;
use pipecat_services::settings::STTSettings;
use pipecat_services::stt::{STTService, STTServiceState, stt_process_frame};
use pipecat_turns::{
    SpeechTimeoutUserTurnStopStrategy, UserTurnStrategies, VadUserTurnStartStrategy,
};
use serde_json::json;
use tokio::task::JoinHandle;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(10);

/// 0.5s of real speech at 16 kHz mono 16-bit PCM (8000 samples, 16000 bytes).
const TEST_SPEECH_PCM: &[u8] = include_bytes!("../fixtures/test_speech_16khz.pcm");

/// Silero at 16 kHz requires 512 samples per VAD chunk = 1024 bytes of int16 PCM.
const VAD_CHUNK_BYTES: usize = 512 * 2;

/// Number of silence chunks to send after speech (~2s at 16kHz/512 samples per chunk).
/// Generous headroom beyond the 0.2s stop threshold.
const SILENCE_CHUNKS: usize = 62;

// ---------------------------------------------------------------------------
// VadBridgeProcessor: wraps real Silero VAD and emits VAD frames
// ---------------------------------------------------------------------------

/// Test-only processor that wraps a real `VadController<SileroVadAnalyzer>`.
///
/// Converts `InputAudioRaw` frames into `VADUserStartedSpeaking`,
/// `VADUserStoppedSpeaking`, and `UserSpeaking` frames — mimicking what
/// the input transport does in production. Forwards all audio downstream
/// regardless of VAD state (matching real transport behavior where the STT
/// receives all audio and manages its own state).
#[derive(Debug)]
struct VadBridgeProcessor {
    base: ProcessorBase,
    controller: Option<VadController<SileroVadAnalyzer>>,
}

impl VadBridgeProcessor {
    fn new() -> Self {
        Self {
            base: ProcessorBase::new("VadBridge"),
            controller: None,
        }
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
                if let Some(controller) = &mut self.controller {
                    for chunk in audio_frame.audio.chunks_exact(VAD_CHUNK_BYTES) {
                        let events = controller.handle_audio(chunk);
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
                }
                // Forward all audio downstream — matching real transport behavior.
                // The STT decides whether to act on audio based on its own VAD state.
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
// Audio helpers: send realistic per-chunk frames
// ---------------------------------------------------------------------------

/// Send speech PCM as individual VAD-chunk-sized frames.
async fn feed_speech_chunks(handle: &ProcessorNodeHandle) {
    for chunk in TEST_SPEECH_PCM.chunks(VAD_CHUNK_BYTES) {
        handle
            .send(
                FrameEnvelope::new(Frame::InputAudioRaw(AudioRawFrame {
                    audio: Bytes::from(chunk.to_vec()),
                    sample_rate: 16000,
                    num_channels: 1,
                })),
                Direction::Downstream,
            )
            .await
            .unwrap();
    }
}

/// Send silence as individual VAD-chunk-sized frames.
async fn feed_silence_chunks(handle: &ProcessorNodeHandle, num_chunks: usize) {
    let silence = vec![0u8; VAD_CHUNK_BYTES];
    for _ in 0..num_chunks {
        handle
            .send(
                FrameEnvelope::new(Frame::InputAudioRaw(AudioRawFrame {
                    audio: Bytes::from(silence.clone()),
                    sample_rate: 16000,
                    num_channels: 1,
                })),
                Direction::Downstream,
            )
            .await
            .unwrap();
    }
}

// ---------------------------------------------------------------------------
// Pipeline construction helper
// ---------------------------------------------------------------------------

struct TestHarness {
    handle: ProcessorNodeHandle,
    down: FrameCollector,
    run: JoinHandle<()>,
    context: LLMContext,
}

/// Build the standard VAD → STT → Aggregators → LLM → TTS pipeline.
fn build_pipeline(stt: Box<dyn FrameProcessor>, llm_tokens: Vec<String>) -> TestHarness {
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

    let pipeline = Pipeline::new(vec![
        Box::new(VadBridgeProcessor::new()),
        stt,
        user_agg,
        Box::new(FakeLLMService::new(llm_tokens)),
        Box::new(FakeTTSService::new()),
        assistant_agg,
    ]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    TestHarness {
        handle,
        down,
        run,
        context: context_ref,
    }
}

/// Run one complete user turn: speech chunks → silence chunks → wait for TTS output.
async fn run_turn(handle: &ProcessorNodeHandle, down: &FrameCollector) {
    feed_speech_chunks(handle).await;
    feed_silence_chunks(handle, SILENCE_CHUNKS).await;
    down.wait_for_frame("TTSStopped").await;
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
// Test 1: Single turn with real audio driving real VAD through full pipeline
// ===========================================================================

#[tokio::test]
async fn single_turn_real_vad_full_pipeline() {
    let h = build_pipeline(
        Box::new(VadGatedSTT::new(vec!["hello from real vad"], 1)),
        vec!["I heard you!".to_string()],
    );

    send_frame(
        &h.handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    run_turn(&h.handle, &h.down).await;

    send_frame(
        &h.handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;
    timeout(TEST_TIMEOUT, h.run).await.unwrap().unwrap();

    // --- Frame ordering: VAD start → VAD stop → TTS cascade ---
    let names = h.down.frame_names();

    assert_order(&names, "VADUserStartedSpeaking", "VADUserStoppedSpeaking");
    assert_order(&names, "VADUserStoppedSpeaking", "TTSStarted");
    assert_order(&names, "TTSStarted", "TTSAudioRaw");
    assert_order(&names, "TTSAudioRaw", "TTSStopped");

    // UserSpeaking should fire while speaking
    assert!(
        names.contains(&"UserSpeaking".to_string()),
        "should emit UserSpeaking activity frames: {names:?}"
    );

    // --- Context accumulated correctly ---
    let messages = h.context.get_messages();
    assert_eq!(
        messages.len(),
        3,
        "expected system + user + assistant = 3 messages, got {}: {messages:?}",
        messages.len()
    );
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"], "hello from real vad");
    assert_eq!(messages[2]["role"], "assistant");
    assert!(
        !messages[2]["content"].as_str().unwrap_or("").is_empty(),
        "assistant message should have content"
    );
}

// ===========================================================================
// Test 2: Two turns — real VAD drives both, context accumulates across turns
// ===========================================================================

#[tokio::test]
async fn two_turns_real_vad_context_accumulation() {
    let h = build_pipeline(
        Box::new(VadGatedSTT::new(vec!["hello", "goodbye"], 1)),
        vec!["Got it.".to_string()],
    );

    send_frame(
        &h.handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // --- Turn 1 ---
    run_turn(&h.handle, &h.down).await;

    let msg_count = h.context.message_count();
    assert_eq!(
        msg_count, 3,
        "after turn 1: expected system + user + assistant = 3, got {msg_count}"
    );
    assert_eq!(h.context.get_messages()[1]["content"], "hello");

    // Verify turn 1 ordering
    let t1_names = h.down.frame_names();
    assert_order(
        &t1_names,
        "VADUserStartedSpeaking",
        "VADUserStoppedSpeaking",
    );
    assert_order(&t1_names, "VADUserStoppedSpeaking", "TTSStarted");
    assert_order(&t1_names, "TTSStarted", "TTSStopped");

    // Record turn 1's last frame index for cross-turn ordering
    let t1_tts_stopped = first_index(&t1_names, "TTSStopped");

    // Clear collector, preserving frame name list for cross-turn assertion
    h.down.take_frames();

    // --- Turn 2 ---
    run_turn(&h.handle, &h.down).await;

    // Shut down
    send_frame(
        &h.handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;
    timeout(TEST_TIMEOUT, h.run).await.unwrap().unwrap();

    // Verify turn 2 ordering (within turn 2's frames)
    let t2_names = h.down.frame_names();
    assert_order(
        &t2_names,
        "VADUserStartedSpeaking",
        "VADUserStoppedSpeaking",
    );
    assert_order(&t2_names, "VADUserStoppedSpeaking", "TTSStarted");
    assert_order(&t2_names, "TTSStarted", "TTSStopped");

    // Turn 1 completed before turn 2 started (TTSStopped was in turn 1's frames,
    // and VADUserStartedSpeaking is the first event of turn 2)
    assert!(
        t1_tts_stopped < t1_names.len(),
        "turn 1 should have completed before take_frames()"
    );

    // --- Final context: 5 messages ---
    let messages = h.context.get_messages();
    assert_eq!(
        messages.len(),
        5,
        "expected system + 2*(user + assistant) = 5, got {}: {messages:?}",
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
