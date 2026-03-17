use std::time::Duration;

use pipecat_context::{LLMContext, LLMContextAggregatorPair, LLMUserAggregatorParams};
use pipecat_core::frame::*;
use pipecat_core::test_utils::*;
use pipecat_integration_tests::helpers::*;
use pipecat_integration_tests::mock_services::*;
use pipecat_pipeline::Pipeline;
use pipecat_turns::{
    SpeechTimeoutUserTurnStopStrategy, UserTurnStrategies, VadUserTurnStartStrategy,
};
use serde_json::json;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Build a pipeline with STT -> UserAggregator -> LLM -> TTS -> AssistantAggregator
/// using VAD-only start strategy and a given speech timeout.
///
/// Returns the pipeline and a reference to the shared LLM context.
fn make_turn_pipeline(
    stt_text: &str,
    stt_chunks: usize,
    speech_timeout_secs: f64,
) -> (Pipeline, LLMContext) {
    let context = LLMContext::new(vec![
        json!({"role": "system", "content": "You are helpful."}),
    ]);
    let context_ref = context.clone();

    let params = LLMUserAggregatorParams {
        user_turn_strategies: UserTurnStrategies {
            start: vec![Box::new(VadUserTurnStartStrategy::new())],
            stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(
                speech_timeout_secs,
            ))],
        },
        user_turn_stop_timeout: Duration::from_secs(5),
    };
    let pair = LLMContextAggregatorPair::new(context, params);
    let (user_agg, assistant_agg) = pair.into_processors();

    let stt = FakeSTTService::new(stt_text, stt_chunks);
    let llm = FakeLLMService::new(vec!["response".to_string()]);
    let tts = FakeTTSService::new();

    let pipeline = Pipeline::new(vec![
        Box::new(stt),
        user_agg,
        Box::new(llm),
        Box::new(tts),
        assistant_agg,
    ]);

    (pipeline, context_ref)
}

// ---------------------------------------------------------------------------
// Test 1: Partial (non-finalized) transcription with VAD should not
//         contribute text to the context
// ---------------------------------------------------------------------------

/// When a turn is started via VAD and partial (non-finalized) transcription
/// arrives, only the finalized transcription should contribute to the context
/// message. This sends VAD start + partial transcription + finalized
/// transcription + VAD stop, and verifies the context message contains only
/// the finalized text.
#[tokio::test]
async fn partial_transcription_text_not_in_context() {
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

    let llm = FakeLLMService::new(vec!["response".to_string()]);
    let tts = FakeTTSService::new();

    let pipeline = Pipeline::new(vec![user_agg, Box::new(llm), Box::new(tts), assistant_agg]);

    let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(pipeline));
    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // VAD start -> triggers turn start
    handle
        .send(make_vad_started(), Direction::Downstream)
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    // Send a non-finalized (partial) transcription — should be consumed but
    // NOT accumulated into the context message
    handle
        .send(
            FrameEnvelope::new(Frame::InterimTranscription(InterimTranscriptionFrame {
                text: "partial stuff".to_string(),
                user_id: "user".to_string(),
                timestamp: None,
                language: None,
                result: None,
            })),
            Direction::Downstream,
        )
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    // Send a finalized transcription — this IS the real user text
    handle
        .send(
            FrameEnvelope::new(Frame::Transcription(TranscriptionFrame {
                text: "hello world".to_string(),
                user_id: "user".to_string(),
                timestamp: None,
                language: None,
                finalized: true,
                result: None,
            })),
            Direction::Downstream,
        )
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    // VAD stop -> speech timeout starts
    handle
        .send(make_vad_stopped(), Direction::Downstream)
        .await
        .unwrap();

    // Wait for speech timeout (50ms) + propagation
    tokio::time::sleep(Duration::from_millis(150)).await;

    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = drain_rx(&mut down_rx).await;
    let names = frame_names(&frames);

    // The turn should have completed and emitted an LLMContext frame
    assert!(
        names.contains(&"LLMContext".to_string()),
        "Finalized transcription with VAD should produce LLMContext frame: {names:?}"
    );

    // The context should contain the finalized text but NOT the partial text
    let msgs = context_ref.get_messages();
    let user_msgs: Vec<_> = msgs.iter().filter(|m| m["role"] == "user").collect();
    assert!(
        !user_msgs.is_empty(),
        "Expected at least one user message in context"
    );

    let all_user_text: String = user_msgs
        .iter()
        .map(|m| m["content"].as_str().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all_user_text.contains("hello world"),
        "Context should contain finalized text 'hello world', got: {all_user_text}"
    );
    assert!(
        !all_user_text.contains("partial stuff"),
        "Context should NOT contain partial text 'partial stuff', got: {all_user_text}"
    );
}

/// Verify that a finalized transcription WITH VAD does trigger a turn
/// (control test to confirm the pipeline is wired correctly).
#[tokio::test]
async fn finalized_transcription_with_vad_triggers_turn() {
    let (pipeline, context_ref) = make_turn_pipeline("hello", 1, 0.05);

    let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(pipeline));
    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // VAD start -> triggers turn start
    handle
        .send(make_vad_started(), Direction::Downstream)
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    // Send audio to trigger STT (FakeSTT emits finalized transcription after 1 chunk)
    handle
        .send(
            make_input_audio_frame(&[0i16; 160], 16000),
            Direction::Downstream,
        )
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    // VAD stop -> speech timeout starts
    handle
        .send(make_vad_stopped(), Direction::Downstream)
        .await
        .unwrap();

    // Wait for speech timeout (50ms) + propagation
    tokio::time::sleep(Duration::from_millis(150)).await;

    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = drain_rx(&mut down_rx).await;
    let names = frame_names(&frames);

    // This IS the happy path: VAD + finalized transcription -> full pipeline
    // Verify LLMContext was emitted (the semantic signal that the LLM gets called)
    assert!(
        names.contains(&"LLMContext".to_string()),
        "Finalized transcription with VAD should produce LLMContext frame: {names:?}"
    );

    // Verify TTS audio was produced (full pipeline fired)
    assert!(
        names.contains(&"TTSAudioRaw".to_string()),
        "Finalized transcription with VAD should produce TTS audio: {names:?}"
    );

    // Context should have system + user + assistant messages
    let msg_count = context_ref.message_count();
    assert!(
        msg_count >= 3,
        "Context should have system + user + assistant, got {msg_count}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Rapid VAD start/stop/start should not produce duplicate LLMContext
//         emissions (the real failure mode is "LLM gets called twice")
// ---------------------------------------------------------------------------

/// Sending VADUserStartedSpeaking -> VADUserStoppedSpeaking -> VADUserStartedSpeaking
/// in quick succession should not produce two LLMContext frames,
/// because the turn hasn't actually stopped yet (speech timeout hasn't fired).
///
/// Uses a very long speech timeout (10s) so it's impossible for the timeout
/// to fire during the test. The point is to test deduplication, not timeouts.
#[tokio::test]
async fn rapid_vad_start_stop_start_no_duplicate_llm_calls() {
    let (pipeline, _context_ref) = make_turn_pipeline("hello", 1, 10.0);

    let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(pipeline));
    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Rapid sequence: VAD start -> VAD stop -> VAD start
    // Speech timeout is 10s, so the turn won't have stopped between these
    handle
        .send(make_vad_started(), Direction::Downstream)
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    handle
        .send(make_vad_stopped(), Direction::Downstream)
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    // Second VAD start — turn is still active (speech timeout hasn't fired)
    handle
        .send(make_vad_started(), Direction::Downstream)
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = drain_rx(&mut down_rx).await;

    // Count LLMContext frames — the semantic signal that the LLM gets called.
    // With deduplication, the turn never stopped so no LLMContext should be emitted
    // (or at most one from the Cancel frame flushing).
    let llm_context_count = frames
        .iter()
        .filter(|f| matches!(&f.frame, Frame::LLMContext(_)))
        .count();
    assert!(
        llm_context_count <= 1,
        "Expected at most 1 LLMContext frame (from cancel flush), got {llm_context_count}. \
         Rapid VAD start/stop/start should not produce duplicate LLM calls."
    );
}

/// Similar to the above but with a longer sequence:
/// VAD start -> VAD stop -> VAD start -> VAD stop -> VAD start
/// All happening faster than the speech timeout. Should still see at most one
/// LLMContext frame (since the turn never actually stopped).
///
/// Uses a very long speech timeout (10s) so it's impossible for the timeout
/// to fire during the test.
#[tokio::test]
async fn triple_vad_bounce_single_llm_call() {
    let (pipeline, _context_ref) = make_turn_pipeline("hello", 1, 10.0);

    let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(pipeline));
    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Rapid bouncing: start -> stop -> start -> stop -> start
    for _ in 0..3 {
        handle
            .send(make_vad_started(), Direction::Downstream)
            .await
            .unwrap();
        tokio::time::sleep(SETTLE).await;

        handle
            .send(make_vad_stopped(), Direction::Downstream)
            .await
            .unwrap();
        tokio::time::sleep(SETTLE).await;
    }

    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = drain_rx(&mut down_rx).await;

    // Count LLMContext frames — the real failure mode of duplicate turns
    // is the LLM getting called twice. The turn never stopped (10s timeout),
    // so no LLMContext should be emitted during the bounce sequence.
    let llm_context_count = frames
        .iter()
        .filter(|f| matches!(&f.frame, Frame::LLMContext(_)))
        .count();
    assert!(
        llm_context_count <= 1,
        "Expected at most 1 LLMContext frame across triple VAD bounce, got {llm_context_count}"
    );

    // Also verify UserStartedSpeaking deduplication (secondary check)
    let user_started_count = frames
        .iter()
        .filter(|f| matches!(&f.frame, Frame::UserStartedSpeaking(_)))
        .count();
    assert_eq!(
        user_started_count, 1,
        "Expected exactly 1 UserStartedSpeaking across triple VAD bounce, got {user_started_count}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: New turn after a completed turn
// ---------------------------------------------------------------------------

/// Verifies that after a complete turn cycle (start -> transcription -> stop -> timeout),
/// a new VAD start properly begins a NEW turn. This confirms that deduplication
/// resets after a turn actually completes.
///
/// Uses a short speech timeout (50ms) and verifies context message counts after
/// each turn.
#[tokio::test]
async fn new_turn_after_completed_turn() {
    let (pipeline, context_ref) = make_turn_pipeline("hello", 1, 0.05);

    let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(pipeline));
    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // --- First turn ---
    handle
        .send(make_vad_started(), Direction::Downstream)
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    handle
        .send(
            make_input_audio_frame(&[0i16; 160], 16000),
            Direction::Downstream,
        )
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    handle
        .send(make_vad_stopped(), Direction::Downstream)
        .await
        .unwrap();

    // Wait for speech timeout to fire and turn to complete
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Verify context after first turn: system + user + assistant = 3 messages
    let msg_count_after_first = context_ref.message_count();
    assert!(
        msg_count_after_first >= 3,
        "After first turn, context should have system + user + assistant (>= 3), got {msg_count_after_first}"
    );

    // --- Second turn ---
    handle
        .send(make_vad_started(), Direction::Downstream)
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    handle
        .send(
            make_input_audio_frame(&[0i16; 160], 16000),
            Direction::Downstream,
        )
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;

    handle
        .send(make_vad_stopped(), Direction::Downstream)
        .await
        .unwrap();

    // Wait for second turn to complete
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Verify context after second turn: should have more messages
    let msg_count_after_second = context_ref.message_count();
    assert!(
        msg_count_after_second > msg_count_after_first,
        "After second turn, context should have more messages than after first turn. \
         First: {msg_count_after_first}, Second: {msg_count_after_second}"
    );

    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = drain_rx(&mut down_rx).await;

    // Should see TWO LLMContext frames — one per complete turn
    let llm_context_count = frames
        .iter()
        .filter(|f| matches!(&f.frame, Frame::LLMContext(_)))
        .count();
    assert!(
        llm_context_count >= 2,
        "Expected at least 2 LLMContext frames (one per completed turn), got {llm_context_count}"
    );

    // Context should have accumulated messages from both turns:
    // system(1) + user(1) + assistant(1) + user(1) + assistant(1) = 5
    let final_msg_count = context_ref.message_count();
    assert!(
        final_msg_count >= 5,
        "Context should have system + 2*(user + assistant) = 5 messages, got {final_msg_count}"
    );
}
