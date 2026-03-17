use std::time::Duration;

use pipecat_context::{LLMContext, LLMContextAggregatorPair, LLMUserAggregatorParams};
use pipecat_core::frame::*;
use pipecat_core::processor::FrameProcessor;
use pipecat_core::test_utils::*;
use pipecat_integration_tests::helpers::*;
use pipecat_integration_tests::mock_services::*;
use pipecat_pipeline::Pipeline;
use pipecat_turns::{
    SpeechTimeoutUserTurnStopStrategy, UserTurnStrategies, VadUserTurnStartStrategy,
};
use serde_json::json;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(10);

fn make_pipeline_with_aggregators(context: LLMContext) -> (Box<dyn FrameProcessor>, LLMContext) {
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

    let stt = FakeSTTService::new("hello", 1);
    let llm = FakeLLMService::new(vec!["Hi there.".to_string()]);
    let tts = FakeTTSService::new();

    // STT before UserAggregator so transcriptions reach the aggregator
    let pipeline = Pipeline::new(vec![
        Box::new(stt),
        user_agg,
        Box::new(llm),
        Box::new(tts),
        assistant_agg,
    ]);

    (Box::new(pipeline), context_ref)
}

// ---------------------------------------------------------------------------
// Helper: run one full turn (VAD start → audio → VAD stop → wait for timeout)
// ---------------------------------------------------------------------------

async fn run_turn(handle: &pipecat_core::node::ProcessorNodeHandle, down: &FrameCollector) {
    // Clear any previously collected frames so we can wait for fresh output
    down.take_frames();

    handle
        .send(make_vad_started(), Direction::Downstream)
        .await
        .unwrap();

    handle
        .send(
            make_input_audio_frame(&[0i16; 160], 16000),
            Direction::Downstream,
        )
        .await
        .unwrap();

    handle
        .send(make_vad_stopped(), Direction::Downstream)
        .await
        .unwrap();

    // Wait for speech timeout (50ms) to fire — this is a real time-based behavior
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Wait for the turn's TTS output to fully propagate through the pipeline
    down.wait_for_frame("TTSStopped").await;
}

// ---------------------------------------------------------------------------
// 5d-1: Interruption stops bot audio and allows new turn
// ---------------------------------------------------------------------------

#[tokio::test]
async fn interruption_allows_new_turn() {
    let context = LLMContext::new(vec![json!({"role": "system", "content": "test"})]);
    let (pipeline, context_ref) = make_pipeline_with_aggregators(context);
    let (node, handle, down_rx, _up_rx) = make_node(pipeline);
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // --- Turn 1: normal flow ---
    run_turn(&handle, &down).await;

    // Verify Turn 1 produced TTS output
    let names = down.frame_names();
    assert!(
        names.contains(&"TTSAudioRaw".to_string()),
        "Turn 1 should produce TTS audio: {names:?}"
    );

    // --- Turn 2: user speaks again (triggers interruption of bot speech) ---
    run_turn(&handle, &down).await;

    // Verify Turn 2 also produced TTS output
    let names_turn2 = down.frame_names();
    assert!(
        names_turn2.contains(&"TTSAudioRaw".to_string()),
        "Turn 2 should produce TTS audio after interruption: {names_turn2:?}"
    );

    // Turn 2 should have generated an interruption (from turn start)
    assert!(
        names_turn2.contains(&"Interruption".to_string()),
        "Turn 2 start should broadcast interruption: {names_turn2:?}"
    );

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    // Context should have system + 2*(user+assistant) = 5 messages
    let msg_count = context_ref.message_count();
    assert!(
        msg_count >= 5,
        "context should have system + 2*(user+assistant) messages, got {msg_count}"
    );
}

// ---------------------------------------------------------------------------
// 5d-2: Interruption frame passes through pipeline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn interruption_passes_through_pipeline() {
    let stt = FakeSTTService::new("hello", 100);
    let llm = FakeLLMService::new(vec!["response".to_string()]);
    let tts = FakeTTSService::new();

    let pipeline = Pipeline::new(vec![Box::new(stt), Box::new(llm), Box::new(tts)]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    send_frame(
        &handle,
        Frame::Interruption(InterruptionFrame),
        Direction::Downstream,
    )
    .await;

    down.wait_for_frame("Interruption").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();

    assert!(
        names.contains(&"Interruption".to_string()),
        "Interruption should pass through all services in the pipeline: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// 5d-3: Interruption drains interruptible frames, preserves uninterruptible
// ---------------------------------------------------------------------------

#[tokio::test]
async fn interruption_drains_interruptible_preserves_uninterruptible() {
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(PassthroughProcessor::new()));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Queue several interruptible frames + one uninterruptible (End) rapidly
    for i in 0..5 {
        handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new(format!("msg{i}")))),
                Direction::Downstream,
            )
            .await
            .unwrap();
    }
    // Queue an uninterruptible frame
    handle
        .send(
            FrameEnvelope::new(Frame::End(EndFrame::default())),
            Direction::Downstream,
        )
        .await
        .unwrap();

    // Send interruption (system priority — processed before normal frames)
    handle
        .send(
            FrameEnvelope::new(Frame::Interruption(InterruptionFrame)),
            Direction::Downstream,
        )
        .await
        .unwrap();

    // Wait for the interruption to propagate through
    down.wait_for_frame("Interruption").await;

    // Also wait for End (uninterruptible) — it should survive
    down.wait_for_frame("End").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();

    // End (uninterruptible) should be preserved
    assert!(
        names.contains(&"End".to_string()),
        "End frame (uninterruptible) should survive interruption: {names:?}"
    );

    // Interruption itself should be present
    assert!(
        names.contains(&"Interruption".to_string()),
        "Interruption frame should be present: {names:?}"
    );
}
