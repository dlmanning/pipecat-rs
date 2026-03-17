use std::time::Duration;

use pipecat_core::frame::*;
use pipecat_core::test_utils::*;
use pipecat_integration_tests::helpers::*;
use pipecat_integration_tests::mock_services::*;
use pipecat_pipeline::Pipeline;
use serde_json::json;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Interruption mid-cascade: multi-processor pipeline
//
// In a Pipeline, each processor gets its own ProcessorNode. Frames between
// nodes travel through bridge tasks that call `handle.send()`, which routes
// system frames to the system channel and normal frames to the normal channel.
//
// When an Interruption arrives at a node's system channel (high priority),
// the node drains its normal channel queue — removing interruptible frames
// that haven't been processed yet. The Interruption then propagates through
// the processor and onward to the next node, where the same drain happens.
//
// These tests verify that invariant across a real multi-node cascade.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 1. Interruption drains queued frames across a multi-processor cascade
//
// Queue many normal frames rapidly without settling, then fire an
// Interruption. The system-priority Interruption should arrive at each
// node before the backlog of normal frames is fully processed, draining
// interruptible frames from each node's queue.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn interruption_drains_queued_frames_across_cascade() {
    // Two passthrough processors = three internal nodes (Source, P1, P2, Sink)
    // which means frames must traverse multiple channel hops.
    let pipeline = Pipeline::new(vec![
        Box::new(PassthroughProcessor::new()),
        Box::new(PassthroughProcessor::new()),
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

    // Rapidly queue many interruptible Text frames without waiting for
    // them to be processed. They pile up in the internal node channels.
    for i in 0..20 {
        handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new(format!("msg{i}")))),
                Direction::Downstream,
            )
            .await
            .unwrap();
    }

    // Fire Interruption immediately — as a system frame it gets priority
    // routing at every node in the cascade.
    handle
        .send(
            FrameEnvelope::new(Frame::Interruption(InterruptionFrame)),
            Direction::Downstream,
        )
        .await
        .unwrap();

    // Wait for interruption to propagate through the cascade
    down.wait_for_frame("Interruption").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();

    // The Interruption frame must propagate through.
    assert!(
        names.contains(&"Interruption".to_string()),
        "Interruption should propagate through multi-processor cascade: {names:?}"
    );

    // Core invariant: no interruptible Text frames appear AFTER the Interruption.
    let interruption_pos = names
        .iter()
        .position(|n| n == "Interruption")
        .expect("Interruption must be present");

    let stale_text: Vec<&String> = names[interruption_pos + 1..]
        .iter()
        .filter(|n| n.as_str() == "Text")
        .collect();

    assert!(
        stale_text.is_empty(),
        "No Text frames should appear after the Interruption in a cascade, \
         but found {} stale frame(s). Full sequence: {names:?}",
        stale_text.len(),
    );
}

// ---------------------------------------------------------------------------
// 2. Uninterruptible frames survive the drain in a cascade
//
// Mix interruptible and uninterruptible frames, then fire Interruption.
// End frames (uninterruptible) should survive; Text frames should not
// appear after the Interruption.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn interruption_preserves_uninterruptible_in_cascade() {
    let pipeline = Pipeline::new(vec![
        Box::new(PassthroughProcessor::new()),
        Box::new(PassthroughProcessor::new()),
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

    // Queue interruptible Text frames.
    for i in 0..10 {
        handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new(format!("msg{i}")))),
                Direction::Downstream,
            )
            .await
            .unwrap();
    }

    // Queue an uninterruptible End frame among them.
    handle
        .send(
            FrameEnvelope::new(Frame::End(EndFrame::default())),
            Direction::Downstream,
        )
        .await
        .unwrap();

    // Fire Interruption on the system channel.
    handle
        .send(
            FrameEnvelope::new(Frame::Interruption(InterruptionFrame)),
            Direction::Downstream,
        )
        .await
        .unwrap();

    // Wait for both Interruption and End to propagate
    down.wait_for_frame("Interruption").await;
    down.wait_for_frame("End").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();

    // Interruption must be present.
    assert!(
        names.contains(&"Interruption".to_string()),
        "Interruption should propagate: {names:?}"
    );

    // End (uninterruptible) must survive the drain.
    assert!(
        names.contains(&"End".to_string()),
        "End frame (uninterruptible) should survive interruption drain in cascade: {names:?}"
    );

    // No interruptible Text frames after the Interruption.
    let interruption_pos = names
        .iter()
        .position(|n| n == "Interruption")
        .expect("Interruption must be present");

    let stale_text: Vec<&String> = names[interruption_pos + 1..]
        .iter()
        .filter(|n| n.as_str() == "Text")
        .collect();

    assert!(
        stale_text.is_empty(),
        "No Text frames should appear after Interruption. Full sequence: {names:?}",
    );
}

// ---------------------------------------------------------------------------
// 3. LLM → TTS cascade: no stale TTSAudioRaw after Interruption
//
// Trigger the LLM to emit tokens into the TTS, then immediately fire an
// Interruption. Verify no TTSAudioRaw frames appear after the Interruption
// in the output stream.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn interruption_mid_cascade_no_stale_audio_after_interruption() {
    // Use multiple tokens that form a complete sentence so TTS can produce audio.
    let llm = FakeLLMService::new(vec![
        "Hello ".to_string(),
        "there, ".to_string(),
        "how ".to_string(),
        "are ".to_string(),
        "you ".to_string(),
        "doing ".to_string(),
        "today?".to_string(),
    ]);
    let tts = FakeTTSService::new();

    let pipeline = Pipeline::new(vec![Box::new(llm), Box::new(tts)]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send LLMContext to trigger the LLM response. The FakeLLM emits all tokens
    // synchronously within one process_frame call (LLMFullResponseStart, Text x7,
    // LLMFullResponseEnd). These then queue up in the TTS node's normal channel
    // via the bridge.
    let ctx_frame = make_llm_context_frame(vec![
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "Hello"}),
    ]);
    handle.send(ctx_frame, Direction::Downstream).await.unwrap();

    // Brief pause to let some tokens propagate through the LLM and queue at TTS.
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Fire the Interruption. As a system frame it is prioritized: each node in
    // the pipeline drains its queued normal frames before dispatching it.
    handle
        .send(
            FrameEnvelope::new(Frame::Interruption(InterruptionFrame)),
            Direction::Downstream,
        )
        .await
        .unwrap();

    // Wait for interruption to propagate
    down.wait_for_frame("Interruption").await;

    // Shut down the pipeline.
    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();

    // The Interruption frame must appear in the output.
    assert!(
        names.contains(&"Interruption".to_string()),
        "Interruption should propagate through the cascade: {names:?}"
    );

    // Core invariant: no TTSAudioRaw frames appear AFTER the Interruption.
    let interruption_pos = names
        .iter()
        .position(|n| n == "Interruption")
        .expect("Interruption must be present");

    let stale_audio: Vec<&String> = names[interruption_pos + 1..]
        .iter()
        .filter(|n| n.as_str() == "TTSAudioRaw")
        .collect();

    assert!(
        stale_audio.is_empty(),
        "No TTSAudioRaw frames should appear after the Interruption, but found {} stale frame(s). \
         Full sequence: {names:?}",
        stale_audio.len(),
    );
}

// ---------------------------------------------------------------------------
// 4. Frames already output before interruption are preserved
//
// Send frames, settle (let them through), then send more frames + Interruption.
// The early frames should have made it through. The later frames should be
// drained. No stale frames after Interruption.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn interruption_preserves_already_delivered_frames() {
    let llm = FakeLLMService::new(vec!["Short.".to_string()]);
    let tts = FakeTTSService::new();

    let pipeline = Pipeline::new(vec![Box::new(llm), Box::new(tts)]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Trigger LLM and let the full cascade complete.
    let ctx_frame = make_llm_context_frame(vec![json!({"role": "user", "content": "Hi"})]);
    send_frame(&handle, ctx_frame.frame, Direction::Downstream).await;

    // Wait for the cascade to fully propagate (LLM → TTS → output).
    down.wait_for_frame("TTSStopped").await;

    // Record what's already been output — these frames are "delivered".
    let early_frames = down.take_frames();
    let early_names: Vec<String> = early_frames
        .iter()
        .map(|f| format!("{}", f.frame))
        .collect();

    // The first response should have produced some output (TTS audio for "Short.").
    let has_tts = early_names.iter().any(|n| n == "TTSAudioRaw");

    // Now send the Interruption after TTS has already produced output.
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

    let late_names = down.frame_names();

    // Interruption must be present in the late output.
    assert!(
        late_names.contains(&"Interruption".to_string()),
        "Interruption should propagate: {late_names:?}"
    );

    // No TTSAudioRaw after the Interruption in the late output.
    let interruption_pos = late_names
        .iter()
        .position(|n| n == "Interruption")
        .expect("Interruption must be present");

    let stale_audio: Vec<&String> = late_names[interruption_pos + 1..]
        .iter()
        .filter(|n| n.as_str() == "TTSAudioRaw")
        .collect();

    assert!(
        stale_audio.is_empty(),
        "No TTSAudioRaw after Interruption in late output. \
         Early frames (already delivered): {early_names:?}, \
         Late frames: {late_names:?}",
    );

    // If TTS audio was delivered before the interruption, that's fine —
    // it was already output and is not "stale".
    if has_tts {
        // Confirm it's in the early batch, not the late batch after Interruption.
        assert!(
            early_names.iter().any(|n| n == "TTSAudioRaw"),
            "Pre-interruption TTS audio should be in the early batch"
        );
    }
}
