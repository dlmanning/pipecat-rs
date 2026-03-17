use std::time::Duration;

use pipecat_core::frame::*;
use pipecat_core::test_utils::*;
use pipecat_integration_tests::mock_services::*;
use pipecat_pipeline::Pipeline;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// 5b-1: SPS audio round-trip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sps_audio_round_trip() {
    let realtime = FakeRealtimeProcessor::new(3);
    let pipeline = Pipeline::new(vec![Box::new(realtime)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send 3 audio chunks to trigger response
    for _ in 0..3 {
        send_frame(
            &handle,
            Frame::InputAudioRaw(AudioRawFrame {
                audio: bytes::Bytes::from(vec![0u8; 320]),
                sample_rate: 16000,
                num_channels: 1,
            }),
            Direction::Downstream,
        )
        .await;
    }

    down.wait_for_frame("TTSAudioRaw").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();

    assert!(
        names.contains(&"TTSAudioRaw".to_string()),
        "SPS should emit TTSAudioRaw after receiving enough audio chunks: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// 5b-2: SPS interruption resets accumulation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sps_interruption_resets() {
    let realtime = FakeRealtimeProcessor::new(3);
    let pipeline = Pipeline::new(vec![Box::new(realtime)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send 2 audio chunks (not enough to trigger response)
    for _ in 0..2 {
        send_frame(
            &handle,
            Frame::InputAudioRaw(AudioRawFrame {
                audio: bytes::Bytes::from(vec![0u8; 320]),
                sample_rate: 16000,
                num_channels: 1,
            }),
            Direction::Downstream,
        )
        .await;
    }

    // Interrupt — should reset chunk counter
    send_frame(
        &handle,
        Frame::Interruption(InterruptionFrame),
        Direction::Downstream,
    )
    .await;

    down.wait_for_frame("Interruption").await;

    // Send 2 more chunks — still not enough (counter was reset)
    for _ in 0..2 {
        send_frame(
            &handle,
            Frame::InputAudioRaw(AudioRawFrame {
                audio: bytes::Bytes::from(vec![0u8; 320]),
                sample_rate: 16000,
                num_channels: 1,
            }),
            Direction::Downstream,
        )
        .await;
    }

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();

    // Should NOT have audio output (reset prevented threshold)
    assert!(
        !names.contains(&"TTSAudioRaw".to_string()),
        "interruption should reset accumulation, no audio should be emitted: {names:?}"
    );

    // Interruption frame should pass through
    assert!(
        names.contains(&"Interruption".to_string()),
        "interruption should pass through: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// 5b-3: SPS consumes audio (does not forward InputAudioRaw)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sps_consumes_audio() {
    let realtime = FakeRealtimeProcessor::new(100); // high threshold, won't emit
    let pipeline = Pipeline::new(vec![Box::new(realtime)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send audio
    send_frame(
        &handle,
        Frame::InputAudioRaw(AudioRawFrame {
            audio: bytes::Bytes::from(vec![0u8; 320]),
            sample_rate: 16000,
            num_channels: 1,
        }),
        Direction::Downstream,
    )
    .await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    // Wait for Cancel to propagate to confirm everything is through
    down.wait_for_frame("Cancel").await;

    let names = down.frame_names();

    // InputAudioRaw should NOT appear in output (consumed by FakeRealtime)
    assert!(
        !names.contains(&"InputAudioRaw".to_string()),
        "FakeRealtime should consume InputAudioRaw, not forward it: {names:?}"
    );

    // Lifecycle frames should still pass through
    assert!(names.contains(&"Start".to_string()));
    assert!(names.contains(&"Cancel".to_string()));
}
