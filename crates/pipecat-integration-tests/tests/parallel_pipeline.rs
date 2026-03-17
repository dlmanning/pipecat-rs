use std::time::Duration;

use pipecat_core::frame::*;
use pipecat_core::processor::FrameProcessor;
use pipecat_core::test_utils::*;
use pipecat_pipeline::{ParallelPipeline, Pipeline};
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// 1: Frame ID deduplication — two passthrough branches
// ---------------------------------------------------------------------------
// Both branches forward the SAME envelope (same frame ID). The coordinator
// should deduplicate so each frame appears exactly once, not twice.

#[tokio::test]
async fn frame_id_deduplication_with_passthrough_branches() {
    let parallel = ParallelPipeline::new(vec![
        vec![Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>],
        vec![Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>],
    ]);

    let pipeline = Pipeline::new(vec![Box::new(parallel)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Wait for Start to fully propagate through the parallel pipeline
    // (lifecycle sync completes once all branches process it). This ensures
    // the coordinator's seen_ids set has been cleared before data frames arrive.
    down.wait_for_frame("Start").await;

    // Send multiple text frames — each will be cloned to both branches but
    // should emerge only once from the coordinator due to ID-based dedup.
    for word in ["alpha", "bravo", "charlie"] {
        send_frame(
            &handle,
            Frame::Text(TextFrame::new(word)),
            Direction::Downstream,
        )
        .await;
    }

    // Wait for the last text frame to propagate through.
    down.wait_for(|f| matches!(f, Frame::Text(t) if t.text == "charlie"))
        .await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = down.frames();
    let names = down.frame_names();

    // Each text frame should appear exactly once (dedup filters the duplicate).
    let text_values: Vec<&str> = frames
        .iter()
        .filter_map(|f| match &f.frame {
            Frame::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(
        text_values.len(),
        3,
        "dedup should produce exactly 3 text frames (not 6): {text_values:?}"
    );
    assert!(text_values.contains(&"alpha"), "{text_values:?}");
    assert!(text_values.contains(&"bravo"), "{text_values:?}");
    assert!(text_values.contains(&"charlie"), "{text_values:?}");

    // Lifecycle frames should also be deduplicated.
    assert_eq!(
        names.iter().filter(|n| *n == "Start").count(),
        1,
        "Start should appear exactly once: {names:?}"
    );
    assert_eq!(
        names.iter().filter(|n| *n == "Cancel").count(),
        1,
        "Cancel should appear exactly once: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// 2: Transforming branches produce independent output (no false dedup)
// ---------------------------------------------------------------------------
// One branch uppercases text (creates new frames with new IDs), while the
// other passes through unchanged. Both outputs should appear because they
// have different frame IDs.

#[tokio::test]
async fn transforming_branches_produce_independent_output() {
    let parallel = ParallelPipeline::new(vec![
        vec![Box::new(UppercaseProcessor::new()) as Box<dyn FrameProcessor>],
        vec![Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>],
    ]);

    let pipeline = Pipeline::new(vec![Box::new(parallel)]);
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
        Frame::Text(TextFrame::new("hello")),
        Direction::Downstream,
    )
    .await;

    send_frame(
        &handle,
        Frame::Text(TextFrame::new("world")),
        Direction::Downstream,
    )
    .await;

    // Wait for both branches to produce output for "world":
    // - Passthrough: "world" (original ID)
    // - Uppercase: "WORLD" (new ID)
    // Wait for the uppercased version of the last word to confirm all frames are through.
    down.wait_for(|f| matches!(f, Frame::Text(t) if t.text == "WORLD"))
        .await;
    // Also wait for the passthrough version
    down.wait_for(|f| matches!(f, Frame::Text(t) if t.text == "world"))
        .await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = down.frames();

    let text_values: Vec<&str> = frames
        .iter()
        .filter_map(|f| match &f.frame {
            Frame::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();

    // Uppercase branch creates new IDs, passthrough keeps original IDs.
    // Dedup correctly distinguishes them: 4 text frames total.
    assert_eq!(
        text_values.len(),
        4,
        "should have 4 text frames (2 original + 2 uppercased): {text_values:?}"
    );
    assert!(text_values.contains(&"hello"), "{text_values:?}");
    assert!(text_values.contains(&"HELLO"), "{text_values:?}");
    assert!(text_values.contains(&"world"), "{text_values:?}");
    assert!(text_values.contains(&"WORLD"), "{text_values:?}");
}

// ---------------------------------------------------------------------------
// 3: Lifecycle ordering — Start before data, data before Cancel
// ---------------------------------------------------------------------------
// Verifies the coordinator's barrier-sync behavior: Start is emitted before
// any data frames, and Cancel is emitted after all data frames.

#[tokio::test]
async fn lifecycle_ordering_start_before_data_cancel_after() {
    let parallel = ParallelPipeline::new(vec![
        vec![
            Box::new(UppercaseProcessor::new()) as Box<dyn FrameProcessor>,
            Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>,
        ],
        vec![
            Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>,
            Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>,
        ],
    ]);

    let pipeline = Pipeline::new(vec![Box::new(parallel)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send several text frames to have enough data to verify ordering.
    for word in ["one", "two", "three"] {
        send_frame(
            &handle,
            Frame::Text(TextFrame::new(word)),
            Direction::Downstream,
        )
        .await;
    }

    // Wait for all text frames from both branches to propagate.
    // Uppercase branch creates new frames so we get 6 text frames total.
    // Wait for the last uppercased word to ensure everything is through.
    down.wait_for(|f| matches!(f, Frame::Text(t) if t.text == "THREE"))
        .await;
    down.wait_for(|f| matches!(f, Frame::Text(t) if t.text == "three"))
        .await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    // Wait a moment for Cancel to be collected
    down.wait_for_frame("Cancel").await;

    let frames = down.frames();
    let names = down.frame_names();

    // Start must be the very first frame.
    assert_eq!(
        names.first().unwrap(),
        "Start",
        "first frame must be Start: {names:?}"
    );

    // Cancel must be the very last frame.
    assert_eq!(
        names.last().unwrap(),
        "Cancel",
        "last frame must be Cancel: {names:?}"
    );

    // All Text frames must be strictly between Start and Cancel.
    let start_pos = names
        .iter()
        .position(|n| n == "Start")
        .expect("Start present");
    let cancel_pos = names
        .iter()
        .position(|n| n == "Cancel")
        .expect("Cancel present");
    let first_text_pos = names
        .iter()
        .position(|n| n == "Text")
        .expect("at least one Text present");
    let last_text_pos = names
        .iter()
        .rposition(|n| n == "Text")
        .expect("at least one Text present");

    assert!(
        start_pos < first_text_pos,
        "Start ({start_pos}) must precede first Text ({first_text_pos}): {names:?}"
    );
    assert!(
        last_text_pos < cancel_pos,
        "last Text ({last_text_pos}) must precede Cancel ({cancel_pos}): {names:?}"
    );

    // Verify exactly one Start and one Cancel (lifecycle dedup).
    assert_eq!(names.iter().filter(|n| *n == "Start").count(), 1);
    assert_eq!(names.iter().filter(|n| *n == "Cancel").count(), 1);

    // Verify we got the expected text content from both branches.
    let text_values: Vec<&str> = frames
        .iter()
        .filter_map(|f| match &f.frame {
            Frame::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();

    // 3 originals from passthrough + 3 uppercased from uppercase = 6 text frames.
    assert_eq!(
        text_values.len(),
        6,
        "should have 6 text frames (3 per branch): {text_values:?}"
    );

    // NOTE: Testing that a slow branch delays the fast branch's output before
    // lifecycle sync completes is inherently racy in async tests. The barrier-sync
    // guarantee is structurally ensured by the coordinator: it buffers all
    // non-lifecycle frames until all branches have reported the lifecycle frame,
    // then flushes. The ordering assertions above validate the observable effect.
}
