//! Integration tests for error propagation across multi-node pipelines.
//!
//! Single-node error tests (ErrorOnTextProcessor, FailingProcessor) live in
//! `pipecat-core/src/node.rs`. These tests focus on multi-node pipeline behavior:
//! - Errors from one processor propagate upstream through the pipeline
//! - ProcessorNode wraps Err returns with its own format string
//! - Non-error frames continue flowing after an error
//! - Fatal errors propagate and cause pipeline cancellation

use std::time::Duration;

use async_trait::async_trait;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_core::test_utils::*;
use pipecat_pipeline::Pipeline;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Test-local processors
// ---------------------------------------------------------------------------

/// Pushes a fatal ErrorFrame upstream on Text frames. Forwards everything else.
struct FatalErrorOnTextProcessor {
    base: ProcessorBase,
}

impl FatalErrorOnTextProcessor {
    fn new() -> Self {
        Self {
            base: ProcessorBase::new("FatalErrorOnText"),
        }
    }
}

#[async_trait]
impl FrameProcessor for FatalErrorOnTextProcessor {
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
            Frame::Text(_) => {
                ctx.push_error("fatal error", true).await?;
            }
            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Test 1: ProcessorNode wraps Err from FailingProcessor with known format
// ---------------------------------------------------------------------------

/// FailingProcessor returns Err("intentional failure") from process_frame.
/// ProcessorNode catches this and wraps it as "Error processing frame: <original>".
/// This test verifies the wrapping in a multi-node pipeline context.
#[tokio::test]
async fn failing_processor_error_wrapped_by_node() {
    // Pipeline: Passthrough -> FailingProcessor
    // Send Text → FailingProcessor returns Err → ProcessorNode wraps and pushes upstream.
    let pipeline = Pipeline::new(vec![
        Box::new(PassthroughProcessor::new()),
        Box::new(FailingProcessor::new()),
    ]);
    let (node, handle, _down_rx, up_rx) = make_node(Box::new(pipeline));
    let up = FrameCollector::spawn(up_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    send_frame(
        &handle,
        Frame::Text(TextFrame::new("trigger")),
        Direction::Downstream,
    )
    .await;

    up.wait_for_frame("Error").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let upstream = up.frames();
    let error_frames: Vec<&ErrorFrame> = upstream
        .iter()
        .filter_map(|f| match &f.frame {
            Frame::Error(e) => Some(e),
            _ => None,
        })
        .collect();

    assert_eq!(
        error_frames.len(),
        1,
        "expected exactly one ErrorFrame upstream, got: {:?}",
        up.frame_names()
    );

    let e = error_frames[0];
    assert!(!e.fatal, "auto-caught errors should be non-fatal");

    // Verify ProcessorNode wrapped the original error with its format string.
    // node.rs line 441: push_error(&format!("Error processing frame: {e}"), false)
    assert!(
        e.error.starts_with("Error processing frame: "),
        "error should be wrapped by ProcessorNode, got: {}",
        e.error
    );
    assert!(
        e.error.contains("intentional failure"),
        "wrapped error should contain the original message, got: {}",
        e.error
    );
}

// ---------------------------------------------------------------------------
// Test 2: Error in processor B doesn't block processor A from receiving
//         subsequent frames
// ---------------------------------------------------------------------------

/// Pipeline: Recorder(A) -> ErrorOnText(B)
/// Send Text (B errors), then send Interruption (B forwards).
/// Recorder A should see both Text and Interruption, proving that an error in B
/// doesn't prevent A from processing subsequent frames.
#[tokio::test]
async fn error_in_later_processor_doesnt_block_earlier() {
    let (recorder, record) = RecorderProcessor::new();

    // Recorder sees frames before they reach ErrorOnText.
    let pipeline = Pipeline::new(vec![
        Box::new(recorder),
        Box::new(ErrorOnTextProcessor::new()),
    ]);
    let (node, handle, down_rx, up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let up = FrameCollector::spawn(up_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // First: Text triggers error in ErrorOnText
    send_frame(
        &handle,
        Frame::Text(TextFrame::new("bad")),
        Direction::Downstream,
    )
    .await;

    // Wait for first error to propagate upstream before sending next frame
    up.wait_for_frame("Error").await;

    // Second: Interruption flows through both processors normally
    send_frame(
        &handle,
        Frame::Interruption(InterruptionFrame),
        Direction::Downstream,
    )
    .await;

    // Wait for Interruption to fully propagate through the pipeline before
    // sending the next Text. Otherwise, the Interruption drain may consume it.
    down.wait_for_frame("Interruption").await;

    // Third: another Text — still errors in B, but A should still see it
    send_frame(
        &handle,
        Frame::Text(TextFrame::new("also bad")),
        Direction::Downstream,
    )
    .await;

    // Wait for the second error to arrive upstream (total of 2 Error frames)
    up.wait_for_count(2).await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let recorded = record.lock().unwrap();

    // Recorder (before ErrorOnText) should have seen both Text frames and the Interruption.
    let text_count = recorded.iter().filter(|s| s.as_str() == "Text").count();
    assert_eq!(
        text_count, 2,
        "Recorder should see both Text frames despite error in later processor: {:?}",
        *recorded
    );
    assert!(
        recorded.iter().any(|s| s == "Interruption"),
        "Recorder should see Interruption after the error: {:?}",
        *recorded
    );

    // Verify ErrorOnText did push errors upstream
    let upstream = up.frames();
    let error_count = upstream
        .iter()
        .filter(|f| matches!(&f.frame, Frame::Error(_)))
        .count();
    assert_eq!(
        error_count,
        2,
        "ErrorOnText should have pushed 2 errors upstream: {:?}",
        up.frame_names()
    );
}

// ---------------------------------------------------------------------------
// Test 3: Non-fatal error propagates through pipeline but doesn't stop it
// ---------------------------------------------------------------------------

/// Pipeline: ErrorOnText -> Passthrough
/// Send Text (ErrorOnText pushes non-fatal error), then send another frame.
/// The pipeline should continue operating after the non-fatal error.
#[tokio::test]
async fn non_fatal_error_does_not_stop_pipeline() {
    let pipeline = Pipeline::new(vec![
        Box::new(ErrorOnTextProcessor::new()),
        Box::new(PassthroughProcessor::new()),
    ]);
    let (node, handle, down_rx, up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let up = FrameCollector::spawn(up_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Text triggers non-fatal error in ErrorOnText
    send_frame(
        &handle,
        Frame::Text(TextFrame::new("bad")),
        Direction::Downstream,
    )
    .await;

    // Wait for error to propagate upstream
    up.wait_for_frame("Error").await;

    // Interruption should still flow through the whole pipeline
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

    // Error should arrive upstream
    let upstream = up.frames();
    assert!(
        upstream
            .iter()
            .any(|f| matches!(&f.frame, Frame::Error(e) if !e.fatal)),
        "non-fatal error should propagate upstream: {:?}",
        up.frame_names()
    );

    // Interruption should arrive downstream (pipeline still working)
    let names = down.frame_names();
    assert!(
        names.contains(&"Interruption".to_string()),
        "pipeline should still forward frames after non-fatal error: {:?}",
        names
    );
}

// ---------------------------------------------------------------------------
// Test 4: Fatal error propagates upstream through multi-node pipeline
// ---------------------------------------------------------------------------

/// Pipeline: Passthrough -> FatalErrorOnText
/// Send Text → FatalErrorOnText pushes fatal error upstream.
/// Verify the error arrives upstream with fatal=true and correct source_processor.
#[tokio::test]
async fn fatal_error_propagates_upstream() {
    let pipeline = Pipeline::new(vec![
        Box::new(PassthroughProcessor::new()),
        Box::new(FatalErrorOnTextProcessor::new()),
    ]);
    let (node, handle, _down_rx, up_rx) = make_node(Box::new(pipeline));
    let up = FrameCollector::spawn(up_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    send_frame(
        &handle,
        Frame::Text(TextFrame::new("trigger fatal")),
        Direction::Downstream,
    )
    .await;

    up.wait_for_frame("Error").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let upstream = up.frames();
    let fatal_errors: Vec<&ErrorFrame> = upstream
        .iter()
        .filter_map(|f| match &f.frame {
            Frame::Error(e) if e.fatal => Some(e),
            _ => None,
        })
        .collect();

    assert_eq!(
        fatal_errors.len(),
        1,
        "expected exactly one fatal ErrorFrame, got: {:?}",
        up.frame_names()
    );

    let e = fatal_errors[0];
    assert!(e.fatal, "error should be fatal");
    assert_eq!(e.error, "fatal error");
    assert_eq!(
        e.source_processor, "FatalErrorOnText",
        "source_processor should identify the originating processor"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Fatal error source_processor is preserved through pipeline
// ---------------------------------------------------------------------------

/// Pipeline: Uppercase -> FatalErrorOnText -> Passthrough
/// Verify the fatal error's source_processor field correctly identifies the
/// originating processor even when it's in the middle of the pipeline.
#[tokio::test]
async fn fatal_error_source_processor_preserved_in_pipeline() {
    let pipeline = Pipeline::new(vec![
        Box::new(UppercaseProcessor::new()),
        Box::new(FatalErrorOnTextProcessor::new()),
        Box::new(PassthroughProcessor::new()),
    ]);
    let (node, handle, _down_rx, up_rx) = make_node(Box::new(pipeline));
    let up = FrameCollector::spawn(up_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Uppercase transforms "trigger" → "TRIGGER", then FatalErrorOnText sees Text and errors
    send_frame(
        &handle,
        Frame::Text(TextFrame::new("trigger")),
        Direction::Downstream,
    )
    .await;

    up.wait_for_frame("Error").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let upstream = up.frames();
    let fatal_errors: Vec<&ErrorFrame> = upstream
        .iter()
        .filter_map(|f| match &f.frame {
            Frame::Error(e) if e.fatal => Some(e),
            _ => None,
        })
        .collect();

    assert_eq!(
        fatal_errors.len(),
        1,
        "expected exactly one fatal ErrorFrame, got: {:?}",
        up.frame_names()
    );

    let e = fatal_errors[0];
    assert_eq!(
        e.source_processor, "FatalErrorOnText",
        "source_processor should be preserved through multi-node pipeline"
    );
    assert_eq!(e.error, "fatal error");
}
