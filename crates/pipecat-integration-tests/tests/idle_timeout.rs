use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_core::test_utils::PassthroughProcessor;
use pipecat_pipeline::{Pipeline, PipelineParams, PipelineTask};
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Idle timeout fires callback and auto-cancels the pipeline
// ---------------------------------------------------------------------------

/// Proves that with a short idle timeout and no activity frames:
/// 1. The `on_idle_timeout` callback fires.
/// 2. The pipeline auto-cancels (cancel_on_idle_timeout defaults to true).
/// 3. `task.run()` returns Ok and `has_finished()` is true.
#[tokio::test]
async fn idle_timeout_fires_callback_and_cancels() {
    let flag = Arc::new(AtomicBool::new(false));
    let flag_clone = Arc::clone(&flag);

    let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
    let mut task = PipelineTask::new(
        Box::new(pipeline),
        PipelineParams {
            enable_heartbeats: false,
            idle_timeout: Some(Duration::from_millis(100)),
            cancel_on_idle_timeout: true,
            ..Default::default()
        },
    )
    .on_idle_timeout(move || {
        let f = Arc::clone(&flag_clone);
        async move {
            f.store(true, Ordering::SeqCst);
        }
    });

    // Don't send any activity frames — idle timeout should fire and cancel.
    let result = timeout(TEST_TIMEOUT, task.run()).await;
    assert!(
        result.is_ok(),
        "pipeline should complete via idle timeout, not hit test timeout"
    );
    assert!(
        result.unwrap().is_ok(),
        "task.run() should return Ok after idle cancellation"
    );
    assert!(task.has_finished(), "task should be marked finished");
    assert!(
        flag.load(Ordering::SeqCst),
        "on_idle_timeout callback should have been invoked"
    );
}

// ---------------------------------------------------------------------------
// cancel_on_idle_timeout=false: callback fires but pipeline stays alive
// ---------------------------------------------------------------------------

/// Proves that when `cancel_on_idle_timeout` is false:
/// 1. The `on_idle_timeout` callback fires (flag is set).
/// 2. The pipeline does NOT auto-cancel — it stays alive after the timeout.
///
/// Strategy: We wrap `task.run()` in a `tokio::time::timeout` that is much
/// longer than the idle timeout (500ms vs 50ms). If the pipeline auto-cancelled,
/// it would exit within ~50ms. We assert:
/// - The outer timeout fires (Err), proving the pipeline was still running.
/// - The callback flag is set, proving the idle callback did fire.
///
/// The generous 500ms observation window (10x the idle timeout) makes this
/// robust against scheduling jitter.
///
/// Limitation: Because `push_tx` is `pub(crate)`, integration tests cannot
/// directly inject a CancelFrame to shut the pipeline down cleanly. The
/// `tokio::time::timeout` drop is the teardown mechanism.
#[tokio::test]
async fn idle_timeout_no_auto_cancel_when_disabled() {
    let flag = Arc::new(AtomicBool::new(false));
    let flag_clone = Arc::clone(&flag);

    let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
    let mut task = PipelineTask::new(
        Box::new(pipeline),
        PipelineParams {
            enable_heartbeats: false,
            idle_timeout: Some(Duration::from_millis(50)),
            cancel_on_idle_timeout: false,
            ..Default::default()
        },
    )
    .on_idle_timeout(move || {
        let f = Arc::clone(&flag_clone);
        async move {
            f.store(true, Ordering::SeqCst);
        }
    });

    // The pipeline should NOT exit on its own since cancel_on_idle_timeout=false.
    // 500ms is 10x the idle timeout — generous margin against scheduling jitter.
    let result = timeout(Duration::from_millis(500), task.run()).await;

    // We expect the timeout to fire (Err) because the pipeline is still running.
    assert!(
        result.is_err(),
        "pipeline should NOT auto-cancel when cancel_on_idle_timeout is false"
    );

    // But the callback should have fired.
    assert!(
        flag.load(Ordering::SeqCst),
        "on_idle_timeout callback should fire even when cancel_on_idle_timeout is false"
    );
}

// ---------------------------------------------------------------------------
// idle_timeout=None: no timeout, no callback
// ---------------------------------------------------------------------------

/// Proves that when `idle_timeout` is None, the idle monitor is never started:
/// the pipeline runs indefinitely and the callback never fires.
///
/// Strategy: Use `tokio::time::timeout` to observe that the pipeline does NOT
/// exit within 300ms. When the timeout fires, the future is dropped, which
/// tears down the pipeline. This is acceptable — the test only needs to prove
/// that nothing happened during the observation window.
#[tokio::test]
async fn no_idle_timeout_when_disabled() {
    let flag = Arc::new(AtomicBool::new(false));
    let flag_clone = Arc::clone(&flag);

    let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
    let mut task = PipelineTask::new(
        Box::new(pipeline),
        PipelineParams {
            enable_heartbeats: false,
            idle_timeout: None,
            ..Default::default()
        },
    )
    .on_idle_timeout(move || {
        let f = Arc::clone(&flag_clone);
        async move {
            f.store(true, Ordering::SeqCst);
        }
    });

    // With idle_timeout=None, the pipeline will never self-cancel.
    // A short timeout proves neither the callback fires nor the pipeline exits.
    // Dropping the future on timeout is the expected teardown mechanism here.
    let result = timeout(Duration::from_millis(300), task.run()).await;

    // Expect timeout — pipeline should still be running.
    assert!(
        result.is_err(),
        "pipeline should NOT exit when idle_timeout is None"
    );
    assert!(
        !flag.load(Ordering::SeqCst),
        "on_idle_timeout should NOT fire when idle_timeout is None"
    );
}

// ---------------------------------------------------------------------------
// Activity frames reset the idle timeout
// ---------------------------------------------------------------------------

/// A processor that, upon receiving a StartFrame, spawns a background task
/// emitting `UserStartedSpeaking` frames at regular intervals for a limited
/// duration. Used to simulate pipeline activity that should reset the idle
/// timer.
struct ActivityEmitter {
    base: ProcessorBase,
    activity_interval: Duration,
    activity_duration: Duration,
}

impl ActivityEmitter {
    fn new(activity_interval: Duration, activity_duration: Duration) -> Self {
        Self {
            base: ProcessorBase::new("ActivityEmitter"),
            activity_interval,
            activity_duration,
        }
    }
}

#[async_trait]
impl FrameProcessor for ActivityEmitter {
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
        if matches!(&envelope.frame, Frame::Start(_)) {
            // Start emitting activity frames in a background task.
            let ctx = ctx.clone();
            let interval = self.activity_interval;
            let duration = self.activity_duration;
            tokio::spawn(async move {
                let start = Instant::now();
                let mut ticker = tokio::time::interval(interval);
                ticker.tick().await; // skip initial tick
                while start.elapsed() < duration {
                    ticker.tick().await;
                    let frame = Frame::UserStartedSpeaking(UserStartedSpeakingFrame);
                    if ctx.send_downstream(frame).await.is_err() {
                        break;
                    }
                }
            });
        }
        // Always forward the original frame.
        ctx.push_frame(envelope, direction).await
    }
}

/// Proves that activity frames reset the idle timeout timer.
///
/// Setup:
/// - Idle timeout: 100ms, cancel_on_idle_timeout: true
/// - ActivityEmitter sends UserStartedSpeaking every 40ms for 400ms total
///
/// Expected behavior:
/// - During the 400ms activity window, the idle timer is continuously reset
///   by UserStartedSpeaking frames reaching the sink monitor.
/// - The pipeline does NOT timeout during activity (would have timed out
///   multiple times at 100ms intervals without resets).
/// - After activity stops (~400ms), the idle timeout fires ~100ms later.
/// - Total runtime should be roughly 400ms + 100ms = 500ms.
/// - If resets weren't working, the pipeline would exit after ~100ms.
#[tokio::test]
async fn activity_frames_reset_idle_timeout() {
    let flag = Arc::new(AtomicBool::new(false));
    let flag_clone = Arc::clone(&flag);

    // ActivityEmitter sends UserStartedSpeaking every 40ms for 400ms.
    // Idle timeout is 100ms. Without resets, the pipeline would exit at ~100ms.
    let emitter = ActivityEmitter::new(
        Duration::from_millis(40),  // activity every 40ms
        Duration::from_millis(400), // for 400ms total
    );

    let pipeline = Pipeline::new(vec![Box::new(emitter)]);
    let mut task = PipelineTask::new(
        Box::new(pipeline),
        PipelineParams {
            enable_heartbeats: false,
            idle_timeout: Some(Duration::from_millis(100)),
            cancel_on_idle_timeout: true,
            ..Default::default()
        },
    )
    .on_idle_timeout(move || {
        let f = Arc::clone(&flag_clone);
        async move {
            f.store(true, Ordering::SeqCst);
        }
    });

    let start = Instant::now();
    let result = timeout(TEST_TIMEOUT, task.run()).await;
    let elapsed = start.elapsed();

    assert!(
        result.is_ok(),
        "pipeline should complete via idle timeout after activity stops"
    );
    assert!(
        result.unwrap().is_ok(),
        "task.run() should return Ok after idle cancellation"
    );
    assert!(task.has_finished(), "task should be marked finished");
    assert!(
        flag.load(Ordering::SeqCst),
        "on_idle_timeout callback should have fired after activity stopped"
    );

    // The pipeline should have survived well past the 100ms idle timeout.
    // Activity runs for 400ms, then idle timeout fires after another ~100ms.
    // Use conservative bounds to avoid flakiness: at least 300ms (proves resets
    // worked) and no more than 3000ms (proves it eventually timed out).
    assert!(
        elapsed >= Duration::from_millis(300),
        "pipeline exited too early ({elapsed:?}); activity resets may not be working"
    );
    assert!(
        elapsed < Duration::from_millis(3000),
        "pipeline took too long ({elapsed:?}); idle timeout may not have fired"
    );
}
