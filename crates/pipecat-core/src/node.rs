use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tracing::{debug, trace};

use crate::error::{PipecatError, Result};
use crate::frame::{Direction, Frame, FrameEnvelope};
use crate::observer::{FrameProcessedEvent, PipelineObserver};
use crate::processor::{FrameProcessor, ProcessorContext};

// ---------------------------------------------------------------------------
// NodeMessage — wraps a frame with its direction
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct NodeMessage {
    envelope: FrameEnvelope,
    direction: Direction,
}

// ---------------------------------------------------------------------------
// ProcessorNodeHandle — the sending side
// ---------------------------------------------------------------------------

/// Handle used by adjacent processors (or the pipeline) to send frames into
/// this node. Clone is cheap — all clones share the same underlying channels.
///
/// Frames are routed to the system or normal channel based on `is_system()`.
/// System frames get priority processing in the node's event loop.
#[derive(Clone, Debug)]
pub struct ProcessorNodeHandle {
    system_tx: mpsc::Sender<NodeMessage>,
    normal_tx: mpsc::Sender<NodeMessage>,
    name: String,
}

impl ProcessorNodeHandle {
    /// The name of the processor this handle sends to.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Send a frame, automatically routing to system or normal channel.
    pub async fn send(&self, envelope: FrameEnvelope, direction: Direction) -> Result<()> {
        let is_system = envelope.frame.is_system();
        let msg = NodeMessage {
            envelope,
            direction,
        };
        if is_system {
            self.system_tx.send(msg).await
        } else {
            self.normal_tx.send(msg).await
        }
        .map_err(|e| PipecatError::ChannelSend(format!("NodeHandle({}): {}", self.name, e)))
    }

    /// Force-send on the system channel regardless of frame type.
    pub async fn send_system(&self, envelope: FrameEnvelope, direction: Direction) -> Result<()> {
        self.system_tx
            .send(NodeMessage {
                envelope,
                direction,
            })
            .await
            .map_err(|e| PipecatError::ChannelSend(format!("NodeHandle({}): {}", self.name, e)))
    }

    /// Force-send on the normal channel regardless of frame type.
    pub async fn send_normal(&self, envelope: FrameEnvelope, direction: Direction) -> Result<()> {
        self.normal_tx
            .send(NodeMessage {
                envelope,
                direction,
            })
            .await
            .map_err(|e| PipecatError::ChannelSend(format!("NodeHandle({}): {}", self.name, e)))
    }
}

// ---------------------------------------------------------------------------
// ProcessorNode — the runtime wrapper
// ---------------------------------------------------------------------------

/// Runtime wrapper that owns a `FrameProcessor` and drives it with a
/// priority-aware event loop.
///
/// # Architecture
///
/// The node receives frames on two channels:
/// - **system channel** — high priority (system frames: Start, Cancel, Interruption, etc.)
/// - **normal channel** — normal priority (data and control frames)
///
/// The `run()` loop uses `tokio::select! { biased; ... }` to always check the
/// system channel first, guaranteeing that system frames preempt queued normal
/// frames.
///
/// # Lifecycle
///
/// 1. Normal frames are NOT processed until a `StartFrame` arrives.
/// 2. An `InterruptionFrame` drains all queued normal frames except uninterruptible ones.
/// 3. `ProcessorPauseUrgent`/`ProcessorResumeUrgent` pause/resume normal frame processing.
///    Normal-priority `ProcessorPause`/`ProcessorResume` (ControlFrames) arrive on the
///    normal channel and are passed through to the processor without the node acting on
///    them — these are for compound processors to forward to sub-processors.
/// 4. A `CancelFrame` exits the run loop after being dispatched to the processor.
/// 5. `EndFrame` is a normal-priority control frame — the node dispatches it to the
///    processor but does not exit. The pipeline is responsible for sending CancelFrame
///    after EndFrame propagates.
///
/// All system frames are passed through to the processor after the node handles
/// its infrastructure side effects. This lets processors react to lifecycle events
/// (e.g., a service initializes on Start, aborts on Cancel).
///
/// # Cleanup
///
/// When the run loop exits (Cancel received, or channels closed), the node calls
/// `processor.cleanup()` to let the processor release resources (close connections,
/// flush buffers, cancel background tasks).
pub struct ProcessorNode {
    processor: Box<dyn FrameProcessor>,
    system_rx: mpsc::Receiver<NodeMessage>,
    normal_rx: mpsc::Receiver<NodeMessage>,
    ctx: ProcessorContext,
    observer: Option<Arc<dyn PipelineObserver>>,
    pending: VecDeque<NodeMessage>,
    started: bool,
    cancelled: bool,
    paused: bool,
}

impl fmt::Debug for ProcessorNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProcessorNode")
            .field("processor", &self.processor.name())
            .field("id", &self.processor.id())
            .field("started", &self.started)
            .field("cancelled", &self.cancelled)
            .field("paused", &self.paused)
            .field("pending", &self.pending.len())
            .finish()
    }
}

impl ProcessorNode {
    /// Create a new ProcessorNode wrapping the given processor.
    ///
    /// Returns `(node, handle)`. The handle is used to send frames into the node.
    /// Call `run()` to start the event loop.
    ///
    /// `downstream_tx` and `upstream_tx` are the channels for the processor to
    /// push frames out (to the next/previous processor in the pipeline).
    pub fn new(
        processor: Box<dyn FrameProcessor>,
        downstream_tx: mpsc::Sender<FrameEnvelope>,
        upstream_tx: mpsc::Sender<FrameEnvelope>,
        channel_size: usize,
    ) -> (Self, ProcessorNodeHandle) {
        Self::build(processor, downstream_tx, upstream_tx, channel_size, None)
    }

    /// Create a ProcessorNode with a pipeline observer attached.
    /// The observer receives `on_process_frame` events from this node
    /// and `on_push_frame` events from the processor's context.
    pub fn with_observer(
        processor: Box<dyn FrameProcessor>,
        downstream_tx: mpsc::Sender<FrameEnvelope>,
        upstream_tx: mpsc::Sender<FrameEnvelope>,
        channel_size: usize,
        observer: Arc<dyn PipelineObserver>,
    ) -> (Self, ProcessorNodeHandle) {
        Self::build(
            processor,
            downstream_tx,
            upstream_tx,
            channel_size,
            Some(observer),
        )
    }

    fn build(
        processor: Box<dyn FrameProcessor>,
        downstream_tx: mpsc::Sender<FrameEnvelope>,
        upstream_tx: mpsc::Sender<FrameEnvelope>,
        channel_size: usize,
        observer: Option<Arc<dyn PipelineObserver>>,
    ) -> (Self, ProcessorNodeHandle) {
        let (system_tx, system_rx) = mpsc::channel(channel_size);
        let (normal_tx, normal_rx) = mpsc::channel(channel_size);

        let ctx = match &observer {
            Some(obs) => ProcessorContext::with_observer(
                downstream_tx,
                upstream_tx,
                processor.id(),
                processor.name().to_string(),
                obs.clone(),
            ),
            None => ProcessorContext::new(
                downstream_tx,
                upstream_tx,
                processor.id(),
                processor.name().to_string(),
            ),
        };

        let handle = ProcessorNodeHandle {
            system_tx,
            normal_tx,
            name: processor.name().to_string(),
        };

        let node = Self {
            processor,
            system_rx,
            normal_rx,
            ctx,
            observer,
            pending: VecDeque::new(),
            started: false,
            cancelled: false,
            paused: false,
        };

        (node, handle)
    }

    /// Run the processor's event loop until Cancel or channel close.
    ///
    /// Consumes the node. Calls `processor.cleanup()` before returning.
    pub async fn run(mut self) {
        debug!(
            processor = self.processor.name(),
            id = self.processor.id(),
            "ProcessorNode starting"
        );

        self.processor.setup().await;

        loop {
            // 1. Process pending frames first (survivors of interruption drain).
            //    Only process if started and not paused — same rules as normal channel.
            if self.started
                && !self.paused
                && let Some(msg) = self.pending.pop_front()
            {
                self.dispatch(msg).await;
                if self.cancelled {
                    break;
                }
                continue;
            }

            // 2. select! with bias towards system frames
            tokio::select! {
                biased;

                msg = self.system_rx.recv() => {
                    match msg {
                        Some(msg) => {
                            self.handle_system_frame(msg).await;
                            if self.cancelled {
                                break;
                            }
                        }
                        None => {
                            // System channel closed — handle was dropped
                            trace!(
                                processor = self.processor.name(),
                                "system channel closed"
                            );
                            break;
                        }
                    }
                }

                msg = self.normal_rx.recv(),
                    if self.started && !self.paused && !self.cancelled => {
                    match msg {
                        Some(msg) => {
                            self.dispatch(msg).await;
                        }
                        None => {
                            // Normal channel closed
                            trace!(
                                processor = self.processor.name(),
                                "normal channel closed"
                            );
                            break;
                        }
                    }
                }

                // If normal_rx is disabled (not started, paused, or cancelled)
                // and system_rx returns None, we break.
            }
        }

        // Cleanup: let the processor release resources
        debug!(
            processor = self.processor.name(),
            id = self.processor.id(),
            cancelled = self.cancelled,
            "ProcessorNode cleaning up"
        );
        self.processor.cleanup().await;

        debug!(
            processor = self.processor.name(),
            id = self.processor.id(),
            "ProcessorNode stopped"
        );
    }

    /// Handle a system frame: apply infrastructure side effects, then dispatch
    /// to the processor.
    async fn handle_system_frame(&mut self, msg: NodeMessage) {
        let frame = &msg.envelope.frame;

        match frame {
            Frame::Start(_) => {
                trace!(processor = self.processor.name(), "StartFrame received");
                self.started = true;
                self.dispatch(msg).await;
                if let Some(ref observer) = self.observer {
                    observer.on_pipeline_started().await;
                }
            }

            Frame::Cancel(_) => {
                trace!(processor = self.processor.name(), "CancelFrame received");
                self.dispatch(msg).await;
                self.cancelled = true;
            }

            Frame::Interruption(_) => {
                // NOTE: Python tracks __process_current_frame to skip task
                // cancellation if the currently-dispatching frame is
                // uninterruptible. In Rust, this is safe-by-design: dispatch()
                // runs sequentially in the select! loop, so the
                // currently-dispatching frame always completes before the next
                // iteration processes the InterruptionFrame. The drain only
                // affects queued frames, never the in-progress one.
                trace!(
                    processor = self.processor.name(),
                    "InterruptionFrame received, draining normal queue"
                );
                self.drain_on_interruption();
                self.dispatch(msg).await;
            }

            Frame::ProcessorPauseUrgent(pf) => {
                if pf.processor_name == self.processor.name() {
                    trace!(
                        processor = self.processor.name(),
                        "pausing normal frame processing"
                    );
                    self.paused = true;
                }
                self.dispatch(msg).await;
            }

            Frame::ProcessorResumeUrgent(rf) => {
                if rf.processor_name == self.processor.name() {
                    trace!(
                        processor = self.processor.name(),
                        "resuming normal frame processing"
                    );
                    self.paused = false;
                }
                self.dispatch(msg).await;
            }

            // All other system frames: just dispatch to the processor.
            _ => {
                self.dispatch(msg).await;
            }
        }
    }

    /// Dispatch a message to the processor. Fires `on_process_frame` on the
    /// observer and `on_before_process`/`on_after_process` hooks on the processor.
    async fn dispatch(&mut self, msg: NodeMessage) {
        trace!(
            processor = self.processor.name(),
            frame = %msg.envelope.frame,
            direction = ?msg.direction,
            "dispatching frame"
        );

        if let Some(ref observer) = self.observer {
            observer
                .on_process_frame(FrameProcessedEvent {
                    processor_name: self.processor.name(),
                    processor_id: self.processor.id(),
                    frame: &msg.envelope.frame,
                    frame_id: msg.envelope.header.id,
                    direction: msg.direction,
                    timestamp: Instant::now(),
                })
                .await;
        }

        self.processor
            .on_before_process(&msg.envelope.frame, msg.direction)
            .await;

        if let Err(e) = self
            .processor
            .process_frame(msg.envelope, msg.direction, &self.ctx)
            .await
        {
            tracing::error!(
                processor = self.processor.name(),
                error = %e,
                "process_frame returned error, pushing ErrorFrame upstream"
            );
            let _ = self
                .ctx
                .push_error(&format!("Error processing frame: {e}"), false)
                .await;
        }

        self.processor.on_after_process().await;
    }

    /// Drain the pending buffer and normal channel on interruption.
    /// Only uninterruptible frames survive. Pending buffer is drained first
    /// (preserving order of previously surviving frames), then the channel.
    fn drain_on_interruption(&mut self) {
        // 1. Filter the pending buffer
        let old_pending = std::mem::take(&mut self.pending);
        let mut kept = 0u32;
        let mut dropped = 0u32;
        for msg in old_pending {
            if msg.envelope.frame.is_uninterruptible() {
                self.pending.push_back(msg);
                kept += 1;
            } else {
                dropped += 1;
            }
        }

        // 2. Drain the normal channel
        while let Ok(msg) = self.normal_rx.try_recv() {
            if msg.envelope.frame.is_uninterruptible() {
                self.pending.push_back(msg);
                kept += 1;
            } else {
                dropped += 1;
            }
        }

        if kept > 0 || dropped > 0 {
            debug!(
                processor = self.processor.name(),
                kept, dropped, "interruption drain complete"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::time::timeout;

    use super::*;
    use crate::frame::*;
    use crate::test_utils::*;

    const TEST_TIMEOUT: Duration = Duration::from_secs(2);

    // -----------------------------------------------------------------------
    // Basic flow tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn passthrough_start_text_end() {
        let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(PassthroughProcessor::new()));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new("hello")),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::End(EndFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let frames = drain_rx(&mut down_rx).await;
        let n = frame_names(&frames);
        assert_eq!(n, vec!["Start", "Text", "End", "Cancel"], "got: {n:?}");
    }

    #[tokio::test]
    async fn frame_ordering_preserved() {
        let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(PassthroughProcessor::new()));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;

        for i in 0..10 {
            handle
                .send(
                    FrameEnvelope::new(Frame::Text(TextFrame::new(format!("msg{i}")))),
                    Direction::Downstream,
                )
                .await
                .unwrap();
        }
        tokio::time::sleep(crate::test_utils::SETTLE).await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let frames = drain_rx(&mut down_rx).await;
        assert_eq!(
            frames.len(),
            12,
            "expected Start + 10 texts + Cancel, got: {:?}",
            frame_names(&frames)
        );
        for i in 0..10 {
            assert!(
                matches!(&frames[i + 1].frame, Frame::Text(t) if t.text == format!("msg{i}")),
                "frame {} should be msg{}: got {:?}",
                i + 1,
                i,
                frame_names(&frames)
            );
        }
    }

    // -----------------------------------------------------------------------
    // Priority tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn system_frames_processed_before_normal() {
        let (recorder, record) = RecorderProcessor::new();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(recorder));

        let run = tokio::spawn(async move { node.run().await });

        // Start and let it settle so started=true
        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;

        // Batch: normal + system frames simultaneously
        for i in 0..5 {
            handle
                .send(
                    FrameEnvelope::new(Frame::Text(TextFrame::new(format!("text{i}")))),
                    Direction::Downstream,
                )
                .await
                .unwrap();
        }
        handle
            .send(
                FrameEnvelope::new(Frame::BotStartedSpeaking(BotStartedSpeakingFrame)),
                Direction::Downstream,
            )
            .await
            .unwrap();
        handle
            .send(
                FrameEnvelope::new(Frame::BotStoppedSpeaking(BotStoppedSpeakingFrame)),
                Direction::Downstream,
            )
            .await
            .unwrap();

        tokio::time::sleep(crate::test_utils::SETTLE).await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let recorded = record.lock().unwrap();
        let bot_start_idx = recorded
            .iter()
            .position(|s| s == "BotStartedSpeaking")
            .expect("BotStartedSpeaking should be recorded");
        let first_text_idx = recorded
            .iter()
            .position(|s| s == "Text")
            .expect("at least one text should be recorded");

        assert!(
            bot_start_idx < first_text_idx,
            "system frames should come before texts: {:?}",
            *recorded
        );
    }

    // -----------------------------------------------------------------------
    // StartFrame gating tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn normal_frames_gated_until_start() {
        let (recorder, record) = RecorderProcessor::new();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(recorder));

        let run = tokio::spawn(async move { node.run().await });

        // Send normal frame BEFORE Start
        handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new("early"))),
                Direction::Downstream,
            )
            .await
            .unwrap();
        tokio::time::sleep(crate::test_utils::SETTLE).await;

        // Verify nothing processed yet
        {
            let recorded = record.lock().unwrap();
            assert!(
                !recorded.iter().any(|s| s == "Text"),
                "text should not process before Start: {:?}",
                *recorded
            );
        }

        // Now send Start — the queued text should process after it
        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let recorded = record.lock().unwrap();
        assert_eq!(recorded[0], "Start");
        assert!(
            recorded.contains(&"Text".to_string()),
            "early text should process after Start: {:?}",
            *recorded
        );
    }

    // -----------------------------------------------------------------------
    // Interruption tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn interruption_drains_normal_queue() {
        let (recorder, record) = RecorderProcessor::new();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(recorder));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;

        // Queue normal frames + interruption simultaneously
        for i in 0..5 {
            handle
                .send(
                    FrameEnvelope::new(Frame::Text(TextFrame::new(format!("pre_interrupt_{i}")))),
                    Direction::Downstream,
                )
                .await
                .unwrap();
        }
        handle
            .send(
                FrameEnvelope::new(Frame::Interruption(InterruptionFrame)),
                Direction::Downstream,
            )
            .await
            .unwrap();

        tokio::time::sleep(crate::test_utils::SETTLE).await;

        // Post-interruption text should arrive
        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new("post_interrupt")),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let recorded = record.lock().unwrap();
        assert!(recorded.contains(&"Start".to_string()));
        assert!(recorded.contains(&"Interruption".to_string()));
        assert!(recorded.contains(&"Cancel".to_string()));
        assert!(
            recorded.iter().any(|s| s == "Text"),
            "post_interrupt text should arrive: {:?}",
            *recorded
        );
        let text_count = recorded.iter().filter(|s| s.as_str() == "Text").count();
        assert!(
            text_count <= 2,
            "most pre-interrupt texts should be drained, got {} texts: {:?}",
            text_count,
            *recorded
        );
    }

    #[tokio::test]
    async fn uninterruptible_frames_survive_drain() {
        let (recorder, record) = RecorderProcessor::new();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(recorder));

        let run = tokio::spawn(async move { node.run().await });
        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;

        // Queue: text, EndFrame (uninterruptible), text, then interrupt
        handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new("will_die"))),
                Direction::Downstream,
            )
            .await
            .unwrap();
        handle
            .send(
                FrameEnvelope::new(Frame::End(EndFrame {
                    reason: Some("survive".into()),
                })),
                Direction::Downstream,
            )
            .await
            .unwrap();
        handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new("also_dies"))),
                Direction::Downstream,
            )
            .await
            .unwrap();
        handle
            .send(
                FrameEnvelope::new(Frame::Interruption(InterruptionFrame)),
                Direction::Downstream,
            )
            .await
            .unwrap();

        tokio::time::sleep(crate::test_utils::SETTLE).await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let recorded = record.lock().unwrap();
        assert!(
            recorded.contains(&"End".to_string()),
            "EndFrame should survive interruption drain: {:?}",
            *recorded
        );
        assert!(recorded.contains(&"Interruption".to_string()));
    }

    #[tokio::test]
    async fn function_call_result_survives_interruption() {
        let (recorder, record) = RecorderProcessor::new();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(recorder));

        let run = tokio::spawn(async move { node.run().await });
        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;

        handle
            .send(
                FrameEnvelope::new(Frame::FunctionCallResult(FunctionCallResultFrame {
                    function_name: "get_weather".into(),
                    tool_call_id: "tc1".into(),
                    arguments: serde_json::Value::Null,
                    result: serde_json::json!({"temp": 72}),
                    run_llm: Some(true),
                    properties: None,
                })),
                Direction::Downstream,
            )
            .await
            .unwrap();
        handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new("expendable"))),
                Direction::Downstream,
            )
            .await
            .unwrap();
        handle
            .send(
                FrameEnvelope::new(Frame::Interruption(InterruptionFrame)),
                Direction::Downstream,
            )
            .await
            .unwrap();

        tokio::time::sleep(crate::test_utils::SETTLE).await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let recorded = record.lock().unwrap();
        assert!(
            recorded.contains(&"FunctionCallResult".to_string()),
            "FunctionCallResult should survive interruption: {:?}",
            *recorded
        );
    }

    // -----------------------------------------------------------------------
    // Cancel tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn cancel_exits_run_loop() {
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(PassthroughProcessor::new()));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        handle
            .send(
                FrameEnvelope::new(Frame::Cancel(CancelFrame::default())),
                Direction::Downstream,
            )
            .await
            .unwrap();

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn frames_after_cancel_not_processed() {
        let (recorder, record) = RecorderProcessor::new();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(recorder));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        let _ = handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new("ignored"))),
                Direction::Downstream,
            )
            .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let recorded = record.lock().unwrap();
        assert!(recorded.contains(&"Start".to_string()));
        assert!(recorded.contains(&"Cancel".to_string()));
        assert!(
            !recorded.iter().any(|s| s == "Text"),
            "frames after cancel should not be processed: {:?}",
            *recorded
        );
    }

    // -----------------------------------------------------------------------
    // Cleanup tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn cleanup_called_on_cancel() {
        let (proc, cleanup_flag) = PassthroughProcessor::with_cleanup_flag();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(proc));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        handle
            .send(
                FrameEnvelope::new(Frame::Cancel(CancelFrame::default())),
                Direction::Downstream,
            )
            .await
            .unwrap();

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        assert!(
            cleanup_flag.load(Ordering::SeqCst),
            "cleanup should be called when node exits via Cancel"
        );
    }

    #[tokio::test]
    async fn cleanup_called_on_handle_drop() {
        let (proc, cleanup_flag) = PassthroughProcessor::with_cleanup_flag();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(proc));

        let run = tokio::spawn(async move { node.run().await });

        drop(handle);

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        assert!(
            cleanup_flag.load(Ordering::SeqCst),
            "cleanup should be called when node exits via channel close"
        );
    }

    // -----------------------------------------------------------------------
    // Pause / Resume tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn pause_stops_normal_processing() {
        let (recorder, record) = RecorderProcessor::new();
        let processor_name = recorder.base.name().to_string();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(recorder));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new("before_pause")),
            Direction::Downstream,
        )
        .await;

        // Pause
        send_and_settle(
            &handle,
            Frame::ProcessorPauseUrgent(ProcessorPauseUrgentFrame {
                processor_name: processor_name.clone(),
            }),
            Direction::Downstream,
        )
        .await;

        // Send text while paused
        handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new("during_pause"))),
                Direction::Downstream,
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Verify paused text has NOT been processed
        {
            let recorded = record.lock().unwrap();
            let text_count = recorded.iter().filter(|s| s.as_str() == "Text").count();
            assert_eq!(
                text_count, 1,
                "only before_pause should be processed while paused: {:?}",
                *recorded
            );
        }

        // Resume
        send_and_settle(
            &handle,
            Frame::ProcessorResumeUrgent(ProcessorResumeUrgentFrame { processor_name }),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;
        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let recorded = record.lock().unwrap();
        let text_count = recorded.iter().filter(|s| s.as_str() == "Text").count();
        assert_eq!(
            text_count, 2,
            "both texts should be processed after resume: {:?}",
            *recorded
        );
    }

    #[tokio::test]
    async fn system_frames_process_while_paused() {
        let (recorder, record) = RecorderProcessor::new();
        let processor_name = recorder.base.name().to_string();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(recorder));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::ProcessorPauseUrgent(ProcessorPauseUrgentFrame {
                processor_name: processor_name.clone(),
            }),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::BotStartedSpeaking(BotStartedSpeakingFrame),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::ProcessorResumeUrgent(ProcessorResumeUrgentFrame { processor_name }),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let recorded = record.lock().unwrap();
        assert!(
            recorded.contains(&"BotStartedSpeaking".to_string()),
            "system frames should process while paused: {:?}",
            *recorded
        );
    }

    #[tokio::test]
    async fn pause_for_different_processor_ignored() {
        let (recorder, record) = RecorderProcessor::new();
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(recorder));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::ProcessorPauseUrgent(ProcessorPauseUrgentFrame {
                processor_name: "OtherProcessor".into(),
            }),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new("still_flowing")),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;
        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let recorded = record.lock().unwrap();
        assert!(
            recorded.iter().any(|s| s == "Text"),
            "pause for different processor should not affect us: {:?}",
            *recorded
        );
    }

    // -----------------------------------------------------------------------
    // Error handling tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn processor_error_propagates_upstream() {
        let (node, handle, _down_rx, mut up_rx) = make_node(Box::new(ErrorOnTextProcessor::new()));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new("bad")),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let errors = drain_rx(&mut up_rx).await;
        assert!(
            errors
                .iter()
                .any(|e| matches!(&e.frame, Frame::Error(ef) if ef.error == "text not allowed")),
            "error should propagate upstream: {:?}",
            frame_names(&errors)
        );
    }

    // -----------------------------------------------------------------------
    // Direction tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn upstream_direction_preserved() {
        let (node, handle, _down_rx, mut up_rx) = make_node(Box::new(PassthroughProcessor::new()));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new("going_up")),
            Direction::Upstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;
        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let upstream = drain_rx(&mut up_rx).await;
        assert!(
            upstream
                .iter()
                .any(|e| matches!(&e.frame, Frame::Text(t) if t.text == "going_up")),
            "upstream frame should arrive on upstream channel: {:?}",
            frame_names(&upstream)
        );
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn handle_dropped_exits_run() {
        let (node, handle, _down_rx, _up_rx) = make_node(Box::new(PassthroughProcessor::new()));

        let run = tokio::spawn(async move { node.run().await });
        drop(handle);

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn send_on_closed_node_returns_error() {
        let (down_tx, _) = mpsc::channel(1);
        let (up_tx, _) = mpsc::channel(1);
        let (node, handle) =
            ProcessorNode::new(Box::new(PassthroughProcessor::new()), down_tx, up_tx, 1);

        drop(node);

        let result = handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new("x"))),
                Direction::Downstream,
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn handle_is_cloneable() {
        let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(PassthroughProcessor::new()));

        let run = tokio::spawn(async move { node.run().await });

        let handle2 = handle.clone();

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle2,
            Frame::Text(TextFrame::new("from_clone")),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;
        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let frames = drain_rx(&mut down_rx).await;
        assert!(
            frames
                .iter()
                .any(|e| matches!(&e.frame, Frame::Text(t) if t.text == "from_clone")),
            "frame from cloned handle should arrive: {:?}",
            frame_names(&frames)
        );
    }

    #[tokio::test]
    async fn debug_impl_works() {
        let (node, _handle, _down_rx, _up_rx) = make_node(Box::new(PassthroughProcessor::new()));
        let debug_str = format!("{node:?}");
        assert!(debug_str.contains("Passthrough"));
        assert!(debug_str.contains("started: false"));
    }

    // -----------------------------------------------------------------------
    // Observer integration tests
    // -----------------------------------------------------------------------

    /// Mock observer for node integration tests.
    struct NodeTestObserver {
        process_frames: Mutex<Vec<String>>,
        push_frames: Mutex<Vec<String>>,
        started: std::sync::atomic::AtomicBool,
    }

    impl NodeTestObserver {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                process_frames: Mutex::new(Vec::new()),
                push_frames: Mutex::new(Vec::new()),
                started: std::sync::atomic::AtomicBool::new(false),
            })
        }
    }

    #[async_trait]
    impl crate::observer::PipelineObserver for NodeTestObserver {
        async fn on_process_frame(&self, event: crate::observer::FrameProcessedEvent<'_>) {
            self.process_frames
                .lock()
                .unwrap()
                .push(format!("{}", event.frame));
        }

        async fn on_push_frame(&self, event: crate::observer::FramePushedEvent<'_>) {
            self.push_frames
                .lock()
                .unwrap()
                .push(format!("{}", event.frame));
        }

        async fn on_pipeline_started(&self) {
            self.started
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn observer_receives_on_process_frame() {
        let obs = NodeTestObserver::new();
        let (node, handle, _down_rx, _up_rx) =
            make_observed_node(Box::new(PassthroughProcessor::new()), obs.clone());

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new("observed")),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let processed = obs.process_frames.lock().unwrap();
        assert!(
            processed.contains(&"Start".to_string()),
            "observer should see Start: {:?}",
            *processed
        );
        assert!(
            processed.contains(&"Text".to_string()),
            "observer should see Text: {:?}",
            *processed
        );
        assert!(
            processed.contains(&"Cancel".to_string()),
            "observer should see Cancel: {:?}",
            *processed
        );
    }

    #[tokio::test]
    async fn observer_receives_on_push_frame() {
        let obs = NodeTestObserver::new();
        let (node, handle, _down_rx, _up_rx) =
            make_observed_node(Box::new(PassthroughProcessor::new()), obs.clone());

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new("pushed")),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        // PassthroughProcessor pushes every frame, so observer should see pushes
        let pushed = obs.push_frames.lock().unwrap();
        assert!(
            pushed.contains(&"Text".to_string()),
            "observer should see pushed Text: {:?}",
            *pushed
        );
    }

    #[tokio::test]
    async fn observer_receives_pipeline_started() {
        let obs = NodeTestObserver::new();
        let (node, handle, _down_rx, _up_rx) =
            make_observed_node(Box::new(PassthroughProcessor::new()), obs.clone());

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        assert!(
            obs.started.load(std::sync::atomic::Ordering::SeqCst),
            "on_pipeline_started should fire after StartFrame"
        );
    }

    #[tokio::test]
    async fn observer_sees_error_pushed_upstream() {
        let obs = NodeTestObserver::new();
        let (node, handle, _down_rx, _up_rx) =
            make_observed_node(Box::new(ErrorOnTextProcessor::new()), obs.clone());

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new("trigger_error")),
            Direction::Downstream,
        )
        .await;
        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        // ErrorOnTextProcessor calls push_error which calls push_upstream → observer sees it
        let pushed = obs.push_frames.lock().unwrap();
        assert!(
            pushed.contains(&"Error".to_string()),
            "observer should see Error pushed upstream: {:?}",
            *pushed
        );
    }

    // -----------------------------------------------------------------------
    // Auto error catching
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn process_frame_err_produces_error_frame_upstream() {
        let (node, handle, mut down_rx, mut up_rx) = make_node(Box::new(FailingProcessor::new()));

        let run = tokio::spawn(async move { node.run().await });

        send_and_settle(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;

        // Text triggers Err in FailingProcessor
        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new("will fail")),
            Direction::Downstream,
        )
        .await;

        // Non-text is forwarded normally
        send_and_settle(
            &handle,
            Frame::Interruption(InterruptionFrame),
            Direction::Downstream,
        )
        .await;

        send_and_settle(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        // The Err should have been caught and an ErrorFrame pushed upstream.
        let upstream = drain_rx(&mut up_rx).await;
        let error_frames: Vec<_> = upstream
            .iter()
            .filter(|f| matches!(&f.frame, Frame::Error(_)))
            .collect();
        assert_eq!(
            error_frames.len(),
            1,
            "expected exactly one ErrorFrame upstream, got: {:?}",
            frame_names(&upstream)
        );
        match &error_frames[0].frame {
            Frame::Error(e) => {
                assert!(!e.fatal, "auto-caught errors should be non-fatal");
                assert!(
                    e.error.contains("intentional failure"),
                    "error message should contain original error: {}",
                    e.error
                );
            }
            _ => unreachable!(),
        }

        // Interruption should still flow downstream.
        let downstream = drain_rx(&mut down_rx).await;
        let names = frame_names(&downstream);
        assert!(
            names.contains(&"Interruption".to_string()),
            "non-failing frames should still flow: {:?}",
            names
        );
    }
}
