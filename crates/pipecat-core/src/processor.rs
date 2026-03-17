use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::{PipecatError, Result};
use crate::frame::{Direction, ErrorFrame, Frame, FrameEnvelope, InterruptionFrame, MetricsFrame};
use crate::metrics::{LlmTokenUsage, ProcessorMetrics};
use crate::observer::{FramePushedEvent, PipelineObserver};

// ---------------------------------------------------------------------------
// Processor ID
// ---------------------------------------------------------------------------

static NEXT_PROCESSOR_ID: AtomicU64 = AtomicU64::new(1);

fn next_processor_id() -> u64 {
    NEXT_PROCESSOR_ID.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// ProcessorContext
// ---------------------------------------------------------------------------

/// The context through which processors push frames. Owns the downstream and
/// upstream channel senders. Clone is cheap (mpsc::Sender and Arc are ref-counted).
///
/// Processors receive this by reference in `process_frame`. Processors that
/// spawn async tasks should clone the context and move it into the task.
///
/// Optionally carries a `PipelineObserver` that is notified on every push.
#[derive(Clone)]
pub struct ProcessorContext {
    downstream: mpsc::Sender<FrameEnvelope>,
    upstream: mpsc::Sender<FrameEnvelope>,
    processor_id: u64,
    processor_name: String,
    observer: Option<Arc<dyn PipelineObserver>>,
    downstream_name: Option<String>,
    upstream_name: Option<String>,
}

impl std::fmt::Debug for ProcessorContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessorContext")
            .field("processor_id", &self.processor_id)
            .field("processor_name", &self.processor_name)
            .field("has_observer", &self.observer.is_some())
            .field("downstream_name", &self.downstream_name)
            .field("upstream_name", &self.upstream_name)
            .finish()
    }
}

impl ProcessorContext {
    pub fn new(
        downstream: mpsc::Sender<FrameEnvelope>,
        upstream: mpsc::Sender<FrameEnvelope>,
        processor_id: u64,
        processor_name: String,
    ) -> Self {
        Self {
            downstream,
            upstream,
            processor_id,
            processor_name,
            observer: None,
            downstream_name: None,
            upstream_name: None,
        }
    }

    /// Create a context with an observer attached.
    pub fn with_observer(
        downstream: mpsc::Sender<FrameEnvelope>,
        upstream: mpsc::Sender<FrameEnvelope>,
        processor_id: u64,
        processor_name: String,
        observer: Arc<dyn PipelineObserver>,
    ) -> Self {
        Self {
            downstream,
            upstream,
            processor_id,
            processor_name,
            observer: Some(observer),
            downstream_name: None,
            upstream_name: None,
        }
    }

    /// Set the name of the downstream processor (called during pipeline wiring).
    pub fn set_downstream_name(&mut self, name: String) {
        self.downstream_name = Some(name);
    }

    /// Set the name of the upstream processor (called during pipeline wiring).
    pub fn set_upstream_name(&mut self, name: String) {
        self.upstream_name = Some(name);
    }

    pub fn processor_id(&self) -> u64 {
        self.processor_id
    }

    pub fn processor_name(&self) -> &str {
        &self.processor_name
    }

    /// Push a frame in the given direction. Notifies the observer before send.
    ///
    /// Observer notification happens before the channel send so we can borrow
    /// the frame without cloning. If the send subsequently fails (channel closed
    /// during shutdown), the observer will have seen a frame that didn't actually
    /// arrive — this matches Python's behavior and only occurs during shutdown.
    pub async fn push_frame(&self, envelope: FrameEnvelope, direction: Direction) -> Result<()> {
        if let Some(ref observer) = self.observer {
            let dest_name = match direction {
                Direction::Downstream => self.downstream_name.as_deref(),
                Direction::Upstream => self.upstream_name.as_deref(),
            };
            observer
                .on_push_frame(FramePushedEvent {
                    source_name: &self.processor_name,
                    source_id: self.processor_id,
                    destination_name: dest_name,
                    frame: &envelope.frame,
                    direction,
                    timestamp: Instant::now(),
                })
                .await;
        }

        let sender = match direction {
            Direction::Downstream => &self.downstream,
            Direction::Upstream => &self.upstream,
        };
        sender.send(envelope).await.map_err(|e| {
            PipecatError::ChannelSend(format!(
                "{}({}): {}",
                self.processor_name, self.processor_id, e
            ))
        })
    }

    /// Push a frame downstream.
    pub async fn push_downstream(&self, envelope: FrameEnvelope) -> Result<()> {
        self.push_frame(envelope, Direction::Downstream).await
    }

    /// Push a frame upstream.
    pub async fn push_upstream(&self, envelope: FrameEnvelope) -> Result<()> {
        self.push_frame(envelope, Direction::Upstream).await
    }

    /// Create a new FrameEnvelope from a Frame and push it in the given direction.
    /// Convenience for the common case of creating and sending a new frame.
    pub async fn send(&self, frame: Frame, direction: Direction) -> Result<()> {
        self.push_frame(FrameEnvelope::new(frame), direction).await
    }

    /// Create a new FrameEnvelope from a Frame and push it downstream.
    pub async fn send_downstream(&self, frame: Frame) -> Result<()> {
        self.push_downstream(FrameEnvelope::new(frame)).await
    }

    /// Create a new FrameEnvelope from a Frame and push it upstream.
    pub async fn send_upstream(&self, frame: Frame) -> Result<()> {
        self.push_upstream(FrameEnvelope::new(frame)).await
    }

    /// Construct an ErrorFrame and push it upstream.
    pub async fn push_error(&self, error: &str, fatal: bool) -> Result<()> {
        self.send_upstream(Frame::Error(ErrorFrame {
            error: error.to_string(),
            fatal,
            source_processor: self.processor_name.clone(),
        }))
        .await
    }

    /// Send a frame both downstream and upstream. The two envelopes are linked
    /// via `broadcast_sibling_id` so observers can correlate them.
    pub async fn broadcast(&self, frame: Frame) -> Result<()> {
        let mut env1 = FrameEnvelope::new(frame.clone());
        let mut env2 = FrameEnvelope::new(frame);
        env1.header.broadcast_sibling_id = Some(env2.header.id);
        env2.header.broadcast_sibling_id = Some(env1.header.id);
        self.push_downstream(env1).await?;
        self.push_upstream(env2).await?;
        Ok(())
    }

    /// Broadcast an InterruptionFrame in both directions.
    pub async fn broadcast_interruption(&self) -> Result<()> {
        self.broadcast(Frame::Interruption(InterruptionFrame)).await
    }

    /// Stop a TTFB measurement and push the resulting MetricsFrame downstream.
    /// Callers should gate on their own metrics-enabled flag; this always pushes
    /// if the timer was started.
    pub async fn push_ttfb(&self, metrics: &mut ProcessorMetrics) -> Result<()> {
        if let Some(data) = metrics.stop_ttfb() {
            self.send_downstream(Frame::Metrics(MetricsFrame { data: vec![data] }))
                .await?;
        }
        Ok(())
    }

    /// Stop a processing measurement and push the resulting MetricsFrame downstream.
    /// Callers should gate on their own metrics-enabled flag; this always pushes
    /// if the timer was started.
    pub async fn push_processing_metrics(&self, metrics: &mut ProcessorMetrics) -> Result<()> {
        if let Some(data) = metrics.stop_processing() {
            self.send_downstream(Frame::Metrics(MetricsFrame { data: vec![data] }))
                .await?;
        }
        Ok(())
    }

    /// Push LLM usage metrics downstream. Always sends; callers should gate
    /// on their own metrics-enabled flag.
    pub async fn push_llm_usage(
        &self,
        metrics: &ProcessorMetrics,
        usage: LlmTokenUsage,
    ) -> Result<()> {
        let data = metrics.record_llm_usage(usage);
        self.send_downstream(Frame::Metrics(MetricsFrame { data: vec![data] }))
            .await
    }

    /// Push TTS usage metrics downstream. Always sends; callers should gate
    /// on their own metrics-enabled flag.
    pub async fn push_tts_usage(&self, metrics: &ProcessorMetrics, text: &str) -> Result<()> {
        let data = metrics.record_tts_usage(text);
        self.send_downstream(Frame::Metrics(MetricsFrame { data: vec![data] }))
            .await
    }
}

// ---------------------------------------------------------------------------
// FrameProcessor trait
// ---------------------------------------------------------------------------

/// The core processor trait. Implement this to process frames in a pipeline.
///
/// Processors receive frames via `process_frame` and push results via the
/// `ProcessorContext`. The ProcessorNode (runtime wrapper) handles priority
/// queuing, interruption draining, and lifecycle — this trait is pure logic.
///
/// # Common patterns
///
/// **Forward everything (passthrough):**
/// ```ignore
/// async fn process_frame(&mut self, envelope: FrameEnvelope, direction: Direction, ctx: &ProcessorContext) {
///     ctx.push_frame(envelope, direction).await.ok();
/// }
/// ```
///
/// **Handle specific frames, forward the rest:**
/// ```ignore
/// async fn process_frame(&mut self, envelope: FrameEnvelope, direction: Direction, ctx: &ProcessorContext) {
///     match &envelope.frame {
///         Frame::Text(t) => {
///             // Transform and send a new frame
///             ctx.send_downstream(Frame::Text(TextFrame::new(t.text.to_uppercase()))).await.ok();
///         }
///         _ => {
///             // Forward everything else unchanged
///             ctx.push_frame(envelope, direction).await.ok();
///         }
///     }
/// }
/// ```
///
/// **Create new frames:**
/// ```ignore
/// ctx.send_downstream(Frame::TTSStarted(TTSStartedFrame { context_id: Some("ctx1".into()) })).await?;
/// ```
#[async_trait]
pub trait FrameProcessor: Send + Sync {
    /// Unique name for this processor instance. Used in logging, error reporting,
    /// and observer events.
    fn name(&self) -> &str;

    /// Unique ID for this processor instance. Default implementation generates
    /// an auto-incrementing ID. Override only if you need deterministic IDs
    /// (e.g., for testing).
    fn id(&self) -> u64 {
        0 // Overridden by ProcessorBase
    }

    /// Process a single frame. Called by the ProcessorNode for every frame
    /// that reaches this processor. The processor should either handle the
    /// frame (producing output via `ctx`) or forward it unchanged.
    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    );

    /// Called once before the processor starts receiving frames.
    /// Override to perform one-time initialization: open connections,
    /// load models, allocate buffers, etc. Default is a no-op.
    ///
    /// Called by `ProcessorNode::run()` before entering the event loop.
    async fn setup(&mut self) {}

    /// Called when the ProcessorNode shuts down (after Cancel, or when channels
    /// close). Override to release resources: close connections, flush buffers,
    /// cancel background tasks, etc. Default is a no-op.
    async fn cleanup(&mut self) {}

    /// Called before `process_frame`. Override for pre-processing hooks
    /// (start metrics timers, validate frames, log incoming frames).
    async fn on_before_process(&mut self, _frame: &Frame, _direction: Direction) {}

    /// Called after `process_frame` completes. Override for post-processing hooks
    /// (stop metrics timers, log completion). The frame is not available here
    /// because `process_frame` consumed the envelope — processors that need
    /// the frame in after-hooks should store it in `on_before_process`.
    async fn on_after_process(&mut self) {}
}

// ---------------------------------------------------------------------------
// ProcessorBase — common state for processor implementations
// ---------------------------------------------------------------------------

/// Common state that processor implementations embed. Provides a unique ID
/// and name. Use this as a field in your processor struct:
///
/// ```ignore
/// struct MyProcessor {
///     base: ProcessorBase,
///     // ... your fields
/// }
///
/// impl FrameProcessor for MyProcessor {
///     fn name(&self) -> &str { self.base.name() }
///     fn id(&self) -> u64 { self.base.id() }
///     // ...
/// }
/// ```
#[derive(Debug)]
pub struct ProcessorBase {
    id: u64,
    name: String,
}

impl ProcessorBase {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: next_processor_id(),
            name: name.into(),
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Frame, InterruptionFrame, StartFrame, TextFrame};
    use crate::test_utils::*;

    // -- ProcessorBase tests --

    #[test]
    fn processor_base_generates_unique_ids() {
        let a = ProcessorBase::new("a");
        let b = ProcessorBase::new("b");
        let c = ProcessorBase::new("c");
        assert_ne!(a.id(), b.id());
        assert_ne!(b.id(), c.id());
    }

    #[test]
    fn processor_base_stores_name() {
        let base = ProcessorBase::new("MyProcessor");
        assert_eq!(base.name(), "MyProcessor");
    }

    // -- ProcessorContext tests --

    fn make_ctx(
        channel_size: usize,
    ) -> (
        ProcessorContext,
        mpsc::Receiver<FrameEnvelope>,
        mpsc::Receiver<FrameEnvelope>,
    ) {
        let (down_tx, down_rx) = mpsc::channel(channel_size);
        let (up_tx, up_rx) = mpsc::channel(channel_size);
        let ctx = ProcessorContext::new(down_tx, up_tx, 99, "test-ctx".into());
        (ctx, down_rx, up_rx)
    }

    #[tokio::test]
    async fn context_is_cloneable() {
        let (ctx, mut down_rx, _up_rx) = make_ctx(16);

        let ctx_clone = ctx.clone();
        let handle = tokio::spawn(async move {
            ctx_clone
                .send_downstream(Frame::Text(TextFrame::new("from task")))
                .await
                .unwrap();
        });

        handle.await.unwrap();
        let received = down_rx.recv().await.unwrap();
        assert!(matches!(received.frame, Frame::Text(t) if t.text == "from task"));
    }

    #[tokio::test]
    async fn context_push_frame_downstream() {
        let (ctx, mut down_rx, _up_rx) = make_ctx(16);

        let envelope = make_text_frame("hello");
        let id = envelope.header.id;
        ctx.push_downstream(envelope).await.unwrap();

        let received = down_rx.recv().await.unwrap();
        assert_eq!(received.header.id, id);
        assert!(matches!(received.frame, Frame::Text(t) if t.text == "hello"));
    }

    #[tokio::test]
    async fn context_push_frame_upstream() {
        let (ctx, mut down_rx, mut up_rx) = make_ctx(16);

        let envelope = make_interruption_frame();
        ctx.push_upstream(envelope).await.unwrap();

        let received = up_rx.recv().await.unwrap();
        assert!(matches!(received.frame, Frame::Interruption(_)));
        assert!(down_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn context_send_creates_envelope() {
        let (ctx, mut down_rx, _up_rx) = make_ctx(16);

        ctx.send_downstream(Frame::Text(TextFrame::new("auto-wrapped")))
            .await
            .unwrap();

        let received = down_rx.recv().await.unwrap();
        assert!(received.header.id > 0);
        assert!(matches!(received.frame, Frame::Text(t) if t.text == "auto-wrapped"));
    }

    #[tokio::test]
    async fn context_push_error_non_fatal() {
        let (ctx, _down_rx, mut up_rx) = make_ctx(16);
        ctx.push_error("something broke", false).await.unwrap();

        let received = up_rx.recv().await.unwrap();
        match &received.frame {
            Frame::Error(e) => {
                assert_eq!(e.error, "something broke");
                assert!(!e.fatal);
                assert_eq!(e.source_processor, "test-ctx");
            }
            other => panic!("expected ErrorFrame, got {other}"),
        }
    }

    #[tokio::test]
    async fn context_push_error_fatal() {
        let (ctx, _down_rx, mut up_rx) = make_ctx(16);
        ctx.push_error("unrecoverable", true).await.unwrap();

        let received = up_rx.recv().await.unwrap();
        match &received.frame {
            Frame::Error(e) => {
                assert_eq!(e.error, "unrecoverable");
                assert!(e.fatal);
            }
            other => panic!("expected ErrorFrame, got {other}"),
        }
    }

    #[tokio::test]
    async fn context_returns_error_on_closed_channel() {
        let (down_tx, _) = mpsc::channel(1);
        let (up_tx, _) = mpsc::channel(1);
        let ctx = ProcessorContext::new(down_tx, up_tx, 1, "broken".into());

        let result = ctx.send_downstream(Frame::Text(TextFrame::new("x"))).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            PipecatError::ChannelSend(msg) => {
                assert!(msg.contains("broken"));
                assert!(msg.contains("1"));
            }
            other => panic!("expected ChannelSend, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn context_exposes_processor_identity() {
        let (ctx, _down_rx, _up_rx) = make_ctx(16);
        assert_eq!(ctx.processor_name(), "test-ctx");
        assert_eq!(ctx.processor_id(), 99);
    }

    // -- FrameProcessor trait tests using run_processor --

    #[tokio::test]
    async fn passthrough_forwards_downstream() {
        let mut proc = PassthroughProcessor::new();
        let (downstream, _upstream) = run_processor(
            &mut proc,
            vec![(Frame::Text(TextFrame::new("hello")), Direction::Downstream)],
        )
        .await;

        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::Text(t) if t.text == "hello"));
    }

    #[tokio::test]
    async fn passthrough_forwards_upstream() {
        let mut proc = PassthroughProcessor::new();
        let (_downstream, upstream) = run_processor(
            &mut proc,
            vec![(Frame::Interruption(InterruptionFrame), Direction::Upstream)],
        )
        .await;

        assert_eq!(upstream.len(), 1);
        assert!(matches!(&upstream[0].frame, Frame::Interruption(_)));
    }

    #[tokio::test]
    async fn collector_stores_all_frames() {
        let (mut proc, collected) = CollectorProcessor::new();
        let _ = run_processor(
            &mut proc,
            (0..5)
                .map(|i| {
                    (
                        Frame::Text(TextFrame::new(format!("msg{i}"))),
                        Direction::Downstream,
                    )
                })
                .collect(),
        )
        .await;

        let frames = collected.lock().unwrap();
        assert_eq!(frames.len(), 5);
        assert!(matches!(&frames[0].frame, Frame::Text(t) if t.text == "msg0"));
        assert!(matches!(&frames[4].frame, Frame::Text(t) if t.text == "msg4"));
    }

    #[tokio::test]
    async fn collector_stores_mixed_frame_types() {
        let (mut proc, collected) = CollectorProcessor::new();
        let _ = run_processor(
            &mut proc,
            vec![
                (Frame::Start(StartFrame::default()), Direction::Downstream),
                (Frame::Text(TextFrame::new("hi")), Direction::Downstream),
                (
                    Frame::Interruption(InterruptionFrame),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        let frames = collected.lock().unwrap();
        assert_eq!(frames.len(), 3);
        assert!(matches!(&frames[0].frame, Frame::Start(_)));
        assert!(matches!(&frames[1].frame, Frame::Text(_)));
        assert!(matches!(&frames[2].frame, Frame::Interruption(_)));
    }

    #[tokio::test]
    async fn uppercase_transforms_text_forwards_others() {
        let mut proc = UppercaseProcessor::new();
        let (downstream, _) = run_processor(
            &mut proc,
            vec![
                (
                    Frame::Text(TextFrame::new("hello world")),
                    Direction::Downstream,
                ),
                (
                    Frame::Interruption(InterruptionFrame),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        assert_eq!(downstream.len(), 2);
        assert!(matches!(&downstream[0].frame, Frame::Text(t) if t.text == "HELLO WORLD"));
        assert!(matches!(&downstream[1].frame, Frame::Interruption(_)));
    }

    #[tokio::test]
    async fn error_on_text_pushes_error_upstream() {
        let mut proc = ErrorOnTextProcessor::new();
        let (downstream, upstream) = run_processor(
            &mut proc,
            vec![
                (Frame::Text(TextFrame::new("bad")), Direction::Downstream),
                (
                    Frame::Interruption(InterruptionFrame),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        // Error goes upstream
        assert_eq!(upstream.len(), 1);
        assert!(matches!(&upstream[0].frame, Frame::Error(e) if e.error == "text not allowed"));

        // Non-text goes downstream
        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::Interruption(_)));
    }

    #[tokio::test]
    async fn processor_base_used_in_trait() {
        let proc = PassthroughProcessor::new();
        assert_eq!(proc.name(), "Passthrough");
        assert!(proc.id() > 0);

        let (proc2, _) = CollectorProcessor::new();
        assert_eq!(proc2.name(), "Collector");
        assert_ne!(proc.id(), proc2.id());
    }
}
