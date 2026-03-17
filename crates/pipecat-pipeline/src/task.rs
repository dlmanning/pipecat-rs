use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::*;
use pipecat_core::node::ProcessorNode;
use pipecat_core::observer::PipelineObserver;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinHandle;
use tracing::{debug, error, warn};

use crate::pipeline::Pipeline;

const CHANNEL_SIZE: usize = 64;
const DEFAULT_HEARTBEAT_PERIOD: Duration = Duration::from_secs(1);
const DEFAULT_HEARTBEAT_MONITOR_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_CANCEL_TIMEOUT: Duration = Duration::from_secs(20);
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// Event callback types
// ---------------------------------------------------------------------------

/// Async callback that takes no arguments.
pub type EventCallback = Box<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Async callback that receives an ErrorFrame.
pub type ErrorCallback =
    Box<dyn Fn(ErrorFrame) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

// ---------------------------------------------------------------------------
// PipelineParams
// ---------------------------------------------------------------------------

/// Parameters for pipeline initialization. Carried in the StartFrame.
#[derive(Debug, Clone)]
pub struct PipelineParams {
    pub audio_in_sample_rate: u32,
    pub audio_out_sample_rate: u32,
    pub allow_interruptions: bool,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
    pub report_only_initial_ttfb: bool,
    pub enable_heartbeats: bool,
    pub heartbeat_period: Duration,
    pub idle_timeout: Option<Duration>,
    /// When true (default), the pipeline is cancelled when the idle timeout fires.
    /// When false, only the `on_idle_timeout` callback is invoked.
    pub cancel_on_idle_timeout: bool,
    pub cancel_timeout: Duration,
}

impl Default for PipelineParams {
    fn default() -> Self {
        Self {
            audio_in_sample_rate: 16000,
            audio_out_sample_rate: 24000,
            allow_interruptions: true,
            enable_metrics: false,
            enable_usage_metrics: false,
            report_only_initial_ttfb: false,
            enable_heartbeats: false,
            heartbeat_period: DEFAULT_HEARTBEAT_PERIOD,
            idle_timeout: Some(DEFAULT_IDLE_TIMEOUT),
            cancel_on_idle_timeout: true,
            cancel_timeout: DEFAULT_CANCEL_TIMEOUT,
        }
    }
}

// ---------------------------------------------------------------------------
// TaskSource — intercepts upstream frames escaping from the user pipeline
// ---------------------------------------------------------------------------

struct TaskSource {
    base: ProcessorBase,
    upstream_tx: mpsc::Sender<FrameEnvelope>,
}

#[async_trait]
impl FrameProcessor for TaskSource {
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
        match direction {
            Direction::Downstream => {
                ctx.push_downstream(envelope).await?;
            }
            Direction::Upstream => {
                self.upstream_tx.send(envelope).await.ok();
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TaskSink — intercepts downstream frames exiting the user pipeline
// ---------------------------------------------------------------------------

struct TaskSink {
    base: ProcessorBase,
    downstream_tx: mpsc::Sender<FrameEnvelope>,
}

#[async_trait]
impl FrameProcessor for TaskSink {
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
        match direction {
            Direction::Downstream => {
                self.downstream_tx.send(envelope).await.ok();
            }
            Direction::Upstream => {
                ctx.push_upstream(envelope).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PipelineTask
// ---------------------------------------------------------------------------

/// Lifecycle manager for a pipeline. Wraps the user's pipeline in a
/// task-level source/sink, manages StartFrame injection, handles upstream
/// errors and task control frames, and optionally runs heartbeat and idle
/// monitoring.
///
/// # Usage
///
/// ```ignore
/// let pipeline = Pipeline::new(vec![...]);
/// let mut task = PipelineTask::new(Box::new(pipeline), PipelineParams::default());
/// task.run().await?;
/// ```
pub struct PipelineTask {
    user_pipeline: Option<Box<dyn FrameProcessor>>,
    params: PipelineParams,
    observer: Option<Arc<dyn PipelineObserver>>,
    pub(crate) push_tx: mpsc::Sender<Frame>,
    push_rx: Option<mpsc::Receiver<Frame>>,
    started: Arc<Notify>,
    ended: Arc<Notify>,
    finished: bool,
    // Event callbacks
    on_started: Option<EventCallback>,
    on_finished: Option<EventCallback>,
    on_error: Option<ErrorCallback>,
    on_idle_timeout: Option<EventCallback>,
}

impl std::fmt::Debug for PipelineTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineTask")
            .field("finished", &self.finished)
            .finish()
    }
}

impl PipelineTask {
    pub fn new(pipeline: Box<dyn FrameProcessor>, params: PipelineParams) -> Self {
        let (push_tx, push_rx) = mpsc::channel(CHANNEL_SIZE);
        Self {
            user_pipeline: Some(pipeline),
            params,
            observer: None,
            push_tx,
            push_rx: Some(push_rx),
            started: Arc::new(Notify::new()),
            ended: Arc::new(Notify::new()),
            finished: false,
            on_started: None,
            on_finished: None,
            on_error: None,
            on_idle_timeout: None,
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn PipelineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Register a callback fired when StartFrame reaches the sink.
    pub fn on_pipeline_started<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_started = Some(Box::new(move || Box::pin(f())));
        self
    }

    /// Register a callback fired when End/Stop reaches the sink.
    pub fn on_pipeline_finished<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_finished = Some(Box::new(move || Box::pin(f())));
        self
    }

    /// Register a callback fired on errors (fatal and non-fatal) from the pipeline.
    pub fn on_pipeline_error<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(ErrorFrame) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_error = Some(Box::new(move |e| Box::pin(f(e))));
        self
    }

    /// Register a callback fired when the idle timeout expires. Called before
    /// cancellation (if `cancel_on_idle_timeout` is true).
    pub fn on_idle_timeout<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_idle_timeout = Some(Box::new(move || Box::pin(f())));
        self
    }

    /// Queue a frame for injection into the pipeline (downstream).
    pub async fn queue_frame(&self, frame: Frame) -> Result<()> {
        self.push_tx
            .send(frame)
            .await
            .map_err(|e| PipecatError::ChannelSend(format!("PipelineTask queue_frame: {e}")))
    }

    /// Graceful shutdown: queue an EndFrame.
    pub async fn stop_when_done(&self) {
        self.queue_frame(Frame::End(EndFrame::default())).await.ok();
    }

    /// Immediate cancellation: queue a CancelFrame.
    pub async fn cancel(&self) {
        self.queue_frame(Frame::Cancel(CancelFrame::default()))
            .await
            .ok();
    }

    pub fn has_finished(&self) -> bool {
        self.finished
    }

    /// Run the pipeline to completion.
    pub async fn run(&mut self) -> Result<()> {
        if self.finished {
            return Err(PipecatError::PipelineError("task already finished".into()));
        }

        let user_pipeline = self
            .user_pipeline
            .take()
            .ok_or_else(|| PipecatError::PipelineError("pipeline already consumed".into()))?;
        let mut push_rx = self
            .push_rx
            .take()
            .ok_or_else(|| PipecatError::PipelineError("push_rx already consumed".into()))?;

        // Channels for TaskSource/TaskSink to send intercepted frames back.
        let (source_up_tx, mut source_up_rx) = mpsc::channel::<FrameEnvelope>(CHANNEL_SIZE);
        let (sink_down_tx, mut sink_down_rx) = mpsc::channel::<FrameEnvelope>(CHANNEL_SIZE);

        let source = TaskSource {
            base: ProcessorBase::new("TaskSource"),
            upstream_tx: source_up_tx,
        };
        let sink = TaskSink {
            base: ProcessorBase::new("TaskSink"),
            downstream_tx: sink_down_tx,
        };

        // Wrap: [TaskSource, user_pipeline, TaskSink]
        let mut task_pipeline = Pipeline::new(vec![
            Box::new(source) as Box<dyn FrameProcessor>,
            user_pipeline,
            Box::new(sink) as Box<dyn FrameProcessor>,
        ]);
        if let Some(ref obs) = self.observer {
            task_pipeline = task_pipeline.with_observer(Arc::clone(obs));
        }

        // Wrap the task pipeline in a ProcessorNode to drive it.
        let (pipeline_down_tx, _pipeline_down_rx) = mpsc::channel(CHANNEL_SIZE);
        let (pipeline_up_tx, _pipeline_up_rx) = mpsc::channel(CHANNEL_SIZE);
        let (node, node_handle) = if let Some(ref obs) = self.observer {
            ProcessorNode::with_observer(
                Box::new(task_pipeline),
                pipeline_down_tx,
                pipeline_up_tx,
                CHANNEL_SIZE,
                Arc::clone(obs),
            )
        } else {
            ProcessorNode::new(
                Box::new(task_pipeline),
                pipeline_down_tx,
                pipeline_up_tx,
                CHANNEL_SIZE,
            )
        };

        let mut node_task: JoinHandle<()> = tokio::spawn(async move { node.run().await });

        // -- Send StartFrame --
        let start_frame = Frame::Start(StartFrame {
            audio_in_sample_rate: self.params.audio_in_sample_rate,
            audio_out_sample_rate: self.params.audio_out_sample_rate,
            allow_interruptions: self.params.allow_interruptions,
            enable_metrics: self.params.enable_metrics,
            enable_usage_metrics: self.params.enable_usage_metrics,
            report_only_initial_ttfb: self.params.report_only_initial_ttfb,
        });

        node_handle
            .send(FrameEnvelope::new(start_frame), Direction::Downstream)
            .await?;
        debug!("PipelineTask: StartFrame sent, waiting for propagation");

        // -- Heartbeat and idle channels --
        let mut background_tasks: Vec<JoinHandle<()>> = Vec::new();
        let (heartbeat_tx, heartbeat_rx) = mpsc::channel::<HeartbeatFrame>(CHANNEL_SIZE);
        let idle_notify = Arc::new(Notify::new());

        // -- Sink monitor --
        let sink_monitor = {
            let started = Arc::clone(&self.started);
            let ended = Arc::clone(&self.ended);
            let heartbeat_tx = heartbeat_tx.clone();
            let idle_notify = Arc::clone(&idle_notify);
            let on_started = self.on_started.take();
            let on_finished = self.on_finished.take();
            tokio::spawn(async move {
                while let Some(envelope) = sink_down_rx.recv().await {
                    match &envelope.frame {
                        Frame::Start(_) => {
                            debug!("PipelineTask: StartFrame reached sink");
                            if let Some(ref cb) = on_started {
                                cb().await;
                            }
                            started.notify_waiters();
                        }
                        Frame::End(_) | Frame::Stop(_) => {
                            debug!(
                                "PipelineTask: terminal frame reached sink: {}",
                                envelope.frame
                            );
                            if let Some(ref cb) = on_finished {
                                cb().await;
                            }
                            ended.notify_waiters();
                            break;
                        }
                        Frame::Cancel(_) => {
                            debug!("PipelineTask: CancelFrame reached sink");
                            if let Some(ref cb) = on_finished {
                                cb().await;
                            }
                            ended.notify_waiters();
                            break;
                        }
                        Frame::Heartbeat(hb) => {
                            heartbeat_tx.send(hb.clone()).await.ok();
                        }
                        // Activity frames reset idle timeout.
                        Frame::BotSpeaking(_)
                        | Frame::UserSpeaking(_)
                        | Frame::UserStartedSpeaking(_)
                        | Frame::BotStartedSpeaking(_) => {
                            idle_notify.notify_waiters();
                        }
                        _ => {}
                    }
                }
            })
        };

        // -- Source monitor --
        let source_monitor = {
            let push_tx = self.push_tx.clone();
            let node_handle_for_source = node_handle.clone();
            let on_error = self.on_error.take();
            tokio::spawn(async move {
                while let Some(envelope) = source_up_rx.recv().await {
                    match envelope.frame {
                        Frame::Error(ref e) => {
                            if let Some(ref cb) = on_error {
                                cb(e.clone()).await;
                            }
                            if e.fatal {
                                error!(
                                    "PipelineTask: fatal error from {}: {}",
                                    e.source_processor, e.error
                                );
                                push_tx
                                    .send(Frame::Cancel(CancelFrame::default()))
                                    .await
                                    .ok();
                            } else {
                                warn!(
                                    "PipelineTask: non-fatal error from {}: {}",
                                    e.source_processor, e.error
                                );
                            }
                        }
                        Frame::EndTask(_) => {
                            push_tx.send(Frame::End(EndFrame::default())).await.ok();
                        }
                        Frame::CancelTask(_) => {
                            push_tx
                                .send(Frame::Cancel(CancelFrame::default()))
                                .await
                                .ok();
                        }
                        Frame::StopTask(_) => {
                            push_tx.send(Frame::Stop(StopFrame)).await.ok();
                        }
                        Frame::InterruptionTask(_) => {
                            node_handle_for_source
                                .send(
                                    FrameEnvelope::new(Frame::Interruption(InterruptionFrame)),
                                    Direction::Downstream,
                                )
                                .await
                                .ok();
                        }
                        _ => {}
                    }
                }
            })
        };

        // Wait for pipeline to start.
        self.started.notified().await;
        debug!("PipelineTask: pipeline started");

        // -- Start heartbeat if enabled --
        if self.params.enable_heartbeats {
            let period = self.params.heartbeat_period;
            let handle_for_hb = node_handle.clone();
            background_tasks.push(tokio::spawn(async move {
                let mut interval = tokio::time::interval(period);
                loop {
                    interval.tick().await;
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;
                    let frame = Frame::Heartbeat(HeartbeatFrame { timestamp: ts });
                    if handle_for_hb
                        .send(FrameEnvelope::new(frame), Direction::Downstream)
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }));
            background_tasks.push(tokio::spawn(heartbeat_monitor(heartbeat_rx)));
        }

        // -- Start idle monitor if configured --
        if let Some(idle_dur) = self.params.idle_timeout {
            let idle_n = Arc::clone(&idle_notify);
            let idle_push_tx = self.push_tx.clone();
            let on_idle = self.on_idle_timeout.take();
            let cancel_on_idle = self.params.cancel_on_idle_timeout;
            // Notify once so the first wait period starts fresh.
            idle_n.notify_waiters();
            background_tasks.push(tokio::spawn(async move {
                loop {
                    if tokio::time::timeout(idle_dur, idle_n.notified())
                        .await
                        .is_err()
                    {
                        warn!("PipelineTask: idle timeout ({}s)", idle_dur.as_secs());
                        if let Some(ref cb) = on_idle {
                            cb().await;
                        }
                        if cancel_on_idle {
                            idle_push_tx
                                .send(Frame::Cancel(CancelFrame::default()))
                                .await
                                .ok();
                        }
                        break;
                    }
                }
            }));
        }

        // -- Main loop: process pushed frames --
        // Use select to also detect node panics.
        let cancel_timeout = self.params.cancel_timeout;
        let mut terminal_frame: Option<Frame> = None;

        loop {
            tokio::select! {
                frame = push_rx.recv() => {
                    let Some(frame) = frame else {
                        debug!("PipelineTask: push channel closed");
                        break;
                    };

                    let is_terminal = matches!(
                        frame,
                        Frame::End(_) | Frame::Cancel(_) | Frame::Stop(_)
                    );
                    let is_cancel = matches!(frame, Frame::Cancel(_));

                    if is_terminal {
                        terminal_frame = Some(frame.clone());
                    }

                    node_handle
                        .send(FrameEnvelope::new(frame), Direction::Downstream)
                        .await
                        .ok();

                    if is_terminal {
                        if is_cancel {
                            let wait = self.ended.notified();
                            if tokio::time::timeout(cancel_timeout, wait).await.is_err() {
                                warn!(
                                    "PipelineTask: cancel timeout ({}s)",
                                    cancel_timeout.as_secs()
                                );
                            }
                        } else {
                            self.ended.notified().await;
                        }
                        break;
                    }
                }
                result = &mut node_task => {
                    // Node exited unexpectedly (panic or early shutdown).
                    if let Err(ref e) = result {
                        error!("PipelineTask: pipeline node panicked: {e}");
                    }
                    // Don't try to interact with the node further.
                    self.finished = true;

                    // Cleanup background tasks.
                    for task in background_tasks {
                        task.abort();
                    }
                    sink_monitor.abort();
                    source_monitor.abort();

                    debug!("PipelineTask: finished (node exited early)");

                    return if result.is_err() {
                        Err(PipecatError::PipelineError("pipeline node panicked".into()))
                    } else {
                        Ok(())
                    };
                }
            }
        }

        debug!("PipelineTask: shutting down");

        // Determine cleanup strategy based on terminal frame type.
        let cleanup_pipeline = !matches!(terminal_frame, Some(Frame::Stop(_)));

        if cleanup_pipeline {
            // Send Cancel to fully shut down nodes (EndFrame alone doesn't exit nodes).
            node_handle
                .send(
                    FrameEnvelope::new(Frame::Cancel(CancelFrame::default())),
                    Direction::Downstream,
                )
                .await
                .ok();

            // Drop the handle to close channels, then wait for the node.
            drop(node_handle);
            let _ = tokio::time::timeout(Duration::from_secs(5), node_task).await;
        } else {
            // StopFrame: don't cleanup the pipeline — leave it alive for potential reuse.
            drop(node_handle);
        }

        // Cleanup background tasks.
        for task in background_tasks {
            task.abort();
        }
        drop(heartbeat_tx);

        // Wait briefly for monitors to exit (their channels should be closing).
        let _ = tokio::time::timeout(Duration::from_millis(100), sink_monitor).await;
        let _ = tokio::time::timeout(Duration::from_millis(100), source_monitor).await;

        self.finished = true;
        debug!("PipelineTask: finished");

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Heartbeat monitor
// ---------------------------------------------------------------------------

async fn heartbeat_monitor(mut rx: mpsc::Receiver<HeartbeatFrame>) {
    loop {
        match tokio::time::timeout(DEFAULT_HEARTBEAT_MONITOR_TIMEOUT, rx.recv()).await {
            Ok(Some(hb)) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                let latency_ms = now.saturating_sub(hb.timestamp) / 1_000_000;
                if latency_ms > 1000 {
                    warn!("PipelineTask: heartbeat latency {}ms", latency_ms);
                }
            }
            Ok(None) => break,
            Err(_) => {
                warn!(
                    "PipelineTask: no heartbeat received for {}s",
                    DEFAULT_HEARTBEAT_MONITOR_TIMEOUT.as_secs()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use pipecat_core::test_utils::*;

    use super::*;

    #[tokio::test]
    async fn basic_lifecycle() {
        let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                ..Default::default()
            },
        );

        let push_tx = task.push_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            push_tx.send(Frame::End(EndFrame::default())).await.ok();
        });

        let result = task.run().await;
        assert!(result.is_ok());
        assert!(task.has_finished());
    }

    #[tokio::test]
    async fn stop_when_done() {
        let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                ..Default::default()
            },
        );

        let task_handle = {
            let push_tx = task.push_tx.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                push_tx.send(Frame::End(EndFrame::default())).await.ok();
            })
        };

        task.run().await.unwrap();
        task_handle.await.unwrap();
        assert!(task.has_finished());
    }

    #[tokio::test]
    async fn cancel_shuts_down() {
        let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                ..Default::default()
            },
        );

        let push_tx = task.push_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            push_tx
                .send(Frame::Cancel(CancelFrame::default()))
                .await
                .ok();
        });

        let result = task.run().await;
        assert!(result.is_ok());
        assert!(task.has_finished());
    }

    #[tokio::test]
    async fn fatal_error_triggers_cancel() {
        struct FatalErrorProcessor(ProcessorBase);
        impl FatalErrorProcessor {
            fn new() -> Self {
                Self(ProcessorBase::new("FatalError"))
            }
        }
        #[async_trait]
        impl FrameProcessor for FatalErrorProcessor {
            fn name(&self) -> &str {
                self.0.name()
            }
            fn id(&self) -> u64 {
                self.0.id()
            }
            async fn process_frame(
                &mut self,
                envelope: FrameEnvelope,
                direction: Direction,
                ctx: &ProcessorContext,
            ) -> Result<()> {
                if matches!(&envelope.frame, Frame::Text(_)) {
                    ctx.push_error("fatal!", true).await?;
                } else {
                    ctx.push_frame(envelope, direction).await?;
                }
                Ok(())
            }
        }

        let pipeline = Pipeline::new(vec![Box::new(FatalErrorProcessor::new())]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                ..Default::default()
            },
        );

        let push_tx = task.push_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            push_tx
                .send(Frame::Text(TextFrame::new("trigger")))
                .await
                .ok();
        });

        let result = tokio::time::timeout(Duration::from_secs(5), task.run()).await;
        assert!(result.is_ok());
        assert!(task.has_finished());
    }

    #[tokio::test]
    async fn task_frame_triggers_end() {
        struct EndTaskProcessor(ProcessorBase);
        impl EndTaskProcessor {
            fn new() -> Self {
                Self(ProcessorBase::new("EndTaskProc"))
            }
        }
        #[async_trait]
        impl FrameProcessor for EndTaskProcessor {
            fn name(&self) -> &str {
                self.0.name()
            }
            fn id(&self) -> u64 {
                self.0.id()
            }
            async fn process_frame(
                &mut self,
                envelope: FrameEnvelope,
                direction: Direction,
                ctx: &ProcessorContext,
            ) -> Result<()> {
                if matches!(&envelope.frame, Frame::Text(_)) {
                    ctx.send_upstream(Frame::EndTask(EndTaskFrame {
                        task_id: "t".into(),
                        handler_id: "h".into(),
                        reason: None,
                    }))
                    .await?;
                } else {
                    ctx.push_frame(envelope, direction).await?;
                }
                Ok(())
            }
        }

        let pipeline = Pipeline::new(vec![Box::new(EndTaskProcessor::new())]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                ..Default::default()
            },
        );

        let push_tx = task.push_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            push_tx
                .send(Frame::Text(TextFrame::new("trigger")))
                .await
                .ok();
        });

        let result = tokio::time::timeout(Duration::from_secs(5), task.run()).await;
        assert!(result.is_ok());
        assert!(task.has_finished());
    }

    #[tokio::test]
    async fn double_run_errors() {
        let pipeline = Pipeline::new(vec![]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                ..Default::default()
            },
        );

        let push_tx = task.push_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            push_tx
                .send(Frame::Cancel(CancelFrame::default()))
                .await
                .ok();
        });

        task.run().await.unwrap();
        let result = task.run().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn event_callbacks_fire() {
        let started_flag = Arc::new(AtomicBool::new(false));
        let finished_flag = Arc::new(AtomicBool::new(false));

        let sf = Arc::clone(&started_flag);
        let ff = Arc::clone(&finished_flag);

        let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                ..Default::default()
            },
        )
        .on_pipeline_started(move || {
            let sf = Arc::clone(&sf);
            async move {
                sf.store(true, Ordering::SeqCst);
            }
        })
        .on_pipeline_finished(move || {
            let ff = Arc::clone(&ff);
            async move {
                ff.store(true, Ordering::SeqCst);
            }
        });

        let push_tx = task.push_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            push_tx.send(Frame::End(EndFrame::default())).await.ok();
        });

        task.run().await.unwrap();

        assert!(
            started_flag.load(Ordering::SeqCst),
            "on_started should fire"
        );
        assert!(
            finished_flag.load(Ordering::SeqCst),
            "on_finished should fire"
        );
    }

    #[tokio::test]
    async fn cancel_fires_finished_callback() {
        let finished_flag = Arc::new(AtomicBool::new(false));
        let ff = Arc::clone(&finished_flag);

        let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                ..Default::default()
            },
        )
        .on_pipeline_finished(move || {
            let ff = Arc::clone(&ff);
            async move {
                ff.store(true, Ordering::SeqCst);
            }
        });

        let push_tx = task.push_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            push_tx
                .send(Frame::Cancel(CancelFrame::default()))
                .await
                .ok();
        });

        task.run().await.unwrap();

        assert!(
            finished_flag.load(Ordering::SeqCst),
            "on_finished should fire on CancelFrame"
        );
    }

    #[tokio::test]
    async fn non_fatal_error_fires_callback() {
        let error_flag = Arc::new(AtomicBool::new(false));
        let ef = Arc::clone(&error_flag);

        struct NonFatalErrorProcessor(ProcessorBase);
        impl NonFatalErrorProcessor {
            fn new() -> Self {
                Self(ProcessorBase::new("NonFatalError"))
            }
        }
        #[async_trait]
        impl FrameProcessor for NonFatalErrorProcessor {
            fn name(&self) -> &str {
                self.0.name()
            }
            fn id(&self) -> u64 {
                self.0.id()
            }
            async fn process_frame(
                &mut self,
                envelope: FrameEnvelope,
                direction: Direction,
                ctx: &ProcessorContext,
            ) -> Result<()> {
                if matches!(&envelope.frame, Frame::Text(_)) {
                    ctx.push_error("non-fatal!", false).await?;
                } else {
                    ctx.push_frame(envelope, direction).await?;
                }
                Ok(())
            }
        }

        let pipeline = Pipeline::new(vec![Box::new(NonFatalErrorProcessor::new())]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                ..Default::default()
            },
        )
        .on_pipeline_error(move |_e| {
            let ef = Arc::clone(&ef);
            async move {
                ef.store(true, Ordering::SeqCst);
            }
        });

        let push_tx = task.push_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            push_tx
                .send(Frame::Text(TextFrame::new("trigger")))
                .await
                .ok();
            tokio::time::sleep(Duration::from_millis(100)).await;
            push_tx
                .send(Frame::Cancel(CancelFrame::default()))
                .await
                .ok();
        });

        task.run().await.unwrap();

        assert!(
            error_flag.load(Ordering::SeqCst),
            "on_error should fire for non-fatal errors"
        );
    }

    #[tokio::test]
    async fn idle_timeout_cancels() {
        let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                idle_timeout: Some(Duration::from_millis(100)),
                ..Default::default()
            },
        );

        // Don't send any activity frames — idle timeout should cancel.
        let result = tokio::time::timeout(Duration::from_secs(5), task.run()).await;
        assert!(result.is_ok(), "should complete via idle timeout");
        assert!(task.has_finished());
    }

    #[tokio::test]
    async fn idle_timeout_fires_callback() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();

        let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
        let mut task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                idle_timeout: Some(Duration::from_millis(100)),
                ..Default::default()
            },
        )
        .on_idle_timeout(move || {
            let f = flag_clone.clone();
            async move {
                f.store(true, Ordering::SeqCst);
            }
        });

        let result = tokio::time::timeout(Duration::from_secs(5), task.run()).await;
        assert!(result.is_ok());
        assert!(flag.load(Ordering::SeqCst), "on_idle_timeout should fire");
    }

    #[tokio::test]
    async fn idle_timeout_no_cancel_when_disabled() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();

        let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);
        let task = PipelineTask::new(
            Box::new(pipeline),
            PipelineParams {
                enable_heartbeats: false,
                idle_timeout: Some(Duration::from_millis(50)),
                cancel_on_idle_timeout: false,
                ..Default::default()
            },
        );

        // Send the cancel from INSIDE the idle callback. This proves:
        // 1. The callback fires (flag is set)
        // 2. The pipeline is still alive when it fires (send succeeds)
        // 3. The idle timeout itself did NOT cancel the pipeline
        let push_tx = task.push_tx.clone();
        let mut task = task.on_idle_timeout(move || {
            let f = flag_clone.clone();
            let tx = push_tx.clone();
            async move {
                f.store(true, Ordering::SeqCst);
                tx.send(Frame::Cancel(CancelFrame::default())).await.ok();
            }
        });

        let result = tokio::time::timeout(Duration::from_secs(5), task.run()).await;
        assert!(result.is_ok());
        assert!(
            flag.load(Ordering::SeqCst),
            "on_idle_timeout should fire even when cancel is disabled"
        );
    }
}
