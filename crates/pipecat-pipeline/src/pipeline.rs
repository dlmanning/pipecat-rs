use std::sync::Arc;

use async_trait::async_trait;
use pipecat_core::error::Result;
use pipecat_core::frame::{Direction, FrameEnvelope};
use pipecat_core::node::ProcessorNode;
use pipecat_core::observer::PipelineObserver;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::debug;

const CHANNEL_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Bridge task — forwards frames from one node's output to the next node's input
// ---------------------------------------------------------------------------

async fn bridge(
    mut rx: mpsc::Receiver<FrameEnvelope>,
    handle: pipecat_core::node::ProcessorNodeHandle,
    direction: Direction,
) {
    while let Some(envelope) = rx.recv().await {
        if handle.send(envelope, direction).await.is_err() {
            break;
        }
    }
}

/// Escape bridge: forwards frames from an escape channel to the outer
/// ProcessorContext so they leave the pipeline boundary.
async fn escape_bridge(
    mut rx: mpsc::Receiver<FrameEnvelope>,
    ctx: ProcessorContext,
    direction: Direction,
) {
    while let Some(envelope) = rx.recv().await {
        if ctx.push_frame(envelope, direction).await.is_err() {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineSource — entry point of the internal chain
// ---------------------------------------------------------------------------

/// Forwards downstream frames into the chain. Upstream frames escape back to
/// the Pipeline's caller.
struct PipelineSource {
    base: ProcessorBase,
    escape_tx: mpsc::Sender<FrameEnvelope>,
}

#[async_trait]
impl FrameProcessor for PipelineSource {
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
                self.escape_tx.send(envelope).await.ok();
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PipelineSink — exit point of the internal chain
// ---------------------------------------------------------------------------

/// Forwards upstream frames back through the chain. Downstream frames escape
/// back to the Pipeline's caller.
struct PipelineSink {
    base: ProcessorBase,
    escape_tx: mpsc::Sender<FrameEnvelope>,
}

#[async_trait]
impl FrameProcessor for PipelineSink {
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
                self.escape_tx.send(envelope).await.ok();
            }
            Direction::Upstream => {
                ctx.push_upstream(envelope).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pipeline state machine
// ---------------------------------------------------------------------------

enum PipelineState {
    /// Processors stored, waiting for first frame to trigger initialization.
    Idle(Vec<Box<dyn FrameProcessor>>),
    /// Internal nodes spawned and running.
    Running {
        source_handle: pipecat_core::node::ProcessorNodeHandle,
        sink_handle: pipecat_core::node::ProcessorNodeHandle,
        tasks: Vec<JoinHandle<()>>,
    },
    /// Shut down (after cleanup).
    Stopped,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A linear chain of processors wired together via tokio channels.
///
/// Pipeline implements `FrameProcessor` so it can be nested inside other
/// pipelines or used directly by `PipelineTask`.
///
/// The internal node graph is created lazily on the first frame (typically
/// `StartFrame`). Each user processor gets wrapped in a `ProcessorNode` with
/// bridge tasks forwarding frames between adjacent nodes. A `PipelineSource`
/// and `PipelineSink` bookend the chain, with escape channels that route
/// frames back to the outer context.
pub struct Pipeline {
    base: ProcessorBase,
    state: PipelineState,
    observer: Option<Arc<dyn PipelineObserver>>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.base.name())
            .field("id", &self.base.id())
            .finish()
    }
}

impl Pipeline {
    pub fn new(processors: Vec<Box<dyn FrameProcessor>>) -> Self {
        Self {
            base: ProcessorBase::new("Pipeline"),
            state: PipelineState::Idle(processors),
            observer: None,
        }
    }

    /// Attach an observer that will be wired to all internal ProcessorNodes.
    pub fn with_observer(mut self, observer: Arc<dyn PipelineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Initialize the internal node graph. Called on the first frame.
    fn initialize(&mut self, ctx: &ProcessorContext) {
        let processors = match std::mem::replace(&mut self.state, PipelineState::Stopped) {
            PipelineState::Idle(p) => p,
            _ => return,
        };

        let mut tasks: Vec<JoinHandle<()>> = Vec::new();

        // Create escape channels for source (upstream) and sink (downstream).
        let (source_escape_tx, source_escape_rx) = mpsc::channel::<FrameEnvelope>(CHANNEL_SIZE);
        let (sink_escape_tx, sink_escape_rx) = mpsc::channel::<FrameEnvelope>(CHANNEL_SIZE);

        // Build the full processor list: [Source, ...user..., Sink]
        let source = PipelineSource {
            base: ProcessorBase::new("PipelineSource"),
            escape_tx: source_escape_tx,
        };
        let sink = PipelineSink {
            base: ProcessorBase::new("PipelineSink"),
            escape_tx: sink_escape_tx,
        };

        let mut all_processors: Vec<Box<dyn FrameProcessor>> =
            Vec::with_capacity(processors.len() + 2);
        all_processors.push(Box::new(source));
        all_processors.extend(processors);
        all_processors.push(Box::new(sink));

        let n = all_processors.len();

        // Create per-node output channels.
        let mut down_txs = Vec::with_capacity(n);
        let mut down_rxs = Vec::with_capacity(n);
        let mut up_txs = Vec::with_capacity(n);
        let mut up_rxs = Vec::with_capacity(n);

        for _ in 0..n {
            let (dtx, drx) = mpsc::channel(CHANNEL_SIZE);
            let (utx, urx) = mpsc::channel(CHANNEL_SIZE);
            down_txs.push(dtx);
            down_rxs.push(drx);
            up_txs.push(utx);
            up_rxs.push(urx);
        }

        // Create all ProcessorNodes.
        let mut handles = Vec::with_capacity(n);
        let mut nodes = Vec::with_capacity(n);

        let mut down_tx_iter = down_txs.into_iter();
        let mut up_tx_iter = up_txs.into_iter();
        for (i, processor) in all_processors.into_iter().enumerate() {
            let dtx = down_tx_iter.next().unwrap();
            let utx = up_tx_iter.next().unwrap();
            let (node, handle) = if let Some(ref obs) = self.observer {
                ProcessorNode::with_observer(processor, dtx, utx, CHANNEL_SIZE, Arc::clone(obs))
            } else {
                ProcessorNode::new(processor, dtx, utx, CHANNEL_SIZE)
            };
            debug!("Pipeline: created node[{i}] = {}", handle.name());
            handles.push(handle);
            nodes.push(node);
        }

        // Wire downstream bridges: down_rx[i] → handle[i+1]
        let mut down_rx_iter = down_rxs.into_iter();
        for i in 0..n - 1 {
            let rx = down_rx_iter.next().unwrap();
            let next_handle = handles[i + 1].clone();
            tasks.push(tokio::spawn(bridge(rx, next_handle, Direction::Downstream)));
        }
        // Sink's downstream output goes through its escape_tx, not down_rx.
        drop(down_rx_iter);

        // Wire upstream bridges: up_rx[i] → handle[i-1] for i in 1..n
        // Source's upstream output goes through its escape_tx, not up_rx[0].
        let mut up_rx_iter = up_rxs.into_iter();
        let _source_up_rx = up_rx_iter.next(); // drop
        for i in 1..n {
            let rx = up_rx_iter.next().unwrap();
            let prev_handle = handles[i - 1].clone();
            tasks.push(tokio::spawn(bridge(rx, prev_handle, Direction::Upstream)));
        }

        // Wire escape bridges back to the outer context.
        tasks.push(tokio::spawn(escape_bridge(
            source_escape_rx,
            ctx.clone(),
            Direction::Upstream,
        )));
        tasks.push(tokio::spawn(escape_bridge(
            sink_escape_rx,
            ctx.clone(),
            Direction::Downstream,
        )));

        let source_handle = handles[0].clone();
        let sink_handle = handles[n - 1].clone();

        // Spawn all node tasks.
        for node in nodes {
            tasks.push(tokio::spawn(async move { node.run().await }));
        }

        debug!(
            "Pipeline initialized: {} processors, {} tasks spawned",
            n,
            tasks.len()
        );

        self.state = PipelineState::Running {
            source_handle,
            sink_handle,
            tasks,
        };
    }
}

#[async_trait]
impl FrameProcessor for Pipeline {
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
        // Lazy initialization on first frame.
        if matches!(self.state, PipelineState::Idle(_)) {
            self.initialize(ctx);
        }

        match &self.state {
            PipelineState::Running {
                source_handle,
                sink_handle,
                ..
            } => {
                let result = match direction {
                    Direction::Downstream => source_handle.send(envelope, direction).await,
                    Direction::Upstream => sink_handle.send(envelope, direction).await,
                };
                if let Err(e) = result {
                    debug!("Pipeline send failed (shutdown?): {e}");
                }
            }
            _ => {
                debug!("Pipeline: frame received in non-running state, dropping");
            }
        }
        Ok(())
    }

    async fn cleanup(&mut self) {
        let tasks = match std::mem::replace(&mut self.state, PipelineState::Stopped) {
            PipelineState::Running { tasks, .. } => tasks,
            _ => return,
        };

        debug!("Pipeline cleanup: waiting for {} tasks", tasks.len());

        // Dropping the handles (via state replacement) closes the channels,
        // which causes nodes and bridges to exit. Then we wait for them.
        for task in tasks {
            let _ = task.await;
        }

        debug!("Pipeline cleanup complete");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use pipecat_core::frame::*;
    use pipecat_core::test_utils::*;
    use tokio::time::timeout;

    use super::*;

    const TEST_TIMEOUT: Duration = Duration::from_secs(5);

    #[tokio::test]
    async fn single_passthrough_processor() {
        let pipeline = Pipeline::new(vec![Box::new(PassthroughProcessor::new())]);

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
            Frame::End(EndFrame::default()),
            Direction::Downstream,
        )
        .await;
        down.wait_for_frame("End").await;
        send_frame(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let names = down.frame_names();
        assert_eq!(names, vec!["Start", "Text", "End", "Cancel"]);
    }

    #[tokio::test]
    async fn multi_processor_chain() {
        let pipeline = Pipeline::new(vec![
            Box::new(UppercaseProcessor::new()),
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
        send_frame(
            &handle,
            Frame::Text(TextFrame::new("hello world")),
            Direction::Downstream,
        )
        .await;
        down.wait_for_frame("Text").await;
        send_frame(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let frames = down.frames();
        let names = down.frame_names();
        assert_eq!(names, vec!["Start", "Text", "Cancel"]);

        // Verify uppercase transformation
        let text_frame = frames
            .iter()
            .find(|f| matches!(f.frame, Frame::Text(_)))
            .unwrap();
        match &text_frame.frame {
            Frame::Text(t) => assert_eq!(t.text, "HELLO WORLD"),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn upstream_error_escapes_pipeline() {
        let pipeline = Pipeline::new(vec![Box::new(ErrorOnTextProcessor::new())]);

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
        send_frame(
            &handle,
            Frame::Text(TextFrame::new("bad")),
            Direction::Downstream,
        )
        .await;
        up.wait_for_frame("Error").await;
        down.wait_for_frame("Start").await;
        send_frame(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        // Error should have escaped upstream through PipelineSource.
        let up_names = up.frame_names();
        assert!(
            up_names.contains(&"Error".to_string()),
            "expected Error upstream, got: {up_names:?}"
        );

        // Start and Cancel should still flow downstream.
        let down_names = down.frame_names();
        assert!(down_names.contains(&"Start".to_string()));
        assert!(down_names.contains(&"Cancel".to_string()));
    }

    #[tokio::test]
    async fn empty_pipeline_passes_through() {
        let pipeline = Pipeline::new(vec![]);

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
            Frame::Text(TextFrame::new("pass")),
            Direction::Downstream,
        )
        .await;
        down.wait_for_frame("Text").await;
        send_frame(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        let names = down.frame_names();
        assert_eq!(names, vec!["Start", "Text", "Cancel"]);
    }

    #[tokio::test]
    async fn cleanup_propagates_to_inner_processors() {
        let (passthrough, cleanup_flag) = PassthroughProcessor::with_cleanup_flag();

        let pipeline = Pipeline::new(vec![Box::new(passthrough)]);
        let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
        let down = FrameCollector::spawn(down_rx);
        let run = tokio::spawn(async move { node.run().await });

        send_frame(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        down.wait_for_frame("Start").await;
        send_frame(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

        // Inner node calls cleanup on the passthrough processor when it exits.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            cleanup_flag.load(std::sync::atomic::Ordering::Relaxed),
            "inner processor cleanup should have been called"
        );
    }
}
