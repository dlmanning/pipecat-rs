use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use pipecat_core::error::Result;
use pipecat_core::frame::{Direction, FrameEnvelope};
use pipecat_core::observer::PipelineObserver;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::debug;

use crate::pipeline::Pipeline;

const CHANNEL_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Tagged envelope — carries branch metadata through coordinator channel
// ---------------------------------------------------------------------------

struct TaggedEnvelope {
    envelope: FrameEnvelope,
    direction: Direction,
}

// ---------------------------------------------------------------------------
// Coordinator — manages lifecycle sync and dedup across branches
// ---------------------------------------------------------------------------

/// Receives all output frames from all branches and forwards them to the
/// outer context after deduplication and lifecycle synchronization.
async fn coordinator(
    num_branches: usize,
    mut rx: mpsc::Receiver<TaggedEnvelope>,
    ctx: ProcessorContext,
    synchronizing: Arc<AtomicBool>,
) {
    let mut seen_ids: HashSet<u64> = HashSet::new();
    let mut frame_counter: HashMap<u64, usize> = HashMap::new();
    let mut buffered: Vec<TaggedEnvelope> = Vec::new();
    let mut pending_lifecycle: HashMap<u64, TaggedEnvelope> = HashMap::new();

    while let Some(tagged) = rx.recv().await {
        let frame_id = tagged.envelope.header.id;
        let is_lifecycle = tagged.envelope.frame.is_lifecycle();

        if is_lifecycle {
            let counter = frame_counter.entry(frame_id).or_insert(0);
            *counter += 1;

            if *counter >= num_branches {
                // All branches have processed this lifecycle frame.
                frame_counter.remove(&frame_id);

                let lifecycle = pending_lifecycle.remove(&frame_id).unwrap_or(tagged);

                // Collect buffered frame IDs BEFORE draining, so we can
                // re-populate seen_ids after the lifecycle clear. This ensures
                // late-arriving duplicates from slower branches are caught.
                let buffered_ids: HashSet<u64> =
                    buffered.iter().map(|b| b.envelope.header.id).collect();

                // For StartFrame: push lifecycle first, then flush buffered.
                // For End/Cancel: flush buffered first, then push lifecycle.
                if matches!(
                    lifecycle.envelope.frame,
                    pipecat_core::frame::Frame::Start(_)
                ) {
                    ctx.push_frame(lifecycle.envelope, lifecycle.direction)
                        .await
                        .ok();
                    for buf in buffered.drain(..) {
                        ctx.push_frame(buf.envelope, buf.direction).await.ok();
                    }
                } else {
                    for buf in buffered.drain(..) {
                        ctx.push_frame(buf.envelope, buf.direction).await.ok();
                    }
                    ctx.push_frame(lifecycle.envelope, lifecycle.direction)
                        .await
                        .ok();
                }

                // Clear dedup set at epoch boundary, then re-insert IDs of
                // frames that were buffered during sync.
                seen_ids.clear();
                seen_ids.extend(buffered_ids);
                synchronizing.store(false, Ordering::SeqCst);
            } else {
                // Not all branches done yet. Store and wait.
                pending_lifecycle.entry(frame_id).or_insert(tagged);
                synchronizing.store(true, Ordering::SeqCst);
            }
        } else {
            // Non-lifecycle: deduplicate by frame ID.
            // Always check seen_ids — even during sync, frames from
            // different branches can race with lifecycle boundaries.
            if seen_ids.contains(&frame_id) {
                continue;
            }
            seen_ids.insert(frame_id);

            if synchronizing.load(Ordering::SeqCst) {
                buffered.push(tagged);
            } else {
                ctx.push_frame(tagged.envelope, tagged.direction).await.ok();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ParallelPipeline state machine
// ---------------------------------------------------------------------------

enum ParallelState {
    Idle(Vec<Vec<Box<dyn FrameProcessor>>>),
    Running {
        branch_source_handles: Vec<pipecat_core::node::ProcessorNodeHandle>,
        tasks: Vec<JoinHandle<()>>,
        synchronizing: Arc<AtomicBool>,
    },
    Stopped,
}

// ---------------------------------------------------------------------------
// ParallelPipeline
// ---------------------------------------------------------------------------

/// Runs multiple processor chains in parallel with lifecycle synchronization.
///
/// Frames are forked to all branches. Lifecycle frames (`Start`, `End`,
/// `Cancel`) are synchronized — they only exit after ALL branches have
/// processed them. Non-lifecycle output frames are deduplicated by frame ID.
///
/// During lifecycle synchronization, non-lifecycle frames are buffered at the
/// ParallelPipeline level (not sent to branches) to prevent ordering issues.
pub struct ParallelPipeline {
    base: ProcessorBase,
    state: ParallelState,
    observer: Option<Arc<dyn PipelineObserver>>,
    /// Frames held back while lifecycle sync is in progress.
    pending_external: Vec<(FrameEnvelope, Direction)>,
}

impl std::fmt::Debug for ParallelPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelPipeline")
            .field("name", &self.base.name())
            .field("id", &self.base.id())
            .finish()
    }
}

impl ParallelPipeline {
    pub fn new(branches: Vec<Vec<Box<dyn FrameProcessor>>>) -> Self {
        assert!(
            !branches.is_empty(),
            "ParallelPipeline requires at least one branch"
        );
        Self {
            base: ProcessorBase::new("ParallelPipeline"),
            state: ParallelState::Idle(branches),
            observer: None,
            pending_external: Vec::new(),
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn PipelineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    fn initialize(&mut self, ctx: &ProcessorContext) {
        let branches = match std::mem::replace(&mut self.state, ParallelState::Stopped) {
            ParallelState::Idle(b) => b,
            _ => return,
        };

        let num_branches = branches.len();
        let mut tasks: Vec<JoinHandle<()>> = Vec::new();
        let mut branch_source_handles = Vec::with_capacity(num_branches);
        let synchronizing = Arc::new(AtomicBool::new(false));

        let (coord_tx, coord_rx) = mpsc::channel::<TaggedEnvelope>(CHANNEL_SIZE * num_branches);

        for (i, processors) in branches.into_iter().enumerate() {
            let mut pipeline = Pipeline::new(processors);
            if let Some(ref obs) = self.observer {
                pipeline = pipeline.with_observer(Arc::clone(obs));
            }

            let coord_tx_down = coord_tx.clone();
            let coord_tx_up = coord_tx.clone();

            let (down_tx, mut down_rx) = mpsc::channel::<FrameEnvelope>(CHANNEL_SIZE);
            let (up_tx, mut up_rx) = mpsc::channel::<FrameEnvelope>(CHANNEL_SIZE);

            let (node, handle) = if let Some(ref obs) = self.observer {
                pipecat_core::node::ProcessorNode::with_observer(
                    Box::new(pipeline),
                    down_tx,
                    up_tx,
                    CHANNEL_SIZE,
                    Arc::clone(obs),
                )
            } else {
                pipecat_core::node::ProcessorNode::new(
                    Box::new(pipeline),
                    down_tx,
                    up_tx,
                    CHANNEL_SIZE,
                )
            };

            debug!("ParallelPipeline: created branch[{i}] = {}", handle.name());
            branch_source_handles.push(handle);

            tasks.push(tokio::spawn(async move {
                while let Some(envelope) = down_rx.recv().await {
                    if coord_tx_down
                        .send(TaggedEnvelope {
                            envelope,
                            direction: Direction::Downstream,
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }));

            tasks.push(tokio::spawn(async move {
                while let Some(envelope) = up_rx.recv().await {
                    if coord_tx_up
                        .send(TaggedEnvelope {
                            envelope,
                            direction: Direction::Upstream,
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }));

            tasks.push(tokio::spawn(async move { node.run().await }));
        }

        drop(coord_tx);

        let coord_sync = Arc::clone(&synchronizing);
        tasks.push(tokio::spawn(coordinator(
            num_branches,
            coord_rx,
            ctx.clone(),
            coord_sync,
        )));

        debug!(
            "ParallelPipeline initialized: {} branches, {} tasks",
            num_branches,
            tasks.len()
        );

        self.state = ParallelState::Running {
            branch_source_handles,
            tasks,
            synchronizing,
        };
    }

    /// Send a frame to all branches. Clones for all but the last branch.
    async fn send_to_branches(
        handles: &[pipecat_core::node::ProcessorNodeHandle],
        envelope: FrameEnvelope,
        direction: Direction,
    ) {
        if handles.is_empty() {
            return;
        }
        let last = handles.len() - 1;
        for (i, handle) in handles[..last].iter().enumerate() {
            if let Err(e) = handle.send(envelope.clone(), direction).await {
                debug!("ParallelPipeline: branch[{i}] send failed: {e}");
            }
        }
        if let Err(e) = handles[last].send(envelope, direction).await {
            debug!("ParallelPipeline: branch[{last}] send failed: {e}");
        }
    }
}

#[async_trait]
impl FrameProcessor for ParallelPipeline {
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
        if matches!(self.state, ParallelState::Idle(_)) {
            self.initialize(ctx);
        }

        match &self.state {
            ParallelState::Running {
                branch_source_handles,
                synchronizing,
                ..
            } => {
                let is_syncing = synchronizing.load(Ordering::SeqCst);
                let is_lifecycle = envelope.frame.is_lifecycle();

                if is_syncing && !is_lifecycle {
                    // Buffer non-lifecycle frames during sync to prevent ordering issues.
                    self.pending_external.push((envelope, direction));
                    return Ok(());
                }

                // Drain any pending external frames first (sync just completed).
                if !is_syncing && !self.pending_external.is_empty() {
                    let pending: Vec<_> = self.pending_external.drain(..).collect();
                    for (env, dir) in pending {
                        Self::send_to_branches(branch_source_handles, env, dir).await;
                    }
                }

                Self::send_to_branches(branch_source_handles, envelope, direction).await;
            }
            _ => {
                debug!("ParallelPipeline: frame in non-running state, dropping");
            }
        }
        Ok(())
    }

    async fn cleanup(&mut self) {
        let tasks = match std::mem::replace(&mut self.state, ParallelState::Stopped) {
            ParallelState::Running { tasks, .. } => tasks,
            _ => return,
        };

        debug!(
            "ParallelPipeline cleanup: waiting for {} tasks",
            tasks.len()
        );
        for task in tasks {
            let _ = task.await;
        }
        debug!("ParallelPipeline cleanup complete");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use pipecat_core::frame::*;
    use pipecat_core::test_utils::*;

    use super::*;

    #[tokio::test]
    async fn two_passthrough_branches() {
        let parallel = ParallelPipeline::new(vec![
            vec![Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>],
            vec![Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>],
        ]);

        let (node, handle, down_rx, _up_rx) = make_node(Box::new(parallel));
        let down = FrameCollector::spawn(down_rx);
        let _run = tokio::spawn(async move { node.run().await });

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

        down.wait_for_frame("Text").await;

        send_frame(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;
        down.wait_for_frame("Cancel").await;

        let names = down.frame_names();
        assert_eq!(
            names.iter().filter(|n| *n == "Start").count(),
            1,
            "Start should appear once: {names:?}"
        );
        assert_eq!(
            names.iter().filter(|n| *n == "Text").count(),
            1,
            "Text should appear once: {names:?}"
        );
        assert_eq!(
            names.iter().filter(|n| *n == "Cancel").count(),
            1,
            "Cancel should appear once: {names:?}"
        );
    }

    #[tokio::test]
    async fn lifecycle_sync_start_before_data() {
        let parallel = ParallelPipeline::new(vec![
            vec![Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>],
            vec![Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>],
        ]);

        let (node, handle, down_rx, _up_rx) = make_node(Box::new(parallel));
        let down = FrameCollector::spawn(down_rx);
        let _run = tokio::spawn(async move { node.run().await });

        send_frame(
            &handle,
            Frame::Start(StartFrame::default()),
            Direction::Downstream,
        )
        .await;
        send_frame(
            &handle,
            Frame::Text(TextFrame::new("after start")),
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
        down.wait_for_frame("Cancel").await;

        let names = down.frame_names();

        let start_pos = names.iter().position(|n| n == "Start");
        let text_pos = names.iter().position(|n| n == "Text");
        assert!(
            start_pos.is_some() && text_pos.is_some(),
            "both Start and Text should be present: {names:?}"
        );
        assert!(
            start_pos.unwrap() < text_pos.unwrap(),
            "Start should come before Text: {names:?}"
        );
    }

    #[tokio::test]
    async fn different_transforms_produce_separate_outputs() {
        let parallel = ParallelPipeline::new(vec![
            vec![Box::new(UppercaseProcessor::new()) as Box<dyn FrameProcessor>],
            vec![Box::new(PassthroughProcessor::new()) as Box<dyn FrameProcessor>],
        ]);

        let (node, handle, down_rx, _up_rx) = make_node(Box::new(parallel));
        let down = FrameCollector::spawn(down_rx);
        let _run = tokio::spawn(async move { node.run().await });

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

        // Wait for both Text frames (passthrough + uppercase) before sending Cancel
        down.wait_for_count(3).await; // Start + 2 Text frames

        send_frame(
            &handle,
            Frame::Cancel(CancelFrame::default()),
            Direction::Downstream,
        )
        .await;

        down.wait_for_frame("Cancel").await;

        let names = down.frame_names();

        assert_eq!(names.iter().filter(|n| *n == "Start").count(), 1);
        assert_eq!(names.iter().filter(|n| *n == "Text").count(), 2);
        assert_eq!(names.iter().filter(|n| *n == "Cancel").count(), 1);

        let frames = down.frames();
        let texts: Vec<&str> = frames
            .iter()
            .filter_map(|f| match &f.frame {
                Frame::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect();
        assert!(texts.contains(&"hello"), "passthrough output: {texts:?}");
        assert!(texts.contains(&"HELLO"), "uppercase output: {texts:?}");
    }
}
