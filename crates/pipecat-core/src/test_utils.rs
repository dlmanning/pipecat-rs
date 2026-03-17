//! Reusable test utilities for pipecat-core and downstream crates.
//!
//! Available within pipecat-core via `#[cfg(test)]`, and to downstream crates
//! via the `test-utils` feature flag:
//!
//! ```toml
//! [dev-dependencies]
//! pipecat-core = { workspace = true, features = ["test-utils"] }
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use crate::error::Result;
use crate::frame::*;
use crate::node::{ProcessorNode, ProcessorNodeHandle};
use crate::observer::PipelineObserver;
use crate::processor::{FrameProcessor, ProcessorBase, ProcessorContext};

// ---------------------------------------------------------------------------
// Frame constructors
// ---------------------------------------------------------------------------

pub fn make_start_frame() -> FrameEnvelope {
    FrameEnvelope::new(Frame::Start(StartFrame::default()))
}

pub fn make_end_frame() -> FrameEnvelope {
    FrameEnvelope::new(Frame::End(EndFrame::default()))
}

pub fn make_cancel_frame() -> FrameEnvelope {
    FrameEnvelope::new(Frame::Cancel(CancelFrame::default()))
}

pub fn make_interruption_frame() -> FrameEnvelope {
    FrameEnvelope::new(Frame::Interruption(InterruptionFrame))
}

pub fn make_text_frame(text: &str) -> FrameEnvelope {
    FrameEnvelope::new(Frame::Text(TextFrame::new(text)))
}

/// Create an audio frame from i16 samples. Encodes as little-endian bytes.
pub fn make_audio_frame(samples: &[i16], sample_rate: u32) -> FrameEnvelope {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    FrameEnvelope::new(Frame::OutputAudioRaw(AudioRawFrame {
        audio: Bytes::from(bytes),
        sample_rate,
        num_channels: 1,
    }))
}

pub fn make_transcription_frame(text: &str, user_id: &str) -> FrameEnvelope {
    FrameEnvelope::new(Frame::Transcription(TranscriptionFrame {
        text: text.to_string(),
        user_id: user_id.to_string(),
        timestamp: None,
        language: None,
        finalized: true,
        result: None,
    }))
}

pub fn make_metrics_frame(data: Vec<MetricsData>) -> FrameEnvelope {
    FrameEnvelope::new(Frame::Metrics(MetricsFrame { data }))
}

// ---------------------------------------------------------------------------
// Test processors
// ---------------------------------------------------------------------------

/// Forwards every frame in the same direction. Optionally tracks cleanup.
pub struct PassthroughProcessor {
    base: ProcessorBase,
    cleaned_up: Arc<AtomicBool>,
}

impl PassthroughProcessor {
    pub fn new() -> Self {
        Self {
            base: ProcessorBase::new("Passthrough"),
            cleaned_up: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create a PassthroughProcessor that sets `flag` to true on cleanup.
    pub fn with_cleanup_flag() -> (Self, Arc<AtomicBool>) {
        let flag = Arc::new(AtomicBool::new(false));
        (
            Self {
                base: ProcessorBase::new("Passthrough"),
                cleaned_up: flag.clone(),
            },
            flag,
        )
    }
}

impl Default for PassthroughProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FrameProcessor for PassthroughProcessor {
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
        ctx.push_frame(envelope, direction).await
    }
    async fn cleanup(&mut self) {
        self.cleaned_up.store(true, Ordering::SeqCst);
    }
}

/// Stores received frames for later assertions. Does NOT forward.
pub struct CollectorProcessor {
    base: ProcessorBase,
    frames: Arc<Mutex<Vec<FrameEnvelope>>>,
}

impl CollectorProcessor {
    pub fn new() -> (Self, Arc<Mutex<Vec<FrameEnvelope>>>) {
        let frames = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                base: ProcessorBase::new("Collector"),
                frames: frames.clone(),
            },
            frames,
        )
    }
}

#[async_trait]
impl FrameProcessor for CollectorProcessor {
    fn name(&self) -> &str {
        self.base.name()
    }
    fn id(&self) -> u64 {
        self.base.id()
    }
    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        _direction: Direction,
        _ctx: &ProcessorContext,
    ) -> Result<()> {
        self.frames.lock().unwrap().push(envelope);
        Ok(())
    }
}

/// Records frame Display names in order AND forwards frames downstream.
pub struct RecorderProcessor {
    pub base: ProcessorBase,
    frames: Arc<Mutex<Vec<String>>>,
}

impl RecorderProcessor {
    pub fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
        let frames = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                base: ProcessorBase::new("Recorder"),
                frames: frames.clone(),
            },
            frames,
        )
    }
}

#[async_trait]
impl FrameProcessor for RecorderProcessor {
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
        self.frames
            .lock()
            .unwrap()
            .push(format!("{}", envelope.frame));
        ctx.push_frame(envelope, direction).await
    }
}

/// Pushes an error upstream when it sees a TextFrame. Forwards everything else.
pub struct ErrorOnTextProcessor {
    base: ProcessorBase,
}

impl ErrorOnTextProcessor {
    pub fn new() -> Self {
        Self {
            base: ProcessorBase::new("ErrorOnText"),
        }
    }
}

impl Default for ErrorOnTextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FrameProcessor for ErrorOnTextProcessor {
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
                ctx.push_error("text not allowed", false).await?;
            }
            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }
        Ok(())
    }
}

/// Uppercases Text frames, forwards everything else.
pub struct UppercaseProcessor {
    base: ProcessorBase,
}

impl UppercaseProcessor {
    pub fn new() -> Self {
        Self {
            base: ProcessorBase::new("Uppercase"),
        }
    }
}

impl Default for UppercaseProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FrameProcessor for UppercaseProcessor {
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
            Frame::Text(t) => {
                ctx.send_downstream(Frame::Text(TextFrame::new(t.text.to_uppercase())))
                    .await?;
            }
            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }
        Ok(())
    }
}

/// Returns Err from process_frame on Text frames. For testing auto error catching.
pub struct FailingProcessor {
    base: ProcessorBase,
}

impl FailingProcessor {
    pub fn new() -> Self {
        Self {
            base: ProcessorBase::new("Failing"),
        }
    }
}

impl Default for FailingProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FrameProcessor for FailingProcessor {
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
            Frame::Text(_) => Err(crate::error::PipecatError::ProcessorError(
                "intentional failure".into(),
            )),
            _ => ctx.push_frame(envelope, direction).await,
        }
    }
}

// ---------------------------------------------------------------------------
// FrameCollector — deterministic frame collection for tests
// ---------------------------------------------------------------------------

/// Collects frames from a channel in a background task, providing deterministic
/// wait methods that replace the racy `sleep` + `drain_rx` pattern.
///
/// # Usage
///
/// ```ignore
/// let (node, handle, down_rx, _up_rx) = make_node(Box::new(processor));
/// let down = FrameCollector::spawn(down_rx);
/// let run = tokio::spawn(async move { node.run().await });
///
/// send_frame(&handle, Frame::Start(StartFrame::default()), Direction::Downstream).await;
/// send_frame(&handle, Frame::Text(TextFrame::new("hello")), Direction::Downstream).await;
/// send_frame(&handle, Frame::Cancel(CancelFrame::default()), Direction::Downstream).await;
///
/// down.wait_for_frame("Cancel").await;
/// assert_eq!(down.frame_names(), vec!["Start", "Text", "Cancel"]);
/// ```
pub struct FrameCollector {
    inner: Arc<Mutex<Vec<FrameEnvelope>>>,
    count_rx: watch::Receiver<usize>,
    _count_tx: Arc<watch::Sender<usize>>,
    _task: JoinHandle<()>,
}

impl FrameCollector {
    /// Spawn a background task that receives frames and stores them.
    /// Takes ownership of the receiver.
    pub fn spawn(mut rx: mpsc::Receiver<FrameEnvelope>) -> Self {
        let inner = Arc::new(Mutex::new(Vec::<FrameEnvelope>::new()));
        let (count_tx, count_rx) = watch::channel(0usize);
        let count_tx = Arc::new(count_tx);

        let inner_clone = inner.clone();
        let count_tx_clone = count_tx.clone();
        let task = tokio::spawn(async move {
            while let Some(env) = rx.recv().await {
                let new_count = {
                    let mut frames = inner_clone.lock().unwrap();
                    frames.push(env);
                    frames.len()
                };
                let _ = count_tx_clone.send(new_count);
            }
        });

        Self {
            inner,
            count_rx,
            _count_tx: count_tx,
            _task: task,
        }
    }

    /// Wait until a frame with the given Display name has been collected.
    /// Panics if the frame doesn't appear within 5 seconds.
    pub async fn wait_for_frame(&self, name: &str) {
        self.wait_for_frame_timeout(name, Duration::from_secs(5))
            .await;
    }

    /// Wait until a frame with the given Display name has been collected,
    /// with a custom timeout.
    pub async fn wait_for_frame_timeout(&self, name: &str, timeout: Duration) {
        let result = tokio::time::timeout(timeout, self.wait_for_frame_inner(name)).await;
        if result.is_err() {
            let names = self.frame_names();
            panic!(
                "FrameCollector: timed out after {timeout:?} waiting for frame '{name}'. \
                 Collected so far: {names:?}"
            );
        }
    }

    async fn wait_for_frame_inner(&self, name: &str) {
        let mut rx = self.count_rx.clone();
        loop {
            rx.borrow_and_update();
            {
                let frames = self.inner.lock().unwrap();
                if frames.iter().any(|f| format!("{}", f.frame) == name) {
                    return;
                }
            }
            if rx.changed().await.is_err() {
                let names = self.frame_names();
                panic!(
                    "FrameCollector: channel closed while waiting for frame '{name}'. \
                     Collected: {names:?}"
                );
            }
        }
    }

    /// Wait until a frame matching the predicate has been collected.
    /// Panics if no match appears within 5 seconds.
    pub async fn wait_for(&self, pred: impl Fn(&Frame) -> bool) {
        let result = tokio::time::timeout(Duration::from_secs(5), self.wait_for_inner(&pred)).await;
        if result.is_err() {
            let names = self.frame_names();
            panic!(
                "FrameCollector: timed out waiting for predicate match. \
                 Collected so far: {names:?}"
            );
        }
    }

    async fn wait_for_inner(&self, pred: &(impl Fn(&Frame) -> bool + ?Sized)) {
        let mut rx = self.count_rx.clone();
        loop {
            rx.borrow_and_update();
            {
                let frames = self.inner.lock().unwrap();
                if frames.iter().any(|f| pred(&f.frame)) {
                    return;
                }
            }
            if rx.changed().await.is_err() {
                let names = self.frame_names();
                panic!(
                    "FrameCollector: channel closed while waiting for predicate. \
                     Collected: {names:?}"
                );
            }
        }
    }

    /// Wait until at least `count` frames have been collected.
    /// Panics if the count isn't reached within 5 seconds.
    pub async fn wait_for_count(&self, count: usize) {
        let result = tokio::time::timeout(Duration::from_secs(5), async {
            let mut rx = self.count_rx.clone();
            loop {
                rx.borrow_and_update();
                if self.count() >= count {
                    return;
                }
                if rx.changed().await.is_err() {
                    panic!(
                        "FrameCollector: channel closed at count {} while waiting for {count}",
                        self.count()
                    );
                }
            }
        })
        .await;
        if result.is_err() {
            panic!(
                "FrameCollector: timed out waiting for count {count}, got {}",
                self.count()
            );
        }
    }

    /// Get the Display names of all collected frames.
    pub fn frame_names(&self) -> Vec<String> {
        let frames = self.inner.lock().unwrap();
        frames.iter().map(|f| format!("{}", f.frame)).collect()
    }

    /// Clone all collected frames.
    pub fn frames(&self) -> Vec<FrameEnvelope> {
        self.inner.lock().unwrap().clone()
    }

    /// Take all collected frames, clearing the internal buffer.
    /// Subsequent waits will only see frames arriving after this call.
    pub fn take_frames(&self) -> Vec<FrameEnvelope> {
        let mut frames = self.inner.lock().unwrap();
        std::mem::take(&mut *frames)
    }

    /// Number of frames collected so far.
    pub fn count(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}

// ---------------------------------------------------------------------------
// Node wiring helpers
// ---------------------------------------------------------------------------

const DEFAULT_CHANNEL_SIZE: usize = 64;

/// Send a frame through a handle without any sleep. Use with `FrameCollector`
/// for deterministic synchronization.
pub async fn send_frame(handle: &ProcessorNodeHandle, frame: Frame, direction: Direction) {
    handle
        .send(FrameEnvelope::new(frame), direction)
        .await
        .unwrap();
}

/// Create a node with downstream/upstream output channels.
pub fn make_node(
    processor: Box<dyn FrameProcessor>,
) -> (
    ProcessorNode,
    ProcessorNodeHandle,
    mpsc::Receiver<FrameEnvelope>,
    mpsc::Receiver<FrameEnvelope>,
) {
    let (down_tx, down_rx) = mpsc::channel(DEFAULT_CHANNEL_SIZE);
    let (up_tx, up_rx) = mpsc::channel(DEFAULT_CHANNEL_SIZE);
    let (node, handle) = ProcessorNode::new(processor, down_tx, up_tx, DEFAULT_CHANNEL_SIZE);
    (node, handle, down_rx, up_rx)
}

/// Create a node with a pipeline observer attached.
pub fn make_observed_node(
    processor: Box<dyn FrameProcessor>,
    observer: Arc<dyn PipelineObserver>,
) -> (
    ProcessorNode,
    ProcessorNodeHandle,
    mpsc::Receiver<FrameEnvelope>,
    mpsc::Receiver<FrameEnvelope>,
) {
    let (down_tx, down_rx) = mpsc::channel(DEFAULT_CHANNEL_SIZE);
    let (up_tx, up_rx) = mpsc::channel(DEFAULT_CHANNEL_SIZE);
    let (node, handle) =
        ProcessorNode::with_observer(processor, down_tx, up_tx, DEFAULT_CHANNEL_SIZE, observer);
    (node, handle, down_rx, up_rx)
}

/// Get Display names of all frames in a slice.
pub fn frame_names(frames: &[FrameEnvelope]) -> Vec<String> {
    frames.iter().map(|f| format!("{}", f.frame)).collect()
}

// ---------------------------------------------------------------------------
// Processor-level test helper (no node)
// ---------------------------------------------------------------------------

/// Run a processor directly with a list of inputs. Returns (downstream, upstream) outputs.
///
/// Creates channels, feeds each input through `process_frame`, then returns
/// all frames that arrived on downstream and upstream channels.
pub async fn run_processor(
    processor: &mut dyn FrameProcessor,
    inputs: Vec<(Frame, Direction)>,
) -> (Vec<FrameEnvelope>, Vec<FrameEnvelope>) {
    let (down_tx, mut down_rx) = mpsc::channel(64);
    let (up_tx, mut up_rx) = mpsc::channel(64);
    let ctx = ProcessorContext::new(down_tx, up_tx, processor.id(), processor.name().to_string());

    for (frame, direction) in inputs {
        let _ = processor
            .process_frame(FrameEnvelope::new(frame), direction, &ctx)
            .await;
    }

    // Drop the context so channels close
    drop(ctx);

    let mut downstream = Vec::new();
    while let Ok(env) = down_rx.try_recv() {
        downstream.push(env);
    }

    let mut upstream = Vec::new();
    while let Ok(env) = up_rx.try_recv() {
        upstream.push(env);
    }

    (downstream, upstream)
}
