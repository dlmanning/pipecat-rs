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
use tokio::sync::mpsc;

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
// Node wiring helpers
// ---------------------------------------------------------------------------

/// Default settle time for send_and_settle. Allows the node's select loop to process.
pub const SETTLE: Duration = Duration::from_millis(30);

const DEFAULT_CHANNEL_SIZE: usize = 64;

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

/// Send a frame and wait for it to settle through the node.
pub async fn send_and_settle(handle: &ProcessorNodeHandle, frame: Frame, direction: Direction) {
    handle
        .send(FrameEnvelope::new(frame), direction)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
}

/// Collect all available frames from a receiver without blocking.
pub async fn drain_rx(rx: &mut mpsc::Receiver<FrameEnvelope>) -> Vec<FrameEnvelope> {
    tokio::task::yield_now().await;
    let mut result = Vec::new();
    while let Ok(env) = rx.try_recv() {
        result.push(env);
    }
    result
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
