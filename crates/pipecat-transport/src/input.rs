use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use pipecat_audio::filter::{AudioFilter, FilterControlFrame};
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;
use tracing;

use crate::params::TransportParams;

/// Timeout for audio input queue reads. If no audio arrives within this
/// duration after the first frame was received, the task exits its wait.
const AUDIO_INPUT_TIMEOUT: Duration = Duration::from_millis(500);

// ---------------------------------------------------------------------------
// BaseInputTransport
// ---------------------------------------------------------------------------

/// Base input transport that receives audio/video from an external source
/// and feeds it into the pipeline.
///
/// Concrete transports call [`push_audio_frame`] and [`push_video_frame`]
/// to inject media. An internal audio task applies the configured filter
/// and pushes frames downstream.
///
/// # Lifecycle
///
/// 1. Pipeline sends `StartFrame` → transport initializes sample rate, starts filter
/// 2. Concrete transport calls `set_transport_ready()` → spawns audio processing task
/// 3. External code calls `push_audio_frame()` → audio queued, filtered, and pushed downstream
/// 4. `EndFrame` or `CancelFrame` → audio task cancelled, filter stopped
pub struct BaseInputTransport {
    base: ProcessorBase,
    params: TransportParams,

    // Runtime state
    sample_rate: u32,
    paused: bool,

    // Audio task communication
    audio_in_tx: Option<mpsc::Sender<AudioRawFrame>>,
    audio_task: Option<JoinHandle<()>>,

    // Filter wrapped for sharing with the audio task.
    // Taken from params on start, wrapped in Arc<Mutex<>>.
    filter: Option<Arc<Mutex<Box<dyn AudioFilter>>>>,

    // Stored context for the audio task (cloned from process_frame)
    ctx: Option<ProcessorContext>,
}

impl BaseInputTransport {
    pub fn new(params: TransportParams) -> Self {
        Self {
            base: ProcessorBase::new("BaseInputTransport"),
            params,
            sample_rate: 0,
            paused: false,
            audio_in_tx: None,
            audio_task: None,
            filter: None,
            ctx: None,
        }
    }

    pub fn with_name(name: impl Into<String>, params: TransportParams) -> Self {
        Self {
            base: ProcessorBase::new(name),
            params,
            sample_rate: 0,
            paused: false,
            audio_in_tx: None,
            audio_task: None,
            filter: None,
            ctx: None,
        }
    }

    /// Current audio sample rate. Returns 0 before StartFrame.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get a clone of the audio input sender, if the audio task is running.
    ///
    /// Concrete transports can use this to push audio from a spawned task
    /// without holding a reference to the entire transport.
    pub fn audio_in_sender(&self) -> Option<mpsc::Sender<AudioRawFrame>> {
        self.audio_in_tx.clone()
    }

    /// Push audio from the external source into the transport.
    ///
    /// Called by concrete transport implementations when audio data arrives.
    /// The audio is queued for processing (filtering, passthrough) by the
    /// internal audio task.
    pub async fn push_audio_frame(&self, frame: AudioRawFrame) {
        if !self.params.audio_in_enabled || self.paused {
            return;
        }
        if let Some(ref tx) = self.audio_in_tx {
            // If the channel is full or closed, drop the frame silently.
            // This matches Python's behavior of non-blocking queue puts.
            tx.send(frame).await.ok();
        }
    }

    /// Push a video frame from the external source into the pipeline.
    ///
    /// Called by concrete transport implementations when a video frame arrives.
    /// Unlike audio, video frames are pushed directly downstream without
    /// an intermediate task.
    pub async fn push_video_frame(&self, frame: ImageRawFrame) {
        if !self.params.video_in_enabled || self.paused {
            return;
        }
        if let Some(ref ctx) = self.ctx {
            ctx.send_downstream(Frame::InputImageRaw(frame)).await.ok();
        }
    }

    /// Signal that the transport is ready to stream.
    ///
    /// Called by concrete transports after their connection is established.
    /// Spawns the internal audio processing task.
    pub async fn set_transport_ready(&mut self) {
        self.create_audio_task();
    }

    /// Enable or disable audio streaming at runtime.
    pub fn enable_audio_in_stream_on_start(&mut self, enabled: bool) {
        self.params.audio_in_stream_on_start = enabled;
    }

    // -- Lifecycle methods --

    async fn start(&mut self, frame: &StartFrame, ctx: &ProcessorContext) {
        self.ctx = Some(ctx.clone());
        self.sample_rate = self
            .params
            .audio_in_sample_rate
            .unwrap_or(frame.audio_in_sample_rate);

        // Take filter from params and wrap in Arc<Mutex<>> for sharing
        // with the audio task.
        if let Some(filter) = self.params.audio_in_filter.take() {
            let filter = Arc::new(Mutex::new(filter));
            filter.lock().await.start(self.sample_rate).await;
            self.filter = Some(filter);
        }
    }

    async fn stop(&mut self) {
        self.cancel_audio_task().await;
        if let Some(ref filter) = self.filter {
            filter.lock().await.stop().await;
        }
    }

    async fn cancel(&mut self) {
        self.cancel_audio_task().await;
        if let Some(ref filter) = self.filter {
            filter.lock().await.stop().await;
        }
    }

    async fn pause(&mut self) {
        self.paused = true;
        // Cancel and recreate the audio task to clear the queue.
        self.cancel_audio_task().await;
        self.create_audio_task();
    }

    // -- Audio task management --

    fn create_audio_task(&mut self) {
        if !self.params.audio_in_enabled {
            return;
        }

        let Some(ref ctx) = self.ctx else {
            tracing::warn!("BaseInputTransport: cannot create audio task before StartFrame");
            return;
        };

        let (tx, rx) = mpsc::channel::<AudioRawFrame>(64);
        self.audio_in_tx = Some(tx);

        let ctx = ctx.clone();
        let passthrough = self.params.audio_in_passthrough;
        let filter = self.filter.clone();

        self.audio_task = Some(tokio::spawn(audio_input_task(ctx, rx, passthrough, filter)));
    }

    async fn cancel_audio_task(&mut self) {
        // Drop the sender to signal the task to exit.
        self.audio_in_tx = None;
        if let Some(handle) = self.audio_task.take() {
            // Give the task a moment to finish, then abort if stuck.
            let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
        }
    }
}

/// Background task that processes queued audio frames and pushes them downstream.
///
/// Applies the audio filter (if configured) before pushing frames. Matches
/// the Python `_audio_task_handler` behavior in `base_input.py`.
async fn audio_input_task(
    ctx: ProcessorContext,
    mut rx: mpsc::Receiver<AudioRawFrame>,
    passthrough: bool,
    filter: Option<Arc<Mutex<Box<dyn AudioFilter>>>>,
) {
    let mut received_first = false;

    loop {
        let mut frame = if received_first {
            // After first frame, use timeout to detect audio gaps.
            match tokio::time::timeout(AUDIO_INPUT_TIMEOUT, rx.recv()).await {
                Ok(Some(frame)) => frame,
                Ok(None) => break, // Channel closed
                Err(_) => {
                    // Timeout — no audio for a while. Continue waiting.
                    continue;
                }
            }
        } else {
            // Wait indefinitely for the first frame.
            match rx.recv().await {
                Some(frame) => {
                    received_first = true;
                    frame
                }
                None => break, // Channel closed
            }
        };

        // Apply audio filter if configured.
        if let Some(ref filter) = filter {
            let filtered_audio = filter.lock().await.filter(frame.audio).await;
            if filtered_audio.is_empty() {
                // Filter is buffering — no output yet.
                continue;
            }
            frame.audio = filtered_audio;
        }

        if passthrough {
            ctx.send_downstream(Frame::InputAudioRaw(frame)).await.ok();
        }
    }
}

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for BaseInputTransport {
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
            Frame::Start(frame) => {
                // Push StartFrame downstream first, then initialize.
                let frame = frame.clone();
                ctx.push_frame(envelope, direction).await?;
                self.start(&frame, ctx).await;
            }

            Frame::End(_) => {
                // Push EndFrame downstream first, then stop.
                ctx.push_frame(envelope, direction).await?;
                self.stop().await;
            }

            Frame::Cancel(_) => {
                // Cancel first, then push CancelFrame downstream.
                self.cancel().await;
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::Stop(_) => {
                // Push StopFrame downstream, then pause.
                ctx.push_frame(envelope, direction).await?;
                self.pause().await;
            }

            Frame::FilterUpdateSettings(f) => {
                if let Some(ref filter) = self.filter {
                    filter
                        .lock()
                        .await
                        .process_frame(FilterControlFrame::UpdateSettings(f.clone()))
                        .await;
                }
                // Forward downstream so other processors with filters also get it.
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::FilterEnable(f) => {
                if let Some(ref filter) = self.filter {
                    filter
                        .lock()
                        .await
                        .process_frame(FilterControlFrame::Enable(f.clone()))
                        .await;
                }
                // Forward downstream.
                ctx.push_frame(envelope, direction).await?;
            }

            _ => {
                // Forward everything else in the same direction.
                ctx.push_frame(envelope, direction).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use bytes::Bytes;
    use pipecat_core::test_utils::*;

    use super::*;

    fn make_input_audio(samples: &[i16], sample_rate: u32) -> AudioRawFrame {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        AudioRawFrame {
            audio: Bytes::from(bytes),
            sample_rate,
            num_channels: 1,
        }
    }

    // -- Mock filter for testing --

    struct MockFilter {
        call_count: Arc<StdMutex<u32>>,
    }

    impl MockFilter {
        fn new() -> (Self, Arc<StdMutex<u32>>) {
            let count = Arc::new(StdMutex::new(0));
            (
                Self {
                    call_count: count.clone(),
                },
                count,
            )
        }
    }

    #[async_trait]
    impl AudioFilter for MockFilter {
        async fn start(&mut self, _sample_rate: u32) {}
        async fn stop(&mut self) {}
        async fn process_frame(&mut self, _frame: FilterControlFrame) {}
        async fn filter(&mut self, audio: Bytes) -> Bytes {
            *self.call_count.lock().unwrap() += 1;
            audio // passthrough
        }
    }

    /// Filter that returns empty bytes to simulate buffering.
    struct BufferingFilter;

    #[async_trait]
    impl AudioFilter for BufferingFilter {
        async fn start(&mut self, _sample_rate: u32) {}
        async fn stop(&mut self) {}
        async fn process_frame(&mut self, _frame: FilterControlFrame) {}
        async fn filter(&mut self, _audio: Bytes) -> Bytes {
            Bytes::new() // buffering, no output
        }
    }

    #[tokio::test]
    async fn start_frame_sets_sample_rate() {
        let params = TransportParams {
            audio_in_enabled: true,
            ..Default::default()
        };
        let mut transport = BaseInputTransport::new(params);
        let start = Frame::Start(StartFrame {
            audio_in_sample_rate: 16000,
            ..Default::default()
        });

        let (downstream, _upstream) =
            run_processor(&mut transport, vec![(start, Direction::Downstream)]).await;

        assert_eq!(transport.sample_rate(), 16000);
        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::Start(_)));
    }

    #[tokio::test]
    async fn params_sample_rate_overrides_start_frame() {
        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_sample_rate: Some(48000),
            ..Default::default()
        };
        let mut transport = BaseInputTransport::new(params);
        let start = Frame::Start(StartFrame {
            audio_in_sample_rate: 16000,
            ..Default::default()
        });

        let _ = run_processor(&mut transport, vec![(start, Direction::Downstream)]).await;

        assert_eq!(transport.sample_rate(), 48000);
    }

    #[tokio::test]
    async fn lifecycle_frames_forwarded() {
        let params = TransportParams::default();
        let mut transport = BaseInputTransport::new(params);

        let (downstream, _) = run_processor(
            &mut transport,
            vec![
                (Frame::Start(StartFrame::default()), Direction::Downstream),
                (Frame::End(EndFrame::default()), Direction::Downstream),
            ],
        )
        .await;

        assert_eq!(downstream.len(), 2);
        assert!(matches!(&downstream[0].frame, Frame::Start(_)));
        assert!(matches!(&downstream[1].frame, Frame::End(_)));
    }

    #[tokio::test]
    async fn cancel_frame_forwarded() {
        let params = TransportParams::default();
        let mut transport = BaseInputTransport::new(params);

        let (downstream, _) = run_processor(
            &mut transport,
            vec![
                (Frame::Start(StartFrame::default()), Direction::Downstream),
                (Frame::Cancel(CancelFrame::default()), Direction::Downstream),
            ],
        )
        .await;

        assert_eq!(downstream.len(), 2);
        assert!(matches!(&downstream[0].frame, Frame::Start(_)));
        assert!(matches!(&downstream[1].frame, Frame::Cancel(_)));
    }

    #[tokio::test]
    async fn non_audio_frames_forwarded() {
        let params = TransportParams::default();
        let mut transport = BaseInputTransport::new(params);

        let (downstream, _) = run_processor(
            &mut transport,
            vec![
                (Frame::Text(TextFrame::new("hello")), Direction::Downstream),
                (
                    Frame::Interruption(InterruptionFrame),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        assert_eq!(downstream.len(), 2);
        assert!(matches!(&downstream[0].frame, Frame::Text(_)));
        assert!(matches!(&downstream[1].frame, Frame::Interruption(_)));
    }

    #[tokio::test]
    async fn audio_passthrough_works() {
        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_passthrough: true,
            ..Default::default()
        };
        let mut transport = BaseInputTransport::new(params);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_in_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        transport.set_transport_ready().await;

        let audio = make_input_audio(&[100, 200, 300], 16000);
        transport.push_audio_frame(audio).await;

        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }

        assert!(frames.len() >= 2);
        assert!(matches!(&frames[0].frame, Frame::Start(_)));
        assert!(matches!(&frames[1].frame, Frame::InputAudioRaw(_)));

        transport.stop().await;
    }

    #[tokio::test]
    async fn audio_disabled_does_not_forward() {
        let params = TransportParams {
            audio_in_enabled: false,
            ..Default::default()
        };
        let mut transport = BaseInputTransport::new(params);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame::default())),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        transport.set_transport_ready().await;

        let audio = make_input_audio(&[100, 200], 16000);
        transport.push_audio_frame(audio).await;

        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }
        assert_eq!(frames.len(), 1);
        assert!(matches!(&frames[0].frame, Frame::Start(_)));

        transport.stop().await;
    }

    #[tokio::test]
    async fn paused_blocks_audio() {
        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_passthrough: true,
            ..Default::default()
        };
        let mut transport = BaseInputTransport::new(params);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_in_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Pause via StopFrame.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Stop(StopFrame)),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Push audio while paused — should be dropped.
        let audio = make_input_audio(&[100], 16000);
        transport.push_audio_frame(audio).await;

        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }

        let audio_frames: Vec<_> = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::InputAudioRaw(_)))
            .collect();
        assert_eq!(audio_frames.len(), 0);

        transport.stop().await;
    }

    #[tokio::test]
    async fn filter_applied_in_audio_task() {
        let (filter, call_count) = MockFilter::new();
        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_passthrough: true,
            audio_in_filter: Some(Box::new(filter)),
            ..Default::default()
        };
        let mut transport = BaseInputTransport::new(params);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_in_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Push audio via the public API — filter runs inside the audio task.
        let audio = make_input_audio(&[100, 200, 300], 16000);
        transport.push_audio_frame(audio).await;

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Filter should have been called once.
        assert_eq!(*call_count.lock().unwrap(), 1);

        // Audio should appear downstream.
        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }
        let audio_frames: Vec<_> = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::InputAudioRaw(_)))
            .collect();
        assert_eq!(audio_frames.len(), 1);

        transport.stop().await;
    }

    #[tokio::test]
    async fn filter_buffering_skips_frame() {
        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_passthrough: true,
            audio_in_filter: Some(Box::new(BufferingFilter)),
            ..Default::default()
        };
        let mut transport = BaseInputTransport::new(params);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_in_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Push audio — filter returns empty, so no frame downstream.
        let audio = make_input_audio(&[100, 200], 16000);
        transport.push_audio_frame(audio).await;

        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }
        let audio_frames: Vec<_> = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::InputAudioRaw(_)))
            .collect();
        assert_eq!(audio_frames.len(), 0, "Buffering filter should skip frame");

        transport.stop().await;
    }

    #[tokio::test]
    async fn filter_control_frames_forwarded_downstream() {
        let (filter, _) = MockFilter::new();
        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_filter: Some(Box::new(filter)),
            ..Default::default()
        };
        let mut transport = BaseInputTransport::new(params);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame::default())),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Send FilterUpdateSettings — should be forwarded downstream.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::FilterUpdateSettings(FilterUpdateSettingsFrame {
                    settings: Default::default(),
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }

        let filter_frames: Vec<_> = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::FilterUpdateSettings(_)))
            .collect();
        assert_eq!(
            filter_frames.len(),
            1,
            "FilterUpdateSettings should be forwarded downstream"
        );

        transport.stop().await;
    }
}
