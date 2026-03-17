use std::{
    cmp::Ordering as CmpOrdering,
    collections::{BinaryHeap, HashMap},
    sync::{Arc, Mutex as StdMutex},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use pipecat_audio::{
    mixer::{AudioMixer, MixerControlFrame},
    resampler::AudioResampler,
};
use pipecat_core::{
    error::Result,
    frame::*,
    processor::{FrameProcessor, ProcessorBase, ProcessorContext},
};
use tokio::{
    sync::{Mutex, mpsc},
    task::JoinHandle,
};
use tracing;

use crate::params::TransportParams;

/// Silence duration (without mixer) before detecting bot stopped speaking.
const BOT_VAD_STOP_SECS: Duration = Duration::from_millis(350);

/// Fallback silence duration (with mixer) before detecting bot stopped speaking.
const BOT_VAD_STOP_FALLBACK_SECS: Duration = Duration::from_secs(3);

/// Period between `BotSpeakingFrame` emissions while bot is speaking.
const BOT_SPEAKING_FRAME_PERIOD: Duration = Duration::from_millis(200);

/// Maximum absolute i16 sample amplitude considered silence.
/// Matches Python's `SPEAKING_THRESHOLD = 20`.
const SPEAKING_THRESHOLD: u16 = 20;

// ---------------------------------------------------------------------------
// OutputTransportCallbacks — virtual methods for concrete transports
// ---------------------------------------------------------------------------

/// Callback trait that concrete output transports implement to handle actual I/O.
///
/// Methods are called by the transport's internal audio/media tasks. The trait
/// is `Send + Sync` so it can be shared with spawned tasks via `Arc`.
#[async_trait]
pub trait OutputTransportCallbacks: Send + Sync + 'static {
    /// Write chunked audio to the external destination.
    ///
    /// Returns `true` if the write succeeded and the frame should be pushed
    /// downstream, `false` to skip the downstream push.
    async fn write_audio_frame(&self, _frame: &AudioRawFrame) -> bool {
        false
    }

    /// Write a video frame to the external destination.
    async fn write_video_frame(&self, _frame: &ImageRawFrame) -> bool {
        false
    }

    /// Send a transport message (e.g. data channel message).
    async fn send_message(&self, _message: &serde_json::Value) {}

    /// Write a DTMF tone to the external destination.
    async fn write_dtmf(&self, _frame: &OutputDTMFFrame) {}

    /// Handle a non-audio frame that was queued through the audio pipeline
    /// for ordering guarantees. Called for frame types not handled by the
    /// transport internally.
    async fn write_transport_frame(&self, _frame: &Frame) {}

    /// Register a named audio output destination with the concrete transport.
    async fn register_audio_destination(&self, _destination: &str) {}

    /// Register a named video output destination with the concrete transport.
    async fn register_video_destination(&self, _destination: &str) {}
}

// ---------------------------------------------------------------------------
// AudioFrameKind — preserves frame variant through buffering
// ---------------------------------------------------------------------------

/// Tracks which Frame variant audio came from, so we can reconstruct
/// the correct variant after chunking.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AudioFrameKind {
    Plain,
    Tts { context_id: Option<String> },
    SpeechOutput,
}

// ---------------------------------------------------------------------------
// ClockEntry — priority queue entry for pts-based frame delivery
// ---------------------------------------------------------------------------

struct ClockEntry {
    pts: u64,
    frame_id: u64,
    envelope: FrameEnvelope,
}

// Min-heap: smallest pts first, then smallest frame_id.
impl Eq for ClockEntry {}
impl PartialEq for ClockEntry {
    fn eq(&self, other: &Self) -> bool {
        self.pts == other.pts && self.frame_id == other.frame_id
    }
}
impl PartialOrd for ClockEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}
impl Ord for ClockEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverse ordering for min-heap (BinaryHeap is max-heap by default).
        other
            .pts
            .cmp(&self.pts)
            .then_with(|| other.frame_id.cmp(&self.frame_id))
    }
}

// ---------------------------------------------------------------------------
// MediaSender — per-destination audio streaming
// ---------------------------------------------------------------------------

/// Manages audio streaming for a single output destination.
///
/// Buffers incoming audio into fixed-size chunks, runs a background task
/// that writes chunks to the transport, and tracks bot speaking state.
struct MediaSender {
    // Destination identifier (None = default sender).
    // Not read yet but stored for future use in logging/callbacks.
    #[allow(dead_code)]
    destination: Option<String>,

    // Configuration
    sample_rate: u32,
    audio_chunk_size: usize,
    audio_out_enabled: bool,
    audio_out_channels: u16,
    audio_out_end_silence_secs: u32,
    has_mixer: bool,

    // Audio buffering (used in process_frame context)
    audio_buffer: BytesMut,
    current_frame_kind: AudioFrameKind,

    // Audio task communication
    audio_tx: Option<mpsc::Sender<FrameEnvelope>>,
    audio_task: Option<JoinHandle<()>>,

    // Clock task communication (pts-based timed delivery)
    clock_tx: Option<mpsc::Sender<FrameEnvelope>>,
    clock_task: Option<JoinHandle<()>>,

    // Video task communication
    video_tx: Option<mpsc::Sender<ImageRawFrame>>,
    video_task: Option<JoinHandle<()>>,

    // Video cycling state (shared with video task for non-live mode)
    video_images: Arc<StdMutex<Vec<ImageRawFrame>>>,

    // Video configuration (from TransportParams)
    video_out_enabled: bool,
    video_out_is_live: bool,
    video_out_width: u32,
    video_out_height: u32,
    video_out_framerate: u32,

    // Shared state
    callbacks: Arc<dyn OutputTransportCallbacks>,
    resampler: Option<Arc<Mutex<Box<dyn AudioResampler>>>>,
    mixer: Option<Arc<Mutex<Box<dyn AudioMixer>>>>,

    // Bot speaking state (tracked via audio task, but need for interruption check)
    bot_speaking: Arc<std::sync::atomic::AtomicBool>,
}

impl MediaSender {
    fn new(
        destination: Option<String>,
        sample_rate: u32,
        audio_chunk_size: usize,
        params: &TransportParams,
        callbacks: Arc<dyn OutputTransportCallbacks>,
        resampler: Option<Arc<Mutex<Box<dyn AudioResampler>>>>,
        mixer: Option<Arc<Mutex<Box<dyn AudioMixer>>>>,
    ) -> Self {
        Self {
            destination,
            sample_rate,
            audio_chunk_size,
            audio_out_enabled: params.audio_out_enabled,
            audio_out_channels: params.audio_out_channels,
            audio_out_end_silence_secs: params.audio_out_end_silence_secs,
            has_mixer: mixer.is_some(),
            audio_buffer: BytesMut::new(),
            current_frame_kind: AudioFrameKind::Plain,
            audio_tx: None,
            audio_task: None,
            clock_tx: None,
            clock_task: None,
            video_tx: None,
            video_task: None,
            video_images: Arc::new(StdMutex::new(Vec::new())),
            video_out_enabled: params.video_out_enabled,
            video_out_is_live: params.video_out_is_live,
            video_out_width: params.video_out_width,
            video_out_height: params.video_out_height,
            video_out_framerate: params.video_out_framerate,
            callbacks,
            resampler,
            mixer,
            bot_speaking: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    async fn start(&mut self, ctx: ProcessorContext) {
        if let Some(ref mixer) = self.mixer {
            mixer.lock().await.start(self.sample_rate).await;
        }
        self.create_audio_task(ctx.clone());
        self.create_clock_task(ctx.clone());
        self.create_video_task(ctx);
    }

    async fn stop(&mut self, _ctx: &ProcessorContext) {
        // Send EndFrame through the audio queue so the task can flush.
        if let Some(ref tx) = self.audio_tx {
            tx.send(FrameEnvelope::new(Frame::End(EndFrame::default())))
                .await
                .ok();
        }

        // Wait for the audio task to finish processing EndFrame.
        if let Some(handle) = self.audio_task.take() {
            let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
        }
        self.audio_tx = None;

        // Send EndFrame to clock queue.
        if let Some(ref tx) = self.clock_tx {
            let mut env = FrameEnvelope::new(Frame::End(EndFrame::default()));
            env.header.pts = Some(u64::MAX);
            tx.send(env).await.ok();
        }
        if let Some(handle) = self.clock_task.take() {
            let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
        }
        self.clock_tx = None;

        // Cancel video task (after audio and clock tasks finish).
        self.video_tx = None;
        if let Some(handle) = self.video_task.take() {
            handle.abort();
            let _ = handle.await;
        }

        if let Some(ref mixer) = self.mixer {
            mixer.lock().await.stop().await;
        }
    }

    async fn cancel(&mut self) {
        // Drop sender to signal task to exit.
        self.audio_tx = None;
        if let Some(handle) = self.audio_task.take() {
            let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
        }

        self.clock_tx = None;
        if let Some(handle) = self.clock_task.take() {
            let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
        }

        self.video_tx = None;
        if let Some(handle) = self.video_task.take() {
            handle.abort();
            let _ = handle.await;
        }
    }

    async fn handle_interruptions(&mut self, allow_interruptions: bool, ctx: &ProcessorContext) {
        if !allow_interruptions {
            return;
        }

        let was_speaking = self.bot_speaking.load(std::sync::atomic::Ordering::Relaxed);

        // Cancel current audio task and recreate with fresh channel.
        self.audio_tx = None;
        if let Some(handle) = self.audio_task.take() {
            handle.abort();
            let _ = handle.await;
        }

        // Cancel current clock task.
        self.clock_tx = None;
        if let Some(handle) = self.clock_task.take() {
            handle.abort();
            let _ = handle.await;
        }

        // Cancel current video task.
        self.video_tx = None;
        if let Some(handle) = self.video_task.take() {
            handle.abort();
            let _ = handle.await;
        }

        // Clear buffered audio.
        self.audio_buffer.clear();

        // Emit BotStoppedSpeaking if we were speaking.
        if was_speaking {
            self.bot_speaking
                .store(false, std::sync::atomic::Ordering::Relaxed);
            broadcast_with_destination(
                ctx,
                Frame::BotStoppedSpeaking(BotStoppedSpeakingFrame),
                &self.destination,
            )
            .await;
        }

        // Recreate the audio, clock, and video tasks.
        self.create_audio_task(ctx.clone());
        self.create_clock_task(ctx.clone());
        self.create_video_task(ctx.clone());
    }

    async fn handle_audio_frame(&mut self, envelope: FrameEnvelope) {
        if !self.audio_out_enabled {
            return;
        }

        // Determine the frame kind and extract audio data.
        let (audio, frame_sample_rate, frame_channels, kind) = match &envelope.frame {
            Frame::TTSAudioRaw(f) => (
                f.audio.clone(),
                f.sample_rate,
                f.num_channels,
                AudioFrameKind::Tts {
                    context_id: f.context_id.clone(),
                },
            ),
            Frame::SpeechOutputAudioRaw(f) => (
                f.audio.clone(),
                f.sample_rate,
                f.num_channels,
                AudioFrameKind::SpeechOutput,
            ),
            Frame::OutputAudioRaw(f) => (
                f.audio.clone(),
                f.sample_rate,
                f.num_channels,
                AudioFrameKind::Plain,
            ),
            _ => return,
        };

        // Resample if needed.
        let audio = if frame_sample_rate != self.sample_rate {
            if let Some(ref resampler) = self.resampler {
                resampler
                    .lock()
                    .await
                    .resample(audio, frame_sample_rate, self.sample_rate)
                    .await
            } else {
                tracing::warn!(
                    "Audio sample rate mismatch ({} != {}) but no resampler configured",
                    frame_sample_rate,
                    self.sample_rate
                );
                audio
            }
        } else {
            audio
        };

        // Update the current frame kind for chunking.
        self.current_frame_kind = kind;

        // Buffer audio and chunk to audio_chunk_size.
        self.audio_buffer.extend_from_slice(&audio);
        while self.audio_buffer.len() >= self.audio_chunk_size {
            let chunk = self.audio_buffer.split_to(self.audio_chunk_size).freeze();
            let chunk_frame = AudioRawFrame {
                audio: chunk,
                sample_rate: self.sample_rate,
                num_channels: frame_channels,
            };

            // Reconstruct the correct Frame variant.
            let frame = match &self.current_frame_kind {
                AudioFrameKind::Tts { context_id } => Frame::TTSAudioRaw(TTSAudioRawFrame {
                    audio: chunk_frame.audio,
                    sample_rate: chunk_frame.sample_rate,
                    num_channels: chunk_frame.num_channels,
                    context_id: context_id.clone(),
                }),
                AudioFrameKind::SpeechOutput => Frame::SpeechOutputAudioRaw(AudioRawFrame {
                    audio: chunk_frame.audio,
                    sample_rate: chunk_frame.sample_rate,
                    num_channels: chunk_frame.num_channels,
                }),
                AudioFrameKind::Plain => Frame::OutputAudioRaw(chunk_frame),
            };

            if let Some(ref tx) = self.audio_tx {
                tx.send(FrameEnvelope::new(frame)).await.ok();
            }
        }
    }

    /// Enqueue a non-audio frame in the audio queue for ordering guarantees.
    async fn enqueue_sync_frame(&self, envelope: FrameEnvelope) {
        if let Some(ref tx) = self.audio_tx {
            tx.send(envelope).await.ok();
        }
    }

    async fn handle_mixer_control(&self, frame: MixerControlFrame) {
        if let Some(ref mixer) = self.mixer {
            mixer.lock().await.process_frame(frame).await;
        }
    }

    fn create_audio_task(&mut self, ctx: ProcessorContext) {
        let (tx, rx) = mpsc::channel::<FrameEnvelope>(64);
        self.audio_tx = Some(tx);

        let callbacks = self.callbacks.clone();
        let mixer = self.mixer.clone();
        let bot_speaking = self.bot_speaking.clone();
        let has_mixer = self.has_mixer;
        let sample_rate = self.sample_rate;
        let channels = self.audio_out_channels;
        let audio_chunk_size = self.audio_chunk_size;
        let end_silence_secs = self.audio_out_end_silence_secs;

        let destination = self.destination.clone();

        self.audio_task = Some(tokio::spawn(audio_output_task(
            ctx,
            rx,
            callbacks,
            mixer,
            bot_speaking,
            has_mixer,
            sample_rate,
            channels,
            audio_chunk_size,
            end_silence_secs,
            destination,
        )));
    }

    fn create_clock_task(&mut self, ctx: ProcessorContext) {
        let (tx, rx) = mpsc::channel::<FrameEnvelope>(64);
        self.clock_tx = Some(tx);
        self.clock_task = Some(tokio::spawn(clock_output_task(ctx, rx)));
    }

    async fn handle_timed_frame(&self, envelope: FrameEnvelope) {
        if let Some(ref tx) = self.clock_tx {
            tx.send(envelope).await.ok();
        }
    }

    fn create_video_task(&mut self, ctx: ProcessorContext) {
        if !self.video_out_enabled {
            return;
        }
        let width = self.video_out_width;
        let height = self.video_out_height;
        if self.video_out_is_live {
            let (tx, rx) = mpsc::channel::<ImageRawFrame>(64);
            self.video_tx = Some(tx);
            self.video_task = Some(tokio::spawn(video_live_task(
                ctx,
                rx,
                self.callbacks.clone(),
                self.video_out_framerate,
                width,
                height,
            )));
        } else {
            self.video_task = Some(tokio::spawn(video_cycling_task(
                ctx,
                self.callbacks.clone(),
                self.video_images.clone(),
                self.video_out_framerate,
                width,
                height,
            )));
        }
    }

    fn set_video_image(&self, image: ImageRawFrame) {
        if let Ok(mut images) = self.video_images.lock() {
            *images = vec![image];
        }
    }

    fn set_video_images(&self, images: Vec<ImageRawFrame>) {
        if let Ok(mut current) = self.video_images.lock() {
            *current = images;
        }
    }

    async fn handle_image_frame(&mut self, envelope: &FrameEnvelope) {
        if !self.video_out_enabled {
            return;
        }
        match &envelope.frame {
            Frame::OutputImageRaw(f) if self.video_out_is_live => {
                if let Some(ref tx) = self.video_tx {
                    tx.send(f.clone()).await.ok();
                }
            }
            Frame::OutputImageRaw(f) => {
                self.set_video_image(f.clone());
            }
            Frame::AssistantImageRaw(f) => {
                let image = ImageRawFrame {
                    image: f.image.clone(),
                    size: f.size,
                    format: f.format.clone(),
                };
                if self.video_out_is_live {
                    if let Some(ref tx) = self.video_tx {
                        tx.send(image).await.ok();
                    }
                } else {
                    self.set_video_image(image);
                }
            }
            Frame::Sprite(f) => {
                self.set_video_images(f.images.clone());
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Audio output task
// ---------------------------------------------------------------------------

/// State tracked inside the audio output task for bot speaking detection.
struct BotSpeakingState {
    speaking: bool,
    tts_audio_received: bool,
    last_speaking_frame_time: Instant,
    last_speech_time: Instant,
}

impl BotSpeakingState {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            speaking: false,
            tts_audio_received: false,
            last_speaking_frame_time: now,
            last_speech_time: now,
        }
    }
}

/// Broadcast a frame both directions with `transport_destination` set.
async fn broadcast_with_destination(
    ctx: &ProcessorContext,
    frame: Frame,
    destination: &Option<String>,
) {
    let mut env1 = FrameEnvelope::new(frame.clone());
    let mut env2 = FrameEnvelope::new(frame);
    env1.header.broadcast_sibling_id = Some(env2.header.id);
    env2.header.broadcast_sibling_id = Some(env1.header.id);
    env1.header.transport_destination = destination.clone();
    env2.header.transport_destination = destination.clone();
    ctx.push_downstream(env1).await.ok();
    ctx.push_upstream(env2).await.ok();
}

async fn bot_started_speaking(
    state: &mut BotSpeakingState,
    bot_speaking_flag: &std::sync::atomic::AtomicBool,
    ctx: &ProcessorContext,
    destination: &Option<String>,
) {
    if !state.speaking {
        state.speaking = true;
        bot_speaking_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        broadcast_with_destination(
            ctx,
            Frame::BotStartedSpeaking(BotStartedSpeakingFrame),
            destination,
        )
        .await;
    }
}

async fn bot_stopped_speaking(
    state: &mut BotSpeakingState,
    bot_speaking_flag: &std::sync::atomic::AtomicBool,
    ctx: &ProcessorContext,
    destination: &Option<String>,
) {
    if state.speaking {
        state.speaking = false;
        state.tts_audio_received = false;
        bot_speaking_flag.store(false, std::sync::atomic::Ordering::Relaxed);
        broadcast_with_destination(
            ctx,
            Frame::BotStoppedSpeaking(BotStoppedSpeakingFrame),
            destination,
        )
        .await;
    }
}

async fn bot_currently_speaking(
    state: &mut BotSpeakingState,
    bot_speaking_flag: &std::sync::atomic::AtomicBool,
    ctx: &ProcessorContext,
    destination: &Option<String>,
) {
    bot_started_speaking(state, bot_speaking_flag, ctx, destination).await;
    state.last_speech_time = Instant::now();

    // Emit periodic BotSpeakingFrame.
    if state.last_speaking_frame_time.elapsed() >= BOT_SPEAKING_FRAME_PERIOD {
        state.last_speaking_frame_time = Instant::now();
        ctx.send_downstream(Frame::BotSpeaking(BotSpeakingFrame))
            .await
            .ok();
    }
}

/// Check if audio data is silence by examining i16 sample amplitudes.
///
/// Interprets bytes as little-endian i16 PCM samples and returns true if
/// the maximum absolute amplitude is at or below [`SPEAKING_THRESHOLD`].
fn is_silence(audio: &[u8]) -> bool {
    let max_abs = audio
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]).unsigned_abs())
        .max()
        .unwrap_or(0);
    max_abs <= SPEAKING_THRESHOLD
}

/// Extract AudioRawFrame from an output audio frame variant.
fn extract_audio(frame: &Frame) -> Option<AudioRawFrame> {
    match frame {
        Frame::OutputAudioRaw(f) => Some(f.clone()),
        Frame::TTSAudioRaw(f) => Some(AudioRawFrame {
            audio: f.audio.clone(),
            sample_rate: f.sample_rate,
            num_channels: f.num_channels,
        }),
        Frame::SpeechOutputAudioRaw(f) => Some(f.clone()),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
async fn audio_output_task(
    ctx: ProcessorContext,
    mut rx: mpsc::Receiver<FrameEnvelope>,
    callbacks: Arc<dyn OutputTransportCallbacks>,
    mixer: Option<Arc<Mutex<Box<dyn AudioMixer>>>>,
    bot_speaking_flag: Arc<std::sync::atomic::AtomicBool>,
    has_mixer: bool,
    sample_rate: u32,
    channels: u16,
    audio_chunk_size: usize,
    end_silence_secs: u32,
    destination: Option<String>,
) {
    let mut state = BotSpeakingState::new();
    let timeout_duration = if has_mixer {
        BOT_VAD_STOP_FALLBACK_SECS
    } else {
        BOT_VAD_STOP_SECS
    };

    loop {
        let envelope = match tokio::time::timeout(timeout_duration, rx.recv()).await {
            Ok(Some(env)) => env,
            Ok(None) => break, // Channel closed
            Err(_) => {
                // Timeout — no audio for a while. If bot was speaking, stop.
                bot_stopped_speaking(&mut state, &bot_speaking_flag, &ctx, &destination).await;
                continue;
            }
        };

        match &envelope.frame {
            // -- EndFrame: send end silence, then exit --
            Frame::End(_) => {
                // Send trailing silence for clean audio teardown.
                if end_silence_secs > 0 {
                    let silence_bytes = (sample_rate as usize)
                        * (channels as usize)
                        * 2
                        * (end_silence_secs as usize);
                    let mut remaining = silence_bytes;
                    while remaining > 0 {
                        let chunk_len = remaining.min(audio_chunk_size);
                        let silence = AudioRawFrame {
                            audio: Bytes::from(vec![0u8; chunk_len]),
                            sample_rate,
                            num_channels: channels,
                        };
                        callbacks.write_audio_frame(&silence).await;
                        remaining -= chunk_len;
                    }
                }
                bot_stopped_speaking(&mut state, &bot_speaking_flag, &ctx, &destination).await;
                break;
            }

            // -- Audio frames: write + bot speaking detection --
            Frame::TTSAudioRaw(_) => {
                state.tts_audio_received = true;
                bot_currently_speaking(&mut state, &bot_speaking_flag, &ctx, &destination).await;

                if let Some(audio) = extract_audio(&envelope.frame) {
                    // Apply mixer if configured.
                    let audio = if let Some(ref mixer) = mixer {
                        let mixed = mixer.lock().await.mix(audio.audio).await;
                        AudioRawFrame {
                            audio: mixed,
                            sample_rate: audio.sample_rate,
                            num_channels: audio.num_channels,
                        }
                    } else {
                        audio
                    };

                    if callbacks.write_audio_frame(&audio).await {
                        ctx.push_frame(envelope, Direction::Downstream).await.ok();
                    }
                }
            }

            Frame::SpeechOutputAudioRaw(f) => {
                if is_silence(&f.audio) {
                    // Check if silence has exceeded the threshold.
                    if state.speaking && state.last_speech_time.elapsed() >= BOT_VAD_STOP_SECS {
                        bot_stopped_speaking(&mut state, &bot_speaking_flag, &ctx, &destination)
                            .await;
                    }
                } else {
                    bot_currently_speaking(&mut state, &bot_speaking_flag, &ctx, &destination)
                        .await;
                }

                if let Some(audio) = extract_audio(&envelope.frame) {
                    let audio = if let Some(ref mixer) = mixer {
                        let mixed = mixer.lock().await.mix(audio.audio).await;
                        AudioRawFrame {
                            audio: mixed,
                            sample_rate: audio.sample_rate,
                            num_channels: audio.num_channels,
                        }
                    } else {
                        audio
                    };

                    if callbacks.write_audio_frame(&audio).await {
                        ctx.push_frame(envelope, Direction::Downstream).await.ok();
                    }
                }
            }

            Frame::OutputAudioRaw(_) => {
                if let Some(audio) = extract_audio(&envelope.frame) {
                    let audio = if let Some(ref mixer) = mixer {
                        let mixed = mixer.lock().await.mix(audio.audio).await;
                        AudioRawFrame {
                            audio: mixed,
                            sample_rate: audio.sample_rate,
                            num_channels: audio.num_channels,
                        }
                    } else {
                        audio
                    };

                    if callbacks.write_audio_frame(&audio).await {
                        ctx.push_frame(envelope, Direction::Downstream).await.ok();
                    }
                }
            }

            // -- TTSStopped: signal bot stopped if TTS audio was received --
            Frame::TTSStopped(_) => {
                if state.tts_audio_received {
                    bot_stopped_speaking(&mut state, &bot_speaking_flag, &ctx, &destination).await;
                }
                ctx.push_frame(envelope, Direction::Downstream).await.ok();
            }

            // -- OutputTransportMessage: send via callback --
            Frame::OutputTransportMessage(f) => {
                callbacks.send_message(&f.message).await;
                ctx.push_frame(envelope, Direction::Downstream).await.ok();
            }

            // -- DTMF: write via callback --
            Frame::OutputDTMF(f) => {
                callbacks.write_dtmf(f).await;
                ctx.push_frame(envelope, Direction::Downstream).await.ok();
            }

            // -- Other sync frames --
            _ => {
                callbacks.write_transport_frame(&envelope.frame).await;
                ctx.push_frame(envelope, Direction::Downstream).await.ok();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Clock output task
// ---------------------------------------------------------------------------

/// Delivers frames at their presentation timestamp (pts).
///
/// Frames are buffered in a min-heap ordered by pts. The task waits until
/// the earliest frame's pts time arrives (relative to when the task started),
/// then pushes it downstream. New frames arriving while waiting are merged
/// into the heap.
async fn clock_output_task(ctx: ProcessorContext, mut rx: mpsc::Receiver<FrameEnvelope>) {
    let start_time = Instant::now();
    let mut heap: BinaryHeap<ClockEntry> = BinaryHeap::new();

    loop {
        if heap.is_empty() {
            // No pending frames — block until we receive one.
            match rx.recv().await {
                Some(env) => {
                    if matches!(&env.frame, Frame::End(_)) {
                        break;
                    }
                    let pts = env.header.pts.unwrap_or(0);
                    let frame_id = env.header.id;
                    heap.push(ClockEntry {
                        pts,
                        frame_id,
                        envelope: env,
                    });
                }
                None => break, // Channel closed
            }
        } else {
            // We have pending frames. Check if the earliest is due.
            let earliest_pts = heap.peek().unwrap().pts;
            let elapsed_ns = start_time.elapsed().as_nanos() as u64;

            if earliest_pts <= elapsed_ns {
                // Frame is due — deliver it.
                let entry = heap.pop().unwrap();
                if matches!(&entry.envelope.frame, Frame::End(_)) {
                    break;
                }
                ctx.push_downstream(entry.envelope).await.ok();
            } else {
                // Wait until the frame is due, or a new frame arrives.
                let wait = Duration::from_nanos(earliest_pts - elapsed_ns);
                tokio::select! {
                    _ = tokio::time::sleep(wait) => {
                        let entry = heap.pop().unwrap();
                        if matches!(&entry.envelope.frame, Frame::End(_)) {
                            break;
                        }
                        ctx.push_downstream(entry.envelope).await.ok();
                    }
                    recv_result = rx.recv() => {
                        match recv_result {
                            Some(env) => {
                                if matches!(&env.frame, Frame::End(_)) {
                                    // Drain remaining frames immediately.
                                    while let Some(entry) = heap.pop() {
                                        ctx.push_downstream(entry.envelope).await.ok();
                                    }
                                    break;
                                }
                                let pts = env.header.pts.unwrap_or(0);
                                let frame_id = env.header.id;
                                heap.push(ClockEntry { pts, frame_id, envelope: env });
                            }
                            None => break,
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Image resize helper
// ---------------------------------------------------------------------------

/// Bytes per pixel for a given format string. Returns None for unknown formats.
fn bytes_per_pixel(format: Option<&str>) -> Option<usize> {
    match format {
        Some("RGB") => Some(3),
        Some("RGBA") | Some("BGRA") | Some("ARGB") => Some(4),
        Some("L") | Some("GRAY") => Some(1),
        _ => None,
    }
}

/// Resize an image using nearest-neighbor interpolation if dimensions don't
/// match the target. Runs synchronously (intended for `spawn_blocking`).
///
/// Returns the original frame unchanged if dimensions already match or if the
/// pixel format is unknown (can't safely resize raw bytes without knowing bpp).
fn resize_image_if_needed(
    frame: ImageRawFrame,
    target_width: u32,
    target_height: u32,
) -> ImageRawFrame {
    let (src_w, src_h) = frame.size;
    if src_w == target_width && src_h == target_height {
        return frame;
    }

    let Some(bpp) = bytes_per_pixel(frame.format.as_deref()) else {
        // Unknown format — pass through without resizing.
        return frame;
    };

    let expected_len = (src_w as usize) * (src_h as usize) * bpp;
    if frame.image.len() != expected_len {
        // Data doesn't match declared dimensions — pass through.
        return frame;
    }

    let src_stride = src_w as usize * bpp;
    let dst_w = target_width as usize;
    let dst_h = target_height as usize;
    let mut dst = vec![0u8; dst_w * dst_h * bpp];

    for dy in 0..dst_h {
        let sy = dy * src_h as usize / dst_h;
        for dx in 0..dst_w {
            let sx = dx * src_w as usize / dst_w;
            let src_offset = sy * src_stride + sx * bpp;
            let dst_offset = dy * dst_w * bpp + dx * bpp;
            dst[dst_offset..dst_offset + bpp]
                .copy_from_slice(&frame.image[src_offset..src_offset + bpp]);
        }
    }

    ImageRawFrame {
        image: Bytes::from(dst),
        size: (target_width, target_height),
        format: frame.format,
    }
}

/// Resize image on a blocking thread if dimensions don't match.
async fn maybe_resize_image(
    frame: ImageRawFrame,
    target_width: u32,
    target_height: u32,
) -> ImageRawFrame {
    if frame.size == (target_width, target_height) {
        return frame;
    }
    tokio::task::spawn_blocking(move || resize_image_if_needed(frame, target_width, target_height))
        .await
        .unwrap_or_else(|_| ImageRawFrame {
            image: Bytes::new(),
            size: (target_width, target_height),
            format: None,
        })
}

// ---------------------------------------------------------------------------
// Video cycling task
// ---------------------------------------------------------------------------

/// Loops through stored images at the configured framerate.
///
/// On each tick the task reads the current image list (set via `set_video_image`
/// or `set_video_images`), picks the next image in round-robin order, and
/// writes it through the transport callback. The task runs until aborted.
async fn video_cycling_task(
    _ctx: ProcessorContext,
    callbacks: Arc<dyn OutputTransportCallbacks>,
    video_images: Arc<StdMutex<Vec<ImageRawFrame>>>,
    framerate: u32,
    target_width: u32,
    target_height: u32,
) {
    let frame_duration = Duration::from_secs_f64(1.0 / framerate as f64);
    let mut index: usize = 0;

    loop {
        let image = {
            let images = video_images.lock().unwrap();
            if images.is_empty() {
                None
            } else {
                let img = images[index % images.len()].clone();
                index = index.wrapping_add(1);
                Some(img)
            }
        };

        if let Some(image) = image {
            let image = maybe_resize_image(image, target_width, target_height).await;
            callbacks.write_video_frame(&image).await;
        }

        tokio::time::sleep(frame_duration).await;
    }
}

// ---------------------------------------------------------------------------
// Video live task
// ---------------------------------------------------------------------------

/// Receives video frames from a channel and writes them with timing control.
///
/// Paces output to the configured framerate. If the sender falls behind by
/// more than 5 frame durations the timing baseline is reset.
async fn video_live_task(
    _ctx: ProcessorContext,
    mut rx: mpsc::Receiver<ImageRawFrame>,
    callbacks: Arc<dyn OutputTransportCallbacks>,
    framerate: u32,
    target_width: u32,
    target_height: u32,
) {
    let frame_duration = Duration::from_secs_f64(1.0 / framerate as f64);
    let frame_reset = frame_duration * 5;
    let mut start_time: Option<Instant> = None;
    let mut frame_index: u64 = 0;

    while let Some(image) = rx.recv().await {
        if start_time.is_none() {
            start_time = Some(Instant::now());
            frame_index = 0;
        }

        let st = start_time.unwrap();
        let real_elapsed = st.elapsed();
        let expected_time = frame_duration * frame_index as u32;

        let delay = if expected_time > real_elapsed {
            expected_time - real_elapsed + frame_duration
        } else {
            // We're behind — no delay, but check for reset.
            Duration::ZERO
        };

        // Reset timing if drift is too large.
        if delay > frame_reset || real_elapsed > expected_time + frame_reset {
            start_time = Some(Instant::now());
            frame_index = 0;
        } else if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }

        frame_index += 1;

        let image = maybe_resize_image(image, target_width, target_height).await;
        callbacks.write_video_frame(&image).await;
    }
}

// ---------------------------------------------------------------------------
// BaseOutputTransport
// ---------------------------------------------------------------------------

/// Base output transport that receives audio/video from the pipeline
/// and writes it to an external destination.
///
/// Concrete transports implement [`OutputTransportCallbacks`] and pass
/// an `Arc` to the constructor. The transport handles buffering, chunking,
/// bot speaking detection, and interruption management.
///
/// # Lifecycle
///
/// 1. Pipeline sends `StartFrame` → transport initializes sample rate, chunk size
/// 2. `set_transport_ready()` → creates MediaSender, spawns audio task, sends
///    `OutputTransportReady` upstream
/// 3. Audio frames arrive → buffered, chunked, written via callbacks
/// 4. `InterruptionFrame` → audio task restarted, buffers cleared
/// 5. `EndFrame` → trailing silence sent, task stopped
pub struct BaseOutputTransport {
    base: ProcessorBase,
    params: TransportParams,
    callbacks: Arc<dyn OutputTransportCallbacks>,

    // Runtime state
    sample_rate: u32,
    audio_chunk_size: usize,
    allow_interruptions: bool,

    // Per-destination media senders (None key = default sender)
    media_senders: HashMap<Option<String>, MediaSender>,

    // Stored context for spawned tasks
    ctx: Option<ProcessorContext>,
}

impl BaseOutputTransport {
    pub fn new(params: TransportParams, callbacks: Arc<dyn OutputTransportCallbacks>) -> Self {
        Self {
            base: ProcessorBase::new("BaseOutputTransport"),
            params,
            callbacks,
            sample_rate: 0,
            audio_chunk_size: 0,
            allow_interruptions: false,
            media_senders: HashMap::new(),
            ctx: None,
        }
    }

    pub fn with_name(
        name: impl Into<String>,
        params: TransportParams,
        callbacks: Arc<dyn OutputTransportCallbacks>,
    ) -> Self {
        Self {
            base: ProcessorBase::new(name),
            params,
            callbacks,
            sample_rate: 0,
            audio_chunk_size: 0,
            allow_interruptions: false,
            media_senders: HashMap::new(),
            ctx: None,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn audio_chunk_size(&self) -> usize {
        self.audio_chunk_size
    }

    /// Signal that the output transport is ready.
    ///
    /// Creates the MediaSender, starts the audio task, and sends
    /// `OutputTransportReady` upstream.
    pub async fn set_transport_ready(&mut self) {
        let Some(ref ctx) = self.ctx else {
            tracing::warn!("BaseOutputTransport: cannot set ready before StartFrame");
            return;
        };

        // Take ownership of resampler from params (shared across all senders).
        let resampler = self
            .params
            .audio_out_resampler
            .take()
            .map(|r| Arc::new(Mutex::new(r)));

        // Take ownership of the default mixer (used by the None destination).
        let default_mixer = self
            .params
            .audio_out_mixer
            .take()
            .map(|m| Arc::new(Mutex::new(m)));

        // Create default sender with the default mixer.
        let mut default_sender = MediaSender::new(
            None,
            self.sample_rate,
            self.audio_chunk_size,
            &self.params,
            self.callbacks.clone(),
            resampler.clone(),
            default_mixer,
        );
        default_sender.start(ctx.clone()).await;
        self.media_senders.insert(None, default_sender);

        // Create per-destination senders.
        let mut destinations = std::collections::HashSet::new();
        for dest in &self.params.audio_out_destinations {
            destinations.insert(dest.clone());
        }
        for dest in &self.params.video_out_destinations {
            destinations.insert(dest.clone());
        }

        for dest in &destinations {
            self.callbacks.register_audio_destination(dest).await;
            self.callbacks.register_video_destination(dest).await;

            // Each named destination gets its own mixer from the map (if present).
            let dest_mixer = self
                .params
                .audio_out_mixer_map
                .remove(dest)
                .map(|m| Arc::new(Mutex::new(m)));

            let mut sender = MediaSender::new(
                Some(dest.clone()),
                self.sample_rate,
                self.audio_chunk_size,
                &self.params,
                self.callbacks.clone(),
                resampler.clone(),
                dest_mixer,
            );
            sender.start(ctx.clone()).await;
            self.media_senders.insert(Some(dest.clone()), sender);
        }

        // Signal readiness upstream.
        ctx.send_upstream(Frame::OutputTransportReady(OutputTransportReadyFrame))
            .await
            .ok();
    }

    // -- Destination routing helpers --

    /// Resolve a destination key: if the exact key exists use it,
    /// otherwise fall back to the default (None) sender.
    fn resolve_dest_key(&self, dest: &Option<String>) -> Option<String> {
        if self.media_senders.contains_key(dest) {
            dest.clone()
        } else {
            if dest.is_some() {
                tracing::warn!("Unknown transport destination: {:?}", dest);
            }
            None
        }
    }

    fn get_sender_mut(&mut self, dest: &Option<String>) -> Option<&mut MediaSender> {
        let key = self.resolve_dest_key(dest);
        self.media_senders.get_mut(&key)
    }

    fn get_sender(&self, dest: &Option<String>) -> Option<&MediaSender> {
        let key = self.resolve_dest_key(dest);
        self.media_senders.get(&key)
    }

    // -- Lifecycle methods --

    async fn start(&mut self, frame: &StartFrame, ctx: &ProcessorContext) {
        self.ctx = Some(ctx.clone());
        self.allow_interruptions = frame.allow_interruptions;
        self.sample_rate = self
            .params
            .audio_out_sample_rate
            .unwrap_or(frame.audio_out_sample_rate);

        // Calculate audio chunk size: (sample_rate / 100) * channels * 2 bytes * chunks
        let bytes_per_10ms =
            (self.sample_rate as usize / 100) * (self.params.audio_out_channels as usize) * 2;
        self.audio_chunk_size = bytes_per_10ms * (self.params.audio_out_10ms_chunks as usize);
    }

    async fn stop(&mut self) {
        if let Some(ref ctx) = self.ctx {
            for sender in self.media_senders.values_mut() {
                sender.stop(ctx).await;
            }
        }
    }

    async fn cancel(&mut self) {
        for sender in self.media_senders.values_mut() {
            sender.cancel().await;
        }
    }
}

#[async_trait]
impl FrameProcessor for BaseOutputTransport {
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
                let frame = frame.clone();
                ctx.push_frame(envelope, direction).await?;
                self.start(&frame, ctx).await;
            }

            Frame::End(_) => {
                self.stop().await;
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::Cancel(_) => {
                self.cancel().await;
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::Interruption(_) => {
                let dest = envelope.header.transport_destination.clone();
                let allow = self.allow_interruptions;
                ctx.push_frame(envelope, direction).await?;
                if let Some(sender) = self.get_sender_mut(&dest) {
                    sender.handle_interruptions(allow, ctx).await;
                }
            }

            Frame::OutputTransportMessageUrgent(f) => {
                self.callbacks.send_message(&f.message).await;
            }

            // Urgent DTMF bypasses audio queue.
            Frame::OutputDTMFUrgent(f) => {
                self.callbacks
                    .write_dtmf(&OutputDTMFFrame { button: f.button })
                    .await;
            }

            // Normal DTMF goes through audio queue for ordering.
            Frame::OutputDTMF(_) => {
                let dest = envelope.header.transport_destination.clone();
                if let Some(sender) = self.get_sender(&dest) {
                    sender.enqueue_sync_frame(envelope).await;
                }
            }

            // Route audio frames to MediaSender.
            Frame::OutputAudioRaw(_) | Frame::TTSAudioRaw(_) | Frame::SpeechOutputAudioRaw(_) => {
                let dest = envelope.header.transport_destination.clone();
                if let Some(sender) = self.get_sender_mut(&dest) {
                    sender.handle_audio_frame(envelope).await;
                }
            }

            // Mixer control frames.
            Frame::MixerUpdateSettings(f) => {
                let dest = envelope.header.transport_destination.clone();
                if let Some(sender) = self.get_sender(&dest) {
                    sender
                        .handle_mixer_control(MixerControlFrame::UpdateSettings(f.clone()))
                        .await;
                }
            }
            Frame::MixerEnable(f) => {
                let dest = envelope.header.transport_destination.clone();
                if let Some(sender) = self.get_sender(&dest) {
                    sender
                        .handle_mixer_control(MixerControlFrame::Enable(f.clone()))
                        .await;
                }
            }

            // Route video frames to MediaSender.
            Frame::OutputImageRaw(_) | Frame::Sprite(_) => {
                let dest = envelope.header.transport_destination.clone();
                if let Some(sender) = self.get_sender_mut(&dest) {
                    sender.handle_image_frame(&envelope).await;
                }
            }

            Frame::AssistantImageRaw(_) => {
                let dest = envelope.header.transport_destination.clone();
                if let Some(sender) = self.get_sender_mut(&dest) {
                    sender.handle_image_frame(&envelope).await;
                    sender.enqueue_sync_frame(envelope).await;
                }
            }

            // TTSStopped and other sync frames go through audio queue for ordering.
            Frame::TTSStopped(_) => {
                let dest = envelope.header.transport_destination.clone();
                if let Some(sender) = self.get_sender(&dest) {
                    sender.enqueue_sync_frame(envelope).await;
                }
            }

            _ => {
                if direction == Direction::Upstream {
                    ctx.push_upstream(envelope).await?;
                } else if envelope.frame.is_system() {
                    ctx.push_frame(envelope, direction).await?;
                } else if envelope.header.pts.is_some() {
                    // Frames with presentation timestamps go through the clock task.
                    let dest = envelope.header.transport_destination.clone();
                    if let Some(sender) = self.get_sender(&dest) {
                        sender.handle_timed_frame(envelope).await;
                    } else {
                        ctx.push_frame(envelope, direction).await?;
                    }
                } else {
                    // Other downstream frames go through audio queue for ordering.
                    let dest = envelope.header.transport_destination.clone();
                    if let Some(sender) = self.get_sender(&dest) {
                        sender.enqueue_sync_frame(envelope).await;
                    } else {
                        ctx.push_frame(envelope, direction).await?;
                    }
                }
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

    use pipecat_core::test_utils::*;

    use super::*;

    // -- Mock callbacks --

    struct MockCallbacks {
        written_audio: Arc<StdMutex<Vec<Bytes>>>,
        written_video: Arc<StdMutex<Vec<ImageRawFrame>>>,
        messages: Arc<StdMutex<Vec<serde_json::Value>>>,
        write_result: bool,
    }

    impl MockCallbacks {
        #[allow(clippy::type_complexity)]
        fn new(
            write_result: bool,
        ) -> (
            Arc<Self>,
            Arc<StdMutex<Vec<Bytes>>>,
            Arc<StdMutex<Vec<serde_json::Value>>>,
        ) {
            let audio = Arc::new(StdMutex::new(Vec::new()));
            let video = Arc::new(StdMutex::new(Vec::new()));
            let msgs = Arc::new(StdMutex::new(Vec::new()));
            let cb = Arc::new(Self {
                written_audio: audio.clone(),
                written_video: video,
                messages: msgs.clone(),
                write_result,
            });
            (cb, audio, msgs)
        }

        #[allow(clippy::type_complexity)]
        fn new_with_video(
            write_result: bool,
        ) -> (
            Arc<Self>,
            Arc<StdMutex<Vec<Bytes>>>,
            Arc<StdMutex<Vec<ImageRawFrame>>>,
            Arc<StdMutex<Vec<serde_json::Value>>>,
        ) {
            let audio = Arc::new(StdMutex::new(Vec::new()));
            let video = Arc::new(StdMutex::new(Vec::new()));
            let msgs = Arc::new(StdMutex::new(Vec::new()));
            let cb = Arc::new(Self {
                written_audio: audio.clone(),
                written_video: video.clone(),
                messages: msgs.clone(),
                write_result,
            });
            (cb, audio, video, msgs)
        }
    }

    #[async_trait]
    impl OutputTransportCallbacks for MockCallbacks {
        async fn write_audio_frame(&self, frame: &AudioRawFrame) -> bool {
            self.written_audio.lock().unwrap().push(frame.audio.clone());
            self.write_result
        }
        async fn write_video_frame(&self, frame: &ImageRawFrame) -> bool {
            self.written_video.lock().unwrap().push(frame.clone());
            self.write_result
        }
        async fn send_message(&self, message: &serde_json::Value) {
            self.messages.lock().unwrap().push(message.clone());
        }
    }

    fn make_output_audio(samples: &[i16], sample_rate: u32) -> Frame {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        Frame::OutputAudioRaw(AudioRawFrame {
            audio: Bytes::from(bytes),
            sample_rate,
            num_channels: 1,
        })
    }

    fn make_tts_audio(samples: &[i16], sample_rate: u32) -> Frame {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        Frame::TTSAudioRaw(TTSAudioRawFrame {
            audio: Bytes::from(bytes),
            sample_rate,
            num_channels: 1,
            context_id: None,
        })
    }

    #[tokio::test]
    async fn start_frame_sets_sample_rate_and_chunk_size() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 4,
            ..Default::default()
        };
        let (cb, _, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let start = Frame::Start(StartFrame {
            audio_out_sample_rate: 24000,
            ..Default::default()
        });
        let (downstream, _) =
            run_processor(&mut transport, vec![(start, Direction::Downstream)]).await;

        assert_eq!(transport.sample_rate(), 24000);
        // 24000/100 * 1 * 2 * 4 = 1920
        assert_eq!(transport.audio_chunk_size(), 1920);
        assert_eq!(downstream.len(), 1);
        assert!(matches!(&downstream[0].frame, Frame::Start(_)));
    }

    #[tokio::test]
    async fn params_sample_rate_overrides_start_frame() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_sample_rate: Some(48000),
            audio_out_channels: 1,
            audio_out_10ms_chunks: 2,
            ..Default::default()
        };
        let (cb, _, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let start = Frame::Start(StartFrame {
            audio_out_sample_rate: 24000,
            ..Default::default()
        });
        let _ = run_processor(&mut transport, vec![(start, Direction::Downstream)]).await;

        assert_eq!(transport.sample_rate(), 48000);
        // 48000/100 * 1 * 2 * 2 = 1920
        assert_eq!(transport.audio_chunk_size(), 1920);
    }

    #[tokio::test]
    async fn lifecycle_frames_forwarded() {
        let (cb, _, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(TransportParams::default(), cb);

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
    async fn audio_chunking_and_write_callback() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1, // 1 chunk = 10ms for easier testing
            ..Default::default()
        };
        let (cb, written_audio, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        // Set up channels.
        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, mut _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        // Start with 16000 Hz sample rate.
        // 10ms chunk = 16000/100 * 1 * 2 = 320 bytes
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(transport.audio_chunk_size(), 320);

        // Set ready to create MediaSender + audio task.
        transport.set_transport_ready().await;

        // Send audio that's exactly 2 chunks worth (640 bytes = 320 samples of i16).
        let samples: Vec<i16> = (0..320).collect();
        transport
            .process_frame(
                FrameEnvelope::new(make_output_audio(&samples, 16000)),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Give audio task time to process.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should have written exactly 2 chunks.
        {
            let written = written_audio.lock().unwrap();
            assert_eq!(written.len(), 2);
            assert_eq!(written[0].len(), 320);
            assert_eq!(written[1].len(), 320);
        }

        // Downstream should have: Start + 2 audio chunks.
        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }
        let audio_frames: Vec<_> = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::OutputAudioRaw(_)))
            .collect();
        assert_eq!(audio_frames.len(), 2);

        // Clean up.
        transport.cancel().await;
    }

    #[tokio::test]
    async fn tts_audio_triggers_bot_started_speaking() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            ..Default::default()
        };
        let (cb, _, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, mut up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        // Start.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Send TTS audio (exactly one chunk = 160 i16 samples = 320 bytes).
        let samples: Vec<i16> = (1..=160).collect();
        transport
            .process_frame(
                FrameEnvelope::new(make_tts_audio(&samples, 16000)),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check for BotStartedSpeaking in downstream frames.
        let mut downstream = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            downstream.push(env);
        }
        let bot_started: Vec<_> = downstream
            .iter()
            .filter(|f| matches!(&f.frame, Frame::BotStartedSpeaking(_)))
            .collect();
        assert_eq!(
            bot_started.len(),
            1,
            "Expected BotStartedSpeaking downstream"
        );

        // Also check upstream.
        let mut upstream = Vec::new();
        while let Ok(env) = up_rx.try_recv() {
            upstream.push(env);
        }
        let bot_started_up: Vec<_> = upstream
            .iter()
            .filter(|f| matches!(&f.frame, Frame::BotStartedSpeaking(_)))
            .collect();
        assert_eq!(
            bot_started_up.len(),
            1,
            "Expected BotStartedSpeaking upstream"
        );

        transport.cancel().await;
    }

    #[tokio::test]
    async fn tts_stopped_triggers_bot_stopped_speaking() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            ..Default::default()
        };
        let (cb, _, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, mut up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Send TTS audio to trigger BotStartedSpeaking.
        let samples: Vec<i16> = (1..=160).collect();
        transport
            .process_frame(
                FrameEnvelope::new(make_tts_audio(&samples, 16000)),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Send TTSStopped.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::TTSStopped(TTSStoppedFrame { context_id: None })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check for BotStoppedSpeaking.
        let mut downstream = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            downstream.push(env);
        }
        let bot_stopped: Vec<_> = downstream
            .iter()
            .filter(|f| matches!(&f.frame, Frame::BotStoppedSpeaking(_)))
            .collect();
        assert_eq!(
            bot_stopped.len(),
            1,
            "Expected BotStoppedSpeaking downstream"
        );

        let mut upstream = Vec::new();
        while let Ok(env) = up_rx.try_recv() {
            upstream.push(env);
        }
        let bot_stopped_up: Vec<_> = upstream
            .iter()
            .filter(|f| matches!(&f.frame, Frame::BotStoppedSpeaking(_)))
            .collect();
        assert_eq!(
            bot_stopped_up.len(),
            1,
            "Expected BotStoppedSpeaking upstream"
        );

        transport.cancel().await;
    }

    #[tokio::test]
    async fn interruption_clears_audio_and_emits_bot_stopped() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            ..Default::default()
        };
        let (cb, _, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, mut down_rx) = mpsc::channel(128);
        let (up_tx, mut up_rx) = mpsc::channel(128);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        // Start with interruptions enabled.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    allow_interruptions: true,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Send TTS audio to get bot speaking.
        let samples: Vec<i16> = (1..=160).collect();
        transport
            .process_frame(
                FrameEnvelope::new(make_tts_audio(&samples, 16000)),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Trigger interruption.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Interruption(InterruptionFrame)),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Check for BotStoppedSpeaking after interruption.
        let mut downstream = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            downstream.push(env);
        }
        let mut upstream = Vec::new();
        while let Ok(env) = up_rx.try_recv() {
            upstream.push(env);
        }

        let bot_stopped_down: Vec<_> = downstream
            .iter()
            .filter(|f| matches!(&f.frame, Frame::BotStoppedSpeaking(_)))
            .collect();
        let bot_stopped_up: Vec<_> = upstream
            .iter()
            .filter(|f| matches!(&f.frame, Frame::BotStoppedSpeaking(_)))
            .collect();

        // Should have BotStoppedSpeaking from the interruption.
        assert!(
            !bot_stopped_down.is_empty() || !bot_stopped_up.is_empty(),
            "Expected BotStoppedSpeaking after interruption"
        );

        transport.cancel().await;
    }

    #[tokio::test]
    async fn output_transport_ready_sent_upstream() {
        let params = TransportParams {
            audio_out_enabled: true,
            ..Default::default()
        };
        let (cb, _, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, mut _down_rx) = mpsc::channel(64);
        let (up_tx, mut up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        let mut upstream = Vec::new();
        while let Ok(env) = up_rx.try_recv() {
            upstream.push(env);
        }

        let ready: Vec<_> = upstream
            .iter()
            .filter(|f| matches!(&f.frame, Frame::OutputTransportReady(_)))
            .collect();
        assert_eq!(ready.len(), 1, "Expected OutputTransportReady upstream");

        transport.cancel().await;
    }

    #[tokio::test]
    async fn urgent_message_sent_directly() {
        let params = TransportParams::default();
        let (cb, _, messages) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let msg = serde_json::json!({"type": "test"});
        let frame = Frame::OutputTransportMessageUrgent(OutputTransportMessageUrgentFrame {
            message: msg.clone(),
        });

        let _ = run_processor(&mut transport, vec![(frame, Direction::Downstream)]).await;

        let sent = messages.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0], msg);
    }

    #[tokio::test]
    async fn end_frame_sends_silence() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            audio_out_end_silence_secs: 1,
            ..Default::default()
        };
        let (cb, written_audio, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, mut _down_rx) = mpsc::channel(256);
        let (up_tx, mut _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        // Start at 16000 Hz, chunk size = 320 bytes.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Send EndFrame.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::End(EndFrame::default())),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Wait for the audio task to process EndFrame and send silence.
        tokio::time::sleep(Duration::from_millis(200)).await;

        let written = written_audio.lock().unwrap();
        // 1 second of silence at 16000 Hz mono 16-bit = 32000 bytes
        // In 320-byte chunks = 100 chunks
        let total_silence: usize = written.iter().map(|b| b.len()).sum();
        assert_eq!(total_silence, 32000, "Expected 1 second of silence");
        // All silence bytes should be zero.
        for chunk in written.iter() {
            assert!(chunk.iter().all(|&b| b == 0), "Silence should be all zeros");
        }
    }

    // -- Silence detection tests --

    fn samples_to_bytes(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn silence_all_zeros() {
        let audio = samples_to_bytes(&[0, 0, 0, 0]);
        assert!(is_silence(&audio));
    }

    #[test]
    fn silence_at_threshold() {
        let audio = samples_to_bytes(&[20, -20, 15, 0]);
        assert!(is_silence(&audio));
    }

    #[test]
    fn not_silence_above_threshold() {
        let audio = samples_to_bytes(&[21, 0, 0, 0]);
        assert!(!is_silence(&audio));
    }

    #[test]
    fn not_silence_negative_above_threshold() {
        let audio = samples_to_bytes(&[0, -21, 0, 0]);
        assert!(!is_silence(&audio));
    }

    #[test]
    fn not_silence_large_amplitude() {
        let audio = samples_to_bytes(&[1000, -5000, 200, 0]);
        assert!(!is_silence(&audio));
    }

    #[test]
    fn silence_empty_audio() {
        assert!(is_silence(&[]));
    }

    #[test]
    fn silence_single_byte_treated_as_empty() {
        // Odd byte count: chunks_exact(2) yields nothing, max defaults to 0
        assert!(is_silence(&[0xFF]));
    }

    // -- Clock task tests --

    #[tokio::test]
    async fn clock_task_delivers_frame_with_pts_zero() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            ..Default::default()
        };
        let (cb, _, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        // Start transport.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Send a frame with pts=0 (should deliver immediately).
        let mut env = FrameEnvelope::new(Frame::Transcription(TranscriptionFrame {
            text: "hello".to_string(),
            user_id: "user1".to_string(),
            timestamp: None,
            language: None,
            finalized: true,
            result: None,
        }));
        env.header.pts = Some(0);
        transport
            .process_frame(env, Direction::Downstream, &ctx)
            .await
            .unwrap();

        // Give clock task time to deliver.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Collect downstream frames.
        let mut downstream = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            downstream.push(env);
        }

        let transcription_frames: Vec<_> = downstream
            .iter()
            .filter(|f| matches!(&f.frame, Frame::Transcription(_)))
            .collect();
        assert_eq!(
            transcription_frames.len(),
            1,
            "Expected one timed frame delivered via clock task"
        );

        transport.cancel().await;
    }

    #[tokio::test]
    async fn clock_task_end_frame_terminates_cleanly() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            ..Default::default()
        };
        let (cb, _, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, _down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        // Start transport.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Stop should send EndFrame to clock task and it should terminate.
        transport.stop().await;

        // Verify that the clock task handle has been consumed (taken).
        assert!(
            transport
                .media_senders
                .get(&None)
                .unwrap()
                .clock_task
                .is_none(),
            "Clock task should be consumed after stop"
        );
    }

    // -- Video task tests --

    fn make_test_image(id: u8) -> ImageRawFrame {
        ImageRawFrame {
            image: Bytes::from(vec![id; 64]),
            size: (8, 8),
            format: Some("RGB".to_string()),
        }
    }

    #[tokio::test]
    async fn video_cycling_writes_frames() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            video_out_enabled: true,
            video_out_is_live: false,
            video_out_framerate: 30,
            ..Default::default()
        };
        let (cb, _, written_video, _) = MockCallbacks::new_with_video(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, _down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        // Start transport.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Send a Sprite frame with 2 images.
        let sprite = Frame::Sprite(SpriteFrame {
            images: vec![make_test_image(1), make_test_image(2)],
        });
        transport
            .process_frame(FrameEnvelope::new(sprite), Direction::Downstream, &ctx)
            .await
            .unwrap();

        // Wait for cycling task to produce some frames (~200ms at 30fps = ~6 frames).
        tokio::time::sleep(Duration::from_millis(200)).await;

        {
            let video = written_video.lock().unwrap();
            assert!(
                video.len() >= 3,
                "Expected at least 3 video frames written by cycling task, got {}",
                video.len()
            );
            assert_eq!(video[0].image[0], 1);
            assert_eq!(video[1].image[0], 2);
            assert_eq!(video[2].image[0], 1);
        }

        transport.cancel().await;
    }

    #[tokio::test]
    async fn video_live_writes_frames() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            video_out_enabled: true,
            video_out_is_live: true,
            video_out_framerate: 30,
            ..Default::default()
        };
        let (cb, _, written_video, _) = MockCallbacks::new_with_video(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, _down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        // Start transport.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Send OutputImageRaw frames.
        for i in 0..3u8 {
            let frame = Frame::OutputImageRaw(make_test_image(i + 10));
            transport
                .process_frame(FrameEnvelope::new(frame), Direction::Downstream, &ctx)
                .await
                .unwrap();
        }

        // Wait for live task to process.
        tokio::time::sleep(Duration::from_millis(200)).await;

        {
            let video = written_video.lock().unwrap();
            assert_eq!(
                video.len(),
                3,
                "Expected 3 video frames written in live mode, got {}",
                video.len()
            );
            assert_eq!(video[0].image[0], 10);
            assert_eq!(video[1].image[0], 11);
            assert_eq!(video[2].image[0], 12);
        }

        transport.cancel().await;
    }

    #[tokio::test]
    async fn video_disabled_ignores_frames() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            video_out_enabled: false,
            ..Default::default()
        };
        let (cb, _, written_video, _) = MockCallbacks::new_with_video(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, _down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Send video frame — should be ignored.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::OutputImageRaw(make_test_image(1))),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        {
            let video = written_video.lock().unwrap();
            assert!(
                video.is_empty(),
                "Expected no video frames when video is disabled"
            );
        }

        transport.cancel().await;
    }

    #[tokio::test]
    async fn video_cycling_single_image_repeats() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            video_out_enabled: true,
            video_out_is_live: false,
            video_out_framerate: 30,
            ..Default::default()
        };
        let (cb, _, written_video, _) = MockCallbacks::new_with_video(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, _down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Send a single OutputImageRaw (sets a single image for cycling).
        transport
            .process_frame(
                FrameEnvelope::new(Frame::OutputImageRaw(make_test_image(42))),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;

        {
            let video = written_video.lock().unwrap();
            assert!(
                video.len() >= 3,
                "Expected at least 3 repeated video frames, got {}",
                video.len()
            );
            for frame in video.iter() {
                assert_eq!(frame.image[0], 42);
            }
        }

        transport.cancel().await;
    }

    #[tokio::test]
    async fn assistant_image_raw_routes_to_video_and_audio_queue() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            video_out_enabled: true,
            video_out_is_live: false,
            video_out_framerate: 30,
            ..Default::default()
        };
        let (cb, _, written_video, _) = MockCallbacks::new_with_video(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Send AssistantImageRaw — should set cycling image AND enqueue for downstream.
        let frame = Frame::AssistantImageRaw(AssistantImageRawFrame {
            image: Bytes::from(vec![99u8; 64]),
            size: (8, 8),
            format: Some("RGB".to_string()),
            original_data: None,
            original_mime_type: None,
        });
        transport
            .process_frame(FrameEnvelope::new(frame), Direction::Downstream, &ctx)
            .await
            .unwrap();

        // Wait for cycling and audio task to process.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Video should be cycling the assistant image.
        {
            let video = written_video.lock().unwrap();
            assert!(
                !video.is_empty(),
                "Expected video frames from AssistantImageRaw"
            );
            assert_eq!(video[0].image[0], 99);
        }

        // AssistantImageRaw should also have been enqueued in the audio task
        // and pushed downstream via write_transport_frame.
        let mut downstream = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            downstream.push(env);
        }
        let assistant_frames: Vec<_> = downstream
            .iter()
            .filter(|f| matches!(&f.frame, Frame::AssistantImageRaw(_)))
            .collect();
        assert_eq!(
            assistant_frames.len(),
            1,
            "Expected AssistantImageRaw to be pushed downstream via audio queue"
        );

        transport.cancel().await;
    }

    #[tokio::test]
    async fn multi_destination_routes_to_correct_sender() {
        let params = TransportParams {
            audio_out_enabled: true,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 1,
            audio_out_destinations: vec!["dest1".to_string()],
            ..Default::default()
        };
        let (cb, _written_audio, _) = MockCallbacks::new(true);
        let mut transport = BaseOutputTransport::new(params, cb);

        let (down_tx, mut _down_rx) = mpsc::channel(128);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_out_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
        transport.set_transport_ready().await;

        // Should have 2 senders: None (default) and Some("dest1").
        assert_eq!(transport.media_senders.len(), 2);
        assert!(transport.media_senders.contains_key(&None));
        assert!(
            transport
                .media_senders
                .contains_key(&Some("dest1".to_string()))
        );

        transport.cancel().await;
    }
}
