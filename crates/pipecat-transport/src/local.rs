#[cfg(feature = "cpal")]
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use bytes::Bytes;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::node::ProcessorNodeHandle;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tokio::sync::mpsc;
use tracing;

use crate::input::BaseInputTransport;
use crate::output::{BaseOutputTransport, OutputTransportCallbacks};
use crate::params::TransportParams;

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Audio file format for reading/writing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AudioFormat {
    /// Raw PCM: signed 16-bit little-endian interleaved samples.
    #[default]
    RawPcm,
    /// WAV container (PCM16 only).
    Wav,
}

/// Source of audio data for the local input transport.
#[derive(Debug)]
pub enum AudioInputSource {
    /// Raw PCM bytes (useful for tests).
    Buffer(Bytes),
    /// Audio file on disk (interpreted according to `AudioFormat`).
    File(PathBuf),
    /// Programmatic injection via a channel (always raw PCM frames).
    Channel(mpsc::Receiver<AudioRawFrame>),
}

/// Configuration for cpal audio device output.
#[cfg(feature = "cpal")]
#[derive(Debug, Clone, Default)]
pub struct DeviceConfig {
    /// Device name to use. `None` selects the system default output device.
    pub device_name: Option<String>,
}

/// Destination for audio data from the local output transport.
pub enum AudioOutputSink {
    /// Collect raw PCM bytes for assertions.
    Buffer(Arc<StdMutex<Vec<u8>>>),
    /// Write to a file on disk (format determined by `AudioFormat`).
    File(PathBuf),
    /// Play audio through a system audio device via cpal.
    #[cfg(feature = "cpal")]
    Device(DeviceConfig),
    /// Discard all output audio.
    Discard,
}

impl std::fmt::Debug for AudioOutputSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buffer(_) => f.debug_tuple("Buffer").finish(),
            Self::File(path) => f.debug_tuple("File").field(path).finish(),
            #[cfg(feature = "cpal")]
            Self::Device(config) => f.debug_tuple("Device").field(config).finish(),
            Self::Discard => write!(f, "Discard"),
        }
    }
}

/// Controls the pacing of audio chunk delivery.
#[derive(Debug, Default)]
pub enum AudioPacing {
    /// Deliver chunks as fast as possible (for tests).
    #[default]
    AsFastAsPossible,
    /// Deliver chunks at real-time rate (~20ms per chunk).
    RealTime,
}

// ---------------------------------------------------------------------------
// LocalAudioInputTransport
// ---------------------------------------------------------------------------

/// Local audio input transport that reads from a buffer, file, or channel
/// and feeds audio into the pipeline via `BaseInputTransport`.
pub struct LocalAudioInputTransport {
    inner: BaseInputTransport,
    source: Option<AudioInputSource>,
    pacing: AudioPacing,
    format: AudioFormat,
    num_channels: u16,
    source_task: Option<tokio::task::JoinHandle<()>>,
    node_handle: Option<ProcessorNodeHandle>,
}

impl LocalAudioInputTransport {
    pub fn new(params: TransportParams, source: AudioInputSource) -> Self {
        let num_channels = params.audio_in_channels;
        Self {
            inner: BaseInputTransport::with_name("LocalAudioInput", params),
            source: Some(source),
            pacing: AudioPacing::default(),
            format: AudioFormat::default(),
            num_channels,
            source_task: None,
            node_handle: None,
        }
    }

    pub fn with_pacing(mut self, pacing: AudioPacing) -> Self {
        self.pacing = pacing;
        self
    }

    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = format;
        self
    }

    fn spawn_source_task(&mut self) {
        let Some(source) = self.source.take() else {
            tracing::warn!("LocalAudioInput: source already consumed");
            return;
        };

        let Some(tx) = self.inner.take_audio_in_sender() else {
            tracing::warn!("LocalAudioInput: audio task not ready, cannot spawn source");
            return;
        };

        let sample_rate = self.inner.sample_rate();
        let num_channels = self.num_channels;
        let realtime = matches!(self.pacing, AudioPacing::RealTime);
        let format = self.format;
        let node_handle = self.node_handle.clone();
        let audio_drained = self.inner.audio_drained_notify();

        self.source_task = Some(tokio::spawn(async move {
            let is_finite = !matches!(&source, AudioInputSource::Channel(_));

            match (source, format) {
                (AudioInputSource::Buffer(data), AudioFormat::RawPcm) => {
                    let chunk_bytes = compute_chunk_bytes(sample_rate, num_channels);
                    feed_chunks(&tx, &data, chunk_bytes, sample_rate, num_channels, realtime).await;
                }
                (AudioInputSource::File(path), AudioFormat::RawPcm) => match std::fs::read(&path) {
                    Ok(data) => {
                        let chunk_bytes = compute_chunk_bytes(sample_rate, num_channels);
                        feed_chunks(&tx, &data, chunk_bytes, sample_rate, num_channels, realtime)
                            .await;
                    }
                    Err(e) => {
                        tracing::error!("LocalAudioInput: failed to read file {:?}: {}", path, e);
                    }
                },
                (AudioInputSource::Buffer(data), AudioFormat::Wav) => {
                    match hound::WavReader::new(std::io::Cursor::new(data.as_ref())) {
                        Ok(reader) => {
                            if let Some((pcm, wav_sr, wav_ch)) = read_wav_to_pcm(reader) {
                                let chunk_bytes = compute_chunk_bytes(wav_sr, wav_ch);
                                feed_chunks(&tx, &pcm, chunk_bytes, wav_sr, wav_ch, realtime).await;
                            }
                        }
                        Err(e) => {
                            tracing::error!("LocalAudioInput: failed to parse WAV buffer: {}", e);
                        }
                    }
                }
                (AudioInputSource::File(path), AudioFormat::Wav) => {
                    match hound::WavReader::open(&path) {
                        Ok(reader) => {
                            if let Some((pcm, wav_sr, wav_ch)) = read_wav_to_pcm(reader) {
                                let chunk_bytes = compute_chunk_bytes(wav_sr, wav_ch);
                                feed_chunks(&tx, &pcm, chunk_bytes, wav_sr, wav_ch, realtime).await;
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                "LocalAudioInput: failed to open WAV file {:?}: {}",
                                path,
                                e
                            );
                        }
                    }
                }
                (AudioInputSource::Channel(mut rx), _) => {
                    // Channel is always raw PCM frames, format ignored.
                    let mut interval = realtime
                        .then(|| tokio::time::interval(std::time::Duration::from_millis(20)));
                    while let Some(frame) = rx.recv().await {
                        if let Some(ref mut iv) = interval {
                            iv.tick().await;
                        }
                        if tx.send(frame).await.is_err() {
                            break;
                        }
                    }
                }
            }

            // Finite sources: drop the audio sender to close the channel,
            // wait for the audio task to drain all queued frames, then send
            // EndFrame so the pipeline shuts down cleanly.
            if is_finite {
                drop(tx);
                audio_drained.notified().await;
                if let Some(handle) = node_handle {
                    handle
                        .send(
                            FrameEnvelope::new(Frame::End(EndFrame::default())),
                            Direction::Downstream,
                        )
                        .await
                        .ok();
                }
            }
        }));
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute 20ms chunk size in bytes for i16 PCM.
fn compute_chunk_bytes(sample_rate: u32, num_channels: u16) -> usize {
    (sample_rate as usize / 50) * (num_channels as usize) * 2
}

/// Feed raw PCM data as chunked AudioRawFrames through the sender.
async fn feed_chunks(
    tx: &mpsc::Sender<AudioRawFrame>,
    data: &[u8],
    chunk_bytes: usize,
    sample_rate: u32,
    num_channels: u16,
    realtime: bool,
) {
    // Use interval (not sleep) for real-time pacing to avoid cumulative
    // timer drift that would slowly drain the output buffer.
    let mut interval =
        realtime.then(|| tokio::time::interval(std::time::Duration::from_millis(20)));

    for chunk in data.chunks(chunk_bytes) {
        if let Some(ref mut iv) = interval {
            iv.tick().await;
        }
        let frame = AudioRawFrame {
            audio: Bytes::copy_from_slice(chunk),
            sample_rate,
            num_channels,
        };
        if tx.send(frame).await.is_err() {
            break;
        }
    }
}

/// Read a WAV file/buffer and return `(pcm_bytes, sample_rate, num_channels)`.
/// Returns `None` if the format is not PCM16.
fn read_wav_to_pcm(reader: hound::WavReader<impl std::io::Read>) -> Option<(Vec<u8>, u32, u16)> {
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        tracing::error!(
            "LocalAudioInput: WAV must be PCM16, got {:?} {}bit",
            spec.sample_format,
            spec.bits_per_sample
        );
        return None;
    }
    let sample_rate = spec.sample_rate;
    let num_channels = spec.channels;
    let mut pcm_bytes = Vec::new();
    for sample in reader.into_samples::<i16>() {
        match sample {
            Ok(s) => pcm_bytes.extend_from_slice(&s.to_le_bytes()),
            Err(e) => {
                tracing::error!("LocalAudioInput: WAV sample read error: {}", e);
                return None;
            }
        }
    }
    Some((pcm_bytes, sample_rate, num_channels))
}

/// Cancel the source task if it's running.
async fn cancel_source_task(source_task: &mut Option<tokio::task::JoinHandle<()>>) {
    if let Some(handle) = source_task.take() {
        handle.abort();
        let _ = handle.await;
    }
}

#[async_trait]
impl FrameProcessor for LocalAudioInputTransport {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn id(&self) -> u64 {
        self.inner.id()
    }

    fn set_node_handle(&mut self, handle: ProcessorNodeHandle) {
        self.node_handle = Some(handle);
    }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let is_start = matches!(&envelope.frame, Frame::Start(_));
        let is_end = matches!(&envelope.frame, Frame::End(_));
        let is_cancel = matches!(&envelope.frame, Frame::Cancel(_));

        // Delegate to inner transport for all frame handling.
        self.inner.process_frame(envelope, direction, ctx).await?;

        if is_start {
            // After inner has initialized, mark transport ready and spawn source.
            self.inner.set_transport_ready().await;
            self.spawn_source_task();
        } else if is_end || is_cancel {
            // Explicitly cancel the source task on shutdown.
            cancel_source_task(&mut self.source_task).await;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LocalOutputCallbacks
// ---------------------------------------------------------------------------

/// The resolved sink used by the callbacks at runtime.
enum ResolvedSink {
    Buffer(Arc<StdMutex<Vec<u8>>>),
    File(StdMutex<std::io::BufWriter<std::fs::File>>),
    WavFile(StdMutex<WavSinkState>),
    #[cfg(feature = "cpal")]
    Device(StdMutex<DeviceSinkState>),
    Discard,
}

/// State machine for lazy WAV writer initialization and finalization.
enum WavSinkState {
    /// Waiting for the first audio frame to determine sample_rate/channels.
    Pending(PathBuf),
    /// Writer is active.
    Active(hound::WavWriter<std::io::BufWriter<std::fs::File>>),
    /// Writer has been finalized.
    Finalized,
}

/// State machine for cpal device output (lazy-initialized from first audio frame).
#[cfg(feature = "cpal")]
enum DeviceSinkState {
    Pending(DeviceConfig),
    Active(DevicePlaybackState),
    Finalized,
}

#[cfg(feature = "cpal")]
impl DeviceSinkState {
    fn as_active_buffer(&self) -> Option<&Arc<StdMutex<VecDeque<f32>>>> {
        match self {
            DeviceSinkState::Active(playback) => Some(&playback.buffer),
            _ => None,
        }
    }
}

/// Wrapper to make `cpal::Stream` usable across async tasks.
///
/// `cpal::Stream` is `!Send` on some platforms (e.g. macOS CoreAudio) because
/// it holds raw pointers internally. However, the stream's audio callback runs
/// on its own dedicated thread, and the only operation we ever perform on the
/// stream from Rust is dropping it (which stops playback). Dropping is safe
/// from any thread, so this `Send + Sync` impl is sound.
#[cfg(feature = "cpal")]
struct SendStream(#[allow(dead_code)] cpal::Stream);

#[cfg(feature = "cpal")]
unsafe impl Send for SendStream {}
#[cfg(feature = "cpal")]
unsafe impl Sync for SendStream {}

#[cfg(feature = "cpal")]
struct DevicePlaybackState {
    /// f32 sample buffer shared with the cpal callback. Samples are already
    /// converted to the device's native rate and channel count.
    buffer: Arc<StdMutex<VecDeque<f32>>>,
    /// Source audio parameters (from the pipeline).
    source_rate: u32,
    source_channels: u16,
    /// Device audio parameters (from the system output device).
    device_rate: u32,
    device_channels: u16,
    stream: SendStream,
    /// Whether `stream.play()` has been called.
    playing: bool,
    /// Number of f32 samples to accumulate before starting playback.
    prefill_threshold: usize,
}

/// Open a cpal f32 output stream using the device's default config.
///
/// The caller is responsible for resampling/remapping source PCM into the
/// device format before pushing into the shared buffer. This avoids
/// requesting an unsupported config from the OS audio system.
#[cfg(feature = "cpal")]
#[allow(clippy::type_complexity)]
fn init_cpal_stream(
    config: &DeviceConfig,
) -> std::result::Result<(Arc<StdMutex<VecDeque<f32>>>, u32, u16, SendStream), String> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();

    let device = if let Some(name) = &config.device_name {
        host.output_devices()
            .map_err(|e| format!("failed to enumerate output devices: {e}"))?
            .find(|d| d.name().map(|n| n == *name).unwrap_or(false))
            .ok_or_else(|| format!("audio output device '{name}' not found"))?
    } else {
        host.default_output_device()
            .ok_or_else(|| "no default audio output device".to_string())?
    };

    let default_config = device
        .default_output_config()
        .map_err(|e| format!("failed to get default output config: {e}"))?;

    let device_rate = default_config.sample_rate().0;
    let device_channels = default_config.channels();

    let stream_config = cpal::StreamConfig {
        channels: device_channels,
        sample_rate: cpal::SampleRate(device_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let buffer: Arc<StdMutex<VecDeque<f32>>> = Arc::new(StdMutex::new(VecDeque::new()));
    let buf_ref = buffer.clone();

    let stream = device
        .build_output_stream(
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if let Ok(mut buf) = buf_ref.try_lock() {
                    for sample in data.iter_mut() {
                        if let Some(s) = buf.pop_front() {
                            *sample = s;
                        } else {
                            *sample = 0.0;
                        }
                    }
                } else {
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }
                }
            },
            move |err| {
                tracing::error!("cpal output stream error: {err}");
            },
            None,
        )
        .map_err(|e| format!("failed to build cpal output stream: {e}"))?;

    // Don't call play() here — the caller starts playback after pre-buffering.
    Ok((buffer, device_rate, device_channels, SendStream(stream)))
}

/// Linear interpolation resampler (per-channel). Good enough for playback.
#[cfg(feature = "cpal")]
fn resample_linear(samples: &[f32], channels: u16, from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ch = channels as usize;
    let num_frames = samples.len() / ch;
    if num_frames == 0 {
        return Vec::new();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_frames = ((num_frames as f64 / ratio).ceil()) as usize;
    let mut out = Vec::with_capacity(out_frames * ch);
    for i in 0..out_frames {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;
        for c in 0..ch {
            let a = samples.get(idx * ch + c).copied().unwrap_or(0.0);
            let b = samples.get((idx + 1) * ch + c).copied().unwrap_or(a);
            out.push(a + (b - a) * frac);
        }
    }
    out
}

/// Map mono→stereo (duplicate) or stereo→mono (average), etc.
#[cfg(feature = "cpal")]
fn remap_channels(samples: &[f32], from_ch: u16, to_ch: u16) -> Vec<f32> {
    let from = from_ch as usize;
    let to = to_ch as usize;
    let num_frames = samples.len() / from;
    let mut out = Vec::with_capacity(num_frames * to);
    for frame_idx in 0..num_frames {
        let src = &samples[frame_idx * from..frame_idx * from + from];
        if to > from {
            // Upmix: copy existing channels, duplicate last for extras.
            for c in 0..to {
                out.push(src[c.min(from - 1)]);
            }
        } else {
            // Downmix: average source channels into each output channel.
            let sum: f32 = src.iter().sum();
            let avg = sum / from as f32;
            for _ in 0..to {
                out.push(avg);
            }
        }
    }
    out
}

/// Callbacks for the local output transport that write audio to a sink.
struct LocalOutputCallbacks {
    sink: ResolvedSink,
}

impl LocalOutputCallbacks {
    fn new(sink: AudioOutputSink, format: AudioFormat) -> std::io::Result<Self> {
        let resolved = match (sink, format) {
            (AudioOutputSink::Buffer(buf), _) => ResolvedSink::Buffer(buf),
            (AudioOutputSink::File(path), AudioFormat::RawPcm) => {
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)?;
                ResolvedSink::File(StdMutex::new(std::io::BufWriter::new(file)))
            }
            (AudioOutputSink::File(path), AudioFormat::Wav) => {
                ResolvedSink::WavFile(StdMutex::new(WavSinkState::Pending(path)))
            }
            #[cfg(feature = "cpal")]
            (AudioOutputSink::Device(config), _) => {
                ResolvedSink::Device(StdMutex::new(DeviceSinkState::Pending(config)))
            }
            (AudioOutputSink::Discard, _) => ResolvedSink::Discard,
        };
        Ok(Self { sink: resolved })
    }

    /// Finalize any stateful sinks. Called on End/Cancel.
    fn finalize(&self) {
        match &self.sink {
            ResolvedSink::WavFile(state) => {
                let mut state = state.lock().unwrap();
                let old = std::mem::replace(&mut *state, WavSinkState::Finalized);
                if let WavSinkState::Active(writer) = old
                    && let Err(e) = writer.finalize()
                {
                    tracing::error!("LocalAudioOutput: WAV finalize error: {}", e);
                }
            }
            #[cfg(feature = "cpal")]
            ResolvedSink::Device(state) => {
                // Let the cpal callback drain remaining buffered audio before
                // dropping the stream. We poll until empty or a timeout elapses.
                let drain_deadline =
                    std::time::Instant::now() + std::time::Duration::from_millis(500);
                loop {
                    let guard = state.lock().unwrap();
                    let empty = guard
                        .as_active_buffer()
                        .map(|b| b.lock().unwrap().is_empty())
                        .unwrap_or(true);
                    drop(guard);
                    if empty || std::time::Instant::now() >= drain_deadline {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                let mut state = state.lock().unwrap();
                *state = DeviceSinkState::Finalized;
            }
            _ => {}
        }
    }
}

#[async_trait]
impl OutputTransportCallbacks for LocalOutputCallbacks {
    async fn write_audio_frame(&self, frame: &AudioRawFrame) -> bool {
        use std::io::Write;
        match &self.sink {
            ResolvedSink::Buffer(buf) => {
                buf.lock().unwrap().extend_from_slice(&frame.audio);
                true
            }
            ResolvedSink::File(writer) => {
                let mut w = writer.lock().unwrap();
                if let Err(e) = w.write_all(&frame.audio) {
                    tracing::error!("LocalAudioOutput: write error: {}", e);
                    return false;
                }
                true
            }
            ResolvedSink::WavFile(state) => {
                let mut state = state.lock().unwrap();
                // Lazy-init writer from first frame's params.
                if matches!(&*state, WavSinkState::Pending(_)) {
                    let old = std::mem::replace(&mut *state, WavSinkState::Finalized);
                    if let WavSinkState::Pending(path) = old {
                        let spec = hound::WavSpec {
                            channels: frame.num_channels,
                            sample_rate: frame.sample_rate,
                            bits_per_sample: 16,
                            sample_format: hound::SampleFormat::Int,
                        };
                        match hound::WavWriter::create(&path, spec) {
                            Ok(writer) => *state = WavSinkState::Active(writer),
                            Err(e) => {
                                tracing::error!(
                                    "LocalAudioOutput: failed to create WAV file: {}",
                                    e
                                );
                                return false;
                            }
                        }
                    }
                }
                if let WavSinkState::Active(writer) = &mut *state {
                    for sample_bytes in frame.audio.chunks_exact(2) {
                        let sample = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
                        if let Err(e) = writer.write_sample(sample) {
                            tracing::error!("LocalAudioOutput: WAV write error: {}", e);
                            return false;
                        }
                    }
                }
                true
            }
            #[cfg(feature = "cpal")]
            ResolvedSink::Device(state) => {
                let mut state = state.lock().unwrap();
                // Lazy-init: open the cpal stream on first audio frame.
                if matches!(&*state, DeviceSinkState::Pending(_)) {
                    let old = std::mem::replace(&mut *state, DeviceSinkState::Finalized);
                    if let DeviceSinkState::Pending(config) = old {
                        match init_cpal_stream(&config) {
                            Ok((buffer, device_rate, device_channels, stream)) => {
                                // Pre-buffer 100ms before starting playback to
                                // absorb pipeline scheduling jitter.
                                let prefill =
                                    (device_rate as usize) * (device_channels as usize) / 10;
                                tracing::info!(
                                    "cpal: stream ready (device {}Hz {}ch, source {}Hz {}ch, prefill {} samples)",
                                    device_rate,
                                    device_channels,
                                    frame.sample_rate,
                                    frame.num_channels,
                                    prefill,
                                );
                                *state = DeviceSinkState::Active(DevicePlaybackState {
                                    buffer,
                                    source_rate: frame.sample_rate,
                                    source_channels: frame.num_channels,
                                    device_rate,
                                    device_channels,
                                    stream,
                                    playing: false,
                                    prefill_threshold: prefill,
                                });
                            }
                            Err(e) => {
                                tracing::error!("cpal: init failed: {e}");
                                return false;
                            }
                        }
                    }
                }
                if let DeviceSinkState::Active(playback) = &mut *state {
                    // Convert i16 LE PCM bytes → f32 samples, resample if needed,
                    // and map to the device channel count.
                    let src_samples: Vec<f32> = frame
                        .audio
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                        .collect();

                    let resampled = if playback.source_rate != playback.device_rate {
                        resample_linear(
                            &src_samples,
                            playback.source_channels,
                            playback.source_rate,
                            playback.device_rate,
                        )
                    } else {
                        src_samples
                    };

                    let mapped = if playback.source_channels != playback.device_channels {
                        remap_channels(
                            &resampled,
                            playback.source_channels,
                            playback.device_channels,
                        )
                    } else {
                        resampled
                    };

                    playback.buffer.lock().unwrap().extend(mapped.iter());

                    // Start playback once we've accumulated enough audio to
                    // absorb scheduling jitter.
                    if !playback.playing {
                        let buf_len = playback.buffer.lock().unwrap().len();
                        if buf_len >= playback.prefill_threshold {
                            use cpal::traits::StreamTrait;
                            if let Err(e) = playback.stream.0.play() {
                                tracing::error!("cpal: failed to start stream: {e}");
                                return false;
                            }
                            playback.playing = true;
                        }
                    }
                }
                true
            }
            ResolvedSink::Discard => true,
        }
    }
}

// ---------------------------------------------------------------------------
// LocalAudioOutputTransport
// ---------------------------------------------------------------------------

/// Local audio output transport that writes audio to a buffer, file, or discards it.
pub struct LocalAudioOutputTransport {
    inner: BaseOutputTransport,
    output_buffer: Option<Arc<StdMutex<Vec<u8>>>>,
    callbacks: Arc<LocalOutputCallbacks>,
}

impl LocalAudioOutputTransport {
    /// Create a new local audio output transport with raw PCM format.
    ///
    /// # Panics
    ///
    /// Panics if `sink` is `AudioOutputSink::File` and the file cannot be opened.
    pub fn new(params: TransportParams, sink: AudioOutputSink) -> Self {
        Self::with_format(params, sink, AudioFormat::RawPcm)
    }

    /// Create a new local audio output transport with the specified format.
    ///
    /// # Panics
    ///
    /// Panics if `sink` is `AudioOutputSink::File` and the file cannot be opened.
    pub fn with_format(
        params: TransportParams,
        sink: AudioOutputSink,
        format: AudioFormat,
    ) -> Self {
        let output_buffer = match &sink {
            AudioOutputSink::Buffer(buf) => Some(buf.clone()),
            _ => None,
        };
        let callbacks = Arc::new(
            LocalOutputCallbacks::new(sink, format)
                .expect("LocalAudioOutput: failed to open output sink"),
        );
        Self {
            inner: BaseOutputTransport::with_name("LocalAudioOutput", params, callbacks.clone()),
            output_buffer,
            callbacks,
        }
    }

    /// Get the output buffer if this transport was configured with `AudioOutputSink::Buffer`.
    pub fn output_buffer(&self) -> Option<Arc<StdMutex<Vec<u8>>>> {
        self.output_buffer.clone()
    }
}

#[async_trait]
impl FrameProcessor for LocalAudioOutputTransport {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn id(&self) -> u64 {
        self.inner.id()
    }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let is_start = matches!(&envelope.frame, Frame::Start(_));
        let is_end = matches!(&envelope.frame, Frame::End(_));
        let is_cancel = matches!(&envelope.frame, Frame::Cancel(_));

        // Delegate to inner transport for all frame handling.
        self.inner.process_frame(envelope, direction, ctx).await?;

        if is_start {
            self.inner.set_transport_ready().await;
        } else if is_end || is_cancel {
            self.callbacks.finalize();
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LocalAudioTransport — convenience builder
// ---------------------------------------------------------------------------

/// Convenience builder that wraps both input and output local transports.
///
/// Since `TransportParams` contains non-Clone trait objects, the builder
/// takes separate params for input and output transports.
pub struct LocalAudioTransport;

impl LocalAudioTransport {
    /// Build an input transport with the given params and source.
    pub fn input(
        mut params: TransportParams,
        source: AudioInputSource,
    ) -> LocalAudioInputTransport {
        params.audio_in_enabled = true;
        params.audio_in_passthrough = true;
        LocalAudioInputTransport::new(params, source)
    }

    /// Build an output transport with the given params and sink.
    pub fn output(mut params: TransportParams, sink: AudioOutputSink) -> LocalAudioOutputTransport {
        params.audio_out_enabled = true;
        LocalAudioOutputTransport::new(params, sink)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use pipecat_core::test_utils::*;

    use super::*;

    fn make_pcm_bytes(num_samples: usize) -> Bytes {
        let samples: Vec<i16> = (0..num_samples).map(|i| (i % 200) as i16).collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        Bytes::from(bytes)
    }

    /// Create a WAV buffer in memory with the given samples.
    fn make_wav_bytes(samples: &[i16], sample_rate: u32, num_channels: u16) -> Bytes {
        let spec = hound::WavSpec {
            channels: num_channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for &s in samples {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        Bytes::from(cursor.into_inner())
    }

    #[tokio::test]
    async fn input_transport_feeds_audio_from_buffer() {
        let pcm_data = make_pcm_bytes(640); // 40ms at 16kHz = 2 chunks
        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_passthrough: true,
            ..Default::default()
        };

        let mut transport =
            LocalAudioInputTransport::new(params, AudioInputSource::Buffer(pcm_data.clone()));

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

        // Wait for source task to deliver frames.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }

        // Should have Start + at least one InputAudioRaw
        assert!(
            frames.len() >= 2,
            "got {} frames: {:?}",
            frames.len(),
            frame_names(&frames)
        );
        assert!(matches!(&frames[0].frame, Frame::Start(_)));

        let audio_count = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::InputAudioRaw(_)))
            .count();
        assert!(audio_count >= 1, "expected audio frames, got {audio_count}");
    }

    #[tokio::test]
    async fn output_transport_collects_audio() {
        let output_buf = Arc::new(StdMutex::new(Vec::new()));
        let params = TransportParams {
            audio_out_enabled: true,
            ..Default::default()
        };

        let mut transport =
            LocalAudioOutputTransport::new(params, AudioOutputSink::Buffer(output_buf.clone()));

        let (down_tx, _down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        // Start the transport.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame::default())),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Default chunk size at 24kHz, 1 channel, 4x10ms = 1920 bytes.
        // Send enough audio data to fill at least one chunk.
        let audio_data = make_pcm_bytes(1920); // 3840 bytes, well over one chunk
        transport
            .process_frame(
                FrameEnvelope::new(Frame::OutputAudioRaw(AudioRawFrame {
                    audio: audio_data.clone(),
                    sample_rate: 24000,
                    num_channels: 1,
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Give the audio task time to process.
        tokio::time::sleep(Duration::from_millis(200)).await;

        let collected = output_buf.lock().unwrap();
        assert!(
            !collected.is_empty(),
            "output buffer should have received audio data"
        );
    }

    #[tokio::test]
    async fn input_transport_channel_source() {
        let (audio_tx, audio_rx) = mpsc::channel(16);
        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_passthrough: true,
            ..Default::default()
        };

        let mut transport =
            LocalAudioInputTransport::new(params, AudioInputSource::Channel(audio_rx));

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

        // Send a frame through the channel.
        audio_tx
            .send(AudioRawFrame {
                audio: make_pcm_bytes(160),
                sample_rate: 16000,
                num_channels: 1,
            })
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }

        let audio_count = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::InputAudioRaw(_)))
            .count();
        assert!(audio_count >= 1, "expected audio from channel source");
    }

    #[tokio::test]
    async fn output_discard_sink() {
        let params = TransportParams {
            audio_out_enabled: true,
            ..Default::default()
        };

        let mut transport = LocalAudioOutputTransport::new(params, AudioOutputSink::Discard);

        let (down_tx, _down_rx) = mpsc::channel(64);
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

        // Should not panic.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::OutputAudioRaw(AudioRawFrame {
                    audio: make_pcm_bytes(320),
                    sample_rate: 24000,
                    num_channels: 1,
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn wav_input_from_buffer() {
        let samples: Vec<i16> = (0..640).map(|i| (i % 200) as i16).collect();
        let wav_data = make_wav_bytes(&samples, 16000, 1);

        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_passthrough: true,
            ..Default::default()
        };

        let mut transport =
            LocalAudioInputTransport::new(params, AudioInputSource::Buffer(wav_data))
                .with_format(AudioFormat::Wav);

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

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }

        let audio_count = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::InputAudioRaw(_)))
            .count();
        assert!(
            audio_count >= 1,
            "WAV input should produce audio frames, got {audio_count}"
        );

        // Verify the total PCM data matches the original samples.
        let total_pcm: Vec<u8> = frames
            .iter()
            .filter_map(|f| match &f.frame {
                Frame::InputAudioRaw(a) => Some(a.audio.to_vec()),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(
            total_pcm.len(),
            samples.len() * 2,
            "total PCM bytes should match input sample count * 2"
        );
    }

    #[tokio::test]
    async fn wav_output_to_file() {
        let dir = std::env::temp_dir().join("pipecat_test_wav_output");
        std::fs::create_dir_all(&dir).unwrap();
        let wav_path = dir.join("test_output.wav");
        // Clean up from previous runs.
        let _ = std::fs::remove_file(&wav_path);

        let params = TransportParams {
            audio_out_enabled: true,
            ..Default::default()
        };

        let mut transport = LocalAudioOutputTransport::with_format(
            params,
            AudioOutputSink::File(wav_path.clone()),
            AudioFormat::Wav,
        );

        let (down_tx, _down_rx) = mpsc::channel(64);
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

        // Send audio with enough data to fill output chunks.
        let samples: Vec<i16> = (0..2400).map(|i| (i % 100) as i16).collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        transport
            .process_frame(
                FrameEnvelope::new(Frame::OutputAudioRaw(AudioRawFrame {
                    audio: Bytes::from(pcm_bytes),
                    sample_rate: 24000,
                    num_channels: 1,
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Give the audio task time to process and write.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Finalize via End frame.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::End(EndFrame::default())),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Read the WAV file back and verify it's valid.
        let reader = hound::WavReader::open(&wav_path).expect("should be a valid WAV file");
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 24000);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);

        let read_samples: Vec<i16> = reader.into_samples::<i16>().map(|s| s.unwrap()).collect();
        assert!(!read_samples.is_empty(), "WAV file should contain samples");

        // Clean up.
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_dir(&dir);
    }
}
