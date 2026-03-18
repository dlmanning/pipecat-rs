use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::input::BaseInputTransport;
use crate::params::TransportParams;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for microphone input capture.
#[derive(Debug, Clone, Default)]
pub struct MicInputConfig {
    /// Device name to use. `None` selects the system default input device.
    pub device_name: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Wrapper to make `cpal::Stream` usable across async tasks.
///
/// `cpal::Stream` is `!Send` on some platforms (e.g. macOS CoreAudio) because
/// it holds raw pointers internally. However, the stream's audio callback runs
/// on its own dedicated thread, and the only operation we ever perform on the
/// stream from Rust is dropping it (which stops capture). This makes the
/// `Send + Sync` impl sound.
struct SendStream(cpal::Stream);

unsafe impl Send for SendStream {}
unsafe impl Sync for SendStream {}

/// Compute the maximum buffer size (~1 second of audio at the given device params).
fn max_buffer_bytes(sample_rate: u32, channels: u16) -> usize {
    sample_rate as usize * channels as usize * 2
}

// ---------------------------------------------------------------------------
// Device enumeration
// ---------------------------------------------------------------------------

/// Information about an available audio input device.
#[derive(Debug, Clone)]
pub struct InputDeviceInfo {
    /// Device name (pass to `MicInputConfig::device_name` to select).
    pub name: String,
    /// Default sample rate.
    pub sample_rate: u32,
    /// Default channel count.
    pub channels: u16,
    /// Whether this is the system default input device.
    pub is_default: bool,
}

/// List all available audio input devices.
pub fn list_input_devices() -> Vec<InputDeviceInfo> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());

    let devices = match host.input_devices() {
        Ok(devices) => devices,
        Err(_) => return Vec::new(),
    };

    devices
        .filter_map(|d| {
            let name = d.name().ok()?;
            let config = d.default_input_config().ok()?;
            Some(InputDeviceInfo {
                is_default: default_name.as_deref() == Some(&name),
                name,
                sample_rate: config.sample_rate().0,
                channels: config.channels(),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// cpal stream setup
// ---------------------------------------------------------------------------

#[allow(clippy::type_complexity)]
fn open_input_stream(
    config: &MicInputConfig,
) -> std::result::Result<(Arc<StdMutex<VecDeque<u8>>>, u32, u16, SendStream), String> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();

    let device = if let Some(name) = &config.device_name {
        host.input_devices()
            .map_err(|e| format!("failed to enumerate input devices: {e}"))?
            .find(|d| d.name().map(|n| n == *name).unwrap_or(false))
            .ok_or_else(|| format!("audio input device '{name}' not found"))?
    } else {
        host.default_input_device()
            .ok_or_else(|| "no default audio input device".to_string())?
    };

    let device_name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
    tracing::info!("MicInput: using device '{device_name}'");

    let default_config = device
        .default_input_config()
        .map_err(|e| format!("failed to get default input config for '{device_name}': {e}"))?;

    let device_rate = default_config.sample_rate().0;
    let device_channels = default_config.channels();

    let stream_config = cpal::StreamConfig {
        channels: device_channels,
        sample_rate: cpal::SampleRate(device_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let max_bytes = max_buffer_bytes(device_rate, device_channels);
    let buffer: Arc<StdMutex<VecDeque<u8>>> = Arc::new(StdMutex::new(VecDeque::new()));
    let buf_ref = buffer.clone();

    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert f32 samples to i16 LE PCM bytes.
                let mut pcm = Vec::with_capacity(data.len() * 2);
                for &sample in data {
                    let clamped = sample.clamp(-1.0, 1.0);
                    let i16_val = (clamped * 32767.0) as i16;
                    pcm.extend_from_slice(&i16_val.to_le_bytes());
                }

                if let Ok(mut buf) = buf_ref.try_lock() {
                    buf.extend(pcm.iter());
                    // Drop oldest data if buffer exceeds max size.
                    if buf.len() > max_bytes {
                        let excess = buf.len() - max_bytes;
                        tracing::warn!(
                            "MicInput: buffer overflow, dropping {} bytes of audio",
                            excess
                        );
                        buf.drain(..excess);
                    }
                }
            },
            move |err| {
                tracing::error!("cpal input stream error: {err}");
            },
            None,
        )
        .map_err(|e| format!("failed to build cpal input stream: {e}"))?;

    Ok((buffer, device_rate, device_channels, SendStream(stream)))
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Linear interpolation resampler for i16 PCM (interleaved channels).
fn resample_i16(samples: &[u8], channels: u16, from_rate: u32, to_rate: u32) -> Vec<u8> {
    let ch = channels as usize;
    let bytes_per_frame = ch * 2;
    let num_frames = samples.len() / bytes_per_frame;
    if num_frames == 0 {
        return Vec::new();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let out_frames = (num_frames as f64 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_frames * bytes_per_frame);

    for i in 0..out_frames {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        for c in 0..ch {
            let offset_a = (idx * ch + c) * 2;
            let a = if offset_a + 1 < samples.len() {
                i16::from_le_bytes([samples[offset_a], samples[offset_a + 1]])
            } else {
                0
            };

            let offset_b = ((idx + 1) * ch + c) * 2;
            let b = if offset_b + 1 < samples.len() {
                i16::from_le_bytes([samples[offset_b], samples[offset_b + 1]])
            } else {
                a
            };

            let interpolated = a as f32 + (b as f32 - a as f32) * frac;
            let result = interpolated.round() as i16;
            out.extend_from_slice(&result.to_le_bytes());
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Chunking task
// ---------------------------------------------------------------------------

/// Background task that drains the shared buffer, chunks into 20ms frames,
/// resamples if needed, and sends to BaseInputTransport.
async fn mic_chunking_task(
    buffer: Arc<StdMutex<VecDeque<u8>>>,
    tx: mpsc::Sender<AudioRawFrame>,
    device_rate: u32,
    device_channels: u16,
    target_rate: u32,
) {
    // 20ms chunk size at device rate.
    let device_chunk_bytes = (device_rate as usize / 50) * (device_channels as usize) * 2;
    let needs_resample = device_rate != target_rate;

    let mut interval = tokio::time::interval(Duration::from_millis(10));
    let mut accumulator: Vec<u8> = Vec::new();

    loop {
        interval.tick().await;

        // Drain all available data from the shared buffer.
        {
            if let Ok(mut buf) = buffer.try_lock()
                && !buf.is_empty()
            {
                accumulator.extend(buf.iter());
                buf.clear();
            }
        }

        // Send complete 20ms chunks.
        while accumulator.len() >= device_chunk_bytes {
            let remainder = accumulator.split_off(device_chunk_bytes);
            let chunk = std::mem::replace(&mut accumulator, remainder);

            let (audio_bytes, sample_rate) = if needs_resample {
                let resampled = resample_i16(&chunk, device_channels, device_rate, target_rate);
                (resampled, target_rate)
            } else {
                (chunk, device_rate)
            };

            let frame = AudioRawFrame {
                audio: Bytes::from(audio_bytes),
                sample_rate,
                num_channels: device_channels,
            };

            if tx.send(frame).await.is_err() {
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MicInput — the FrameProcessor
// ---------------------------------------------------------------------------

/// Captures audio from the system microphone and feeds it into the pipeline.
///
/// Uses cpal to open an input stream on the default (or named) device.
/// Audio is captured in a background thread, chunked into 20ms frames,
/// resampled to the pipeline's target sample rate if needed, and pushed
/// through `BaseInputTransport` for filtering and downstream delivery.
///
/// # Example
///
/// ```ignore
/// let params = TransportParams {
///     audio_in_enabled: true,
///     audio_in_passthrough: true,
///     ..Default::default()
/// };
/// let mic = MicInput::new(params, MicInputConfig::default());
/// // Add to pipeline: Pipeline::new(vec![mic, ...])
/// ```
pub struct MicInput {
    inner: BaseInputTransport,
    config: MicInputConfig,
    stream: Option<SendStream>,
    chunking_task: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for MicInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MicInput")
            .field("config", &self.config)
            .field("stream_active", &self.stream.is_some())
            .finish()
    }
}

impl MicInput {
    pub fn new(mut params: TransportParams, config: MicInputConfig) -> Self {
        params.audio_in_enabled = true;
        params.audio_in_passthrough = true;
        Self {
            inner: BaseInputTransport::with_name("MicInput", params),
            config,
            stream: None,
            chunking_task: None,
        }
    }

    async fn start_capture(&mut self) {
        // Stop any existing capture first (defensive against double StartFrame).
        if self.stream.is_some() {
            self.stop_capture().await;
        }

        let Some(tx) = self.inner.audio_in_sender() else {
            tracing::warn!("MicInput: audio task not ready, cannot start capture");
            return;
        };

        let target_rate = self.inner.sample_rate();
        if target_rate == 0 {
            tracing::error!("MicInput: sample rate not set (StartFrame not received?)");
            return;
        }

        match open_input_stream(&self.config) {
            Ok((buffer, device_rate, device_channels, stream)) => {
                tracing::info!(
                    "MicInput: stream ready (device {}Hz {}ch, target {}Hz)",
                    device_rate,
                    device_channels,
                    target_rate,
                );

                // Start the cpal stream.
                use cpal::traits::StreamTrait;
                if let Err(e) = stream.0.play() {
                    tracing::error!("MicInput: failed to start stream: {e}");
                    return;
                }

                // Spawn the chunking task.
                self.chunking_task = Some(tokio::spawn(mic_chunking_task(
                    buffer,
                    tx,
                    device_rate,
                    device_channels,
                    target_rate,
                )));

                self.stream = Some(stream);
            }
            Err(e) => {
                tracing::error!("MicInput: {e}");
            }
        }
    }

    async fn stop_capture(&mut self) {
        // Dropping the stream stops the cpal capture.
        self.stream = None;
        if let Some(handle) = self.chunking_task.take() {
            handle.abort();
            let _ = handle.await;
        }
    }
}

#[async_trait]
impl FrameProcessor for MicInput {
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
            self.start_capture().await;
        } else if is_end || is_cancel {
            self.stop_capture().await;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_i16_same_rate_passthrough() {
        let samples: Vec<u8> = [100i16, 200, 300, 400]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let result = resample_i16(&samples, 1, 16000, 16000);
        assert_eq!(result, samples);
    }

    #[test]
    fn resample_i16_downsample() {
        // 4 mono samples at 48000 → should produce ~1-2 samples at 16000 (ratio 3:1)
        let samples: Vec<u8> = [1000i16, 2000, 3000, 4000]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let result = resample_i16(&samples, 1, 48000, 16000);
        // At 3:1 ratio, 4 input frames → ceil(4/3) = 2 output frames
        assert_eq!(result.len(), 4); // 2 samples * 2 bytes
    }

    #[test]
    fn resample_i16_upsample() {
        // 2 mono samples at 8000 → should produce ~4 samples at 16000
        let samples: Vec<u8> = [1000i16, 3000]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let result = resample_i16(&samples, 1, 8000, 16000);
        // At 0.5:1 ratio, 2 input frames → ceil(2/0.5) = 4 output frames
        assert_eq!(result.len(), 8); // 4 samples * 2 bytes

        // First sample should be 1000, last should approach 3000
        let first = i16::from_le_bytes([result[0], result[1]]);
        assert_eq!(first, 1000);
    }

    #[test]
    fn resample_i16_stereo() {
        // 2 stereo frames at 16000 → 4 stereo frames at 32000
        let samples: Vec<u8> = [100i16, 200, 300, 400]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let result = resample_i16(&samples, 2, 16000, 32000);
        // 2 input frames → ceil(2/0.5) = 4 output frames * 2 channels * 2 bytes = 16
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn resample_i16_empty() {
        let result = resample_i16(&[], 1, 48000, 16000);
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn chunking_task_produces_frames() {
        let buffer: Arc<StdMutex<VecDeque<u8>>> = Arc::new(StdMutex::new(VecDeque::new()));
        let (tx, mut rx) = mpsc::channel::<AudioRawFrame>(64);

        // Pre-fill buffer with 40ms of mono 16kHz audio (2 chunks worth).
        // 20ms at 16kHz mono = 320 samples * 2 bytes = 640 bytes
        let chunk_bytes = 640;
        let total_bytes = chunk_bytes * 2;
        {
            let mut buf = buffer.lock().unwrap();
            for i in 0..total_bytes {
                buf.push_back(i as u8);
            }
        }

        // Spawn the chunking task.
        let task = tokio::spawn(mic_chunking_task(buffer, tx, 16000, 1, 16000));

        // Wait for chunks to be produced.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut frames = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            frames.push(frame);
        }

        task.abort();

        assert_eq!(frames.len(), 2, "expected 2 x 20ms chunks");
        assert_eq!(frames[0].audio.len(), chunk_bytes);
        assert_eq!(frames[0].sample_rate, 16000);
        assert_eq!(frames[0].num_channels, 1);
    }

    #[tokio::test]
    async fn chunking_task_resamples() {
        let buffer: Arc<StdMutex<VecDeque<u8>>> = Arc::new(StdMutex::new(VecDeque::new()));
        let (tx, mut rx) = mpsc::channel::<AudioRawFrame>(64);

        // Pre-fill with 20ms of mono 48kHz audio.
        // 20ms at 48kHz mono = 960 samples * 2 bytes = 1920 bytes
        let device_chunk = 1920;
        {
            let mut buf = buffer.lock().unwrap();
            for _ in 0..device_chunk {
                buf.push_back(0);
            }
        }

        let task = tokio::spawn(mic_chunking_task(buffer, tx, 48000, 1, 16000));

        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut frames = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            frames.push(frame);
        }

        task.abort();

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].sample_rate, 16000);
        // 20ms at 16kHz mono = 320 samples * 2 bytes = 640 bytes
        assert_eq!(frames[0].audio.len(), 640);
    }

    #[test]
    fn config_default() {
        let config = MicInputConfig::default();
        assert!(config.device_name.is_none());
    }
}
