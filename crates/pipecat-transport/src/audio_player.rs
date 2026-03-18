use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for audio device playback.
#[derive(Debug, Clone, Default)]
pub struct AudioPlayerConfig {
    /// Device name to use. `None` selects the system default output device.
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
/// stream from Rust is dropping it (which stops playback) or calling `play()`.
/// Both are safe from any thread, so this `Send + Sync` impl is sound.
struct SendStream(cpal::Stream);

unsafe impl Send for SendStream {}
unsafe impl Sync for SendStream {}

struct ActivePlayback {
    /// f32 sample buffer shared with the cpal callback.
    buffer: Arc<StdMutex<VecDeque<f32>>>,
    /// Source audio parameters (from the first frame).
    source_rate: u32,
    source_channels: u16,
    /// Device audio parameters.
    device_rate: u32,
    device_channels: u16,
    stream: SendStream,
    playing: bool,
    /// Number of f32 samples to accumulate before starting playback.
    prefill_threshold: usize,
}

enum PlaybackState {
    Pending(AudioPlayerConfig),
    Active(ActivePlayback),
    Finalized,
}

// ---------------------------------------------------------------------------
// cpal stream setup
// ---------------------------------------------------------------------------

#[allow(clippy::type_complexity)]
fn open_stream(
    config: &AudioPlayerConfig,
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

    Ok((buffer, device_rate, device_channels, SendStream(stream)))
}

// ---------------------------------------------------------------------------
// Resampling / channel mapping
// ---------------------------------------------------------------------------

/// Linear interpolation resampler (per-channel).
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
fn remap_channels(samples: &[f32], from_ch: u16, to_ch: u16) -> Vec<f32> {
    let from = from_ch as usize;
    let to = to_ch as usize;
    let num_frames = samples.len() / from;
    let mut out = Vec::with_capacity(num_frames * to);
    for frame_idx in 0..num_frames {
        let src = &samples[frame_idx * from..frame_idx * from + from];
        if to > from {
            for c in 0..to {
                out.push(src[c.min(from - 1)]);
            }
        } else {
            let sum: f32 = src.iter().sum();
            let avg = sum / from as f32;
            for _ in 0..to {
                out.push(avg);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// AudioPlayer — the FrameProcessor
// ---------------------------------------------------------------------------

/// Plays `InputAudioRaw` frames through a system audio device via cpal.
///
/// All frames pass through unchanged so downstream processors still see
/// the audio. The player is a tap, not a sink.
///
/// Lazy-initializes the cpal stream on the first `InputAudioRaw` frame
/// and defers `play()` until enough audio has been pre-buffered to absorb
/// scheduling jitter.
#[derive(Debug)]
pub struct AudioPlayer {
    base: ProcessorBase,
    state: PlaybackState,
}

impl std::fmt::Debug for PlaybackState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending(_) => write!(f, "Pending"),
            Self::Active(_) => write!(f, "Active"),
            Self::Finalized => write!(f, "Finalized"),
        }
    }
}

impl AudioPlayer {
    pub fn new(config: AudioPlayerConfig) -> Self {
        Self {
            base: ProcessorBase::new("AudioPlayer"),
            state: PlaybackState::Pending(config),
        }
    }

    fn push_audio(&mut self, audio: &AudioRawFrame) {
        // Lazy-init the cpal stream.
        if let PlaybackState::Pending(_) = &self.state {
            let old = std::mem::replace(&mut self.state, PlaybackState::Finalized);
            if let PlaybackState::Pending(config) = old {
                match open_stream(&config) {
                    Ok((buffer, device_rate, device_channels, stream)) => {
                        let prefill = (device_rate as usize) * (device_channels as usize) / 10;
                        tracing::info!(
                            "AudioPlayer: stream ready (device {}Hz {}ch, source {}Hz {}ch)",
                            device_rate,
                            device_channels,
                            audio.sample_rate,
                            audio.num_channels,
                        );
                        self.state = PlaybackState::Active(ActivePlayback {
                            buffer,
                            source_rate: audio.sample_rate,
                            source_channels: audio.num_channels,
                            device_rate,
                            device_channels,
                            stream,
                            playing: false,
                            prefill_threshold: prefill,
                        });
                    }
                    Err(e) => {
                        tracing::error!("AudioPlayer: {e}");
                        return;
                    }
                }
            }
        }

        let PlaybackState::Active(playback) = &mut self.state else {
            return;
        };

        // Convert i16 LE PCM → f32.
        let src_samples: Vec<f32> = audio
            .audio
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();

        // Resample if needed.
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

        // Remap channels if needed.
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

        // Start playback once pre-buffer threshold is reached.
        if !playback.playing {
            let buf_len = playback.buffer.lock().unwrap().len();
            if buf_len >= playback.prefill_threshold {
                use cpal::traits::StreamTrait;
                if let Err(e) = playback.stream.0.play() {
                    tracing::error!("AudioPlayer: failed to start stream: {e}");
                    return;
                }
                playback.playing = true;
            }
        }
    }

    async fn drain_and_finalize(&mut self) {
        if let PlaybackState::Active(ref playback) = self.state {
            let drain_deadline =
                tokio::time::Instant::now() + tokio::time::Duration::from_millis(500);
            loop {
                let empty = playback.buffer.lock().unwrap().is_empty();
                if empty || tokio::time::Instant::now() >= drain_deadline {
                    break;
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        }
        self.state = PlaybackState::Finalized;
    }
}

#[async_trait]
impl FrameProcessor for AudioPlayer {
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
            Frame::InputAudioRaw(audio) => {
                self.push_audio(audio);
            }
            Frame::End(_) | Frame::Cancel(_) => {
                self.drain_and_finalize().await;
            }
            _ => {}
        }
        ctx.push_frame(envelope, direction).await
    }
}
