//! Acoustic Echo Cancellation using [aec3-rs](https://github.com/RubyBit/aec3-rs),
//! a Rust port of WebRTC's AEC3.
//!
//! Creates a paired filter + sink:
//! - The **filter** ([`AudioFilter`]) goes on the input transport to process mic audio.
//! - The **sink** ([`EchoReferenceSink`]) goes on the output transport to capture speaker audio.
//!
//! ```rust,ignore
//! let (filter, sink) = Aec3Filter::create(Aec3Config::default()).unwrap();
//! let params = TransportParams {
//!     audio_in_filter: Some(filter),
//!     echo_reference_sink: Some(sink),
//!     ..Default::default()
//! };
//! ```
//!
//! Render (speaker) audio is buffered and fed to the AEC engine synchronously
//! at capture time. The AEC3 engine's internal delay estimator handles the
//! alignment between render and capture streams.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use aec3::voip::VoipAec3;
pub use aec3::voip::VoipAec3Error;
use async_trait::async_trait;
use bytes::Bytes;

use crate::echo::EchoReferenceSink;
use crate::filter::{AudioFilter, FilterControlFrame};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the AEC3 filter.
#[derive(Debug, Clone)]
pub struct Aec3Config {
    /// Audio sample rate in Hz. Must be 16000, 32000, or 48000.
    pub sample_rate: u32,
}

impl Default for Aec3Config {
    fn default() -> Self {
        Self { sample_rate: 16000 }
    }
}

// ---------------------------------------------------------------------------
// Inner state (owned exclusively by the capture side)
// ---------------------------------------------------------------------------

struct Aec3Inner {
    aec: VoipAec3,
    /// Configured sample rate in Hz.
    sample_rate: u32,
    /// Expected number of f32 samples per 10ms frame.
    frame_size: usize,
    /// Accumulated capture (mic) samples awaiting full 10ms frames.
    capture_buf: Vec<f32>,
    /// Accumulated processed i16 output bytes.
    output_buf: Vec<u8>,
    /// Render samples waiting to be fed to the AEC, one frame per capture
    /// frame to match real-time cadence. VecDeque for O(1) drain from front.
    render_pending: VecDeque<f32>,
    /// Set once the first render audio arrives. Until then, the filter
    /// bypasses the AEC engine entirely — there's no echo to cancel, and
    /// the AEC's signal processing (high-pass filter, gain control) would
    /// modify the audio unnecessarily.
    has_render: bool,
    /// Whether capture sample rate mismatches the configured AEC rate.
    /// When true, the filter passes audio through unchanged.
    rate_mismatch: bool,
}

// SAFETY: VoipAec3 contains trait objects that lack Send/Sync bounds in the
// upstream aec3-rs crate. However, the underlying types are pure DSP
// computation ported from deterministic C++ (WebRTC AEC3). Aec3Inner is owned
// exclusively by Aec3FilterCapture (via &mut self in AudioFilter), so there is
// no concurrent access.
// TODO: Remove when aec3-rs adds Send/Sync bounds upstream.
unsafe impl Send for Aec3Inner {}
unsafe impl Sync for Aec3Inner {}

// ---------------------------------------------------------------------------
// Format conversion helpers
// ---------------------------------------------------------------------------

/// Convert i16 LE PCM bytes to f32 samples normalized to [-1.0, 1.0].
fn i16_bytes_to_f32(pcm: &[u8]) -> Vec<f32> {
    pcm.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

/// Convert f32 samples to i16 LE PCM bytes.
fn f32_to_i16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&clamped.to_le_bytes());
    }
    out
}

/// Downmix interleaved i16 LE PCM from N channels to mono by averaging.
fn downmix_to_mono(audio: &[u8], num_channels: u16) -> Vec<u8> {
    let ch = num_channels as usize;
    let frame_bytes = ch * 2; // 2 bytes per i16 sample per channel
    let num_frames = audio.len() / frame_bytes;
    let mut out = Vec::with_capacity(num_frames * 2);

    for i in 0..num_frames {
        let mut sum: i32 = 0;
        for c in 0..ch {
            let offset = i * frame_bytes + c * 2;
            let sample = i16::from_le_bytes([audio[offset], audio[offset + 1]]);
            sum += sample as i32;
        }
        let mono = (sum / ch as i32) as i16;
        out.extend_from_slice(&mono.to_le_bytes());
    }

    out
}

// ---------------------------------------------------------------------------
// Capture side — implements AudioFilter
// ---------------------------------------------------------------------------

/// The capture (microphone) side of the AEC filter.
///
/// Implements [`AudioFilter`] for use as `TransportParams::audio_in_filter`.
/// Owns the AEC engine exclusively — render audio is received via a shared
/// buffer that the render sink writes to.
struct Aec3FilterCapture {
    inner: Aec3Inner,
    /// Shared render (speaker) audio buffer. The render sink pushes f32
    /// samples here; this side drains them before processing capture frames.
    render_buf: Arc<Mutex<Vec<f32>>>,
}

#[async_trait]
impl AudioFilter for Aec3FilterCapture {
    async fn start(&mut self, sample_rate: u32) {
        if sample_rate != self.inner.sample_rate {
            tracing::warn!(
                expected = self.inner.sample_rate,
                actual = sample_rate,
                "AEC3 capture sample rate mismatch — filter will pass through"
            );
            self.inner.rate_mismatch = true;
        }
    }

    async fn stop(&mut self) {
        // No explicit teardown needed.
    }

    async fn process_frame(&mut self, _frame: FilterControlFrame) {
        // No runtime settings to update.
    }

    async fn filter(&mut self, audio: Bytes) -> Bytes {
        // Pass through unchanged when sample rate doesn't match.
        if self.inner.rate_mismatch {
            return audio;
        }

        // Drain render audio from the shared buffer into the pending queue.
        // Lock is held only for the duration of a Vec→VecDeque transfer.
        {
            let mut buf = self.render_buf.lock().unwrap();
            if !buf.is_empty() {
                self.inner.has_render = true;
                self.inner.render_pending.extend(buf.iter());
                buf.clear();
            }
        }

        // Bypass the AEC engine entirely until the first render audio
        // arrives. Without render reference there's no echo to cancel,
        // and the AEC's internal signal processing (high-pass filter,
        // gain control) would modify the audio unnecessarily.
        if !self.inner.has_render {
            return audio;
        }

        // Convert incoming i16 PCM to f32 and accumulate.
        let samples = i16_bytes_to_f32(&audio);
        self.inner.capture_buf.extend_from_slice(&samples);

        let frame_size = self.inner.frame_size;

        // Process complete 10ms capture frames, interleaving one render
        // frame per capture frame to match real-time cadence. This prevents
        // burst-feeding render from confusing the AEC's delay estimator.
        let mut render_frame_buf = vec![0.0f32; frame_size];
        while self.inner.capture_buf.len() >= frame_size {
            // Feed one render frame (if available) before each capture frame.
            if self.inner.render_pending.len() >= frame_size {
                for s in render_frame_buf.iter_mut() {
                    *s = self.inner.render_pending.pop_front().unwrap();
                }
                if let Err(e) = self.inner.aec.handle_render_frame(&render_frame_buf) {
                    tracing::warn!("AEC3 handle_render_frame error: {e}");
                }
            }

            let frame: Vec<f32> = self.inner.capture_buf.drain(..frame_size).collect();
            let mut output = vec![0.0f32; frame_size];

            match self
                .inner
                .aec
                .process_capture_frame(&frame, false, &mut output)
            {
                Ok(_metrics) => {
                    self.inner
                        .output_buf
                        .extend_from_slice(&f32_to_i16_bytes(&output));
                }
                Err(e) => {
                    tracing::warn!("AEC3 process_capture_frame error: {e}");
                    // Pass through original audio on error.
                    self.inner
                        .output_buf
                        .extend_from_slice(&f32_to_i16_bytes(&frame));
                }
            }
        }

        if self.inner.output_buf.is_empty() {
            // Still buffering — no complete frame yet.
            Bytes::new()
        } else {
            Bytes::from(std::mem::take(&mut self.inner.output_buf))
        }
    }
}

// ---------------------------------------------------------------------------
// Render side — implements EchoReferenceSink
// ---------------------------------------------------------------------------

/// The render (speaker) side of the AEC filter.
///
/// Implements [`EchoReferenceSink`] for use as `TransportParams::echo_reference_sink`.
/// Buffers render audio in a shared `Vec<f32>` that the capture side drains
/// synchronously during `filter()`. No background task is needed.
struct Aec3RenderSink {
    /// Configured sample rate for mismatch detection.
    sample_rate: u32,
    /// Shared render buffer — capture side drains this.
    render_buf: Arc<Mutex<Vec<f32>>>,
    /// Maximum f32 samples to buffer (~2 seconds of audio).
    max_samples: usize,
    /// Warn-once flag for sample rate mismatch.
    warned_rate_mismatch: AtomicBool,
}

#[async_trait]
impl EchoReferenceSink for Aec3RenderSink {
    async fn write_render(&self, audio: &[u8], sample_rate: u32, num_channels: u16) {
        // Skip buffering entirely when sample rate doesn't match — the AEC
        // can't use audio at a different rate.
        if sample_rate != self.sample_rate {
            if !self.warned_rate_mismatch.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    expected = self.sample_rate,
                    actual = sample_rate,
                    "AEC3 render audio sample rate mismatch — dropping render audio"
                );
            }
            return;
        }

        // Downmix to mono if the render audio has multiple channels.
        let mono_audio;
        let pcm = if num_channels > 1 {
            mono_audio = downmix_to_mono(audio, num_channels);
            &mono_audio
        } else {
            audio
        };

        // Convert to f32 outside the lock.
        let samples = i16_bytes_to_f32(pcm);

        // Push to shared buffer, capping at max_samples.
        let mut buf = self.render_buf.lock().unwrap();
        let total = buf.len() + samples.len();
        if total > self.max_samples {
            // Drop oldest data: first from the existing buffer, then from
            // the head of the incoming chunk if it alone exceeds the cap.
            let to_drop = total - self.max_samples;
            let drain_existing = to_drop.min(buf.len());
            buf.drain(..drain_existing);
            let skip_incoming = to_drop - drain_existing;
            buf.extend_from_slice(&samples[skip_incoming..]);
        } else {
            buf.extend_from_slice(&samples);
        }
    }
}

// ---------------------------------------------------------------------------
// Public constructor
// ---------------------------------------------------------------------------

/// Factory for creating a paired AEC3 filter and echo reference sink.
pub struct Aec3Filter;

impl Aec3Filter {
    /// Create a new AEC3 filter pair.
    ///
    /// Returns `(filter, sink)`:
    /// - `filter` implements [`AudioFilter`] — pass it to `TransportParams::audio_in_filter`
    /// - `sink` implements [`EchoReferenceSink`] — pass it to `TransportParams::echo_reference_sink`
    ///
    /// # Errors
    ///
    /// Returns [`VoipAec3Error`] if the AEC3 engine cannot be created (e.g.
    /// unsupported sample rate).
    #[allow(clippy::type_complexity)]
    pub fn create(
        config: Aec3Config,
    ) -> Result<(Box<dyn AudioFilter>, Arc<dyn EchoReferenceSink>), VoipAec3Error> {
        let aec = VoipAec3::builder(config.sample_rate as usize, 1, 1).build()?;
        let frame_size = aec.capture_frame_samples();

        // Shared render buffer — just a Vec<f32> behind std::sync::Mutex.
        let render_buf = Arc::new(Mutex::new(Vec::<f32>::new()));

        // 30 seconds of mono audio at the configured rate. Must be large
        // enough to hold an entire TTS utterance since synthesis pushes all
        // chunks at once, while the capture side drains at real-time rate.
        let max_samples = config.sample_rate as usize * 30;

        let inner = Aec3Inner {
            aec,
            sample_rate: config.sample_rate,
            frame_size,
            capture_buf: Vec::new(),
            output_buf: Vec::new(),
            render_pending: VecDeque::new(),
            has_render: false,
            rate_mismatch: false,
        };

        let filter = Box::new(Aec3FilterCapture {
            inner,
            render_buf: render_buf.clone(),
        });

        let sink = Arc::new(Aec3RenderSink {
            sample_rate: config.sample_rate,
            render_buf,
            max_samples,
            warned_rate_mismatch: AtomicBool::new(false),
        });

        Ok((filter, sink))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_conversion_roundtrip() {
        let original: Vec<i16> = vec![0, 100, -100, i16::MAX, i16::MIN + 1];
        let bytes: Vec<u8> = original.iter().flat_map(|s| s.to_le_bytes()).collect();

        let f32_samples = i16_bytes_to_f32(&bytes);
        let roundtripped = f32_to_i16_bytes(&f32_samples);

        let result: Vec<i16> = roundtripped
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(original, result);
    }

    #[test]
    fn downmix_stereo_to_mono() {
        // Stereo: L=100, R=200 → mono = 150
        let left: i16 = 100;
        let right: i16 = 200;
        let mut stereo = Vec::new();
        stereo.extend_from_slice(&left.to_le_bytes());
        stereo.extend_from_slice(&right.to_le_bytes());

        let mono = downmix_to_mono(&stereo, 2);
        let result = i16::from_le_bytes([mono[0], mono[1]]);
        assert_eq!(result, 150);
    }

    #[test]
    fn downmix_mono_passthrough() {
        // Single channel: should produce identical output.
        let sample: i16 = 500;
        let input: Vec<u8> = sample.to_le_bytes().to_vec();
        let output = downmix_to_mono(&input, 1);
        assert_eq!(input, output);
    }

    #[test]
    fn format_conversion_silence() {
        let silence = vec![0u8; 320]; // 160 samples of silence
        let f32_samples = i16_bytes_to_f32(&silence);
        assert!(f32_samples.iter().all(|&s| s == 0.0));

        let back = f32_to_i16_bytes(&f32_samples);
        assert_eq!(silence, back);
    }

    #[tokio::test]
    async fn aec3_filter_creates_successfully() {
        let (mut filter, _sink) = Aec3Filter::create(Aec3Config::default()).unwrap();
        filter.start(16000).await;
        filter.stop().await;
    }

    #[tokio::test]
    async fn aec3_filter_passthrough_without_render() {
        let (mut filter, _sink) = Aec3Filter::create(Aec3Config::default()).unwrap();
        filter.start(16000).await;

        // Without any render audio, filter should pass through unchanged.
        let input = Bytes::from(vec![1u8, 2, 3, 4]);
        let result = filter.filter(input.clone()).await;
        assert_eq!(result, input, "Should pass through when no render received");
    }

    #[tokio::test]
    async fn aec3_filter_buffers_small_input() {
        let (mut filter, sink) = Aec3Filter::create(Aec3Config::default()).unwrap();
        filter.start(16000).await;

        // Feed render audio to activate AEC processing.
        let render = vec![0u8; 320];
        sink.write_render(&render, 16000, 1).await;

        // Send less than 10ms at 16kHz (160 samples = 320 bytes).
        // Only send 50 samples = 100 bytes.
        let small_input: Vec<u8> = vec![0u8; 100];
        let result = filter.filter(Bytes::from(small_input)).await;
        assert!(
            result.is_empty(),
            "Should return empty when buffering partial frame"
        );
    }

    #[tokio::test]
    async fn aec3_filter_processes_full_frame() {
        let (mut filter, sink) = Aec3Filter::create(Aec3Config::default()).unwrap();
        filter.start(16000).await;

        // Feed render audio to activate AEC processing.
        let render = vec![0u8; 320];
        sink.write_render(&render, 16000, 1).await;

        // Send exactly 10ms at 16kHz = 160 samples = 320 bytes.
        let full_frame: Vec<u8> = vec![0u8; 320];
        let result = filter.filter(Bytes::from(full_frame)).await;

        // With silence input and silence render, should get 320 bytes back.
        assert_eq!(result.len(), 320, "Should output one full 10ms frame");
    }

    #[tokio::test]
    async fn aec3_render_sink_accepts_audio() {
        let (_filter, sink) = Aec3Filter::create(Aec3Config::default()).unwrap();

        // Feed 10ms of render audio (silence).
        let render_audio = vec![0u8; 320];
        sink.write_render(&render_audio, 16000, 1).await;
        // Should not panic or error.
    }

    #[tokio::test]
    async fn aec3_roundtrip_with_render_and_capture() {
        let (mut filter, sink) = Aec3Filter::create(Aec3Config::default()).unwrap();
        filter.start(16000).await;

        // Generate a tone as render (speaker) audio — 10ms at 16kHz.
        let render_samples: Vec<i16> = (0..160)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (f32::sin(2.0 * std::f32::consts::PI * 440.0 * t) * 16000.0) as i16
            })
            .collect();
        let render_bytes: Vec<u8> = render_samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();

        // Feed render audio — buffered in the shared Vec, no background task.
        sink.write_render(&render_bytes, 16000, 1).await;

        // Feed the same tone as capture (simulating echo).
        // Render audio is drained and fed to the AEC synchronously in filter().
        let result = filter.filter(Bytes::from(render_bytes.clone())).await;
        assert_eq!(result.len(), 320, "Should produce output");

        // The AEC won't fully cancel echo on the first frame (needs convergence),
        // but it should produce valid output without panicking.
    }

    #[tokio::test]
    async fn aec3_render_sink_handles_stereo() {
        let (_filter, sink) = Aec3Filter::create(Aec3Config::default()).unwrap();

        // Build stereo audio: 160 frames × 2 channels = 320 samples = 640 bytes.
        let mut stereo_bytes = Vec::with_capacity(640);
        for i in 0..160 {
            let left = (i * 100) as i16;
            let right = (i * 50) as i16;
            stereo_bytes.extend_from_slice(&left.to_le_bytes());
            stereo_bytes.extend_from_slice(&right.to_le_bytes());
        }

        // Should downmix to mono internally without panicking.
        sink.write_render(&stereo_bytes, 16000, 2).await;
    }

    #[tokio::test]
    async fn render_buffer_capped() {
        let (mut filter, sink) = Aec3Filter::create(Aec3Config::default()).unwrap();
        filter.start(16000).await;

        // Write 3 seconds of render audio (over the 2s cap).
        // 16000 samples/sec * 3 sec = 48000 samples = 96000 bytes.
        let three_sec = vec![0u8; 96000];
        sink.write_render(&three_sec, 16000, 1).await;

        // Process one capture frame to drain — should not OOM or panic.
        let frame = vec![0u8; 320];
        let _ = filter.filter(Bytes::from(frame)).await;
    }

    #[tokio::test]
    async fn sample_rate_mismatch_passthrough() {
        let (mut filter, _sink) = Aec3Filter::create(Aec3Config { sample_rate: 16000 }).unwrap();
        // Start with a different rate to trigger mismatch.
        filter.start(48000).await;

        let input = Bytes::from(vec![1u8, 2, 3, 4]);
        let output = filter.filter(input.clone()).await;
        assert_eq!(
            output, input,
            "Should pass through unchanged on rate mismatch"
        );
    }

    #[tokio::test]
    async fn render_rate_mismatch_skips_buffering() {
        let (_filter, sink) = Aec3Filter::create(Aec3Config { sample_rate: 16000 }).unwrap();

        // Write render at wrong rate — should be silently dropped.
        let audio = vec![0u8; 320];
        sink.write_render(&audio, 48000, 1).await;
        // Should not panic.
    }
}
