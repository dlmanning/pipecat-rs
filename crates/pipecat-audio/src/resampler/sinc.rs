use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use super::AudioResampler;

/// Duration after which the stream resampler clears its internal state,
/// matching Python's `CLEAR_STREAM_AFTER_SECS = 0.2`.
const CLEAR_STREAM_AFTER_SECS: f64 = 0.2;

/// Default chunk size (in frames) for rubato processing.
///
/// 256 frames = ~16ms at 16kHz, balancing latency vs quality.
const DEFAULT_CHUNK_SIZE: usize = 256;

/// High-quality sinc interpolation stream resampler.
///
/// Wraps [`rubato::SincFixedIn`] for production-quality sample rate conversion
/// with anti-aliasing. Suitable for voice and music audio at any rate
/// combination.
///
/// Maintains streaming state across calls for smooth chunk boundaries,
/// matching the behavior of Python's `SOXRStreamAudioResampler`.
///
/// Requires the `sinc-resampler` cargo feature.
///
/// # Latency
///
/// The sinc resampler uses a fixed internal chunk size. If the input chunk is
/// smaller than this, samples are buffered until enough accumulate. With the
/// default chunk size of 256 frames, this introduces up to ~16ms of latency
/// at 16kHz. Use [`SincResampler::with_chunk_size`] to tune this trade-off.
///
/// # Safety
///
/// `SincFixedIn<f64>` is `!Sync` because it contains `Box<dyn SincInterpolator>`.
/// We implement `Sync` manually because `SincResampler` is only ever accessed via
/// `&mut self` (the `AudioResampler` trait has no `&self` methods), so no shared
/// references to the inner resampler are ever created across threads.
pub struct SincResampler {
    in_rate: Option<u32>,
    out_rate: Option<u32>,
    last_resample_time: Option<Instant>,
    resampler: Option<SincFixedIn<f64>>,
    /// Accumulation buffer for input samples (f64, normalized to [-1, 1]).
    input_buf: Vec<f64>,
    chunk_size: usize,
}

// SAFETY: See doc comment on `SincResampler` above. The inner `SincFixedIn` is `Send`
// and only accessed through `&mut self`, so sharing `&SincResampler` across threads is safe.
unsafe impl Sync for SincResampler {}

// SincFixedIn<f64> doesn't impl Debug, so we implement it manually.
impl std::fmt::Debug for SincResampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SincResampler")
            .field("in_rate", &self.in_rate)
            .field("out_rate", &self.out_rate)
            .field("chunk_size", &self.chunk_size)
            .field("input_buf_len", &self.input_buf.len())
            .field("initialized", &self.resampler.is_some())
            .finish()
    }
}

impl SincResampler {
    /// Create a new sinc resampler with the default chunk size (256 frames).
    pub fn new() -> Self {
        Self {
            in_rate: None,
            out_rate: None,
            last_resample_time: None,
            resampler: None,
            input_buf: Vec::new(),
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }

    /// Create a sinc resampler with a custom chunk size.
    ///
    /// Smaller chunk sizes reduce latency but may slightly reduce quality.
    /// Larger chunk sizes improve quality at the cost of higher latency.
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            ..Self::new()
        }
    }

    fn create_resampler(in_rate: u32, out_rate: u32, chunk_size: usize) -> SincFixedIn<f64> {
        let ratio = out_rate as f64 / in_rate as f64;
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            oversampling_factor: 128,
            interpolation: SincInterpolationType::Cubic,
            window: WindowFunction::BlackmanHarris2,
        };
        SincFixedIn::<f64>::new(ratio, 2.0, params, chunk_size, 1)
            .expect("valid sinc resampler parameters")
    }

    fn initialize(&mut self, in_rate: u32, out_rate: u32) {
        self.in_rate = Some(in_rate);
        self.out_rate = Some(out_rate);
        self.resampler = Some(Self::create_resampler(in_rate, out_rate, self.chunk_size));
        self.input_buf.clear();
    }

    fn clear(&mut self) {
        if let Some(ref mut r) = self.resampler {
            r.reset();
        }
        self.input_buf.clear();
    }

    fn maybe_clear_state(&mut self) {
        if let Some(last_time) = self.last_resample_time
            && last_time.elapsed().as_secs_f64() > CLEAR_STREAM_AFTER_SECS
        {
            self.clear();
        }
    }
}

impl Default for SincResampler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AudioResampler for SincResampler {
    async fn resample(&mut self, audio: Bytes, in_rate: u32, out_rate: u32) -> Bytes {
        if in_rate == out_rate {
            return audio;
        }

        // Initialize or validate rates (same pattern as LinearResampler).
        match (self.in_rate, self.out_rate) {
            (None, None) => self.initialize(in_rate, out_rate),
            (Some(ir), Some(or)) => {
                self.maybe_clear_state();
                if ir != in_rate || or != out_rate {
                    tracing::warn!(
                        "SincResampler: rate change ({ir}->{or} to {in_rate}->{out_rate}), reinitializing"
                    );
                    self.initialize(in_rate, out_rate);
                }
            }
            _ => unreachable!(),
        }
        self.last_resample_time = Some(Instant::now());

        // Convert i16 LE PCM bytes to f64 samples normalized to [-1, 1].
        let in_samples: Vec<f64> = audio
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0)
            .collect();

        if in_samples.is_empty() {
            return Bytes::new();
        }

        // Append to internal accumulation buffer.
        self.input_buf.extend_from_slice(&in_samples);

        let resampler = self.resampler.as_mut().unwrap();
        let mut output_samples: Vec<f64> = Vec::new();

        // Process as many full chunks as we can.
        loop {
            let needed = resampler.input_frames_next();
            if self.input_buf.len() < needed {
                break;
            }
            let result = resampler
                .process(&[&self.input_buf[..needed]], None)
                .expect("rubato processing should not fail with valid input");
            self.input_buf.drain(..needed);
            output_samples.extend_from_slice(&result[0]);
        }

        if output_samples.is_empty() {
            return Bytes::new();
        }

        // Convert f64 samples back to i16 LE PCM bytes.
        let mut out_bytes = Vec::with_capacity(output_samples.len() * 2);
        for &s in &output_samples {
            let sample = (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
            out_bytes.extend_from_slice(&sample.to_le_bytes());
        }
        Bytes::from(out_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn same_rate_passthrough() {
        let mut r = SincResampler::new();
        let input = Bytes::from(vec![0u8; 100]);
        let output = r.resample(input.clone(), 16000, 16000).await;
        assert_eq!(input, output);
    }

    #[tokio::test]
    async fn downsample_approximate_length() {
        let mut r = SincResampler::with_chunk_size(64);
        // 1024 samples at 16kHz → ~512 samples at 8kHz
        // Sinc filter has internal delay, so output is shorter than theoretical.
        let samples: Vec<i16> = (0..1024).map(|i| ((i % 100) * 100) as i16).collect();
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), 16000, 8000).await;
        let out_samples = output.len() / 2;
        assert!(
            (350..=550).contains(&out_samples),
            "Expected roughly 350-550 samples (512 minus filter delay), got {out_samples}"
        );
    }

    #[tokio::test]
    async fn upsample_approximate_length() {
        let mut r = SincResampler::with_chunk_size(64);
        // 512 samples at 8kHz → ~1024 samples at 16kHz
        // Sinc filter has internal delay, so output is shorter than theoretical.
        let samples: Vec<i16> = (0..512).map(|i| ((i % 100) * 100) as i16).collect();
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), 8000, 16000).await;
        let out_samples = output.len() / 2;
        assert!(
            (700..=1100).contains(&out_samples),
            "Expected roughly 700-1100 samples (1024 minus filter delay), got {out_samples}"
        );
    }

    #[tokio::test]
    async fn empty_input_returns_empty() {
        let mut r = SincResampler::new();
        let output = r.resample(Bytes::new(), 16000, 8000).await;
        assert!(output.is_empty());
    }

    #[tokio::test]
    async fn dc_signal_preserved() {
        let mut r = SincResampler::with_chunk_size(64);
        // Constant value should remain approximately constant after resampling.
        // Use enough samples to fill multiple chunks and get past the filter startup.
        let samples: Vec<i16> = vec![1000i16; 1024];
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), 24000, 16000).await;

        let out_samples: Vec<i16> = output
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        // Skip the first few samples (filter startup transient) and check the rest.
        let stable = &out_samples[out_samples.len() / 4..];
        for (i, &s) in stable.iter().enumerate() {
            assert!(
                (990..=1010).contains(&s),
                "DC not preserved at stable sample {i}: got {s}"
            );
        }
    }

    #[tokio::test]
    async fn streaming_continuity_across_chunks() {
        let mut r = SincResampler::with_chunk_size(64);

        // Send multiple chunks of a ramp signal and verify smooth output.
        let chunk1: Vec<i16> = (0..512).map(|i| (i * 50) as i16).collect();
        let chunk2: Vec<i16> = (512..1024).map(|i| (i * 50) as i16).collect();

        let input1: Vec<u8> = chunk1.iter().flat_map(|s| s.to_le_bytes()).collect();
        let input2: Vec<u8> = chunk2.iter().flat_map(|s| s.to_le_bytes()).collect();

        let out1 = r.resample(Bytes::from(input1), 16000, 8000).await;
        let out2 = r.resample(Bytes::from(input2), 16000, 8000).await;

        let samples1: Vec<i16> = out1
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let samples2: Vec<i16> = out2
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        if !samples1.is_empty() && !samples2.is_empty() {
            let last = *samples1.last().unwrap();
            let first = samples2[0];
            // Verify no large discontinuity (click) at boundary.
            let diff = (first as i32 - last as i32).unsigned_abs();
            assert!(
                diff < 500,
                "Discontinuity at chunk boundary: last={last}, first={first}, diff={diff}"
            );
        }
    }

    #[tokio::test]
    async fn common_tts_conversion_24k_to_16k() {
        let mut r = SincResampler::with_chunk_size(64);
        // Send enough data to produce output.
        let samples: Vec<i16> = (0..960).map(|i| (i * 30) as i16).collect();
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), 24000, 16000).await;
        let out_samples = output.len() / 2;
        // 960 samples at 24kHz = 40ms → ~640 samples at 16kHz
        // Sinc filter delay reduces actual output.
        assert!(
            (450..=700).contains(&out_samples),
            "Expected roughly 450-700 samples (640 minus filter delay), got {out_samples}"
        );
    }

    #[tokio::test]
    async fn small_input_accumulates() {
        let mut r = SincResampler::with_chunk_size(256);
        // Send input smaller than chunk_size — should accumulate, no output yet.
        let samples: Vec<i16> = (0..100).collect();
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), 16000, 8000).await;
        assert!(
            output.is_empty(),
            "Expected no output for input smaller than chunk_size"
        );

        // Send more data to exceed chunk_size — should produce output.
        let samples2: Vec<i16> = (100..512).collect();
        let input2: Vec<u8> = samples2.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output2 = r.resample(Bytes::from(input2), 16000, 8000).await;
        assert!(
            !output2.is_empty(),
            "Expected output after accumulating past chunk_size"
        );
    }

    #[tokio::test]
    async fn sine_wave_frequency_preserved() {
        let mut r = SincResampler::with_chunk_size(64);
        let in_rate = 48000u32;
        let out_rate = 16000u32;
        let freq = 440.0; // Hz
        let duration_samples = 4800; // 100ms at 48kHz

        // Generate 440Hz sine wave at 48kHz.
        let samples: Vec<i16> = (0..duration_samples)
            .map(|i| {
                let t = i as f64 / in_rate as f64;
                (f64::sin(2.0 * std::f64::consts::PI * freq * t) * 16000.0) as i16
            })
            .collect();
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), in_rate, out_rate).await;

        let out_samples: Vec<f64> = output
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
            .collect();

        // Verify output has energy (not silent).
        let rms: f64 =
            (out_samples.iter().map(|s| s * s).sum::<f64>() / out_samples.len() as f64).sqrt();
        assert!(rms > 5000.0, "Output RMS too low: {rms}");

        // Verify approximate frequency by counting zero crossings.
        let zero_crossings = out_samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        let duration_secs = out_samples.len() as f64 / out_rate as f64;
        let estimated_freq = zero_crossings as f64 / (2.0 * duration_secs);
        assert!(
            (400.0..=480.0).contains(&estimated_freq),
            "Expected ~440Hz, got {estimated_freq}Hz"
        );
    }
}
