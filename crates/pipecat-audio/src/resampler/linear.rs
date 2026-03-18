use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;

use super::AudioResampler;

/// Duration after which the stream resampler clears its internal state,
/// matching Python's `CLEAR_STREAM_AFTER_SECS = 0.2`.
const CLEAR_STREAM_AFTER_SECS: f64 = 0.2;

/// Linear interpolation stream resampler.
///
/// Converts between sample rates using linear interpolation with internal
/// state tracking to avoid clicks at chunk boundaries. Suitable for voice
/// audio at common TTS/telephony rates (e.g., 24kHz→16kHz, 48kHz→16kHz).
///
/// **Limitations:** No anti-aliasing filter — downsampling may introduce
/// aliasing artifacts on wideband content. For production-quality resampling,
/// use [`SincResampler`](super::SincResampler) (requires `sinc-resampler` feature)
/// or implement [`AudioResampler`] with a library like `libsoxr`.
///
/// Matches the streaming behavior of Python's `SOXRStreamAudioResampler`:
/// - Maintains state across calls for smooth chunk boundaries
/// - Clears state after 200ms of inactivity
/// - Warns and reinitializes if sample rates change
#[derive(Debug)]
pub struct LinearResampler {
    in_rate: Option<u32>,
    out_rate: Option<u32>,
    last_resample_time: Option<Instant>,
    /// Last sample from the previous chunk, for interpolation continuity.
    last_sample: i16,
    /// Fractional position carried over from the previous chunk.
    frac_pos: f64,
    /// Pre-allocated buffer for decoded input samples.
    in_buf: Vec<i16>,
    /// Pre-allocated buffer for encoded output bytes.
    out_buf: Vec<u8>,
}

impl LinearResampler {
    pub fn new() -> Self {
        Self {
            in_rate: None,
            out_rate: None,
            last_resample_time: None,
            last_sample: 0,
            frac_pos: 0.0,
            in_buf: Vec::new(),
            out_buf: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.last_sample = 0;
        self.frac_pos = 0.0;
    }

    fn maybe_clear_state(&mut self) {
        if let Some(last_time) = self.last_resample_time
            && last_time.elapsed().as_secs_f64() > CLEAR_STREAM_AFTER_SECS
        {
            self.clear();
        }
    }
}

impl Default for LinearResampler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AudioResampler for LinearResampler {
    async fn resample(&mut self, audio: Bytes, in_rate: u32, out_rate: u32) -> Bytes {
        if in_rate == out_rate {
            return audio;
        }

        // Initialize or validate rates.
        match (self.in_rate, self.out_rate) {
            (None, None) => {
                self.in_rate = Some(in_rate);
                self.out_rate = Some(out_rate);
            }
            (Some(ir), Some(or)) => {
                self.maybe_clear_state();
                if ir != in_rate || or != out_rate {
                    tracing::warn!(
                        "LinearResampler: rate change ({ir}->{or} to {in_rate}->{out_rate}), reinitializing"
                    );
                    self.in_rate = Some(in_rate);
                    self.out_rate = Some(out_rate);
                    self.clear();
                }
            }
            _ => unreachable!(),
        }
        self.last_resample_time = Some(Instant::now());

        // Parse input samples into reusable buffer.
        self.in_buf.clear();
        self.in_buf.extend(
            audio
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]])),
        );

        if self.in_buf.is_empty() {
            return Bytes::new();
        }

        let ratio = in_rate as f64 / out_rate as f64;

        let mut pos = self.frac_pos;
        let prev = self.last_sample;

        // Encode output samples directly into reusable byte buffer.
        self.out_buf.clear();

        while (pos as usize) < self.in_buf.len() {
            let idx = pos as usize;
            let frac = pos - idx as f64;

            let current = self.in_buf[idx];
            let sample = if frac < 1e-9 {
                current
            } else if idx == 0 {
                // Interpolate between last chunk's final sample and first sample.
                let val = prev as f64 * (1.0 - frac) + current as f64 * frac;
                val.round() as i16
            } else {
                let val = self.in_buf[idx - 1] as f64 * (1.0 - frac) + current as f64 * frac;
                val.round() as i16
            };

            self.out_buf.extend_from_slice(&sample.to_le_bytes());
            pos += ratio;
        }

        // Save state for next chunk.
        self.last_sample = *self.in_buf.last().unwrap();
        self.frac_pos = pos - self.in_buf.len() as f64;

        Bytes::copy_from_slice(&self.out_buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn same_rate_passthrough() {
        let mut r = LinearResampler::new();
        let input = Bytes::from(vec![0u8; 100]);
        let output = r.resample(input.clone(), 16000, 16000).await;
        assert_eq!(input, output);
    }

    #[tokio::test]
    async fn downsample_halves_length() {
        let mut r = LinearResampler::new();
        // 100 samples at 16kHz → ~50 samples at 8kHz
        let samples: Vec<i16> = (0..100).collect();
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), 16000, 8000).await;
        let out_samples = output.len() / 2;
        assert_eq!(out_samples, 50);
    }

    #[tokio::test]
    async fn upsample_doubles_length() {
        let mut r = LinearResampler::new();
        // 50 samples at 8kHz → ~100 samples at 16kHz
        let samples: Vec<i16> = (0..50).collect();
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), 8000, 16000).await;
        let out_samples = output.len() / 2;
        assert_eq!(out_samples, 100);
    }

    #[tokio::test]
    async fn empty_input_returns_empty() {
        let mut r = LinearResampler::new();
        let output = r.resample(Bytes::new(), 16000, 8000).await;
        assert!(output.is_empty());
    }

    #[tokio::test]
    async fn common_tts_conversion_24k_to_16k() {
        let mut r = LinearResampler::new();
        // 240 samples at 24kHz (10ms) → 160 samples at 16kHz (10ms)
        let samples: Vec<i16> = (0..240).map(|i| (i * 100) as i16).collect();
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), 24000, 16000).await;
        let out_samples = output.len() / 2;
        assert_eq!(out_samples, 160);
    }

    #[tokio::test]
    async fn non_integer_ratio_44100_to_16000() {
        let mut r = LinearResampler::new();
        // 441 samples at 44100Hz (10ms) → 160 samples at 16000Hz (10ms)
        let samples: Vec<i16> = (0..441).map(|i| (i * 50) as i16).collect();
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), 44100, 16000).await;
        let out_samples = output.len() / 2;
        assert_eq!(out_samples, 160);
    }

    #[tokio::test]
    async fn dc_signal_preserved() {
        let mut r = LinearResampler::new();
        // Constant value should remain constant after resampling.
        let samples: Vec<i16> = vec![1000i16; 100];
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let output = r.resample(Bytes::from(input), 24000, 16000).await;

        let out_samples: Vec<i16> = output
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        // All output samples should be 1000 (DC preserved).
        for (i, &s) in out_samples.iter().enumerate() {
            assert_eq!(s, 1000, "DC not preserved at output sample {i}: got {s}");
        }
    }

    #[tokio::test]
    async fn streaming_continuity_across_chunks() {
        let mut r = LinearResampler::new();

        // Send two chunks and verify no clicks at boundary.
        let chunk1: Vec<i16> = (0..100).map(|i| (i * 100) as i16).collect();
        let chunk2: Vec<i16> = (100..200).map(|i| (i * 100) as i16).collect();

        let input1: Vec<u8> = chunk1.iter().flat_map(|s| s.to_le_bytes()).collect();
        let input2: Vec<u8> = chunk2.iter().flat_map(|s| s.to_le_bytes()).collect();

        let out1 = r.resample(Bytes::from(input1), 16000, 8000).await;
        let out2 = r.resample(Bytes::from(input2), 16000, 8000).await;

        // Parse output samples.
        let samples1: Vec<i16> = out1
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let samples2: Vec<i16> = out2
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        // The last sample of chunk1 and first sample of chunk2 should be
        // monotonically increasing (no click/discontinuity).
        let last = *samples1.last().unwrap();
        let first = samples2[0];
        assert!(
            first > last,
            "Expected continuity: last={last}, first={first}"
        );
    }
}
