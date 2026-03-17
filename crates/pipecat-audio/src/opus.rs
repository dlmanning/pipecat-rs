use audiopus::coder::{Decoder as OpusDecoderInner, Encoder as OpusEncoderInner, GenericCtl};
use audiopus::{Application, Channels, SampleRate};
use bytes::Bytes;

use crate::codec::{AudioDecoder, AudioEncoder, CodecError};

/// Opus application mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusApplication {
    /// Best for VoIP/videoconference (default for pipecat).
    Voip,
    /// Best for broadcast/high-fidelity audio.
    Audio,
    /// Lowest achievable latency.
    LowDelay,
}

/// Opus audio encoder.
///
/// Wraps `audiopus::coder::Encoder` with configuration appropriate for
/// real-time voice. Default: 48kHz mono, VoIP application mode.
///
/// Requires the `opus` cargo feature.
pub struct OpusEncoder {
    inner: OpusEncoderInner,
    /// Pre-allocated output buffer (max Opus packet ~4000 bytes).
    encode_buf: Vec<u8>,
}

// OpusEncoderInner doesn't impl Debug.
impl std::fmt::Debug for OpusEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusEncoder").finish_non_exhaustive()
    }
}

impl OpusEncoder {
    /// Create a new Opus encoder.
    ///
    /// # Supported sample rates
    ///
    /// 8000, 12000, 16000, 24000, 48000 Hz.
    ///
    /// # Supported channel counts
    ///
    /// 1 (mono) or 2 (stereo).
    pub fn new(
        sample_rate: u32,
        channels: u16,
        application: OpusApplication,
    ) -> Result<Self, CodecError> {
        let sr = map_sample_rate(sample_rate)?;
        let ch = map_channels(channels)?;
        let app = map_application(application);
        let inner = OpusEncoderInner::new(sr, ch, app)
            .map_err(|e| CodecError::InvalidConfig(e.to_string()))?;
        Ok(Self {
            inner,
            encode_buf: vec![0u8; 4000],
        })
    }

    /// Set the encoder bitrate in bits per second.
    ///
    /// Typical values: 6000–510000. For VoIP voice, 16000–32000 is common.
    /// The encoder defaults to a bitrate based on the application mode.
    pub fn set_bitrate(&mut self, bits_per_second: i32) -> Result<(), CodecError> {
        self.inner
            .set_bitrate(audiopus::Bitrate::BitsPerSecond(bits_per_second))
            .map_err(|e| CodecError::InvalidConfig(e.to_string()))
    }
}

impl AudioEncoder for OpusEncoder {
    fn encode(&mut self, pcm: &[i16]) -> Result<Bytes, CodecError> {
        let len = self
            .inner
            .encode(pcm, &mut self.encode_buf)
            .map_err(|e| CodecError::Encode(e.to_string()))?;
        Ok(Bytes::copy_from_slice(&self.encode_buf[..len]))
    }

    fn reset(&mut self) -> Result<(), CodecError> {
        self.inner
            .reset_state()
            .map_err(|e: audiopus::Error| CodecError::Encode(e.to_string()))
    }
}

/// Opus audio decoder.
///
/// Wraps `audiopus::coder::Decoder` with configuration for real-time voice.
/// Supports packet loss concealment (PLC) via `decode(None)`.
///
/// The output buffer is sized for the maximum Opus frame (120ms) so it can
/// decode any valid packet. For PLC (packet loss concealment), the decoder
/// uses the `frame_duration_ms` from construction to determine how many
/// samples to synthesize.
///
/// Requires the `opus` cargo feature.
pub struct OpusDecoder {
    inner: OpusDecoderInner,
    channels: u16,
    /// Frame size in samples per channel, used for PLC output sizing.
    plc_frame_size: usize,
    /// Pre-allocated output buffer, sized for the maximum Opus frame (120ms).
    decode_buf: Vec<i16>,
}

// OpusDecoderInner doesn't impl Debug.
impl std::fmt::Debug for OpusDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusDecoder")
            .field("channels", &self.channels)
            .field("plc_frame_size", &self.plc_frame_size)
            .finish_non_exhaustive()
    }
}

/// Maximum Opus frame duration is 120ms. At 48kHz that's 5760 samples per channel.
/// Lower sample rates produce fewer samples, so 5760 * channels is always sufficient.
const MAX_OPUS_FRAME_SAMPLES: usize = 5760;

impl OpusDecoder {
    /// Create a new Opus decoder.
    ///
    /// `frame_duration_ms` sets the expected frame duration (typically 20ms).
    /// This controls how many samples PLC (packet loss concealment) produces
    /// when `decode(None)` is called. Regular decode handles any valid Opus
    /// frame size regardless of this parameter.
    ///
    /// # Supported sample rates
    ///
    /// 8000, 12000, 16000, 24000, 48000 Hz.
    ///
    /// # Supported channel counts
    ///
    /// 1 (mono) or 2 (stereo).
    pub fn new(
        sample_rate: u32,
        channels: u16,
        frame_duration_ms: u32,
    ) -> Result<Self, CodecError> {
        let sr = map_sample_rate(sample_rate)?;
        let ch = map_channels(channels)?;
        let inner =
            OpusDecoderInner::new(sr, ch).map_err(|e| CodecError::InvalidConfig(e.to_string()))?;
        let plc_frame_size = (sample_rate as usize * frame_duration_ms as usize) / 1000;
        let max_total = MAX_OPUS_FRAME_SAMPLES * channels as usize;
        Ok(Self {
            inner,
            channels,
            plc_frame_size,
            decode_buf: vec![0i16; max_total],
        })
    }
}

impl AudioDecoder for OpusDecoder {
    fn decode(&mut self, data: Option<&[u8]>) -> Result<Vec<i16>, CodecError> {
        // For PLC (None), use a buffer sized to plc_frame_size so the decoder
        // produces the expected number of samples. For regular decode, use the
        // full buffer so any valid Opus frame fits.
        let buf_len = match data {
            None => self.plc_frame_size * self.channels as usize,
            Some(_) => self.decode_buf.len(),
        };
        let len = self
            .inner
            .decode(data, &mut self.decode_buf[..buf_len], false)
            .map_err(|e| CodecError::Decode(e.to_string()))?;
        let total = len * self.channels as usize;
        Ok(self.decode_buf[..total].to_vec())
    }

    fn reset(&mut self) -> Result<(), CodecError> {
        self.inner
            .reset_state()
            .map_err(|e: audiopus::Error| CodecError::Decode(e.to_string()))
    }
}

fn map_sample_rate(rate: u32) -> Result<SampleRate, CodecError> {
    match rate {
        8000 => Ok(SampleRate::Hz8000),
        12000 => Ok(SampleRate::Hz12000),
        16000 => Ok(SampleRate::Hz16000),
        24000 => Ok(SampleRate::Hz24000),
        48000 => Ok(SampleRate::Hz48000),
        _ => Err(CodecError::InvalidConfig(format!(
            "unsupported Opus sample rate: {rate}Hz (supported: 8000, 12000, 16000, 24000, 48000)"
        ))),
    }
}

fn map_channels(channels: u16) -> Result<Channels, CodecError> {
    match channels {
        1 => Ok(Channels::Mono),
        2 => Ok(Channels::Stereo),
        _ => Err(CodecError::InvalidConfig(format!(
            "unsupported channel count: {channels} (supported: 1, 2)"
        ))),
    }
}

fn map_application(app: OpusApplication) -> Application {
    match app {
        OpusApplication::Voip => Application::Voip,
        OpusApplication::Audio => Application::Audio,
        OpusApplication::LowDelay => Application::LowDelay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_silence(frame_size: usize) -> Vec<i16> {
        vec![0i16; frame_size]
    }

    fn make_tone(sample_rate: u32, freq: f64, duration_ms: u32) -> Vec<i16> {
        let num_samples = (sample_rate as usize * duration_ms as usize) / 1000;
        (0..num_samples)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (f64::sin(2.0 * std::f64::consts::PI * freq * t) * 16000.0) as i16
            })
            .collect()
    }

    #[test]
    fn round_trip_mono_48k() {
        let mut enc = OpusEncoder::new(48000, 1, OpusApplication::Voip).unwrap();
        let mut dec = OpusDecoder::new(48000, 1, 20).unwrap();

        let input = make_tone(48000, 440.0, 20); // 960 samples = 20ms at 48kHz
        let encoded = enc.encode(&input).unwrap();
        assert!(!encoded.is_empty());

        let decoded = dec.decode(Some(&encoded)).unwrap();
        assert_eq!(decoded.len(), 960);

        // Opus is lossy — verify correlation, not exact match.
        let correlation: f64 = input
            .iter()
            .zip(decoded.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum::<f64>();
        assert!(
            correlation > 0.0,
            "Decoded audio should correlate with input"
        );
    }

    #[test]
    fn silence_round_trip() {
        let mut enc = OpusEncoder::new(48000, 1, OpusApplication::Voip).unwrap();
        let mut dec = OpusDecoder::new(48000, 1, 20).unwrap();

        let input = make_silence(960);
        let encoded = enc.encode(&input).unwrap();
        let decoded = dec.decode(Some(&encoded)).unwrap();

        // Decoded silence should be near-silent.
        let max_amplitude = decoded.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        assert!(
            max_amplitude < 100,
            "Decoded silence too loud: max={max_amplitude}"
        );
    }

    #[test]
    fn packet_loss_concealment() {
        let mut enc = OpusEncoder::new(48000, 1, OpusApplication::Voip).unwrap();
        let mut dec = OpusDecoder::new(48000, 1, 20).unwrap();

        // Decode a real packet first so the decoder has state.
        let input = make_tone(48000, 440.0, 20);
        let encoded = enc.encode(&input).unwrap();
        let _ = dec.decode(Some(&encoded)).unwrap();

        // Now simulate packet loss.
        let plc_output = dec.decode(None).unwrap();
        assert_eq!(plc_output.len(), 960, "PLC should produce a full frame");
    }

    #[test]
    fn invalid_sample_rate() {
        let result = OpusEncoder::new(44100, 1, OpusApplication::Voip);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("44100"), "Error should mention the bad rate");
    }

    #[test]
    fn invalid_channels() {
        let result = OpusEncoder::new(48000, 3, OpusApplication::Voip);
        assert!(result.is_err());
    }

    #[test]
    fn set_bitrate() {
        let mut enc = OpusEncoder::new(48000, 1, OpusApplication::Voip).unwrap();
        enc.set_bitrate(24000).unwrap();
        // Encoding should still work at the new bitrate.
        let input = make_tone(48000, 440.0, 20);
        let encoded = enc.encode(&input).unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn encoder_reset() {
        let mut enc = OpusEncoder::new(48000, 1, OpusApplication::Voip).unwrap();
        let input = make_tone(48000, 440.0, 20);
        let _ = enc.encode(&input).unwrap();
        enc.reset().unwrap();
        // After reset, encoding should still work.
        let encoded = enc.encode(&input).unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn decoder_reset() {
        let mut enc = OpusEncoder::new(48000, 1, OpusApplication::Voip).unwrap();
        let mut dec = OpusDecoder::new(48000, 1, 20).unwrap();
        let input = make_tone(48000, 440.0, 20);
        let encoded = enc.encode(&input).unwrap();
        let _ = dec.decode(Some(&encoded)).unwrap();
        dec.reset().unwrap();
        // After reset, decoding should still work.
        let decoded = dec.decode(Some(&encoded)).unwrap();
        assert_eq!(decoded.len(), 960);
    }

    #[test]
    fn different_frame_sizes() {
        let mut enc = OpusEncoder::new(48000, 1, OpusApplication::Voip).unwrap();
        let mut dec = OpusDecoder::new(48000, 1, 20).unwrap();

        // 10ms at 48kHz = 480 samples
        let input_10ms = make_tone(48000, 440.0, 10);
        assert_eq!(input_10ms.len(), 480);
        let encoded = enc.encode(&input_10ms).unwrap();
        let decoded = dec.decode(Some(&encoded)).unwrap();
        assert_eq!(decoded.len(), 480);

        // 40ms at 48kHz = 1920 samples — same decoder handles it.
        let input_40ms = make_tone(48000, 440.0, 40);
        assert_eq!(input_40ms.len(), 1920);
        let encoded = enc.encode(&input_40ms).unwrap();
        let decoded = dec.decode(Some(&encoded)).unwrap();
        assert_eq!(decoded.len(), 1920);
    }

    #[test]
    fn stereo_round_trip() {
        let mut enc = OpusEncoder::new(48000, 2, OpusApplication::Voip).unwrap();
        let mut dec = OpusDecoder::new(48000, 2, 20).unwrap();

        // 960 frames * 2 channels = 1920 interleaved samples
        let input: Vec<i16> = (0..1920)
            .map(|i| {
                let t = (i / 2) as f64 / 48000.0;
                (f64::sin(2.0 * std::f64::consts::PI * 440.0 * t) * 16000.0) as i16
            })
            .collect();

        let encoded = enc.encode(&input).unwrap();
        let decoded = dec.decode(Some(&encoded)).unwrap();
        assert_eq!(decoded.len(), 1920);
    }

    #[test]
    fn multiple_sample_rates() {
        for rate in [8000, 12000, 16000, 24000, 48000] {
            let mut enc = OpusEncoder::new(rate, 1, OpusApplication::Voip).unwrap();
            let mut dec = OpusDecoder::new(rate, 1, 20).unwrap();
            let frame_size = (rate as usize * 20) / 1000;
            let input = make_tone(rate, 440.0, 20);
            assert_eq!(input.len(), frame_size, "rate={rate}");
            let encoded = enc.encode(&input).unwrap();
            let decoded = dec.decode(Some(&encoded)).unwrap();
            assert_eq!(decoded.len(), frame_size, "rate={rate}");
        }
    }
}
