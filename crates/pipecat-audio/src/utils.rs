/// Normal speech amplitude is 500–5000; this threshold is well below real speech.
pub const SPEAKING_THRESHOLD: i16 = 20;

/// Mix two 16-bit LE PCM streams by adding samples with clipping.
/// If lengths differ, the shorter stream is zero-padded.
pub fn mix_audio(audio1: &[u8], audio2: &[u8]) -> Vec<u8> {
    let len = audio1.len().max(audio2.len());
    // Ensure even length for i16 alignment
    let sample_count = len / 2;
    let mut result = vec![0u8; sample_count * 2];

    for i in 0..sample_count {
        let s1 = sample_at(audio1, i);
        let s2 = sample_at(audio2, i);
        let mixed = (s1 as i32 + s2 as i32).clamp(-32768, 32767) as i16;
        let bytes = mixed.to_le_bytes();
        result[i * 2] = bytes[0];
        result[i * 2 + 1] = bytes[1];
    }

    result
}

/// Calculate audio volume using RMS loudness, normalized to \[0.0, 1.0\].
///
/// This is a simplified approximation of the Python implementation's EBU R128
/// loudness (via pyloudnorm). The Python version normalizes the EBU R128
/// integrated loudness from \[-20, 80\] dB to \[0, 1\]. We use RMS-to-dBFS
/// mapped to the same range, which is sufficient for the VAD's relative
/// threshold comparisons with exponential smoothing.
pub fn calculate_audio_volume(audio: &[u8], _sample_rate: u32) -> f64 {
    let sample_count = audio.len() / 2;
    if sample_count == 0 {
        return 0.0;
    }

    // Compute RMS in float64 domain
    let mut sum_sq: f64 = 0.0;
    for i in 0..sample_count {
        let s = sample_at(audio, i) as f64;
        sum_sq += s * s;
    }
    let rms = (sum_sq / sample_count as f64).sqrt();

    // Convert to dB scale without normalizing to full-scale first.
    //
    // The Python implementation passes raw int16 values (range ±32768) to
    // pyloudnorm as float64 WITHOUT dividing by 32768. pyloudnorm treats
    // them as-is, producing loudness values in a high dB range. We match
    // this by computing 20*log10(rms) on the raw sample values.
    //
    // Example: amplitude 5000 → 20*log10(5000) ≈ 74 dB → normalized ≈ 0.94
    // This matches Python's output and works correctly with default
    // min_volume=0.6 (which corresponds to amplitude ≈ 100).
    let db = if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -100.0
    };

    // Normalize to [0, 1] using same range as Python (-20 to 80 dB)
    // and clamp to bounds.
    normalize_value(db, -20.0, 80.0)
}

/// Exponential smoothing: `prev + factor * (value - prev)`.
pub fn exp_smoothing(value: f64, prev_value: f64, factor: f64) -> f64 {
    prev_value + factor * (value - prev_value)
}

/// Returns `true` if the max absolute amplitude is at or below [`SPEAKING_THRESHOLD`].
pub fn is_silence(pcm_bytes: &[u8]) -> bool {
    let sample_count = pcm_bytes.len() / 2;
    let mut max_abs: i16 = 0;
    for i in 0..sample_count {
        let s = sample_at(pcm_bytes, i);
        // i16::MIN.abs() overflows, so use wrapping and handle it
        let abs = s.saturating_abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    max_abs <= SPEAKING_THRESHOLD
}

/// Interleave two mono audio streams into stereo (L, R, L, R, ...).
///
/// Both inputs are 16-bit LE PCM. If the channels have different lengths,
/// both are truncated to the shorter length.
pub fn interleave_stereo_audio(left: &[u8], right: &[u8]) -> Vec<u8> {
    let left_samples = left.len() / 2;
    let right_samples = right.len() / 2;
    let count = left_samples.min(right_samples);

    let mut result = Vec::with_capacity(count * 4); // 2 channels × 2 bytes
    for i in 0..count {
        let l = sample_at(left, i);
        let r = sample_at(right, i);
        result.extend_from_slice(&l.to_le_bytes());
        result.extend_from_slice(&r.to_le_bytes());
    }
    result
}

/// Normalize a value from [min_value, max_value] to [0, 1], clamped.
fn normalize_value(value: f64, min_value: f64, max_value: f64) -> f64 {
    let normalized = (value - min_value) / (max_value - min_value);
    normalized.clamp(0.0, 1.0)
}

/// Read the i-th 16-bit LE sample from a byte slice, returning 0 if out of bounds.
#[inline]
fn sample_at(data: &[u8], index: usize) -> i16 {
    let offset = index * 2;
    if offset + 1 < data.len() {
        i16::from_le_bytes([data[offset], data[offset + 1]])
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mix_audio_equal_length() {
        // Two samples: 100 and 200 -> 300
        let a1 = 100i16.to_le_bytes();
        let a2 = 200i16.to_le_bytes();
        let mixed = mix_audio(&a1, &a2);
        let result = i16::from_le_bytes([mixed[0], mixed[1]]);
        assert_eq!(result, 300);
    }

    #[test]
    fn test_mix_audio_different_length() {
        // a1 has 2 samples, a2 has 1 sample (second is zero-padded)
        let mut a1 = Vec::new();
        a1.extend_from_slice(&100i16.to_le_bytes());
        a1.extend_from_slice(&200i16.to_le_bytes());
        let a2 = 50i16.to_le_bytes();

        let mixed = mix_audio(&a1, &a2);
        assert_eq!(mixed.len(), 4);
        let s0 = i16::from_le_bytes([mixed[0], mixed[1]]);
        let s1 = i16::from_le_bytes([mixed[2], mixed[3]]);
        assert_eq!(s0, 150); // 100 + 50
        assert_eq!(s1, 200); // 200 + 0
    }

    #[test]
    fn test_mix_audio_clipping() {
        let a1 = 30000i16.to_le_bytes();
        let a2 = 10000i16.to_le_bytes();
        let mixed = mix_audio(&a1, &a2);
        let result = i16::from_le_bytes([mixed[0], mixed[1]]);
        // 30000 + 10000 = 40000 > 32767, should clip
        assert_eq!(result, 32767);

        // Negative clipping
        let a1 = (-30000i16).to_le_bytes();
        let a2 = (-10000i16).to_le_bytes();
        let mixed = mix_audio(&a1, &a2);
        let result = i16::from_le_bytes([mixed[0], mixed[1]]);
        assert_eq!(result, -32768);
    }

    #[test]
    fn test_exp_smoothing() {
        assert!((exp_smoothing(1.0, 0.0, 0.2) - 0.2).abs() < 1e-10);
        assert!((exp_smoothing(1.0, 0.5, 0.5) - 0.75).abs() < 1e-10);
        assert!((exp_smoothing(0.0, 1.0, 0.2) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_is_silence_quiet() {
        // All zeros
        let audio = vec![0u8; 20];
        assert!(is_silence(&audio));
    }

    #[test]
    fn test_is_silence_at_threshold() {
        let audio = SPEAKING_THRESHOLD.to_le_bytes();
        assert!(is_silence(&audio));
    }

    #[test]
    fn test_is_silence_above_threshold() {
        let audio = (SPEAKING_THRESHOLD + 1).to_le_bytes();
        assert!(!is_silence(&audio));
    }

    #[test]
    fn test_is_silence_speaking() {
        let audio = 5000i16.to_le_bytes();
        assert!(!is_silence(&audio));
    }

    #[test]
    fn test_calculate_audio_volume_silence() {
        let audio = vec![0u8; 200];
        let vol = calculate_audio_volume(&audio, 16000);
        assert_eq!(vol, 0.0);
    }

    #[test]
    fn test_calculate_audio_volume_loud() {
        // Max amplitude signal
        let mut audio = Vec::new();
        for _ in 0..100 {
            audio.extend_from_slice(&32767i16.to_le_bytes());
        }
        let vol = calculate_audio_volume(&audio, 16000);
        // Should be close to 1.0 (very loud)
        assert!(vol > 0.1, "loud signal volume {vol} should be > 0.1");
    }

    #[test]
    fn test_calculate_audio_volume_empty() {
        let vol = calculate_audio_volume(&[], 16000);
        assert_eq!(vol, 0.0);
    }

    #[test]
    fn test_mix_audio_empty() {
        let mixed = mix_audio(&[], &[]);
        assert!(mixed.is_empty());
    }

    #[test]
    fn test_interleave_stereo_equal_length() {
        let left = [100i16.to_le_bytes(), 200i16.to_le_bytes()].concat();
        let right = [300i16.to_le_bytes(), 400i16.to_le_bytes()].concat();
        let stereo = interleave_stereo_audio(&left, &right);
        assert_eq!(stereo.len(), 8); // 2 samples × 2 channels × 2 bytes
        let s0 = i16::from_le_bytes([stereo[0], stereo[1]]);
        let s1 = i16::from_le_bytes([stereo[2], stereo[3]]);
        let s2 = i16::from_le_bytes([stereo[4], stereo[5]]);
        let s3 = i16::from_le_bytes([stereo[6], stereo[7]]);
        assert_eq!(s0, 100); // L
        assert_eq!(s1, 300); // R
        assert_eq!(s2, 200); // L
        assert_eq!(s3, 400); // R
    }

    #[test]
    fn test_interleave_stereo_different_length() {
        // Left has 3 samples, right has 2 — truncate to 2
        let left = [100i16.to_le_bytes(), 200i16.to_le_bytes(), 300i16.to_le_bytes()].concat();
        let right = [400i16.to_le_bytes(), 500i16.to_le_bytes()].concat();
        let stereo = interleave_stereo_audio(&left, &right);
        assert_eq!(stereo.len(), 8); // 2 samples × 2 channels × 2 bytes
    }

    #[test]
    fn test_interleave_stereo_empty() {
        let stereo = interleave_stereo_audio(&[], &[]);
        assert!(stereo.is_empty());
    }
}
