use std::time::Instant;

use ndarray::{Array1, Array3, ArrayView2};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use tracing::{debug, error};

use super::analyzer::VadAnalyzer;

/// How often to reset internal model state to prevent drift (seconds).
const MODEL_RESET_STATES_SECS: f64 = 5.0;

/// Bundled Silero VAD ONNX model bytes.
pub const SILERO_MODEL_BYTES: &[u8] = include_bytes!("data/silero_vad.onnx");

/// Error type for Silero VAD operations.
#[derive(Debug, thiserror::Error)]
pub enum SileroError {
    #[error("ONNX Runtime error: {0}")]
    Ort(String),

    #[error("invalid sample rate {0}: Silero VAD requires 8000 or 16000 Hz")]
    InvalidSampleRate(u32),

    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
}

impl<R> From<ort::Error<R>> for SileroError {
    fn from(e: ort::Error<R>) -> Self {
        SileroError::Ort(e.to_string())
    }
}

/// Silero VAD ONNX model backend.
///
/// Implements [`VadAnalyzer`] for use with `VadAnalyzerBase<SileroVadAnalyzer>`.
/// Runs the Silero ONNX model for voice activity detection, managing LSTM state
/// and audio context buffers between inference calls.
///
/// Supports 8 kHz and 16 kHz sample rates.
///
/// # Example
///
/// ```no_run
/// use pipecat_audio::vad::{SileroVadAnalyzer, VadAnalyzerBase};
///
/// let analyzer = SileroVadAnalyzer::new(16000).unwrap();
/// let mut base = VadAnalyzerBase::new(analyzer, Some(16000), None);
/// base.set_sample_rate(16000);
/// ```
pub struct SileroVadAnalyzer {
    session: Session,
    sample_rate: u32,
    num_samples: usize,
    context_size: usize,

    // LSTM state: shape (2, 1, 128), persisted across calls.
    state: Array3<f32>,

    // Rolling context buffer of float32 samples.
    context: Vec<f32>,

    // Periodic state reset timer.
    last_reset_time: Instant,

    // Pre-allocated buffers reused across inference calls.
    input_buffer: Vec<f32>,
    sr_array: Array1<i64>,
}

impl std::fmt::Debug for SileroVadAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SileroVadAnalyzer")
            .field("sample_rate", &self.sample_rate)
            .field("num_samples", &self.num_samples)
            .field("context_size", &self.context_size)
            .finish_non_exhaustive()
    }
}

impl SileroVadAnalyzer {
    /// Create a new Silero VAD analyzer using the bundled ONNX model.
    ///
    /// # Supported sample rates
    ///
    /// 8000 Hz or 16000 Hz.
    pub fn new(sample_rate: u32) -> Result<Self, SileroError> {
        Self::from_model_bytes(SILERO_MODEL_BYTES, sample_rate)
    }

    /// Create a Silero VAD analyzer from externally-provided model bytes.
    pub fn from_model_bytes(model_bytes: &[u8], sample_rate: u32) -> Result<Self, SileroError> {
        if sample_rate != 16000 && sample_rate != 8000 {
            return Err(SileroError::InvalidSampleRate(sample_rate));
        }

        debug!("Loading Silero VAD model...");

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_memory(model_bytes)?;

        let (num_samples, context_size) = if sample_rate == 16000 {
            (512, 64)
        } else {
            (256, 32)
        };

        debug!("Loaded Silero VAD");

        Ok(Self {
            session,
            sample_rate,
            num_samples,
            context_size,
            state: Array3::<f32>::zeros((2, 1, 128)),
            context: vec![0.0f32; context_size],
            last_reset_time: Instant::now(),
            input_buffer: Vec::with_capacity(context_size + num_samples),
            sr_array: Array1::from_vec(vec![sample_rate as i64]),
        })
    }

    /// Reset internal LSTM state and context buffer to zeros.
    fn reset_states(&mut self) {
        self.state = Array3::<f32>::zeros((2, 1, 128));
        self.context = vec![0.0f32; self.context_size];
    }

    /// Run inference on a single audio chunk.
    ///
    /// `audio_f32` must have exactly `self.num_samples` elements.
    /// Returns the voice probability in \[0.0, 1.0\].
    fn forward(&mut self, audio_f32: &[f32]) -> Result<f32, SileroError> {
        let input_len = self.context_size + self.num_samples;

        // Build input in pre-allocated buffer: [context || audio]
        self.input_buffer.clear();
        self.input_buffer.extend_from_slice(&self.context);
        self.input_buffer.extend_from_slice(audio_f32);

        // Create views over pre-allocated data (no allocation)
        let input_view = ArrayView2::from_shape((1, input_len), &self.input_buffer)?;

        let input_tensor = TensorRef::from_array_view(input_view)?;
        let state_tensor = TensorRef::from_array_view(self.state.view())?;
        let sr_tensor = TensorRef::from_array_view(self.sr_array.view())?;

        // Run ONNX inference
        let outputs = self.session.run(ort::inputs![
            "input" => input_tensor,
            "state" => state_tensor,
            "sr" => sr_tensor,
        ])?;

        // Extract probability: output[0], shape (1, 1)
        let prob_view = outputs[0].try_extract_array::<f32>()?;
        let probability = prob_view[[0, 0]];

        // Extract updated state: output[1], shape (2, 1, 128)
        let state_view = outputs[1].try_extract_array::<f32>()?;
        self.state = state_view.to_owned().into_shape_with_order((2, 1, 128))?;

        // Update context: last context_size samples of [context || audio]
        self.context.clear();
        self.context
            .extend_from_slice(&self.input_buffer[input_len - self.context_size..]);

        Ok(probability)
    }
}

impl VadAnalyzer for SileroVadAnalyzer {
    fn num_frames_required(&self) -> usize {
        self.num_samples
    }

    fn voice_confidence(&mut self, buffer: &[u8]) -> f64 {
        // Convert int16 LE PCM bytes → float32 normalized to [-1.0, 1.0]
        let audio_f32: Vec<f32> = buffer
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();

        match self.forward(&audio_f32) {
            Ok(confidence) => {
                // Periodic state reset to prevent drift.
                // Happens AFTER inference, matching Python (silero.py lines 216-220).
                if self.last_reset_time.elapsed().as_secs_f64() >= MODEL_RESET_STATES_SECS {
                    self.reset_states();
                    self.last_reset_time = Instant::now();
                }
                confidence as f64
            }
            Err(e) => {
                error!(%e, "Silero VAD inference error");
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::analyzer::{VadAnalyzerBase, VadState};

    fn make_silent_pcm(num_samples: usize) -> Vec<u8> {
        vec![0u8; num_samples * 2]
    }

    #[test]
    fn test_new_16khz() {
        let analyzer = SileroVadAnalyzer::new(16000).unwrap();
        assert_eq!(analyzer.num_frames_required(), 512);
    }

    #[test]
    fn test_new_8khz() {
        let analyzer = SileroVadAnalyzer::new(8000).unwrap();
        assert_eq!(analyzer.num_frames_required(), 256);
    }

    #[test]
    fn test_invalid_sample_rate() {
        let result = SileroVadAnalyzer::new(44100);
        assert!(result.is_err());
    }

    #[test]
    fn test_silence_low_confidence() {
        let mut analyzer = SileroVadAnalyzer::new(16000).unwrap();
        let silence = make_silent_pcm(512);
        let confidence = analyzer.voice_confidence(&silence);
        assert!(
            confidence < 0.5,
            "silence should have low confidence, got {confidence}"
        );
    }

    #[test]
    fn test_confidence_in_range() {
        let mut analyzer = SileroVadAnalyzer::new(16000).unwrap();
        let silence = make_silent_pcm(512);
        let confidence = analyzer.voice_confidence(&silence);
        assert!(
            (0.0..=1.0).contains(&confidence),
            "confidence out of range: {confidence}"
        );
    }

    #[test]
    fn test_multiple_calls_dont_panic() {
        let mut analyzer = SileroVadAnalyzer::new(16000).unwrap();
        for _ in 0..20 {
            let silence = make_silent_pcm(512);
            let _ = analyzer.voice_confidence(&silence);
        }
    }

    #[test]
    fn test_works_with_analyzer_base() {
        let analyzer = SileroVadAnalyzer::new(16000).unwrap();
        let mut base = VadAnalyzerBase::new(analyzer, Some(16000), None);
        base.set_sample_rate(16000);

        let silence = make_silent_pcm(512 * 5);
        let (state, _) = base.analyze_audio(&silence);
        assert_eq!(state, VadState::Quiet);
    }

    #[test]
    fn test_8khz_silence() {
        let mut analyzer = SileroVadAnalyzer::new(8000).unwrap();
        let silence = make_silent_pcm(256);
        let confidence = analyzer.voice_confidence(&silence);
        assert!(
            confidence < 0.5,
            "8kHz silence should have low confidence, got {confidence}"
        );
    }

    #[test]
    fn test_from_model_bytes() {
        let analyzer = SileroVadAnalyzer::from_model_bytes(SILERO_MODEL_BYTES, 16000).unwrap();
        assert_eq!(analyzer.num_frames_required(), 512);
    }
}
