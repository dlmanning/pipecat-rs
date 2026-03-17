use async_trait::async_trait;
use bytes::Bytes;

/// Trait for audio resamplers that convert between sample rates.
///
/// Concrete implementations (e.g. using libsoxr) handle the actual
/// resampling. The transport uses this trait to convert audio to the
/// output sample rate.
#[async_trait]
pub trait AudioResampler: Send + Sync {
    /// Resample audio data from one sample rate to another.
    ///
    /// The audio is 16-bit LE PCM. Returns the resampled audio data.
    async fn resample(&mut self, audio: Bytes, in_rate: u32, out_rate: u32) -> Bytes;
}
