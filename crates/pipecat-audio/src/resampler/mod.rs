mod linear;

#[cfg(feature = "sinc-resampler")]
mod sinc;

pub use linear::LinearResampler;

#[cfg(feature = "sinc-resampler")]
pub use sinc::SincResampler;

use async_trait::async_trait;
use bytes::Bytes;

/// Trait for audio resamplers that convert between sample rates.
///
/// Concrete implementations handle the actual resampling. The transport
/// uses this trait to convert audio to the output sample rate.
#[async_trait]
pub trait AudioResampler: Send + Sync {
    /// Resample audio data from one sample rate to another.
    ///
    /// The audio is 16-bit LE PCM mono. Returns the resampled audio data.
    async fn resample(&mut self, audio: Bytes, in_rate: u32, out_rate: u32) -> Bytes;
}
