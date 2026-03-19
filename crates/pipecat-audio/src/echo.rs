use async_trait::async_trait;

/// Trait for feeding render (speaker) audio to an echo canceller.
///
/// The output transport calls [`write_render`] with every chunk of audio
/// it sends to the speaker. An AEC filter on the input side uses this
/// reference signal to subtract the echo from the microphone capture.
///
/// Takes `&self` because it's called from the output transport's audio
/// task via `Arc<dyn EchoReferenceSink>`. Implementations use interior
/// mutability.
#[async_trait]
pub trait EchoReferenceSink: Send + Sync {
    /// Feed render (speaker) audio into the sink.
    ///
    /// `audio` is interleaved i16 LE PCM at the given `sample_rate` with
    /// `num_channels` channels. Implementations should downmix to mono if
    /// the AEC operates on a single channel.
    async fn write_render(&self, audio: &[u8], sample_rate: u32, num_channels: u16);
}
