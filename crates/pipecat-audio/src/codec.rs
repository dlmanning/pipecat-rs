use bytes::Bytes;

/// Errors from audio encoding or decoding operations.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("encode error: {0}")]
    Encode(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Trait for encoding PCM audio to a compressed format.
///
/// Input is interleaved 16-bit signed PCM. Implementations define the
/// expected frame size and channel count via their constructors.
pub trait AudioEncoder: Send {
    /// Encode PCM samples into compressed audio data.
    ///
    /// The number of samples must match the encoder's configured frame size
    /// and channel count. Returns the encoded packet as `Bytes`.
    fn encode(&mut self, pcm: &[i16]) -> Result<Bytes, CodecError>;

    /// Reset the encoder state (e.g., after a discontinuity or interruption).
    fn reset(&mut self) -> Result<(), CodecError>;
}

/// Trait for decoding compressed audio to PCM.
pub trait AudioDecoder: Send {
    /// Decode a compressed audio packet into PCM samples.
    ///
    /// Returns interleaved 16-bit signed PCM samples.
    /// Pass `None` for `data` to request packet loss concealment (PLC),
    /// which generates synthetic audio to mask the gap.
    fn decode(&mut self, data: Option<&[u8]>) -> Result<Vec<i16>, CodecError>;

    /// Reset the decoder state.
    fn reset(&mut self) -> Result<(), CodecError>;
}
