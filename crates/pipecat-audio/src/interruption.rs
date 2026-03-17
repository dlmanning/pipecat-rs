use async_trait::async_trait;

/// Trait for interruption strategies that determine when users can
/// interrupt bot speech.
///
/// Concrete implementations analyze accumulated audio and/or text
/// to decide whether the user's input constitutes a real interruption
/// (vs. background noise or a brief acknowledgment).
#[async_trait]
pub trait InterruptionStrategy: Send + Sync {
    /// Append audio data for analysis.
    ///
    /// Default implementation does nothing — override if the strategy
    /// analyzes audio characteristics.
    async fn append_audio(&mut self, _audio: &[u8], _sample_rate: u32) {}

    /// Append transcribed text for analysis.
    ///
    /// Default implementation does nothing — override if the strategy
    /// analyzes text content (e.g. minimum word count).
    async fn append_text(&mut self, _text: &str) {}

    /// Determine whether the user should interrupt the bot.
    async fn should_interrupt(&mut self) -> bool;

    /// Reset accumulated audio and text data.
    async fn reset(&mut self);
}
