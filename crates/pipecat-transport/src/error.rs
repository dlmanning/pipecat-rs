use thiserror::Error;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("audio write failed: {0}")]
    AudioWrite(String),
    #[error("video write failed: {0}")]
    VideoWrite(String),
    #[error("message send failed: {0}")]
    MessageSend(String),
    #[error("internal channel closed: {0}")]
    ChannelClosed(String),
    #[error("transport not started")]
    NotStarted,
}

pub type Result<T> = std::result::Result<T, TransportError>;
