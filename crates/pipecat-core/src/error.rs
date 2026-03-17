use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipecatError {
    #[error("channel send failed: {0}")]
    ChannelSend(String),
    #[error("channel receive failed: {0}")]
    ChannelRecv(String),
    #[error("processor error: {0}")]
    ProcessorError(String),
    #[error("pipeline error: {0}")]
    PipelineError(String),
}

pub type Result<T> = std::result::Result<T, PipecatError>;
