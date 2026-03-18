pub mod function_call;
pub mod llm;
pub mod observers;
pub mod service_base;
pub mod settings;
pub mod stt;
pub mod text_aggregator;
pub mod tts;

#[cfg(feature = "aws")]
pub mod aws;
#[cfg(feature = "azure")]
pub mod azure;
#[cfg(feature = "deepgram")]
pub mod deepgram;
#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "whisper")]
pub mod whisper;

pub use function_call::{FunctionCallHandler, FunctionCallParams, FunctionCallRegistry};
pub use service_base::ServiceBase;
pub use settings::{LLMSettings, STTSettings, TTSSettings};
pub use text_aggregator::{AggregatedText, SimpleTextAggregator, TextAggregationMode};
