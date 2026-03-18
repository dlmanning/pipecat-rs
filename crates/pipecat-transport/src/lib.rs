pub mod error;
pub mod input;
pub mod local;
pub mod output;
pub mod params;

#[cfg(feature = "cpal")]
pub mod audio_player;

#[cfg(feature = "cpal")]
pub use audio_player::{AudioPlayer, AudioPlayerConfig};
pub use error::TransportError;
pub use input::BaseInputTransport;
pub use local::{
    AudioFormat, AudioInputSource, AudioOutputSink, AudioPacing, LocalAudioInputTransport,
    LocalAudioOutputTransport, LocalAudioTransport,
};
pub use output::{BaseOutputTransport, OutputTransportCallbacks};
pub use params::TransportParams;
