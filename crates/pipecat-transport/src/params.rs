use std::collections::HashMap;
use std::fmt;

use pipecat_audio::filter::AudioFilter;
use pipecat_audio::mixer::AudioMixer;
use pipecat_audio::resampler::AudioResampler;

/// Configuration for input and output transports.
///
/// Audio and video parameters control what media the transport processes.
/// Trait-object fields (`audio_in_filter`, `audio_in_resampler`, `audio_out_mixer`,
/// `audio_out_resampler`) are optional hooks for concrete implementations.
pub struct TransportParams {
    // -- Audio input --
    pub audio_in_enabled: bool,
    /// Sample rate for input audio. `None` uses the rate from `StartFrame`.
    pub audio_in_sample_rate: Option<u32>,
    pub audio_in_channels: u16,
    pub audio_in_filter: Option<Box<dyn AudioFilter>>,
    pub audio_in_resampler: Option<Box<dyn AudioResampler>>,
    /// Start streaming audio immediately when the transport starts.
    pub audio_in_stream_on_start: bool,
    /// Pass input audio downstream through the pipeline.
    pub audio_in_passthrough: bool,

    // -- Audio output --
    pub audio_out_enabled: bool,
    /// Sample rate for output audio. `None` uses the rate from `StartFrame`.
    pub audio_out_sample_rate: Option<u32>,
    pub audio_out_channels: u16,
    pub audio_out_bitrate: u32,
    /// Number of 10ms chunks to buffer before sending (default 4 = 40ms).
    pub audio_out_10ms_chunks: u32,
    /// Default audio mixer (used by the default destination sender).
    /// For per-destination mixers, use `audio_out_mixer_map` instead.
    pub audio_out_mixer: Option<Box<dyn AudioMixer>>,
    /// Per-destination audio mixers. Each named destination looks up its
    /// mixer here; destinations not in the map get no mixer.
    pub audio_out_mixer_map: HashMap<String, Box<dyn AudioMixer>>,
    pub audio_out_resampler: Option<Box<dyn AudioResampler>>,
    /// Seconds of silence to send after EndFrame for clean audio teardown.
    pub audio_out_end_silence_secs: u32,
    /// Named audio output destinations (in addition to the default sender).
    pub audio_out_destinations: Vec<String>,

    // -- Video input --
    pub video_in_enabled: bool,

    // -- Video output --
    pub video_out_enabled: bool,
    /// Named video output destinations (in addition to the default sender).
    pub video_out_destinations: Vec<String>,
    pub video_out_is_live: bool,
    pub video_out_width: u32,
    pub video_out_height: u32,
    pub video_out_bitrate: u32,
    pub video_out_framerate: u32,
    pub video_out_color_format: String,
    pub video_out_codec: Option<String>,
}

impl Default for TransportParams {
    fn default() -> Self {
        Self {
            audio_in_enabled: false,
            audio_in_sample_rate: None,
            audio_in_channels: 1,
            audio_in_filter: None,
            audio_in_resampler: None,
            audio_in_stream_on_start: true,
            audio_in_passthrough: true,

            audio_out_enabled: false,
            audio_out_sample_rate: None,
            audio_out_channels: 1,
            audio_out_bitrate: 96_000,
            audio_out_10ms_chunks: 4,
            audio_out_mixer: None,
            audio_out_mixer_map: HashMap::new(),
            audio_out_resampler: None,
            audio_out_end_silence_secs: 2,
            audio_out_destinations: Vec::new(),

            video_in_enabled: false,

            video_out_enabled: false,
            video_out_destinations: Vec::new(),
            video_out_is_live: false,
            video_out_width: 1024,
            video_out_height: 768,
            video_out_bitrate: 800_000,
            video_out_framerate: 30,
            video_out_color_format: "RGB".to_string(),
            video_out_codec: None,
        }
    }
}

impl fmt::Debug for TransportParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransportParams")
            .field("audio_in_enabled", &self.audio_in_enabled)
            .field("audio_in_sample_rate", &self.audio_in_sample_rate)
            .field("audio_in_channels", &self.audio_in_channels)
            .field("audio_in_filter", &self.audio_in_filter.is_some())
            .field("audio_in_resampler", &self.audio_in_resampler.is_some())
            .field("audio_in_stream_on_start", &self.audio_in_stream_on_start)
            .field("audio_in_passthrough", &self.audio_in_passthrough)
            .field("audio_out_enabled", &self.audio_out_enabled)
            .field("audio_out_sample_rate", &self.audio_out_sample_rate)
            .field("audio_out_channels", &self.audio_out_channels)
            .field("audio_out_bitrate", &self.audio_out_bitrate)
            .field("audio_out_10ms_chunks", &self.audio_out_10ms_chunks)
            .field("audio_out_mixer", &self.audio_out_mixer.is_some())
            .field("audio_out_mixer_map_len", &self.audio_out_mixer_map.len())
            .field("audio_out_resampler", &self.audio_out_resampler.is_some())
            .field(
                "audio_out_end_silence_secs",
                &self.audio_out_end_silence_secs,
            )
            .field("audio_out_destinations", &self.audio_out_destinations)
            .field("video_in_enabled", &self.video_in_enabled)
            .field("video_out_enabled", &self.video_out_enabled)
            .field("video_out_destinations", &self.video_out_destinations)
            .field("video_out_is_live", &self.video_out_is_live)
            .field("video_out_width", &self.video_out_width)
            .field("video_out_height", &self.video_out_height)
            .field("video_out_bitrate", &self.video_out_bitrate)
            .field("video_out_framerate", &self.video_out_framerate)
            .field("video_out_color_format", &self.video_out_color_format)
            .field("video_out_codec", &self.video_out_codec)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params() {
        let p = TransportParams::default();
        assert!(!p.audio_in_enabled);
        assert!(!p.audio_out_enabled);
        assert_eq!(p.audio_in_channels, 1);
        assert_eq!(p.audio_out_channels, 1);
        assert_eq!(p.audio_out_10ms_chunks, 4);
        assert_eq!(p.audio_out_bitrate, 96_000);
        assert_eq!(p.audio_out_end_silence_secs, 2);
        assert!(p.audio_in_passthrough);
        assert!(p.audio_in_stream_on_start);
        assert!(p.audio_in_filter.is_none());
        assert!(p.audio_in_resampler.is_none());
        assert!(p.audio_out_mixer.is_none());
        assert!(p.audio_out_mixer_map.is_empty());
        assert!(p.audio_out_resampler.is_none());
        assert!(p.audio_out_destinations.is_empty());
        assert!(!p.video_in_enabled);
        assert!(!p.video_out_enabled);
        assert!(p.video_out_destinations.is_empty());
        assert_eq!(p.video_out_width, 1024);
        assert_eq!(p.video_out_height, 768);
        assert_eq!(p.video_out_framerate, 30);
    }

    #[test]
    fn debug_impl_works() {
        let p = TransportParams::default();
        let debug = format!("{:?}", p);
        assert!(debug.contains("TransportParams"));
        assert!(debug.contains("audio_in_enabled: false"));
        // Trait objects show as bool (present/absent)
        assert!(debug.contains("audio_in_filter: false"));
        assert!(debug.contains("audio_out_mixer: false"));
    }
}
