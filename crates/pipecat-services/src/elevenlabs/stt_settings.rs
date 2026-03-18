use crate::settings::STTSettings;

/// Commit strategy for ElevenLabs realtime STT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitStrategy {
    /// Pipecat's VAD controls commits. On VADUserStoppedSpeaking, send commit.
    Manual,
    /// ElevenLabs' built-in VAD controls commits.
    Vad,
}

impl CommitStrategy {
    /// Returns the query parameter value for this strategy.
    pub fn as_str(&self) -> &'static str {
        match self {
            CommitStrategy::Manual => "manual",
            CommitStrategy::Vad => "vad",
        }
    }
}

/// ElevenLabs-specific realtime STT settings.
#[derive(Debug, Clone)]
pub struct ElevenLabsSTTSettings {
    pub base: STTSettings,
    pub commit_strategy: CommitStrategy,
}

impl Default for ElevenLabsSTTSettings {
    fn default() -> Self {
        Self {
            base: STTSettings {
                model: Some("scribe_v2_realtime".into()),
                language: None,
            },
            commit_strategy: CommitStrategy::Manual,
        }
    }
}

/// Map a sample rate to an ElevenLabs audio format string for STT.
pub fn audio_format_from_sample_rate(sample_rate: u32) -> String {
    match sample_rate {
        8000 => "pcm_8000".into(),
        16000 => "pcm_16000".into(),
        22050 => "pcm_22050".into(),
        24000 => "pcm_24000".into(),
        44100 => "pcm_44100".into(),
        48000 => "pcm_48000".into(),
        _ => {
            tracing::warn!(
                "ElevenLabs STT: no audio format for sample rate {}, defaulting to pcm_16000",
                sample_rate
            );
            "pcm_16000".into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings() {
        let s = ElevenLabsSTTSettings::default();
        assert_eq!(s.base.model.as_deref(), Some("scribe_v2_realtime"));
        assert!(s.base.language.is_none());
        assert_eq!(s.commit_strategy, CommitStrategy::Manual);
    }

    #[test]
    fn commit_strategy_str() {
        assert_eq!(CommitStrategy::Manual.as_str(), "manual");
        assert_eq!(CommitStrategy::Vad.as_str(), "vad");
    }

    #[test]
    fn audio_format_mapping() {
        assert_eq!(audio_format_from_sample_rate(8000), "pcm_8000");
        assert_eq!(audio_format_from_sample_rate(16000), "pcm_16000");
        assert_eq!(audio_format_from_sample_rate(22050), "pcm_22050");
        assert_eq!(audio_format_from_sample_rate(24000), "pcm_24000");
        assert_eq!(audio_format_from_sample_rate(44100), "pcm_44100");
        assert_eq!(audio_format_from_sample_rate(48000), "pcm_48000");
        assert_eq!(audio_format_from_sample_rate(32000), "pcm_16000"); // fallback
    }
}
