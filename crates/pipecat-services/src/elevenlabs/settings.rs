use crate::settings::TTSSettings;

/// ElevenLabs-specific TTS settings.
#[derive(Debug, Clone)]
pub struct ElevenLabsTTSSettings {
    pub base: TTSSettings,
    pub stability: Option<f64>,
    pub similarity_boost: Option<f64>,
    pub style: Option<f64>,
    pub use_speaker_boost: Option<bool>,
    pub speed: Option<f64>,
}

impl Default for ElevenLabsTTSSettings {
    fn default() -> Self {
        Self {
            base: TTSSettings {
                model: Some("eleven_turbo_v2".into()),
                voice: Some("21m00Tcm4TlvDq8ikWAM".into()), // Rachel
                language: None,
            },
            stability: None,
            similarity_boost: None,
            style: None,
            use_speaker_boost: None,
            speed: None,
        }
    }
}

impl ElevenLabsTTSSettings {
    /// Build the voice_settings JSON object if any voice settings are configured.
    pub fn build_voice_settings(&self) -> Option<serde_json::Value> {
        let mut vs = serde_json::Map::new();

        if let Some(s) = self.stability {
            vs.insert("stability".into(), serde_json::json!(s));
        }
        if let Some(s) = self.similarity_boost {
            vs.insert("similarity_boost".into(), serde_json::json!(s));
        }
        if let Some(s) = self.style {
            vs.insert("style".into(), serde_json::json!(s));
        }
        if let Some(b) = self.use_speaker_boost {
            vs.insert("use_speaker_boost".into(), serde_json::json!(b));
        }
        if let Some(s) = self.speed {
            vs.insert("speed".into(), serde_json::json!(s));
        }

        if vs.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(vs))
        }
    }
}

/// Map a sample rate to an ElevenLabs output format string.
pub fn output_format_from_sample_rate(sample_rate: u32) -> String {
    match sample_rate {
        8000 => "pcm_8000".into(),
        16000 => "pcm_16000".into(),
        22050 => "pcm_22050".into(),
        24000 => "pcm_24000".into(),
        44100 => "pcm_44100".into(),
        _ => {
            tracing::warn!(
                "ElevenLabs: no output format for sample rate {}, defaulting to pcm_24000",
                sample_rate
            );
            "pcm_24000".into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings() {
        let s = ElevenLabsTTSSettings::default();
        assert_eq!(s.base.model.as_deref(), Some("eleven_turbo_v2"));
        assert!(s.stability.is_none());
    }

    #[test]
    fn voice_settings_empty_when_no_fields() {
        let s = ElevenLabsTTSSettings::default();
        assert!(s.build_voice_settings().is_none());
    }

    #[test]
    fn voice_settings_includes_set_fields() {
        let mut s = ElevenLabsTTSSettings::default();
        s.stability = Some(0.5);
        s.similarity_boost = Some(0.75);
        s.speed = Some(1.1);

        let vs = s.build_voice_settings().unwrap();
        assert_eq!(vs["stability"], 0.5);
        assert_eq!(vs["similarity_boost"], 0.75);
        assert_eq!(vs["speed"], 1.1);
        assert!(vs.get("style").is_none());
        assert!(vs.get("use_speaker_boost").is_none());
    }

    #[test]
    fn output_format_mapping() {
        assert_eq!(output_format_from_sample_rate(8000), "pcm_8000");
        assert_eq!(output_format_from_sample_rate(16000), "pcm_16000");
        assert_eq!(output_format_from_sample_rate(22050), "pcm_22050");
        assert_eq!(output_format_from_sample_rate(24000), "pcm_24000");
        assert_eq!(output_format_from_sample_rate(44100), "pcm_44100");
        assert_eq!(output_format_from_sample_rate(48000), "pcm_24000"); // fallback
    }
}
