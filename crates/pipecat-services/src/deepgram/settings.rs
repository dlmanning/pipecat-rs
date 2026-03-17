use crate::settings::STTSettings;

/// Deepgram-specific STT settings.
#[derive(Debug, Clone)]
pub struct DeepgramSTTSettings {
    pub base: STTSettings,
    pub encoding: String,
    pub interim_results: bool,
    pub punctuate: bool,
    pub smart_format: bool,
    pub endpointing: Option<u32>,
    pub utterance_end_ms: Option<u32>,
    pub vad_events: bool,
}

impl Default for DeepgramSTTSettings {
    fn default() -> Self {
        Self {
            base: STTSettings {
                model: Some("nova-2".into()),
                language: Some("en".into()),
            },
            encoding: "linear16".into(),
            interim_results: true,
            punctuate: true,
            smart_format: true,
            endpointing: Some(25),
            utterance_end_ms: Some(1000),
            vad_events: false,
        }
    }
}

impl DeepgramSTTSettings {
    /// Build the WebSocket URL query string from these settings.
    pub fn build_query_params(&self, sample_rate: u32) -> String {
        let mut params = vec![
            format!("encoding={}", self.encoding),
            format!("sample_rate={sample_rate}"),
            format!("channels=1"),
            format!("interim_results={}", self.interim_results),
            format!("punctuate={}", self.punctuate),
            format!("smart_format={}", self.smart_format),
        ];

        if let Some(ref model) = self.base.model {
            params.push(format!("model={model}"));
        }
        if let Some(ref lang) = self.base.language {
            params.push(format!("language={lang}"));
        }
        if let Some(ep) = self.endpointing {
            params.push(format!("endpointing={ep}"));
        }
        if let Some(ue) = self.utterance_end_ms {
            params.push(format!("utterance_end_ms={ue}"));
        }
        if self.vad_events {
            params.push("vad_events=true".to_string());
        }

        params.join("&")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings() {
        let s = DeepgramSTTSettings::default();
        assert_eq!(s.base.model.as_deref(), Some("nova-2"));
        assert_eq!(s.encoding, "linear16");
        assert!(s.interim_results);
        assert!(s.punctuate);
    }

    #[test]
    fn build_query_params_default() {
        let s = DeepgramSTTSettings::default();
        let params = s.build_query_params(16000);
        assert!(params.contains("encoding=linear16"));
        assert!(params.contains("sample_rate=16000"));
        assert!(params.contains("model=nova-2"));
        assert!(params.contains("language=en"));
        assert!(params.contains("endpointing=25"));
        assert!(params.contains("utterance_end_ms=1000"));
        assert!(!params.contains("vad_events"));
    }

    #[test]
    fn build_query_params_with_vad() {
        let mut s = DeepgramSTTSettings::default();
        s.vad_events = true;
        let params = s.build_query_params(8000);
        assert!(params.contains("vad_events=true"));
        assert!(params.contains("sample_rate=8000"));
    }
}
