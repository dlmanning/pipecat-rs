use crate::settings::STTSettings;

/// AWS Transcribe-specific STT settings.
#[derive(Debug, Clone)]
pub struct AWSTranscribeSTTSettings {
    pub base: STTSettings,
    /// AWS access key ID.
    pub access_key_id: String,
    /// AWS secret access key.
    pub secret_access_key: String,
    /// Optional AWS session token (for temporary credentials).
    pub session_token: Option<String>,
    /// AWS region (default: "us-east-1").
    pub region: String,
}

impl AWSTranscribeSTTSettings {
    pub fn new(access_key_id: impl Into<String>, secret_access_key: impl Into<String>) -> Self {
        Self {
            base: STTSettings {
                model: None,
                language: Some("en-US".into()),
            },
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: None,
            region: "us-east-1".into(),
        }
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    pub fn with_session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }

    /// Clamp sample rate to 8000 or 16000 (only rates supported by AWS Transcribe).
    pub fn clamp_sample_rate(sample_rate: u32) -> u32 {
        match sample_rate {
            8000 => 8000,
            16000 => 16000,
            _ => {
                tracing::warn!(
                    "AWS Transcribe: sample rate {} not supported, clamping to 16000",
                    sample_rate
                );
                16000
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings() {
        let s = AWSTranscribeSTTSettings::new("key", "secret");
        assert_eq!(s.access_key_id, "key");
        assert_eq!(s.secret_access_key, "secret");
        assert_eq!(s.region, "us-east-1");
        assert_eq!(s.base.language.as_deref(), Some("en-US"));
        assert!(s.session_token.is_none());
    }

    #[test]
    fn with_region() {
        let s = AWSTranscribeSTTSettings::new("key", "secret").with_region("eu-west-1");
        assert_eq!(s.region, "eu-west-1");
    }

    #[test]
    fn with_session_token() {
        let s = AWSTranscribeSTTSettings::new("key", "secret").with_session_token("token123");
        assert_eq!(s.session_token.as_deref(), Some("token123"));
    }

    #[test]
    fn clamp_sample_rate_valid() {
        assert_eq!(AWSTranscribeSTTSettings::clamp_sample_rate(8000), 8000);
        assert_eq!(AWSTranscribeSTTSettings::clamp_sample_rate(16000), 16000);
    }

    #[test]
    fn clamp_sample_rate_invalid() {
        assert_eq!(AWSTranscribeSTTSettings::clamp_sample_rate(24000), 16000);
        assert_eq!(AWSTranscribeSTTSettings::clamp_sample_rate(44100), 16000);
    }
}
