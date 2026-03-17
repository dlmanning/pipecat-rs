use std::collections::HashMap;

use pipecat_core::frame::Settings;

/// LLM service settings.
#[derive(Debug, Clone, Default)]
pub struct LLMSettings {
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub seed: Option<u32>,
    pub system_instruction: Option<String>,
}

impl LLMSettings {
    /// Apply a delta from a `Settings` map. Returns the list of field names that changed.
    pub fn apply_delta(&mut self, delta: &Settings) -> Vec<String> {
        let mut changed = Vec::new();
        apply_string(delta, "model", &mut self.model, &mut changed);
        apply_f64(delta, "temperature", &mut self.temperature, &mut changed);
        apply_u32(delta, "max_tokens", &mut self.max_tokens, &mut changed);
        apply_f64(delta, "top_p", &mut self.top_p, &mut changed);
        apply_f64(
            delta,
            "frequency_penalty",
            &mut self.frequency_penalty,
            &mut changed,
        );
        apply_f64(
            delta,
            "presence_penalty",
            &mut self.presence_penalty,
            &mut changed,
        );
        apply_u32(delta, "seed", &mut self.seed, &mut changed);
        apply_string(
            delta,
            "system_instruction",
            &mut self.system_instruction,
            &mut changed,
        );
        changed
    }
}

/// TTS service settings.
#[derive(Debug, Clone, Default)]
pub struct TTSSettings {
    pub model: Option<String>,
    pub voice: Option<String>,
    pub language: Option<String>,
}

impl TTSSettings {
    /// Apply a delta from a `Settings` map. Returns the list of field names that changed.
    pub fn apply_delta(&mut self, delta: &Settings) -> Vec<String> {
        let mut changed = Vec::new();
        apply_string(delta, "model", &mut self.model, &mut changed);
        apply_string(delta, "voice", &mut self.voice, &mut changed);
        apply_string(delta, "language", &mut self.language, &mut changed);
        changed
    }
}

/// STT service settings.
#[derive(Debug, Clone, Default)]
pub struct STTSettings {
    pub model: Option<String>,
    pub language: Option<String>,
}

impl STTSettings {
    /// Apply a delta from a `Settings` map. Returns the list of field names that changed.
    pub fn apply_delta(&mut self, delta: &Settings) -> Vec<String> {
        let mut changed = Vec::new();
        apply_string(delta, "model", &mut self.model, &mut changed);
        apply_string(delta, "language", &mut self.language, &mut changed);
        changed
    }
}

// ---------------------------------------------------------------------------
// Helpers for extracting typed values from Settings
// ---------------------------------------------------------------------------

fn apply_string(
    delta: &HashMap<String, serde_json::Value>,
    key: &str,
    field: &mut Option<String>,
    changed: &mut Vec<String>,
) {
    if let Some(val) = delta.get(key)
        && let Some(s) = val.as_str()
    {
        let new_val = Some(s.to_string());
        if *field != new_val {
            *field = new_val;
            changed.push(key.to_string());
        }
    }
}

fn apply_f64(
    delta: &HashMap<String, serde_json::Value>,
    key: &str,
    field: &mut Option<f64>,
    changed: &mut Vec<String>,
) {
    if let Some(val) = delta.get(key)
        && let Some(n) = val.as_f64()
        && *field != Some(n)
    {
        *field = Some(n);
        changed.push(key.to_string());
    }
}

fn apply_u32(
    delta: &HashMap<String, serde_json::Value>,
    key: &str,
    field: &mut Option<u32>,
    changed: &mut Vec<String>,
) {
    if let Some(val) = delta.get(key)
        && let Some(n) = val.as_u64()
    {
        let n = n as u32;
        if *field != Some(n) {
            *field = Some(n);
            changed.push(key.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn llm_settings_partial_delta() {
        let mut settings = LLMSettings {
            model: Some("gpt-4".into()),
            temperature: Some(0.7),
            ..Default::default()
        };

        let mut delta = Settings::new();
        delta.insert("temperature".into(), json!(0.9));
        delta.insert("max_tokens".into(), json!(1024));

        let changed = settings.apply_delta(&delta);
        assert_eq!(settings.temperature, Some(0.9));
        assert_eq!(settings.max_tokens, Some(1024));
        assert_eq!(settings.model, Some("gpt-4".into())); // unchanged
        assert!(changed.contains(&"temperature".to_string()));
        assert!(changed.contains(&"max_tokens".to_string()));
        assert!(!changed.contains(&"model".to_string()));
    }

    #[test]
    fn llm_settings_no_change_on_same_value() {
        let mut settings = LLMSettings {
            model: Some("gpt-4".into()),
            ..Default::default()
        };
        let mut delta = Settings::new();
        delta.insert("model".into(), json!("gpt-4"));

        let changed = settings.apply_delta(&delta);
        assert!(changed.is_empty());
    }

    #[test]
    fn tts_settings_delta() {
        let mut settings = TTSSettings::default();
        let mut delta = Settings::new();
        delta.insert("voice".into(), json!("alloy"));
        delta.insert("language".into(), json!("en"));

        let changed = settings.apply_delta(&delta);
        assert_eq!(settings.voice, Some("alloy".into()));
        assert_eq!(settings.language, Some("en".into()));
        assert_eq!(changed.len(), 2);
    }

    #[test]
    fn stt_settings_delta() {
        let mut settings = STTSettings::default();
        let mut delta = Settings::new();
        delta.insert("model".into(), json!("nova-2"));

        let changed = settings.apply_delta(&delta);
        assert_eq!(settings.model, Some("nova-2".into()));
        assert_eq!(changed, vec!["model"]);
    }

    #[test]
    fn ignores_unknown_keys() {
        let mut settings = LLMSettings::default();
        let mut delta = Settings::new();
        delta.insert("unknown_field".into(), json!("value"));

        let changed = settings.apply_delta(&delta);
        assert!(changed.is_empty());
    }

    #[test]
    fn ignores_wrong_types() {
        let mut settings = LLMSettings::default();
        let mut delta = Settings::new();
        delta.insert("temperature".into(), json!("not_a_number"));
        delta.insert("model".into(), json!(42));

        let changed = settings.apply_delta(&delta);
        assert!(changed.is_empty());
        assert!(settings.temperature.is_none());
        assert!(settings.model.is_none());
    }
}
