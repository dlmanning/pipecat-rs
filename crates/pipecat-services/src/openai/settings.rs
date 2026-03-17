use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::settings::LLMSettings;

/// OpenAI-specific LLM settings.
#[derive(Debug, Clone)]
pub struct OpenAILLMSettings {
    pub base: LLMSettings,
    pub max_completion_tokens: Option<u32>,
    pub extra: HashMap<String, serde_json::Value>,
}

impl OpenAILLMSettings {
    pub fn new(base: LLMSettings) -> Self {
        Self {
            base,
            max_completion_tokens: None,
            extra: HashMap::new(),
        }
    }
}

impl Default for OpenAILLMSettings {
    fn default() -> Self {
        Self::new(LLMSettings::default())
    }
}

// ---------------------------------------------------------------------------
// OpenAI Realtime API settings
// ---------------------------------------------------------------------------

/// Settings for the OpenAI Realtime service.
#[derive(Debug, Clone)]
pub struct OpenAIRealtimeSettings {
    pub base: LLMSettings,
    pub session_properties: SessionProperties,
}

impl OpenAIRealtimeSettings {
    pub fn new(base: LLMSettings) -> Self {
        Self {
            base,
            session_properties: SessionProperties::default(),
        }
    }
}

impl Default for OpenAIRealtimeSettings {
    fn default() -> Self {
        Self::new(LLMSettings::default())
    }
}

/// Session configuration sent to the Realtime API via `session.update`.
///
/// All fields are `Option` with `skip_serializing_if` so only explicitly set
/// values appear in the JSON payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionProperties {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioConfiguration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracing: Option<serde_json::Value>,
}

/// Audio configuration for the Realtime session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioConfiguration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<AudioInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<AudioOutput>,
}

/// Input audio configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcription: Option<InputAudioTranscription>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_reduction: Option<InputAudioNoiseReduction>,
    /// Turn detection configuration.
    ///
    /// - `None` → field omitted (API default: server VAD)
    /// - `Some(None)` → serialized as `null` (disable turn detection)
    /// - `Some(Some(td))` → serialized as the turn detection config object
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_turn_detection",
        deserialize_with = "deserialize_turn_detection"
    )]
    pub turn_detection: Option<Option<TurnDetection>>,
}

/// Serialize `Option<Option<TurnDetection>>`: `Some(None)` → JSON `null`.
fn serialize_turn_detection<S: serde::Serializer>(
    val: &Option<Option<TurnDetection>>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match val {
        None => serializer.serialize_none(),
        Some(None) => serializer.serialize_none(),
        Some(Some(td)) => td.serialize(serializer),
    }
}

/// Deserialize turn_detection: JSON `null`/`false` → `Some(None)`, object → `Some(Some(td))`.
fn deserialize_turn_detection<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Option<TurnDetection>>, D::Error> {
    let val: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    match val {
        None => Ok(Some(None)),
        Some(serde_json::Value::Null) => Ok(Some(None)),
        Some(serde_json::Value::Bool(false)) => Ok(Some(None)),
        Some(v) => {
            let td: TurnDetection = serde_json::from_value(v).map_err(serde::de::Error::custom)?;
            Ok(Some(Some(td)))
        }
    }
}

/// Output audio configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
}

/// Input audio transcription settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioTranscription {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

/// Noise reduction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioNoiseReduction {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub noise_type: Option<String>,
}

/// Turn detection configuration. Tagged by `type` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TurnDetection {
    #[serde(rename = "server_vad")]
    ServerVad {
        #[serde(skip_serializing_if = "Option::is_none")]
        threshold: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prefix_padding_ms: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        silence_duration_ms: Option<u32>,
    },
    #[serde(rename = "semantic_vad")]
    SemanticVad {
        #[serde(skip_serializing_if = "Option::is_none")]
        eagerness: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        create_response: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        interrupt_response: Option<bool>,
    },
}

/// Response-level properties for `response.create`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseProperties {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn session_properties_skip_none_fields() {
        let props = SessionProperties::default();
        let json = serde_json::to_value(&props).unwrap();
        assert_eq!(json, json!({}));
    }

    #[test]
    fn session_properties_includes_set_fields() {
        let props = SessionProperties {
            instructions: Some("You are helpful.".into()),
            output_modalities: Some(vec!["text".into(), "audio".into()]),
            ..Default::default()
        };
        let json = serde_json::to_value(&props).unwrap();
        assert_eq!(json["instructions"], "You are helpful.");
        assert_eq!(json["output_modalities"], json!(["text", "audio"]));
        assert!(json.get("model").is_none());
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn turn_detection_server_vad_serialization() {
        let td = TurnDetection::ServerVad {
            threshold: Some(0.5),
            prefix_padding_ms: Some(300),
            silence_duration_ms: None,
        };
        let json = serde_json::to_value(&td).unwrap();
        assert_eq!(json["type"], "server_vad");
        assert_eq!(json["threshold"], 0.5);
        assert_eq!(json["prefix_padding_ms"], 300);
        assert!(json.get("silence_duration_ms").is_none());
    }

    #[test]
    fn turn_detection_semantic_vad_serialization() {
        let td = TurnDetection::SemanticVad {
            eagerness: Some("medium".into()),
            create_response: Some(true),
            interrupt_response: None,
        };
        let json = serde_json::to_value(&td).unwrap();
        assert_eq!(json["type"], "semantic_vad");
        assert_eq!(json["eagerness"], "medium");
        assert_eq!(json["create_response"], true);
    }

    #[test]
    fn turn_detection_round_trip() {
        let td = TurnDetection::ServerVad {
            threshold: Some(0.8),
            prefix_padding_ms: None,
            silence_duration_ms: Some(500),
        };
        let json = serde_json::to_string(&td).unwrap();
        let back: TurnDetection = serde_json::from_str(&json).unwrap();
        match back {
            TurnDetection::ServerVad {
                threshold,
                silence_duration_ms,
                ..
            } => {
                assert_eq!(threshold, Some(0.8));
                assert_eq!(silence_duration_ms, Some(500));
            }
            _ => panic!("expected ServerVad"),
        }
    }

    #[test]
    fn audio_input_turn_detection_disabled() {
        let input = AudioInput {
            turn_detection: Some(None),
            ..Default::default()
        };
        let json = serde_json::to_value(&input).unwrap();
        assert!(json.get("turn_detection").is_some());
        assert!(json["turn_detection"].is_null());
    }

    #[test]
    fn audio_input_turn_detection_omitted() {
        let input = AudioInput {
            turn_detection: None,
            ..Default::default()
        };
        let json = serde_json::to_value(&input).unwrap();
        assert!(json.get("turn_detection").is_none());
    }

    #[test]
    fn response_properties_default_empty() {
        let props = ResponseProperties::default();
        let json = serde_json::to_value(&props).unwrap();
        assert_eq!(json, json!({}));
    }

    #[test]
    fn response_properties_with_modalities() {
        let props = ResponseProperties {
            output_modalities: Some(vec!["audio".into()]),
            ..Default::default()
        };
        let json = serde_json::to_value(&props).unwrap();
        assert_eq!(json["output_modalities"], json!(["audio"]));
    }
}
