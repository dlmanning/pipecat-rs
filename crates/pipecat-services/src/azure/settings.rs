use crate::settings::{STTSettings, TTSSettings};

/// Azure-specific STT settings.
#[derive(Debug, Clone)]
pub struct AzureSTTSettings {
    pub base: STTSettings,
    /// Azure Speech Services API key.
    pub api_key: String,
    /// Azure region (e.g. "eastus").
    pub region: String,
    /// Optional custom endpoint ID for custom speech models.
    pub endpoint_id: Option<String>,
}

impl AzureSTTSettings {
    pub fn new(api_key: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            base: STTSettings {
                model: None,
                language: Some("en-US".into()),
            },
            api_key: api_key.into(),
            region: region.into(),
            endpoint_id: None,
        }
    }

    /// Build the WebSocket URL for Azure Speech STT.
    pub fn build_url(&self) -> String {
        let lang = self.base.language.as_deref().unwrap_or("en-US");
        let mut url = format!(
            "wss://{}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language={}&format=detailed",
            self.region, lang
        );

        if let Some(ref endpoint_id) = self.endpoint_id {
            url.push_str(&format!("&cid={endpoint_id}"));
        }

        url
    }
}

/// Azure-specific TTS settings.
#[derive(Debug, Clone)]
pub struct AzureTTSSettings {
    pub base: TTSSettings,
    /// Azure Speech Services API key.
    pub api_key: String,
    /// Azure region (e.g. "eastus").
    pub region: String,
    /// SSML prosody: pitch (e.g. "+10%", "high").
    pub pitch: Option<String>,
    /// SSML prosody: rate (e.g. "1.0", "fast").
    pub rate: Option<String>,
    /// SSML prosody: volume (e.g. "+20%", "loud").
    pub volume: Option<String>,
    /// SSML emphasis level (e.g. "strong", "moderate", "reduced").
    pub emphasis: Option<String>,
    /// SSML express-as style (e.g. "cheerful", "sad").
    pub style: Option<String>,
    /// SSML express-as style degree (0.01 to 2.0).
    pub style_degree: Option<f64>,
    /// SSML express-as role (e.g. "YoungAdultFemale").
    pub role: Option<String>,
}

impl AzureTTSSettings {
    pub fn new(api_key: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            base: TTSSettings {
                model: None,
                voice: Some("en-US-SaraNeural".into()),
                language: Some("en-US".into()),
            },
            api_key: api_key.into(),
            region: region.into(),
            pitch: None,
            rate: None,
            volume: None,
            emphasis: None,
            style: None,
            style_degree: None,
            role: None,
        }
    }

    /// Build the REST endpoint URL for Azure TTS.
    pub fn build_url(&self) -> String {
        format!(
            "https://{}.tts.speech.microsoft.com/cognitiveservices/v1",
            self.region
        )
    }

    /// Build the output format header value from a sample rate.
    pub fn output_format_from_sample_rate(sample_rate: u32) -> &'static str {
        match sample_rate {
            8000 => "raw-8khz-16bit-mono-pcm",
            16000 => "raw-16khz-16bit-mono-pcm",
            22050 => "raw-22050hz-16bit-mono-pcm",
            24000 => "raw-24khz-16bit-mono-pcm",
            44100 => "raw-44100hz-16bit-mono-pcm",
            48000 => "raw-48khz-16bit-mono-pcm",
            _ => "raw-24khz-16bit-mono-pcm",
        }
    }

    /// Construct SSML XML for the given text.
    pub fn build_ssml(&self, text: &str) -> String {
        let voice = self.base.voice.as_deref().unwrap_or("en-US-SaraNeural");
        let language = self.base.language.as_deref().unwrap_or("en-US");
        let escaped_text = escape_xml(text);

        let mut inner = escaped_text.to_string();

        // Wrap in emphasis if set
        if let Some(ref emphasis) = self.emphasis {
            inner = format!("<emphasis level=\"{emphasis}\">{inner}</emphasis>");
        }

        // Wrap in prosody if any prosody settings are set
        let has_prosody = self.pitch.is_some() || self.rate.is_some() || self.volume.is_some();
        if has_prosody {
            let mut attrs = String::new();
            if let Some(ref rate) = self.rate {
                attrs.push_str(&format!(" rate=\"{rate}\""));
            }
            if let Some(ref pitch) = self.pitch {
                attrs.push_str(&format!(" pitch=\"{pitch}\""));
            }
            if let Some(ref volume) = self.volume {
                attrs.push_str(&format!(" volume=\"{volume}\""));
            }
            inner = format!("<prosody{attrs}>{inner}</prosody>");
        }

        // Wrap in express-as if style is set
        if let Some(ref style) = self.style {
            let mut attrs = format!(" style=\"{style}\"");
            if let Some(degree) = self.style_degree {
                attrs.push_str(&format!(" styledegree=\"{degree}\""));
            }
            if let Some(ref role) = self.role {
                attrs.push_str(&format!(" role=\"{role}\""));
            }
            inner = format!("<mstts:express-as{attrs}>{inner}</mstts:express-as>");
        }

        format!(
            "<speak version='1.0' xml:lang='{language}' \
             xmlns='http://www.w3.org/2001/10/synthesis' \
             xmlns:mstts='http://www.w3.org/2001/mstts'>\
             <voice name='{voice}'>\
             <mstts:silence type='Sentenceboundary' value='20ms' />\
             {inner}\
             </voice>\
             </speak>"
        )
    }
}

/// Escape special XML characters.
fn escape_xml(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(c),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stt_settings_default() {
        let s = AzureSTTSettings::new("key", "eastus");
        assert_eq!(s.api_key, "key");
        assert_eq!(s.region, "eastus");
        assert_eq!(s.base.language.as_deref(), Some("en-US"));
        assert!(s.endpoint_id.is_none());
    }

    #[test]
    fn stt_build_url() {
        let s = AzureSTTSettings::new("key", "westus2");
        let url = s.build_url();
        assert!(url.contains("westus2.stt.speech.microsoft.com"));
        assert!(url.contains("language=en-US"));
        assert!(url.contains("format=detailed"));
    }

    #[test]
    fn stt_build_url_with_endpoint_id() {
        let mut s = AzureSTTSettings::new("key", "eastus");
        s.endpoint_id = Some("my-endpoint".into());
        let url = s.build_url();
        assert!(url.contains("cid=my-endpoint"));
    }

    #[test]
    fn tts_settings_default() {
        let s = AzureTTSSettings::new("key", "eastus");
        assert_eq!(s.base.voice.as_deref(), Some("en-US-SaraNeural"));
        assert_eq!(s.base.language.as_deref(), Some("en-US"));
    }

    #[test]
    fn tts_build_url() {
        let s = AzureTTSSettings::new("key", "westus2");
        let url = s.build_url();
        assert_eq!(
            url,
            "https://westus2.tts.speech.microsoft.com/cognitiveservices/v1"
        );
    }

    #[test]
    fn output_format_mapping() {
        assert_eq!(
            AzureTTSSettings::output_format_from_sample_rate(8000),
            "raw-8khz-16bit-mono-pcm"
        );
        assert_eq!(
            AzureTTSSettings::output_format_from_sample_rate(16000),
            "raw-16khz-16bit-mono-pcm"
        );
        assert_eq!(
            AzureTTSSettings::output_format_from_sample_rate(24000),
            "raw-24khz-16bit-mono-pcm"
        );
        assert_eq!(
            AzureTTSSettings::output_format_from_sample_rate(48000),
            "raw-48khz-16bit-mono-pcm"
        );
        assert_eq!(
            AzureTTSSettings::output_format_from_sample_rate(99999),
            "raw-24khz-16bit-mono-pcm"
        );
    }

    #[test]
    fn ssml_basic() {
        let s = AzureTTSSettings::new("key", "eastus");
        let ssml = s.build_ssml("Hello world");
        assert!(ssml.contains("xml:lang='en-US'"));
        assert!(ssml.contains("name='en-US-SaraNeural'"));
        assert!(ssml.contains("Hello world"));
        assert!(ssml.contains("Sentenceboundary"));
    }

    #[test]
    fn ssml_escapes_xml() {
        let s = AzureTTSSettings::new("key", "eastus");
        let ssml = s.build_ssml("Tom & Jerry <3 \"quotes\" & 'apostrophes'");
        assert!(
            ssml.contains("Tom &amp; Jerry &lt;3 &quot;quotes&quot; &amp; &apos;apostrophes&apos;")
        );
    }

    #[test]
    fn ssml_with_prosody() {
        let mut s = AzureTTSSettings::new("key", "eastus");
        s.rate = Some("1.25".into());
        s.pitch = Some("+10%".into());
        let ssml = s.build_ssml("Hello");
        assert!(ssml.contains("<prosody"));
        assert!(ssml.contains("rate=\"1.25\""));
        assert!(ssml.contains("pitch=\"+10%\""));
        assert!(ssml.contains(">Hello</prosody>"));
    }

    #[test]
    fn ssml_with_emphasis() {
        let mut s = AzureTTSSettings::new("key", "eastus");
        s.emphasis = Some("strong".into());
        let ssml = s.build_ssml("Important");
        assert!(ssml.contains("<emphasis level=\"strong\">Important</emphasis>"));
    }

    #[test]
    fn ssml_with_style() {
        let mut s = AzureTTSSettings::new("key", "eastus");
        s.style = Some("cheerful".into());
        s.style_degree = Some(1.5);
        let ssml = s.build_ssml("Yay");
        assert!(ssml.contains("<mstts:express-as"));
        assert!(ssml.contains("style=\"cheerful\""));
        assert!(ssml.contains("styledegree=\"1.5\""));
        assert!(ssml.contains(">Yay</mstts:express-as>"));
    }

    #[test]
    fn ssml_with_all_options() {
        let mut s = AzureTTSSettings::new("key", "eastus");
        s.style = Some("excited".into());
        s.style_degree = Some(2.0);
        s.role = Some("YoungAdultFemale".into());
        s.rate = Some("fast".into());
        s.pitch = Some("high".into());
        s.volume = Some("loud".into());
        s.emphasis = Some("moderate".into());
        let ssml = s.build_ssml("Wow");
        // Should have all wrappers nested: express-as > prosody > emphasis > text
        assert!(ssml.contains("express-as"));
        assert!(ssml.contains("prosody"));
        assert!(ssml.contains("emphasis"));
        assert!(ssml.contains("role=\"YoungAdultFemale\""));
    }

    #[test]
    fn escape_xml_all_chars() {
        assert_eq!(escape_xml("&<>\"'"), "&amp;&lt;&gt;&quot;&apos;");
        assert_eq!(escape_xml("normal text"), "normal text");
        assert_eq!(escape_xml(""), "");
    }
}
