use crate::settings::TTSSettings;

/// Settings for the macOS `say` TTS service.
#[derive(Debug, Clone)]
pub struct MacOSSaySettings {
    /// Base TTS settings. `voice` maps to `-v`.
    pub base: TTSSettings,

    /// Speech rate in words per minute (`-r`). Default: `None` (system default).
    pub rate: Option<u32>,
}

impl MacOSSaySettings {
    pub fn new(base: TTSSettings) -> Self {
        Self { base, rate: None }
    }
}

impl Default for MacOSSaySettings {
    fn default() -> Self {
        Self::new(TTSSettings::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings() {
        let settings = MacOSSaySettings::default();
        assert!(settings.base.voice.is_none());
        assert!(settings.rate.is_none());
    }
}
