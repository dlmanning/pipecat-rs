use std::time::Instant;

use crate::frame::MetricsData;

// ---------------------------------------------------------------------------
// LlmTokenUsage — detailed token counts for LLM calls
// ---------------------------------------------------------------------------

/// Detailed token usage from an LLM call.
#[derive(Debug, Clone, Default)]
pub struct LlmTokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub cache_read_input_tokens: Option<u32>,
    pub cache_creation_input_tokens: Option<u32>,
    pub reasoning_tokens: Option<u32>,
}

// ---------------------------------------------------------------------------
// ProcessorMetrics — timing helper for services
// ---------------------------------------------------------------------------

/// Timing helper that services embed to measure TTFB, processing time,
/// text aggregation time, and usage metrics.
///
/// # Usage
///
/// ```ignore
/// // In your service:
/// struct MyLLMService {
///     metrics: ProcessorMetrics,
///     // ...
/// }
///
/// // When starting an LLM call:
/// self.metrics.start_ttfb();
///
/// // When first token arrives:
/// if let Some(data) = self.metrics.stop_ttfb() {
///     ctx.send_downstream(Frame::Metrics(MetricsFrame { data: vec![data] })).await?;
/// }
/// ```
#[derive(Debug)]
pub struct ProcessorMetrics {
    processor_name: String,
    model: Option<String>,

    ttfb_start: Option<Instant>,
    processing_start: Option<Instant>,
    text_aggregation_start: Option<Instant>,

    report_only_initial_ttfb: bool,
    initial_ttfb_reported: bool,
}

impl ProcessorMetrics {
    pub fn new(processor_name: impl Into<String>, model: Option<String>) -> Self {
        Self {
            processor_name: processor_name.into(),
            model,
            ttfb_start: None,
            processing_start: None,
            text_aggregation_start: None,
            report_only_initial_ttfb: false,
            initial_ttfb_reported: false,
        }
    }

    /// Set the model name (e.g., after connecting to the service).
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = Some(model.into());
    }

    /// If true, only the first TTFB measurement is reported. Subsequent
    /// `stop_ttfb()` calls return None. Set from `StartFrame.enable_metrics`.
    pub fn set_report_only_initial_ttfb(&mut self, value: bool) {
        self.report_only_initial_ttfb = value;
    }

    // -- TTFB --

    /// Begin a TTFB (time-to-first-byte) measurement.
    pub fn start_ttfb(&mut self) {
        self.ttfb_start = Some(Instant::now());
    }

    /// Begin a TTFB measurement with a custom start time.
    pub fn start_ttfb_at(&mut self, start: Instant) {
        self.ttfb_start = Some(start);
    }

    /// End TTFB measurement and return the metric. Returns `None` if:
    /// - `start_ttfb()` was never called
    /// - `report_only_initial_ttfb` is true and the first TTFB was already reported
    pub fn stop_ttfb(&mut self) -> Option<MetricsData> {
        let start = self.ttfb_start.take()?;

        if self.report_only_initial_ttfb && self.initial_ttfb_reported {
            return None;
        }

        let duration = start.elapsed();
        self.initial_ttfb_reported = true;

        Some(MetricsData::Ttfb {
            processor: self.processor_name.clone(),
            model: self.model.clone(),
            value_secs: duration.as_secs_f64(),
        })
    }

    // -- Processing --

    /// Begin a processing time measurement.
    pub fn start_processing(&mut self) {
        self.processing_start = Some(Instant::now());
    }

    /// Begin a processing time measurement with a custom start time.
    pub fn start_processing_at(&mut self, start: Instant) {
        self.processing_start = Some(start);
    }

    /// End processing measurement and return the metric.
    pub fn stop_processing(&mut self) -> Option<MetricsData> {
        let start = self.processing_start.take()?;
        let duration = start.elapsed();

        Some(MetricsData::Processing {
            processor: self.processor_name.clone(),
            model: self.model.clone(),
            value_secs: duration.as_secs_f64(),
        })
    }

    // -- Text aggregation --

    /// Begin a text aggregation time measurement (time from first LLM token
    /// to first complete sentence).
    pub fn start_text_aggregation(&mut self) {
        self.text_aggregation_start = Some(Instant::now());
    }

    /// End text aggregation measurement and return the metric.
    pub fn stop_text_aggregation(&mut self) -> Option<MetricsData> {
        let start = self.text_aggregation_start.take()?;
        let duration = start.elapsed();

        Some(MetricsData::TextAggregation {
            processor: self.processor_name.clone(),
            model: self.model.clone(),
            value_secs: duration.as_secs_f64(),
        })
    }

    // -- Usage (immediate, no timing) --

    /// Record LLM token usage. Returns a MetricsData immediately.
    pub fn record_llm_usage(&self, usage: LlmTokenUsage) -> MetricsData {
        MetricsData::LlmUsage {
            processor: self.processor_name.clone(),
            model: self.model.clone(),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
        }
    }

    /// Record TTS character usage. Counts characters in the given text.
    pub fn record_tts_usage(&self, text: &str) -> MetricsData {
        MetricsData::TtsUsage {
            processor: self.processor_name.clone(),
            model: self.model.clone(),
            characters: text.chars().count() as u32,
        }
    }

    /// Record a turn detection result.
    pub fn record_turn(
        &self,
        is_complete: bool,
        probability: f64,
        e2e_processing_time_ms: f64,
    ) -> MetricsData {
        MetricsData::Turn {
            processor: self.processor_name.clone(),
            model: self.model.clone(),
            is_complete,
            probability,
            e2e_processing_time_ms,
        }
    }

    /// Reset all in-progress measurements. Called on interruption.
    pub fn reset(&mut self) {
        self.ttfb_start = None;
        self.processing_start = None;
        self.text_aggregation_start = None;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use super::*;

    fn make_metrics() -> ProcessorMetrics {
        ProcessorMetrics::new("test_service", Some("gpt-4".into()))
    }

    #[test]
    fn ttfb_start_stop_produces_metric() {
        let mut m = make_metrics();
        m.start_ttfb();
        thread::sleep(Duration::from_millis(5));
        let data = m.stop_ttfb();

        assert!(data.is_some());
        match data.unwrap() {
            MetricsData::Ttfb {
                processor,
                model,
                value_secs,
            } => {
                assert_eq!(processor, "test_service");
                assert_eq!(model.as_deref(), Some("gpt-4"));
                assert!(value_secs > 0.0);
                assert!(value_secs < 1.0); // sanity: should be ~5ms
            }
            other => panic!("expected Ttfb, got {other:?}"),
        }
    }

    #[test]
    fn ttfb_stop_without_start_returns_none() {
        let mut m = make_metrics();
        assert!(m.stop_ttfb().is_none());
    }

    #[test]
    fn ttfb_report_only_initial() {
        let mut m = make_metrics();
        m.set_report_only_initial_ttfb(true);

        // First measurement
        m.start_ttfb();
        let first = m.stop_ttfb();
        assert!(first.is_some());

        // Second measurement — should return None
        m.start_ttfb();
        let second = m.stop_ttfb();
        assert!(second.is_none());
    }

    #[test]
    fn ttfb_reports_every_time_by_default() {
        let mut m = make_metrics();

        m.start_ttfb();
        assert!(m.stop_ttfb().is_some());

        m.start_ttfb();
        assert!(m.stop_ttfb().is_some());

        m.start_ttfb();
        assert!(m.stop_ttfb().is_some());
    }

    #[test]
    fn processing_start_stop() {
        let mut m = make_metrics();
        m.start_processing();
        thread::sleep(Duration::from_millis(5));
        let data = m.stop_processing();

        assert!(data.is_some());
        match data.unwrap() {
            MetricsData::Processing {
                processor,
                value_secs,
                ..
            } => {
                assert_eq!(processor, "test_service");
                assert!(value_secs > 0.0);
            }
            other => panic!("expected Processing, got {other:?}"),
        }
    }

    #[test]
    fn processing_stop_without_start_returns_none() {
        let mut m = make_metrics();
        assert!(m.stop_processing().is_none());
    }

    #[test]
    fn text_aggregation_start_stop() {
        let mut m = make_metrics();
        m.start_text_aggregation();
        thread::sleep(Duration::from_millis(5));
        let data = m.stop_text_aggregation();

        assert!(data.is_some());
        match data.unwrap() {
            MetricsData::TextAggregation {
                processor,
                value_secs,
                ..
            } => {
                assert_eq!(processor, "test_service");
                assert!(value_secs > 0.0);
            }
            other => panic!("expected TextAggregation, got {other:?}"),
        }
    }

    #[test]
    fn text_aggregation_stop_without_start_returns_none() {
        let mut m = make_metrics();
        assert!(m.stop_text_aggregation().is_none());
    }

    #[test]
    fn llm_usage_records_tokens() {
        let m = make_metrics();
        let data = m.record_llm_usage(LlmTokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cache_read_input_tokens: Some(20),
            cache_creation_input_tokens: None,
            reasoning_tokens: Some(10),
        });

        match data {
            MetricsData::LlmUsage {
                processor,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cache_read_input_tokens,
                cache_creation_input_tokens,
            } => {
                assert_eq!(processor, "test_service");
                assert_eq!(model.as_deref(), Some("gpt-4"));
                assert_eq!(prompt_tokens, 100);
                assert_eq!(completion_tokens, 50);
                assert_eq!(total_tokens, 150);
                assert_eq!(cache_read_input_tokens, Some(20));
                assert_eq!(cache_creation_input_tokens, None);
            }
            other => panic!("expected LlmUsage, got {other:?}"),
        }
    }

    #[test]
    fn tts_usage_counts_characters() {
        let m = make_metrics();

        let data = m.record_tts_usage("Hello, world!");
        match data {
            MetricsData::TtsUsage { characters, .. } => {
                assert_eq!(characters, 13);
            }
            other => panic!("expected TtsUsage, got {other:?}"),
        }
    }

    #[test]
    fn tts_usage_counts_multibyte_characters() {
        let m = make_metrics();

        // Multi-byte: "Héllo 世界" = 8 characters (not 12 bytes)
        let data = m.record_tts_usage("Héllo 世界");
        match data {
            MetricsData::TtsUsage { characters, .. } => {
                assert_eq!(characters, 8);
            }
            other => panic!("expected TtsUsage, got {other:?}"),
        }
    }

    #[test]
    fn turn_records_correctly() {
        let m = make_metrics();
        let data = m.record_turn(true, 0.95, 123.4);

        match data {
            MetricsData::Turn {
                processor,
                is_complete,
                probability,
                e2e_processing_time_ms,
                ..
            } => {
                assert_eq!(processor, "test_service");
                assert!(is_complete);
                assert!((probability - 0.95).abs() < f64::EPSILON);
                assert!((e2e_processing_time_ms - 123.4).abs() < f64::EPSILON);
            }
            other => panic!("expected Turn, got {other:?}"),
        }
    }

    #[test]
    fn model_name_propagates() {
        let m = ProcessorMetrics::new("svc", Some("claude-4".into()));

        let data = m.record_tts_usage("hi");
        match data {
            MetricsData::TtsUsage { model, .. } => {
                assert_eq!(model.as_deref(), Some("claude-4"));
            }
            other => panic!("expected TtsUsage, got {other:?}"),
        }
    }

    #[test]
    fn no_model_is_none() {
        let m = ProcessorMetrics::new("svc", None);

        let data = m.record_tts_usage("hi");
        match data {
            MetricsData::TtsUsage { model, .. } => {
                assert!(model.is_none());
            }
            other => panic!("expected TtsUsage, got {other:?}"),
        }
    }

    #[test]
    fn set_model_after_creation() {
        let mut m = ProcessorMetrics::new("svc", None);
        m.set_model("new-model");

        let data = m.record_tts_usage("hi");
        match data {
            MetricsData::TtsUsage { model, .. } => {
                assert_eq!(model.as_deref(), Some("new-model"));
            }
            other => panic!("expected TtsUsage, got {other:?}"),
        }
    }

    #[test]
    fn reset_clears_all_timers() {
        let mut m = make_metrics();
        m.start_ttfb();
        m.start_processing();
        m.start_text_aggregation();

        m.reset();

        assert!(m.stop_ttfb().is_none());
        assert!(m.stop_processing().is_none());
        assert!(m.stop_text_aggregation().is_none());
    }

    #[test]
    fn custom_start_time() {
        let mut m = make_metrics();
        let earlier = Instant::now();
        thread::sleep(Duration::from_millis(10));
        m.start_ttfb_at(earlier);
        let data = m.stop_ttfb().unwrap();

        match data {
            MetricsData::Ttfb { value_secs, .. } => {
                assert!(
                    value_secs >= 0.01,
                    "should include sleep time: {value_secs}"
                );
            }
            other => panic!("expected Ttfb, got {other:?}"),
        }
    }
}
