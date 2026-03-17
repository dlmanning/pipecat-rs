use pipecat_core::frame::StartFrame;
use pipecat_core::metrics::ProcessorMetrics;
use pipecat_core::processor::ProcessorBase;

/// Common state shared by all service processors.
///
/// Combines `ProcessorBase` (identity) with `ProcessorMetrics` (timing/usage).
/// Each service type embeds a `ServiceBase`.
#[derive(Debug)]
pub struct ServiceBase {
    pub processor: ProcessorBase,
    pub metrics: ProcessorMetrics,
    metrics_enabled: bool,
    usage_metrics_enabled: bool,
}

impl ServiceBase {
    pub fn new(name: impl Into<String>, model: Option<String>) -> Self {
        let name = name.into();
        Self {
            metrics: ProcessorMetrics::new(&name, model),
            processor: ProcessorBase::new(name),
            metrics_enabled: false,
            usage_metrics_enabled: false,
        }
    }

    /// Handle a StartFrame: enable/disable metrics based on the frame's flags.
    pub fn handle_start_frame(&mut self, frame: &StartFrame) {
        self.metrics_enabled = frame.enable_metrics;
        self.usage_metrics_enabled = frame.enable_usage_metrics;
        self.metrics
            .set_report_only_initial_ttfb(frame.report_only_initial_ttfb);
    }

    /// Set the model name on the metrics helper.
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.metrics.set_model(model);
    }

    /// Whether metrics reporting is enabled.
    pub fn metrics_enabled(&self) -> bool {
        self.metrics_enabled
    }

    /// Whether usage metrics reporting is enabled.
    pub fn usage_metrics_enabled(&self) -> bool {
        self.usage_metrics_enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_with_model() {
        let base = ServiceBase::new("MyService", Some("gpt-4".into()));
        assert_eq!(base.processor.name(), "MyService");
        assert!(!base.metrics_enabled());
        assert!(!base.usage_metrics_enabled());
    }

    #[test]
    fn handle_start_frame() {
        let mut base = ServiceBase::new("svc", None);
        let start = StartFrame {
            enable_metrics: true,
            enable_usage_metrics: true,
            report_only_initial_ttfb: true,
            ..Default::default()
        };
        base.handle_start_frame(&start);
        assert!(base.metrics_enabled());
        assert!(base.usage_metrics_enabled());
    }
}
