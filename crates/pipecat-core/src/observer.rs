use std::time::Instant;

use async_trait::async_trait;

use crate::frame::{Direction, Frame};

// ---------------------------------------------------------------------------
// Observer event types
// ---------------------------------------------------------------------------

/// Event fired when a frame is about to be processed by a processor.
#[derive(Debug)]
pub struct FrameProcessedEvent<'a> {
    pub processor_name: &'a str,
    pub processor_id: u64,
    pub frame: &'a Frame,
    pub frame_id: u64,
    pub direction: Direction,
    pub timestamp: Instant,
}

/// Event fired when a frame is pushed from one processor toward another.
#[derive(Debug)]
pub struct FramePushedEvent<'a> {
    pub source_name: &'a str,
    pub source_id: u64,
    pub destination_name: Option<&'a str>,
    pub frame: &'a Frame,
    pub frame_id: u64,
    pub direction: Direction,
    pub timestamp: Instant,
}

// ---------------------------------------------------------------------------
// PipelineObserver trait
// ---------------------------------------------------------------------------

/// Observer for pipeline-level events. Implement this to monitor frame flow,
/// measure latency, collect metrics, or trace pipeline behavior.
///
/// All methods have default no-op implementations — override only the events
/// you care about.
///
/// # Thread safety
///
/// Observers are shared across all processors in a pipeline via `Arc<dyn PipelineObserver>`.
/// Methods take `&self`, so use interior mutability (`Mutex`, `AtomicU64`, etc.) for
/// mutable state.
///
/// # Performance
///
/// Observer methods are awaited inline before/after frame processing. Keep them
/// fast — do not perform I/O or expensive computation. If you need async work,
/// buffer events and process them in a background task.
///
/// # Frame references
///
/// Event types carry `&Frame` references. The frame is borrowed for the duration
/// of the observer call. If you need to store frame data across calls (e.g.,
/// accumulating metrics), extract and clone the specific data you need.
#[async_trait]
pub trait PipelineObserver: Send + Sync {
    /// Called when a frame is about to be dispatched to a processor's `process_frame`.
    async fn on_process_frame(&self, _event: FrameProcessedEvent<'_>) {}

    /// Called when a frame is pushed from a processor toward downstream or upstream.
    async fn on_push_frame(&self, _event: FramePushedEvent<'_>) {}

    /// Called after StartFrame has been fully dispatched to the processor.
    async fn on_pipeline_started(&self) {}
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::frame::*;

    /// Recorded push event for test assertions.
    #[derive(Debug)]
    struct RecordedPush {
        source_name: String,
        frame_name: String,
        direction: Direction,
    }

    /// Mock observer that records event counts, frame names, and directions.
    struct TestObserver {
        process_count: AtomicU32,
        push_count: AtomicU32,
        started_count: AtomicU32,
        process_frames: Mutex<Vec<String>>,
        push_events: Mutex<Vec<RecordedPush>>,
    }

    impl TestObserver {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                process_count: AtomicU32::new(0),
                push_count: AtomicU32::new(0),
                started_count: AtomicU32::new(0),
                process_frames: Mutex::new(Vec::new()),
                push_events: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl PipelineObserver for TestObserver {
        async fn on_process_frame(&self, event: FrameProcessedEvent<'_>) {
            self.process_count.fetch_add(1, Ordering::SeqCst);
            self.process_frames
                .lock()
                .unwrap()
                .push(format!("{}", event.frame));
        }

        async fn on_push_frame(&self, event: FramePushedEvent<'_>) {
            self.push_count.fetch_add(1, Ordering::SeqCst);
            self.push_events.lock().unwrap().push(RecordedPush {
                source_name: event.source_name.to_string(),
                frame_name: format!("{}", event.frame),
                direction: event.direction,
            });
        }

        async fn on_pipeline_started(&self) {
            self.started_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Observer with no overrides — default no-ops should not panic.
    struct NoopObserver;

    #[async_trait]
    impl PipelineObserver for NoopObserver {}

    #[tokio::test]
    async fn noop_observer_does_not_panic() {
        let obs = NoopObserver;
        obs.on_process_frame(FrameProcessedEvent {
            processor_name: "test",
            processor_id: 1,
            frame: &Frame::Interruption(InterruptionFrame),
            frame_id: 0,
            direction: Direction::Downstream,
            timestamp: Instant::now(),
        })
        .await;

        obs.on_push_frame(FramePushedEvent {
            source_name: "test",
            source_id: 1,
            destination_name: None,
            frame: &Frame::Interruption(InterruptionFrame),
            frame_id: 0,
            direction: Direction::Downstream,
            timestamp: Instant::now(),
        })
        .await;

        obs.on_pipeline_started().await;
    }

    #[tokio::test]
    async fn test_observer_records_process_events() {
        let obs = TestObserver::new();

        obs.on_process_frame(FrameProcessedEvent {
            processor_name: "proc1",
            processor_id: 1,
            frame: &Frame::Text(TextFrame::new("hello")),
            frame_id: 1,
            direction: Direction::Downstream,
            timestamp: Instant::now(),
        })
        .await;

        obs.on_process_frame(FrameProcessedEvent {
            processor_name: "proc1",
            processor_id: 1,
            frame: &Frame::Interruption(InterruptionFrame),
            frame_id: 2,
            direction: Direction::Downstream,
            timestamp: Instant::now(),
        })
        .await;

        assert_eq!(obs.process_count.load(Ordering::SeqCst), 2);
        let frames = obs.process_frames.lock().unwrap();
        assert_eq!(frames[0], "Text");
        assert_eq!(frames[1], "Interruption");
    }

    #[tokio::test]
    async fn test_observer_records_push_events() {
        let obs = TestObserver::new();

        obs.on_push_frame(FramePushedEvent {
            source_name: "proc1",
            source_id: 1,
            destination_name: None,
            frame: &Frame::Text(TextFrame::new("hello")),
            frame_id: 3,
            direction: Direction::Downstream,
            timestamp: Instant::now(),
        })
        .await;

        assert_eq!(obs.push_count.load(Ordering::SeqCst), 1);
        let pushes = obs.push_events.lock().unwrap();
        assert_eq!(pushes[0].source_name, "proc1");
        assert_eq!(pushes[0].frame_name, "Text");
        assert_eq!(pushes[0].direction, Direction::Downstream);
    }

    #[tokio::test]
    async fn test_observer_records_pipeline_started() {
        let obs = TestObserver::new();
        obs.on_pipeline_started().await;
        assert_eq!(obs.started_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn observer_can_inspect_metrics_data() {
        /// Observer that extracts TTFB values from MetricsFrame.
        struct MetricsInspector {
            ttfb_values: Mutex<Vec<f64>>,
        }

        #[async_trait]
        impl PipelineObserver for MetricsInspector {
            async fn on_push_frame(&self, event: FramePushedEvent<'_>) {
                if let Frame::Metrics(m) = event.frame {
                    for data in &m.data {
                        if let MetricsData::Ttfb { value_secs, .. } = data {
                            self.ttfb_values.lock().unwrap().push(*value_secs);
                        }
                    }
                }
            }
        }

        let obs = MetricsInspector {
            ttfb_values: Mutex::new(Vec::new()),
        };

        let metrics_frame = Frame::Metrics(MetricsFrame {
            data: vec![MetricsData::Ttfb {
                processor: "llm".into(),
                model: Some("gpt-4".into()),
                value_secs: 0.42,
            }],
        });

        obs.on_push_frame(FramePushedEvent {
            source_name: "llm_service",
            source_id: 5,
            destination_name: None,
            frame: &metrics_frame,
            frame_id: 4,
            direction: Direction::Downstream,
            timestamp: Instant::now(),
        })
        .await;

        let values = obs.ttfb_values.lock().unwrap();
        assert_eq!(values.len(), 1);
        assert!((values[0] - 0.42).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn observer_records_direction() {
        let obs = TestObserver::new();

        obs.on_push_frame(FramePushedEvent {
            source_name: "proc1",
            source_id: 1,
            destination_name: None,
            frame: &Frame::Text(TextFrame::new("up")),
            frame_id: 5,
            direction: Direction::Upstream,
            timestamp: Instant::now(),
        })
        .await;

        let pushes = obs.push_events.lock().unwrap();
        assert_eq!(pushes[0].direction, Direction::Upstream);
    }

    #[tokio::test]
    async fn observer_behind_arc_works() {
        let obs: Arc<dyn PipelineObserver> = TestObserver::new();

        obs.on_process_frame(FrameProcessedEvent {
            processor_name: "test",
            processor_id: 1,
            frame: &Frame::Start(StartFrame::default()),
            frame_id: 6,
            direction: Direction::Downstream,
            timestamp: Instant::now(),
        })
        .await;

        obs.on_pipeline_started().await;
        // No panic — Arc<dyn PipelineObserver> works
    }
}
