use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::sync::Mutex;

use pipecat_core::frame::{Direction, Frame, MetricsData};
use pipecat_core::observer::{FramePushedEvent, PipelineObserver};

// ---------------------------------------------------------------------------
// Breakdown types
// ---------------------------------------------------------------------------

/// TTFB measurement with timestamp for timeline placement.
#[derive(Debug, Clone)]
pub struct TTFBBreakdownMetrics {
    pub processor: String,
    pub model: Option<String>,
    /// Unix timestamp (seconds) when the TTFB measurement started.
    pub start_time: f64,
    pub duration_secs: f64,
}

/// Text aggregation measurement with timestamp for timeline placement.
#[derive(Debug, Clone)]
pub struct TextAggregationBreakdownMetrics {
    pub processor: String,
    /// Unix timestamp (seconds) when text aggregation started.
    pub start_time: f64,
    pub duration_secs: f64,
}

/// Latency for a single function call execution.
#[derive(Debug, Clone)]
pub struct FunctionCallMetrics {
    pub function_name: String,
    /// Unix timestamp (seconds) when execution started.
    pub start_time: f64,
    pub duration_secs: f64,
}

/// Per-service latency breakdown for a single user-to-bot cycle.
///
/// Collected between `VADUserStoppedSpeakingFrame` and `BotStartedSpeakingFrame`
/// when `enable_metrics=true` in pipeline params.
#[derive(Debug, Clone, Default)]
pub struct LatencyBreakdown {
    /// Time-to-first-byte metrics from each service in the pipeline.
    pub ttfb: Vec<TTFBBreakdownMetrics>,
    /// First text aggregation measurement — the latency cost of sentence
    /// aggregation in the TTS pipeline.
    pub text_aggregation: Option<TextAggregationBreakdownMetrics>,
    /// Unix timestamp when the user actually stopped speaking (adjusted for
    /// VAD `stop_secs`). `None` if no `VADUserStoppedSpeakingFrame` was observed.
    pub user_turn_start_time: Option<f64>,
    /// Duration from when the user stopped speaking to when the turn was released
    /// (`UserStoppedSpeakingFrame`). Includes VAD silence detection, STT
    /// finalization, and any turn analyzer wait.
    pub user_turn_secs: Option<f64>,
    /// Latency for each function call executed during this cycle.
    pub function_calls: Vec<FunctionCallMetrics>,
}

impl LatencyBreakdown {
    /// Return human-readable event labels sorted chronologically by start time.
    pub fn chronological_events(&self) -> Vec<String> {
        let mut events: Vec<(f64, String)> = Vec::new();

        if let (Some(start), Some(secs)) = (self.user_turn_start_time, self.user_turn_secs) {
            events.push((start, format!("User turn: {secs:.3}s")));
        }

        for t in &self.ttfb {
            events.push((
                t.start_time,
                format!("{}: TTFB {:.3}s", t.processor, t.duration_secs),
            ));
        }

        for fc in &self.function_calls {
            events.push((
                fc.start_time,
                format!("{}: {:.3}s", fc.function_name, fc.duration_secs),
            ));
        }

        if let Some(ref ta) = self.text_aggregation {
            events.push((
                ta.start_time,
                format!(
                    "{}: text aggregation {:.3}s",
                    ta.processor, ta.duration_secs
                ),
            ));
        }

        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        events.into_iter().map(|(_, label)| label).collect()
    }
}

// ---------------------------------------------------------------------------
// Handler trait
// ---------------------------------------------------------------------------

/// Callback trait for latency events. Implement this to log, trace, or collect
/// latency data from the observer.
///
/// All methods have default no-op implementations — override only what you need.
#[async_trait]
pub trait UserBotLatencyHandler: Send + Sync {
    /// Called when user-to-bot latency is measured (user stop → bot start).
    async fn on_latency_measured(&self, _latency_secs: f64) {}

    /// Called with a per-service latency breakdown at each measurement point.
    async fn on_latency_breakdown(&self, _breakdown: LatencyBreakdown) {}

    /// Called once: time from client connection to first bot speech.
    async fn on_first_bot_speech_latency(&self, _latency_secs: f64) {}
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct LatencyState {
    /// Wall-clock time when the user actually stopped speaking.
    user_stopped_time: Option<f64>,
    /// Wall-clock time for the start of the user turn (same as user_stopped_time).
    user_turn_start_time: Option<f64>,
    /// Duration from user silence to turn release.
    user_turn: Option<f64>,

    /// Wall-clock time of client connection (first occurrence).
    client_connected_time: Option<f64>,
    /// Whether the first-bot-speech measurement has been made (or abandoned).
    first_bot_speech_measured: bool,

    // Frame deduplication (bounded VecDeque + HashSet, matching Python's
    // deque(maxlen=N) + set pattern)
    processed_frames: HashSet<u64>,
    frame_history: VecDeque<u64>,
    max_frames: usize,

    // Per-cycle metric accumulators
    ttfb: Vec<TTFBBreakdownMetrics>,
    text_aggregation: Option<TextAggregationBreakdownMetrics>,
    function_call_starts: HashMap<String, (String, f64)>,
    function_call_metrics: Vec<FunctionCallMetrics>,
}

impl LatencyState {
    fn new(max_frames: usize) -> Self {
        Self {
            user_stopped_time: None,
            user_turn_start_time: None,
            user_turn: None,
            client_connected_time: None,
            first_bot_speech_measured: false,
            processed_frames: HashSet::with_capacity(max_frames),
            frame_history: VecDeque::with_capacity(max_frames),
            max_frames,
            ttfb: Vec::new(),
            text_aggregation: None,
            function_call_starts: HashMap::new(),
            function_call_metrics: Vec::new(),
        }
    }

    fn reset_accumulators(&mut self) {
        self.ttfb.clear();
        self.text_aggregation = None;
        self.user_turn_start_time = None;
        self.user_turn = None;
        self.function_call_starts.clear();
        self.function_call_metrics.clear();
    }
}

// ---------------------------------------------------------------------------
// Observer
// ---------------------------------------------------------------------------

/// Observer that tracks user-to-bot response latency.
///
/// Measures the time between when a user stops speaking
/// (`VADUserStoppedSpeakingFrame`) and when the bot starts speaking
/// (`BotStartedSpeakingFrame`). Optionally collects per-service latency
/// breakdown metrics when `enable_metrics=true`.
///
/// # Events
///
/// - `on_latency_measured`: Fires with the user→bot latency in seconds.
/// - `on_latency_breakdown`: Fires with a [`LatencyBreakdown`] containing
///   per-service TTFB, text aggregation, user turn duration, and function
///   call metrics collected during the cycle.
/// - `on_first_bot_speech_latency`: Fires once with the time from client
///   connection to first bot speech (greeting latency).
///
/// # Usage
///
/// ```ignore
/// let handler = Arc::new(MyHandler);
/// let observer = UserBotLatencyObserver::new(handler);
/// let task = PipelineTask::new(pipeline, params)
///     .with_observer(Arc::new(observer));
/// ```
pub struct UserBotLatencyObserver {
    state: Mutex<LatencyState>,
    handler: Arc<dyn UserBotLatencyHandler>,
    max_frames: usize,
}

impl std::fmt::Debug for UserBotLatencyObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserBotLatencyObserver")
            .field("max_frames", &self.max_frames)
            .finish()
    }
}

impl UserBotLatencyObserver {
    /// Create a new latency observer with the given event handler.
    pub fn new(handler: Arc<dyn UserBotLatencyHandler>) -> Self {
        let max_frames = 100;
        Self {
            state: Mutex::new(LatencyState::new(max_frames)),
            handler,
            max_frames,
        }
    }

    /// Set the maximum number of frame IDs kept for deduplication.
    pub fn with_max_frames(mut self, max: usize) -> Self {
        self.max_frames = max;
        self.state = Mutex::new(LatencyState::new(max));
        self
    }
}

/// Returns the current wall-clock time as seconds since UNIX epoch.
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[async_trait]
impl PipelineObserver for UserBotLatencyObserver {
    async fn on_push_frame(&self, event: FramePushedEvent<'_>) {
        // Only process downstream frames
        if event.direction != Direction::Downstream {
            return;
        }

        let mut state = self.state.lock().await;

        // Frame deduplication (bounded deque + set, matching Python's
        // deque(maxlen=N) + set pattern). VecDeque doesn't auto-evict
        // like Python's deque, so we manually enforce the bound.
        if state.processed_frames.contains(&event.frame_id) {
            return;
        }
        state.processed_frames.insert(event.frame_id);
        state.frame_history.push_back(event.frame_id);
        if state.frame_history.len() > state.max_frames {
            // Evict oldest and rebuild set from current deque
            state.frame_history.pop_front();
            state.processed_frames = state.frame_history.iter().copied().collect();
        }

        match event.frame {
            Frame::ClientConnected(_) => {
                if state.client_connected_time.is_none() {
                    state.client_connected_time = Some(now_secs());
                }
            }

            Frame::VADUserStartedSpeaking(_) => {
                state.user_stopped_time = None;
                state.reset_accumulators();
                // If user speaks before the bot's first speech, abandon the
                // first-bot-speech measurement — it's only meaningful for greetings.
                state.first_bot_speech_measured = true;
            }

            Frame::VADUserStoppedSpeaking(f) => {
                // The actual time the user stopped speaking: VAD determination
                // time minus the stop_secs silence duration that had to elapse.
                let actual_stop = f.timestamp - f.stop_secs;
                state.user_stopped_time = Some(actual_stop);
                state.user_turn_start_time = Some(actual_stop);
            }

            Frame::UserStoppedSpeaking(_) => {
                // Measure user turn duration: from actual user silence to turn release.
                if let Some(stopped) = state.user_stopped_time {
                    state.user_turn = Some(now_secs() - stopped);
                }
            }

            Frame::Interruption(_) => {
                state.reset_accumulators();
            }

            Frame::FunctionCallInProgress(f) => {
                state.function_call_starts.insert(
                    f.tool_call_id.clone(),
                    (f.function_name.clone(), now_secs()),
                );
            }

            Frame::FunctionCallResult(f) => {
                if let Some((function_name, start_time)) =
                    state.function_call_starts.remove(&f.tool_call_id)
                {
                    state.function_call_metrics.push(FunctionCallMetrics {
                        function_name,
                        start_time,
                        duration_secs: now_secs() - start_time,
                    });
                }
            }

            Frame::Metrics(m) => {
                // Only accumulate during an active measurement cycle
                let waiting_for_first_speech =
                    state.client_connected_time.is_some() && !state.first_bot_speech_measured;
                if state.user_stopped_time.is_none() && !waiting_for_first_speech {
                    return;
                }

                let now = now_secs();
                for data in &m.data {
                    match data {
                        MetricsData::Ttfb {
                            processor,
                            model,
                            value_secs,
                        } if *value_secs > 0.0 => {
                            state.ttfb.push(TTFBBreakdownMetrics {
                                processor: processor.clone(),
                                model: model.clone(),
                                start_time: now - value_secs,
                                duration_secs: *value_secs,
                            });
                        }
                        MetricsData::TextAggregation {
                            processor,
                            value_secs,
                            ..
                        } => {
                            // Keep first measurement only
                            if state.text_aggregation.is_none() {
                                state.text_aggregation = Some(TextAggregationBreakdownMetrics {
                                    processor: processor.clone(),
                                    start_time: now - value_secs,
                                    duration_secs: *value_secs,
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }

            Frame::BotStartedSpeaking(_) => {
                // Release the lock before calling handler methods
                let (first_speech_latency, user_latency, breakdown) =
                    Self::compute_results(&mut state);
                drop(state);

                if let Some(latency) = first_speech_latency {
                    self.handler.on_first_bot_speech_latency(latency).await;
                }
                if let Some(latency) = user_latency {
                    self.handler.on_latency_measured(latency).await;
                }
                if let Some(bd) = breakdown {
                    self.handler.on_latency_breakdown(bd).await;
                }
                return;
            }

            _ => {}
        }
    }
}

impl UserBotLatencyObserver {
    /// Compute latency results and reset state. Returns:
    /// (first_speech_latency, user_latency, breakdown)
    fn compute_results(
        state: &mut LatencyState,
    ) -> (Option<f64>, Option<f64>, Option<LatencyBreakdown>) {
        let now = now_secs();
        let mut emit_breakdown = false;

        // One-time first bot speech measurement (client connect → first speech)
        let first_speech_latency =
            if state.client_connected_time.is_some() && !state.first_bot_speech_measured {
                state.first_bot_speech_measured = true;
                let latency = now - state.client_connected_time.unwrap();
                emit_breakdown = true;
                Some(latency)
            } else {
                None
            };

        // User-to-bot latency
        let user_latency = if let Some(stopped) = state.user_stopped_time.take() {
            let latency = now - stopped;
            emit_breakdown = true;
            Some(latency)
        } else {
            None
        };

        let breakdown = if emit_breakdown {
            let bd = LatencyBreakdown {
                ttfb: state.ttfb.clone(),
                text_aggregation: state.text_aggregation.clone(),
                user_turn_start_time: state.user_turn_start_time,
                user_turn_secs: state.user_turn,
                function_calls: state.function_call_metrics.clone(),
            };
            state.reset_accumulators();
            Some(bd)
        } else {
            None
        };

        (first_speech_latency, user_latency, breakdown)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use pipecat_core::frame::*;

    /// Test handler that records all events.
    struct TestHandler {
        latencies: Mutex<Vec<f64>>,
        breakdowns: Mutex<Vec<LatencyBreakdown>>,
        first_speech: Mutex<Vec<f64>>,
    }

    impl TestHandler {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                latencies: Mutex::new(Vec::new()),
                breakdowns: Mutex::new(Vec::new()),
                first_speech: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl UserBotLatencyHandler for TestHandler {
        async fn on_latency_measured(&self, latency_secs: f64) {
            self.latencies.lock().await.push(latency_secs);
        }

        async fn on_latency_breakdown(&self, breakdown: LatencyBreakdown) {
            self.breakdowns.lock().await.push(breakdown);
        }

        async fn on_first_bot_speech_latency(&self, latency_secs: f64) {
            self.first_speech.lock().await.push(latency_secs);
        }
    }

    /// Create a push event for a frame with a given ID and downstream direction.
    fn push_event(frame: &Frame, frame_id: u64) -> FramePushedEvent<'_> {
        FramePushedEvent {
            source_name: "test",
            source_id: 1,
            destination_name: None,
            frame,
            frame_id,
            direction: Direction::Downstream,
            timestamp: Instant::now(),
        }
    }

    /// Create a push event with upstream direction.
    fn push_event_upstream(frame: &Frame, frame_id: u64) -> FramePushedEvent<'_> {
        FramePushedEvent {
            source_name: "test",
            source_id: 1,
            destination_name: None,
            frame,
            frame_id,
            direction: Direction::Upstream,
            timestamp: Instant::now(),
        }
    }

    #[tokio::test]
    async fn basic_latency_measurement() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 2)).await;

        let latencies = handler.latencies.lock().await;
        assert_eq!(latencies.len(), 1);
        assert!(latencies[0] > 0.0);
        assert!(latencies[0] < 1.0); // sanity

        let breakdowns = handler.breakdowns.lock().await;
        assert_eq!(breakdowns.len(), 1);
    }

    #[tokio::test]
    async fn breakdown_with_ttfb_metrics() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        let metrics = Frame::Metrics(MetricsFrame {
            data: vec![MetricsData::Ttfb {
                processor: "llm".into(),
                model: Some("gpt-4".into()),
                value_secs: 0.25,
            }],
        });
        obs.on_push_frame(push_event(&metrics, 2)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 3)).await;

        let breakdowns = handler.breakdowns.lock().await;
        assert_eq!(breakdowns.len(), 1);
        assert_eq!(breakdowns[0].ttfb.len(), 1);
        assert_eq!(breakdowns[0].ttfb[0].processor, "llm");
        assert!((breakdowns[0].ttfb[0].duration_secs - 0.25).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn first_bot_speech_latency() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let connected = Frame::ClientConnected(ClientConnectedFrame);
        obs.on_push_frame(push_event(&connected, 1)).await;

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 2)).await;

        let first = handler.first_speech.lock().await;
        assert_eq!(first.len(), 1);
        assert!(first[0] > 0.0);
        assert!(first[0] < 1.0);

        // Latency should NOT fire (no VADUserStoppedSpeaking)
        let latencies = handler.latencies.lock().await;
        assert!(latencies.is_empty());
    }

    #[tokio::test]
    async fn first_bot_speech_abandoned_when_user_speaks() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let connected = Frame::ClientConnected(ClientConnectedFrame);
        obs.on_push_frame(push_event(&connected, 1)).await;

        // User speaks before bot → abandon first-bot-speech measurement
        let user_start = Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
            start_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&user_start, 2)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 3)).await;

        let first = handler.first_speech.lock().await;
        assert!(first.is_empty()); // abandoned
    }

    #[tokio::test]
    async fn frame_dedup_prevents_double_counting() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        let metrics = Frame::Metrics(MetricsFrame {
            data: vec![MetricsData::Ttfb {
                processor: "llm".into(),
                model: None,
                value_secs: 0.1,
            }],
        });
        // Same frame_id pushed twice (simulating multiple hops)
        obs.on_push_frame(push_event(&metrics, 2)).await;
        obs.on_push_frame(push_event(&metrics, 2)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 3)).await;

        let breakdowns = handler.breakdowns.lock().await;
        assert_eq!(breakdowns[0].ttfb.len(), 1); // not 2
    }

    #[tokio::test]
    async fn user_turn_measurement() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let user_stop = Frame::UserStoppedSpeaking(UserStoppedSpeakingFrame);
        obs.on_push_frame(push_event(&user_stop, 2)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 3)).await;

        let breakdowns = handler.breakdowns.lock().await;
        assert!(breakdowns[0].user_turn_secs.is_some());
        assert!(breakdowns[0].user_turn_secs.unwrap() > 0.0);
        assert!(breakdowns[0].user_turn_start_time.is_some());
    }

    #[tokio::test]
    async fn interruption_resets_accumulators() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        let metrics = Frame::Metrics(MetricsFrame {
            data: vec![MetricsData::Ttfb {
                processor: "llm".into(),
                model: None,
                value_secs: 0.1,
            }],
        });
        obs.on_push_frame(push_event(&metrics, 2)).await;

        // Interruption discards accumulated metrics
        let interruption = Frame::Interruption(InterruptionFrame);
        obs.on_push_frame(push_event(&interruption, 3)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 4)).await;

        // No latency should be measured (user_stopped_time was in the
        // accumulators but the interruption path doesn't clear it — checking
        // that bot_started still fires if user_stopped_time is set).
        // Actually, interruption resets accumulators but NOT user_stopped_time.
        // So latency IS measured but breakdown has empty TTFB.
        let latencies = handler.latencies.lock().await;
        assert_eq!(latencies.len(), 1);

        let breakdowns = handler.breakdowns.lock().await;
        assert!(breakdowns[0].ttfb.is_empty()); // cleared by interruption
    }

    #[tokio::test]
    async fn function_call_tracking() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        let fc_start = Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
            function_name: "get_weather".into(),
            tool_call_id: "call_123".into(),
            arguments: serde_json::json!({}),
            cancel_on_interruption: false,
        });
        obs.on_push_frame(push_event(&fc_start, 2)).await;

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let fc_result = Frame::FunctionCallResult(FunctionCallResultFrame {
            function_name: "get_weather".into(),
            tool_call_id: "call_123".into(),
            arguments: serde_json::json!({}),
            result: serde_json::json!({"temp": 72}),
            run_llm: None,
            properties: None,
        });
        obs.on_push_frame(push_event(&fc_result, 3)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 4)).await;

        let breakdowns = handler.breakdowns.lock().await;
        assert_eq!(breakdowns[0].function_calls.len(), 1);
        assert_eq!(breakdowns[0].function_calls[0].function_name, "get_weather");
        assert!(breakdowns[0].function_calls[0].duration_secs > 0.0);
    }

    #[tokio::test]
    async fn upstream_frames_ignored() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        // Upstream BotStartedSpeaking should be ignored
        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event_upstream(&bot_start, 2)).await;

        let latencies = handler.latencies.lock().await;
        assert!(latencies.is_empty());
    }

    #[tokio::test]
    async fn metrics_not_accumulated_without_active_cycle() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        // Push metrics without prior VADUserStoppedSpeaking
        let metrics = Frame::Metrics(MetricsFrame {
            data: vec![MetricsData::Ttfb {
                processor: "llm".into(),
                model: None,
                value_secs: 0.5,
            }],
        });
        obs.on_push_frame(push_event(&metrics, 1)).await;

        // Now start a real cycle
        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 2)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 3)).await;

        let breakdowns = handler.breakdowns.lock().await;
        assert!(breakdowns[0].ttfb.is_empty()); // metrics from before cycle are not accumulated
    }

    #[tokio::test]
    async fn vad_stop_secs_adjustment() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let now = now_secs();
        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.3,       // 300ms of silence had to pass
            timestamp: now + 0.3, // VAD determination time is 300ms after actual silence
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 2)).await;

        let breakdowns = handler.breakdowns.lock().await;
        // user_turn_start_time should be (now + 0.3) - 0.3 = now
        assert!(breakdowns[0].user_turn_start_time.is_some());
        let start = breakdowns[0].user_turn_start_time.unwrap();
        assert!((start - now).abs() < 0.05); // within 50ms tolerance
    }

    #[tokio::test]
    async fn text_aggregation_metrics_collected() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        let metrics = Frame::Metrics(MetricsFrame {
            data: vec![MetricsData::TextAggregation {
                processor: "tts".into(),
                model: None,
                value_secs: 0.05,
            }],
        });
        obs.on_push_frame(push_event(&metrics, 2)).await;

        // Second text aggregation should be ignored (first only)
        let metrics2 = Frame::Metrics(MetricsFrame {
            data: vec![MetricsData::TextAggregation {
                processor: "tts".into(),
                model: None,
                value_secs: 0.03,
            }],
        });
        obs.on_push_frame(push_event(&metrics2, 3)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 4)).await;

        let breakdowns = handler.breakdowns.lock().await;
        assert!(breakdowns[0].text_aggregation.is_some());
        let ta = breakdowns[0].text_aggregation.as_ref().unwrap();
        assert_eq!(ta.processor, "tts");
        assert!((ta.duration_secs - 0.05).abs() < f64::EPSILON); // first, not second
    }

    #[tokio::test]
    async fn multiple_turns_measure_independently() {
        let handler = TestHandler::new();
        let obs = UserBotLatencyObserver::new(handler.clone());

        // First turn
        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 2)).await;

        // Second turn — user speaks again
        let user_start = Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
            start_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&user_start, 3)).await;

        let vad_stop2 = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop2, 4)).await;

        let bot_start2 = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start2, 5)).await;

        let latencies = handler.latencies.lock().await;
        assert_eq!(latencies.len(), 2); // two independent measurements

        let breakdowns = handler.breakdowns.lock().await;
        assert_eq!(breakdowns.len(), 2);
    }

    #[tokio::test]
    async fn dedup_history_bounded() {
        let handler = TestHandler::new();
        // Use a very small max_frames to test bounding
        let obs = UserBotLatencyObserver::new(handler.clone()).with_max_frames(3);

        let vad_stop = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop, 1)).await;

        // Push 4 frames (exceeding max_frames=3), which should evict frame 1
        let text = Frame::Text(TextFrame::new("a"));
        obs.on_push_frame(push_event(&text, 2)).await;
        let text2 = Frame::Text(TextFrame::new("b"));
        obs.on_push_frame(push_event(&text2, 3)).await;
        let text3 = Frame::Text(TextFrame::new("c"));
        obs.on_push_frame(push_event(&text3, 4)).await;

        // Frame 1 should have been evicted, so pushing it again should succeed
        // (re-processing the VADUserStoppedSpeaking resets user_stopped_time)
        let vad_stop2 = Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now_secs(),
        });
        obs.on_push_frame(push_event(&vad_stop2, 1)).await; // same ID=1, but evicted

        let bot_start = Frame::BotStartedSpeaking(BotStartedSpeakingFrame);
        obs.on_push_frame(push_event(&bot_start, 5)).await;

        let latencies = handler.latencies.lock().await;
        assert_eq!(latencies.len(), 1);

        // Verify the internal state is bounded
        let state = obs.state.lock().await;
        assert!(state.frame_history.len() <= 3);
        assert!(state.processed_frames.len() <= 3);
    }

    #[test]
    fn chronological_events_sorted() {
        let breakdown = LatencyBreakdown {
            ttfb: vec![
                TTFBBreakdownMetrics {
                    processor: "llm".into(),
                    model: None,
                    start_time: 100.5,
                    duration_secs: 0.2,
                },
                TTFBBreakdownMetrics {
                    processor: "tts".into(),
                    model: None,
                    start_time: 100.8,
                    duration_secs: 0.1,
                },
            ],
            text_aggregation: Some(TextAggregationBreakdownMetrics {
                processor: "tts".into(),
                start_time: 100.7,
                duration_secs: 0.05,
            }),
            user_turn_start_time: Some(100.0),
            user_turn_secs: Some(0.4),
            function_calls: vec![],
        };

        let events = breakdown.chronological_events();
        assert_eq!(events.len(), 4);
        assert!(events[0].contains("User turn"));
        assert!(events[1].contains("llm"));
        assert!(events[2].contains("text aggregation"));
        assert!(events[3].contains("tts") && events[3].contains("TTFB"));
    }
}
