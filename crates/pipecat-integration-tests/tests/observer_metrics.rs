use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use pipecat_core::frame::*;
use pipecat_core::observer::*;
use pipecat_core::test_utils::*;
use pipecat_integration_tests::helpers::*;
use pipecat_integration_tests::mock_services::*;
use pipecat_pipeline::Pipeline;
use pipecat_services::observers::{UserBotLatencyHandler, UserBotLatencyObserver};
use serde_json::json;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Recording observer
// ---------------------------------------------------------------------------

/// Recorded process event for test assertions.
#[derive(Debug)]
struct RecordedProcess {
    processor_name: String,
    frame_name: String,
    direction: Direction,
}

/// Recorded push event for test assertions.
#[derive(Debug)]
#[allow(dead_code)]
struct RecordedPush {
    source_name: String,
    frame_name: String,
    direction: Direction,
    destination_name: Option<String>,
}

/// Observer that records all events for assertion in tests.
struct RecordingObserver {
    process_events: Mutex<Vec<RecordedProcess>>,
    push_events: Mutex<Vec<RecordedPush>>,
    process_count: AtomicU32,
    push_count: AtomicU32,
    started_count: AtomicU32,
}

impl RecordingObserver {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            process_events: Mutex::new(Vec::new()),
            push_events: Mutex::new(Vec::new()),
            process_count: AtomicU32::new(0),
            push_count: AtomicU32::new(0),
            started_count: AtomicU32::new(0),
        })
    }
}

#[async_trait]
impl PipelineObserver for RecordingObserver {
    async fn on_process_frame(&self, event: FrameProcessedEvent<'_>) {
        self.process_count.fetch_add(1, Ordering::SeqCst);
        self.process_events.lock().unwrap().push(RecordedProcess {
            processor_name: event.processor_name.to_string(),
            frame_name: format!("{}", event.frame),
            direction: event.direction,
        });
    }

    async fn on_push_frame(&self, event: FramePushedEvent<'_>) {
        self.push_count.fetch_add(1, Ordering::SeqCst);
        self.push_events.lock().unwrap().push(RecordedPush {
            source_name: event.source_name.to_string(),
            frame_name: format!("{}", event.frame),
            direction: event.direction,
            destination_name: event.destination_name.map(|s| s.to_string()),
        });
    }

    async fn on_pipeline_started(&self) {
        self.started_count.fetch_add(1, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Test 1: Observer on a single processor node
// ---------------------------------------------------------------------------

#[tokio::test]
async fn single_node_observer_records_process_and_push_events() {
    let obs = RecordingObserver::new();
    let (node, handle, _down_rx, _up_rx) =
        make_observed_node(Box::new(PassthroughProcessor::new()), obs.clone());

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("hello")),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("world")),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    // Verify on_process_frame events
    let process_events = obs.process_events.lock().unwrap();
    assert!(
        process_events.len() >= 4,
        "should have at least Start + 2*Text + Cancel process events, got {}",
        process_events.len()
    );

    let process_frame_names: Vec<&str> = process_events
        .iter()
        .map(|e| e.frame_name.as_str())
        .collect();
    assert!(
        process_frame_names.contains(&"Start"),
        "observer should see Start processed: {process_frame_names:?}"
    );
    assert!(
        process_frame_names.contains(&"Text"),
        "observer should see Text processed: {process_frame_names:?}"
    );
    assert!(
        process_frame_names.contains(&"Cancel"),
        "observer should see Cancel processed: {process_frame_names:?}"
    );

    // Verify all process events have the correct processor name
    for event in process_events.iter() {
        assert_eq!(event.processor_name, "Passthrough");
        assert_eq!(event.direction, Direction::Downstream);
    }

    // Verify on_push_frame events (Passthrough forwards everything)
    let push_events = obs.push_events.lock().unwrap();
    let push_frame_names: Vec<&str> = push_events.iter().map(|e| e.frame_name.as_str()).collect();
    assert!(
        push_frame_names.contains(&"Text"),
        "observer should see Text pushed: {push_frame_names:?}"
    );

    for event in push_events.iter() {
        assert_eq!(event.source_name, "Passthrough");
    }
}

// ---------------------------------------------------------------------------
// Test 2: Observer on an Uppercase processor
// ---------------------------------------------------------------------------

#[tokio::test]
async fn observer_sees_uppercase_transformation() {
    let obs = RecordingObserver::new();
    let (node, handle, mut down_rx, _up_rx) =
        make_observed_node(Box::new(UppercaseProcessor::new()), obs.clone());

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("hello")),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    // on_process_frame sees the original input
    {
        let process_events = obs.process_events.lock().unwrap();
        assert!(
            process_events.iter().any(|e| e.frame_name == "Text"),
            "should see Text in process events"
        );
        assert_eq!(
            process_events
                .iter()
                .filter(|e| e.processor_name == "Uppercase")
                .count(),
            process_events.len(),
            "all process events should be from Uppercase"
        );
    }

    // on_push_frame sees the transformed output
    {
        let push_events = obs.push_events.lock().unwrap();
        let text_pushes: Vec<&RecordedPush> = push_events
            .iter()
            .filter(|e| e.frame_name == "Text")
            .collect();
        assert!(
            !text_pushes.is_empty(),
            "should have at least one Text push event"
        );
    }

    // Verify the transformed text arrives downstream
    let frames = drain_rx(&mut down_rx).await;
    let text_frame = frames
        .iter()
        .find_map(|f| match &f.frame {
            Frame::Text(t) => Some(t),
            _ => None,
        })
        .expect("should have a text frame downstream");
    assert_eq!(text_frame.text, "HELLO");
}

// ---------------------------------------------------------------------------
// Test 3: Observer on ErrorOnText processor sees upstream error push
// ---------------------------------------------------------------------------

#[tokio::test]
async fn observer_records_error_pushed_upstream() {
    let obs = RecordingObserver::new();
    let (node, handle, _down_rx, _up_rx) =
        make_observed_node(Box::new(ErrorOnTextProcessor::new()), obs.clone());

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("trigger")),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let push_events = obs.push_events.lock().unwrap();

    // ErrorOnTextProcessor pushes Error upstream when it sees Text
    let error_pushes: Vec<&RecordedPush> = push_events
        .iter()
        .filter(|e| e.frame_name == "Error" && e.direction == Direction::Upstream)
        .collect();
    assert!(
        !error_pushes.is_empty(),
        "observer should see Error pushed upstream: {:?}",
        push_events
            .iter()
            .map(|e| (&e.frame_name, &e.direction))
            .collect::<Vec<_>>()
    );
    assert_eq!(error_pushes[0].source_name, "ErrorOnText");
}

// ---------------------------------------------------------------------------
// Test 4: on_pipeline_started fires after StartFrame (single node)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn observer_pipeline_started_fires() {
    // NOTE: This test uses a single ProcessorNode (not a Pipeline), so
    // on_pipeline_started fires exactly once. When using Pipeline.with_observer(),
    // the observer is wired to every internal ProcessorNode (PipelineSource +
    // user processors + PipelineSink). Each node that receives StartFrame calls
    // on_pipeline_started independently, resulting in N+2 calls for a Pipeline
    // with N user processors. See `pipeline_observer_started_fires_per_node`
    // below for that behavior.
    let obs = RecordingObserver::new();
    let (node, handle, _down_rx, _up_rx) =
        make_observed_node(Box::new(PassthroughProcessor::new()), obs.clone());

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    assert_eq!(
        obs.started_count.load(Ordering::SeqCst),
        1,
        "on_pipeline_started should fire exactly once on a single node after StartFrame"
    );
}

// ---------------------------------------------------------------------------
// Test 4b: on_pipeline_started fires N+2 times in a multi-processor Pipeline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pipeline_observer_started_fires_per_node() {
    // Pipeline.with_observer() wires the observer to ALL internal ProcessorNodes:
    // PipelineSource + user processors + PipelineSink. Each node independently
    // calls on_pipeline_started when it processes StartFrame, so for a Pipeline
    // with N user processors, we get N+2 on_pipeline_started calls.
    let obs = RecordingObserver::new();

    let pipeline = Pipeline::new(vec![
        Box::new(UppercaseProcessor::new()),
        Box::new(PassthroughProcessor::new()),
    ])
    .with_observer(obs.clone());

    let (node, handle, _down_rx, _up_rx) = make_node(Box::new(pipeline));
    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    // 2 user processors + PipelineSource + PipelineSink = 4 nodes, each fires
    // on_pipeline_started when it processes StartFrame.
    let started = obs.started_count.load(Ordering::SeqCst);
    assert_eq!(
        started, 4,
        "on_pipeline_started should fire N+2 times (PipelineSource + 2 user processors + PipelineSink), got {started}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Pipeline-level observer sees events from all processors
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pipeline_observer_sees_all_processors() {
    // Pipeline.with_observer() wires the observer to every internal ProcessorNode,
    // including PipelineSource and PipelineSink. Events from all of them appear
    // in the observer callbacks. Tests below filter by user processor names to
    // verify the user-visible processors are observed.
    let obs = RecordingObserver::new();

    let pipeline = Pipeline::new(vec![
        Box::new(UppercaseProcessor::new()),
        Box::new(PassthroughProcessor::new()),
    ])
    .with_observer(obs.clone());

    let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(pipeline));

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("test")),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    {
        let process_events = obs.process_events.lock().unwrap();
        let processor_names: Vec<&str> = process_events
            .iter()
            .map(|e| e.processor_name.as_str())
            .collect();

        // Both user processors in the pipeline should appear in observer events.
        // PipelineSource and PipelineSink also appear since the observer is wired
        // to all internal nodes.
        assert!(
            processor_names.contains(&"Uppercase"),
            "observer should see Uppercase processor: {processor_names:?}"
        );
        assert!(
            processor_names.contains(&"Passthrough"),
            "observer should see Passthrough processor: {processor_names:?}"
        );
    }

    // Verify the pipeline end result
    let frames = drain_rx(&mut down_rx).await;
    let text_frame = frames
        .iter()
        .find_map(|f| match &f.frame {
            Frame::Text(t) => Some(t),
            _ => None,
        })
        .expect("should receive text frame through pipeline");
    assert_eq!(text_frame.text, "TEST");
}

// ---------------------------------------------------------------------------
// Test 6: Observer event ordering within a pipeline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pipeline_observer_events_in_order() {
    let obs = RecordingObserver::new();

    let pipeline = Pipeline::new(vec![
        Box::new(UppercaseProcessor::new()),
        Box::new(PassthroughProcessor::new()),
    ])
    .with_observer(obs.clone());

    let (node, handle, _down_rx, _up_rx) = make_node(Box::new(pipeline));

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("order")),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    // A Pipeline internally wires processors with a PipelineSource at position 0
    // and a PipelineSink at the end. The observer is wired to all internal nodes,
    // so events come from PipelineSource, user processors, and PipelineSink.
    //
    // For a Text frame flowing downstream through [PipelineSource, Uppercase, Passthrough, PipelineSink]:
    // 1. PipelineSource processes Text, pushes downstream
    // 2. Uppercase processes Text, pushes transformed Text downstream
    // 3. Passthrough processes Text, pushes downstream
    // 4. PipelineSink processes Text, escapes to outer context
    //
    // Verify Uppercase process event comes before Passthrough for Text,
    // filtering out internal pipeline processors (PipelineSource/PipelineSink).
    let process_events = obs.process_events.lock().unwrap();
    let text_process_events: Vec<&RecordedProcess> = process_events
        .iter()
        .filter(|e| {
            e.frame_name == "Text"
                && (e.processor_name == "Uppercase" || e.processor_name == "Passthrough")
        })
        .collect();

    assert!(
        text_process_events.len() >= 2,
        "should have Text processed by both Uppercase and Passthrough, got {}: {:?}",
        text_process_events.len(),
        text_process_events
    );

    let first_text_processor = &text_process_events[0].processor_name;
    let second_text_processor = &text_process_events[1].processor_name;
    assert_eq!(
        first_text_processor, "Uppercase",
        "Uppercase should process Text first"
    );
    assert_eq!(
        second_text_processor, "Passthrough",
        "Passthrough should process Text second"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Shared observer across pipeline records correct counts
// ---------------------------------------------------------------------------

#[tokio::test]
async fn observer_counts_match_expected() {
    let obs = RecordingObserver::new();

    // Single Passthrough: every frame processed = one process event + one push event.
    // We use >= assertions because the framework may inject internal frames
    // (e.g., metrics) that we should tolerate without breaking the test.
    let (node, handle, _down_rx, _up_rx) =
        make_observed_node(Box::new(PassthroughProcessor::new()), obs.clone());

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send exactly 3 text frames
    for i in 0..3 {
        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new(format!("msg{i}"))),
            Direction::Downstream,
        )
        .await;
    }

    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    // Verify we see at least the expected frame types rather than asserting
    // exact counts, which would be brittle if the framework adds internal frames.
    let process_events = obs.process_events.lock().unwrap();

    let start_count = process_events
        .iter()
        .filter(|e| e.frame_name == "Start")
        .count();
    let text_count = process_events
        .iter()
        .filter(|e| e.frame_name == "Text")
        .count();
    let cancel_count = process_events
        .iter()
        .filter(|e| e.frame_name == "Cancel")
        .count();

    assert!(
        start_count >= 1,
        "should have at least 1 Start process event, got {start_count}"
    );
    assert!(
        text_count >= 3,
        "should have at least 3 Text process events, got {text_count}"
    );
    assert!(
        cancel_count >= 1,
        "should have at least 1 Cancel process event, got {cancel_count}"
    );

    // Total should be at least Start + 3 Text + Cancel = 5
    let process_count = obs.process_count.load(Ordering::SeqCst);
    assert!(
        process_count >= 5,
        "expected at least 5 process events (Start + 3 Text + Cancel), got {process_count}"
    );

    // Passthrough forwards everything, so push count should be at least as many
    let push_count = obs.push_count.load(Ordering::SeqCst);
    assert!(
        push_count >= 5,
        "expected at least 5 push events (Passthrough forwards all), got {push_count}"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Observer with cascade pipeline (STT -> LLM -> TTS)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn observer_on_cascade_pipeline() {
    // Pipeline.with_observer() wires the observer to all internal nodes:
    // PipelineSource, FakeSTT, FakeLLM, FakeTTS, and PipelineSink. Events from
    // all five nodes appear in the observer callbacks.
    let obs = RecordingObserver::new();

    let stt = FakeSTTService::new("hello", 1);
    let llm = FakeLLMService::new(vec!["Hi!".to_string()]);
    let tts = FakeTTSService::new();

    let pipeline =
        Pipeline::new(vec![Box::new(stt), Box::new(llm), Box::new(tts)]).with_observer(obs.clone());

    let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(pipeline));
    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send audio to trigger STT -> Transcription
    let samples = vec![0i16; 160];
    send_and_settle(
        &handle,
        Frame::InputAudioRaw(AudioRawFrame {
            audio: bytes::Bytes::from(
                samples
                    .iter()
                    .flat_map(|s| s.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            sample_rate: 16000,
            num_channels: 1,
        }),
        Direction::Downstream,
    )
    .await;

    // Also send LLMContext to trigger LLM -> TTS
    let ctx_frame = make_llm_context_frame(vec![json!({"role": "user", "content": "hello"})]);
    send_and_settle(&handle, ctx_frame.frame.clone(), Direction::Downstream).await;

    // Wait for pipeline propagation through all 3 services. The cascade
    // STT -> LLM -> TTS involves async processing across multiple nodes with
    // bridge tasks forwarding frames between them. 100ms is sufficient because
    // all services are fake (no I/O), and each hop is a tokio channel send.
    // The sleep ensures frames have propagated through the entire chain before
    // we send Cancel, which shuts down the pipeline.
    tokio::time::sleep(Duration::from_millis(100)).await;

    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    {
        let process_events = obs.process_events.lock().unwrap();

        // Verify all three user services appear in observer events.
        // PipelineSource and PipelineSink also appear since the observer is
        // wired to all internal nodes via Pipeline.with_observer().
        let processor_names: Vec<&str> = process_events
            .iter()
            .map(|e| e.processor_name.as_str())
            .collect();
        assert!(
            processor_names.contains(&"FakeSTT"),
            "observer should see FakeSTT: {processor_names:?}"
        );
        assert!(
            processor_names.contains(&"FakeLLM"),
            "observer should see FakeLLM: {processor_names:?}"
        );
        assert!(
            processor_names.contains(&"FakeTTS"),
            "observer should see FakeTTS: {processor_names:?}"
        );
    }

    {
        // Verify LLM-related frames appear in push events
        let push_events = obs.push_events.lock().unwrap();
        let push_frame_names: Vec<&str> =
            push_events.iter().map(|e| e.frame_name.as_str()).collect();
        assert!(
            push_frame_names.contains(&"LLMFullResponseStart"),
            "observer should see LLMFullResponseStart pushed: {push_frame_names:?}"
        );
        assert!(
            push_frame_names.contains(&"Text"),
            "observer should see Text pushed: {push_frame_names:?}"
        );
    }

    // Verify the pipeline actually produced output
    let frames = drain_rx(&mut down_rx).await;
    let names = frame_names(&frames);
    assert!(
        names.contains(&"TTSAudioRaw".to_string()),
        "cascade pipeline should produce TTS audio: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 9: UserBotLatencyObserver integration with pipeline
// ---------------------------------------------------------------------------

/// Test handler that records latency measurements.
struct TestLatencyHandler {
    latencies: tokio::sync::Mutex<Vec<f64>>,
    breakdowns: tokio::sync::Mutex<Vec<pipecat_services::observers::LatencyBreakdown>>,
}

impl TestLatencyHandler {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            latencies: tokio::sync::Mutex::new(Vec::new()),
            breakdowns: tokio::sync::Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl UserBotLatencyHandler for TestLatencyHandler {
    async fn on_latency_measured(&self, latency_secs: f64) {
        self.latencies.lock().await.push(latency_secs);
    }

    async fn on_latency_breakdown(&self, breakdown: pipecat_services::observers::LatencyBreakdown) {
        self.breakdowns.lock().await.push(breakdown);
    }
}

#[tokio::test]
async fn latency_observer_measures_user_to_bot_latency() {
    let handler = TestLatencyHandler::new();
    let latency_obs = Arc::new(UserBotLatencyObserver::new(handler.clone()));

    // Use a Passthrough so frames flow through to the latency observer's push handler.
    let (node, handle, _down_rx, _up_rx) =
        make_observed_node(Box::new(PassthroughProcessor::new()), latency_obs);

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Simulate VAD user stopped speaking
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    send_and_settle(
        &handle,
        Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.0,
            timestamp: now,
        }),
        Direction::Downstream,
    )
    .await;

    // Small delay to ensure measurable latency
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Simulate bot started speaking
    send_and_settle(
        &handle,
        Frame::BotStartedSpeaking(BotStartedSpeakingFrame),
        Direction::Downstream,
    )
    .await;

    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let latencies = handler.latencies.lock().await;
    assert_eq!(latencies.len(), 1, "should have one latency measurement");
    assert!(
        latencies[0] > 0.0,
        "latency should be positive: {}",
        latencies[0]
    );
    assert!(
        latencies[0] < 1.0,
        "latency should be under 1 second: {}",
        latencies[0]
    );

    let breakdowns = handler.breakdowns.lock().await;
    assert_eq!(breakdowns.len(), 1, "should have one breakdown");
}

// ---------------------------------------------------------------------------
// Test 10: Observer sees mixed directions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn observer_records_upstream_frames() {
    let obs = RecordingObserver::new();
    let (node, handle, _down_rx, _up_rx) =
        make_observed_node(Box::new(PassthroughProcessor::new()), obs.clone());

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send a frame upstream
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("upstream_msg")),
        Direction::Upstream,
    )
    .await;

    // Send a frame downstream
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("downstream_msg")),
        Direction::Downstream,
    )
    .await;

    send_and_settle(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let process_events = obs.process_events.lock().unwrap();
    let upstream_events: Vec<&RecordedProcess> = process_events
        .iter()
        .filter(|e| e.direction == Direction::Upstream)
        .collect();
    let downstream_events: Vec<&RecordedProcess> = process_events
        .iter()
        .filter(|e| e.direction == Direction::Downstream)
        .collect();

    assert!(
        !upstream_events.is_empty(),
        "should have upstream process events"
    );
    assert!(
        !downstream_events.is_empty(),
        "should have downstream process events"
    );

    // Push events should also reflect direction
    let push_events = obs.push_events.lock().unwrap();
    let upstream_pushes: Vec<&RecordedPush> = push_events
        .iter()
        .filter(|e| e.direction == Direction::Upstream)
        .collect();
    assert!(
        !upstream_pushes.is_empty(),
        "should have upstream push events"
    );
}
