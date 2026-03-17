use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::time::timeout;

use pipecat_core::frame::*;
use pipecat_core::metrics::ProcessorMetrics;
use pipecat_core::node::ProcessorNode;
use pipecat_core::observer::{FrameProcessedEvent, FramePushedEvent, PipelineObserver};
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_core::test_utils::*;

const TEST_TIMEOUT: Duration = Duration::from_secs(2);

// ---------------------------------------------------------------------------
// Test observer for integration tests
// ---------------------------------------------------------------------------

struct IntegrationObserver {
    process_frames: Mutex<Vec<String>>,
    push_frames: Mutex<Vec<String>>,
    started: std::sync::atomic::AtomicBool,
}

impl IntegrationObserver {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            process_frames: Mutex::new(Vec::new()),
            push_frames: Mutex::new(Vec::new()),
            started: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

#[async_trait]
impl PipelineObserver for IntegrationObserver {
    async fn on_process_frame(&self, event: FrameProcessedEvent<'_>) {
        self.process_frames
            .lock()
            .unwrap()
            .push(format!("{}", event.frame));
    }

    async fn on_push_frame(&self, event: FramePushedEvent<'_>) {
        self.push_frames
            .lock()
            .unwrap()
            .push(format!("{}", event.frame));
    }

    async fn on_pipeline_started(&self) {
        self.started.store(true, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Custom processor: emits a MetricsFrame using ProcessorMetrics
// ---------------------------------------------------------------------------

struct MetricsEmittingProcessor {
    base: ProcessorBase,
    metrics: ProcessorMetrics,
}

impl MetricsEmittingProcessor {
    fn new() -> Self {
        Self {
            base: ProcessorBase::new("MetricsEmitter"),
            metrics: ProcessorMetrics::new("MetricsEmitter", Some("test-model".into())),
        }
    }
}

#[async_trait]
impl FrameProcessor for MetricsEmittingProcessor {
    fn name(&self) -> &str {
        self.base.name()
    }
    fn id(&self) -> u64 {
        self.base.id()
    }
    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) {
        match &envelope.frame {
            Frame::Text(_) => {
                // Simulate TTFB: start when we get text, immediately stop
                self.metrics.start_ttfb();
                ctx.push_ttfb(&mut self.metrics).await.ok();
                // Forward the text too
                ctx.push_frame(envelope, direction).await.ok();
            }
            _ => {
                ctx.push_frame(envelope, direction).await.ok();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_lifecycle_through_node() {
    let (node, handle, mut down_rx, _up_rx) = make_node(Box::new(PassthroughProcessor::new()));

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    for i in 0..5 {
        send_and_settle(
            &handle,
            Frame::Text(TextFrame::new(format!("msg{i}"))),
            Direction::Downstream,
        )
        .await;
    }
    send_and_settle(
        &handle,
        Frame::End(EndFrame::default()),
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

    let frames = drain_rx(&mut down_rx).await;
    let n = frame_names(&frames);
    assert_eq!(
        n,
        vec![
            "Start", "Text", "Text", "Text", "Text", "Text", "End", "Cancel"
        ],
        "full lifecycle: {n:?}"
    );
}

#[tokio::test]
async fn multi_node_chain() {
    // Wire: Uppercase → Passthrough
    // Uppercase's downstream feeds into Passthrough's input.
    let (down_tx, mut final_rx) = mpsc::channel(64);
    let (up_tx, _up_rx) = mpsc::channel(64);

    // Create Passthrough node (second in chain)
    let (pass_node, pass_handle) =
        ProcessorNode::new(Box::new(PassthroughProcessor::new()), down_tx, up_tx, 64);

    // Create Uppercase node (first in chain)
    // Its downstream goes to the Passthrough node's handle
    let (mid_tx, mut mid_rx) = mpsc::channel(64);
    let (up_tx2, _up_rx2) = mpsc::channel(64);
    let (upper_node, upper_handle) =
        ProcessorNode::new(Box::new(UppercaseProcessor::new()), mid_tx, up_tx2, 64);

    // Spawn both nodes
    let pass_run = tokio::spawn(async move { pass_node.run().await });
    let upper_run = tokio::spawn(async move { upper_node.run().await });

    // Bridge: forward frames from uppercase's downstream to passthrough's input
    let bridge = tokio::spawn(async move {
        while let Some(env) = mid_rx.recv().await {
            let is_cancel = matches!(&env.frame, Frame::Cancel(_));
            pass_handle.send(env, Direction::Downstream).await.ok();
            if is_cancel {
                break;
            }
        }
    });

    // Send frames into uppercase node
    send_and_settle(
        &upper_handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &upper_handle,
        Frame::Text(TextFrame::new("hello world")),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &upper_handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, upper_run).await.unwrap().unwrap();
    timeout(TEST_TIMEOUT, bridge).await.unwrap().unwrap();
    timeout(TEST_TIMEOUT, pass_run).await.unwrap().unwrap();

    let frames = drain_rx(&mut final_rx).await;
    // Text should be uppercased by first node, passed through by second
    assert!(
        frames
            .iter()
            .any(|f| matches!(&f.frame, Frame::Text(t) if t.text == "HELLO WORLD")),
        "text should be uppercased through chain: {:?}",
        frame_names(&frames)
    );
}

#[tokio::test]
async fn interruption_with_observer() {
    let obs = IntegrationObserver::new();
    let (node, handle, _down_rx, _up_rx) =
        make_observed_node(Box::new(PassthroughProcessor::new()), obs.clone());

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Queue texts and interruption simultaneously
    for i in 0..5 {
        handle
            .send(
                FrameEnvelope::new(Frame::Text(TextFrame::new(format!("pre_{i}")))),
                Direction::Downstream,
            )
            .await
            .unwrap();
    }
    handle
        .send(
            FrameEnvelope::new(Frame::Interruption(InterruptionFrame)),
            Direction::Downstream,
        )
        .await
        .unwrap();

    tokio::time::sleep(SETTLE).await;

    // Post-interrupt text
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("post")),
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

    let processed = obs.process_frames.lock().unwrap();
    assert!(
        processed.contains(&"Interruption".to_string()),
        "observer should see Interruption: {:?}",
        *processed
    );
    assert!(
        processed.contains(&"Start".to_string()),
        "observer should see Start: {:?}",
        *processed
    );
}

#[tokio::test]
async fn metrics_flow_end_to_end() {
    let obs = IntegrationObserver::new();
    let (node, handle, mut down_rx, _up_rx) =
        make_observed_node(Box::new(MetricsEmittingProcessor::new()), obs.clone());

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("trigger metrics")),
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

    let frames = drain_rx(&mut down_rx).await;

    // Should have: Start, MetricsFrame, Text, Cancel
    assert!(
        frames
            .iter()
            .any(|f| matches!(&f.frame, Frame::Metrics(m) if !m.data.is_empty())),
        "MetricsFrame should arrive downstream: {:?}",
        frame_names(&frames)
    );

    // Observer should see the MetricsFrame push
    let pushed = obs.push_frames.lock().unwrap();
    assert!(
        pushed.contains(&"Metrics".to_string()),
        "observer should see Metrics push: {:?}",
        *pushed
    );
}

#[tokio::test]
async fn error_propagation_through_node() {
    let (node, handle, _down_rx, mut up_rx) = make_node(Box::new(ErrorOnTextProcessor::new()));

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("trigger error")),
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

    let errors = drain_rx(&mut up_rx).await;
    assert!(
        errors
            .iter()
            .any(|f| matches!(&f.frame, Frame::Error(e) if e.error == "text not allowed")),
        "ErrorFrame should propagate upstream: {:?}",
        frame_names(&errors)
    );
}

#[tokio::test]
async fn pause_resume_with_observer() {
    let obs = IntegrationObserver::new();
    let processor_name = "Recorder"; // RecorderProcessor's default name
    let (recorder, record) = RecorderProcessor::new();
    let (node, handle, _down_rx, _up_rx) = make_observed_node(Box::new(recorder), obs.clone());

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Pause
    send_and_settle(
        &handle,
        Frame::ProcessorPauseUrgent(ProcessorPauseUrgentFrame {
            processor_name: processor_name.into(),
        }),
        Direction::Downstream,
    )
    .await;

    // System frame while paused — observer should still see it
    send_and_settle(
        &handle,
        Frame::BotStartedSpeaking(BotStartedSpeakingFrame),
        Direction::Downstream,
    )
    .await;

    // Resume
    send_and_settle(
        &handle,
        Frame::ProcessorResumeUrgent(ProcessorResumeUrgentFrame {
            processor_name: processor_name.into(),
        }),
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

    // Observer saw system frames during pause
    let processed = obs.process_frames.lock().unwrap();
    assert!(
        processed.contains(&"BotStartedSpeaking".to_string()),
        "observer should see system frames while paused: {:?}",
        *processed
    );

    // Recorder also saw them (dispatched by node)
    let recorded = record.lock().unwrap();
    assert!(
        recorded.contains(&"BotStartedSpeaking".to_string()),
        "processor should receive system frames while paused: {:?}",
        *recorded
    );
}

#[tokio::test]
async fn cleanup_on_both_exit_paths() {
    // Cancel path
    let (proc1, flag1) = PassthroughProcessor::with_cleanup_flag();
    let (node1, handle1, _d1, _u1) = make_node(Box::new(proc1));
    let run1 = tokio::spawn(async move { node1.run().await });
    send_and_settle(
        &handle1,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    handle1
        .send(
            FrameEnvelope::new(Frame::Cancel(CancelFrame::default())),
            Direction::Downstream,
        )
        .await
        .unwrap();
    timeout(TEST_TIMEOUT, run1).await.unwrap().unwrap();
    assert!(flag1.load(Ordering::SeqCst), "cleanup on Cancel");

    // Handle-drop path
    let (proc2, flag2) = PassthroughProcessor::with_cleanup_flag();
    let (node2, handle2, _d2, _u2) = make_node(Box::new(proc2));
    let run2 = tokio::spawn(async move { node2.run().await });
    drop(handle2);
    timeout(TEST_TIMEOUT, run2).await.unwrap().unwrap();
    assert!(flag2.load(Ordering::SeqCst), "cleanup on handle drop");
}

// ---------------------------------------------------------------------------
// TextFrame::new() defaults
// ---------------------------------------------------------------------------

#[test]
fn text_frame_new_defaults() {
    let f = TextFrame::new("hello");
    assert_eq!(f.text, "hello");
    assert!(f.skip_tts.is_none());
    assert!(!f.includes_inter_frame_spaces);
    assert!(f.append_to_context);
}

#[test]
fn text_frame_custom_fields() {
    let f = TextFrame {
        text: "fn output".into(),
        skip_tts: Some(true),
        includes_inter_frame_spaces: true,
        append_to_context: false,
    };
    assert_eq!(f.skip_tts, Some(true));
    assert!(f.includes_inter_frame_spaces);
    assert!(!f.append_to_context);
}

// ---------------------------------------------------------------------------
// broadcast() sends both directions with sibling IDs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn broadcast_sends_both_directions() {
    let (down_tx, mut down_rx) = tokio::sync::mpsc::channel(16);
    let (up_tx, mut up_rx) = tokio::sync::mpsc::channel(16);
    let ctx = ProcessorContext::new(down_tx, up_tx, 1, "test".into());

    ctx.broadcast(Frame::BotStartedSpeaking(BotStartedSpeakingFrame))
        .await
        .unwrap();

    let down = down_rx.recv().await.unwrap();
    let up = up_rx.recv().await.unwrap();

    assert!(matches!(down.frame, Frame::BotStartedSpeaking(_)));
    assert!(matches!(up.frame, Frame::BotStartedSpeaking(_)));

    // Sibling IDs cross-reference each other
    assert_eq!(down.header.broadcast_sibling_id, Some(up.header.id));
    assert_eq!(up.header.broadcast_sibling_id, Some(down.header.id));
}

// ---------------------------------------------------------------------------
// broadcast_interruption() sends InterruptionFrame both ways
// ---------------------------------------------------------------------------

#[tokio::test]
async fn broadcast_interruption_sends_both_ways() {
    let (down_tx, mut down_rx) = tokio::sync::mpsc::channel(16);
    let (up_tx, mut up_rx) = tokio::sync::mpsc::channel(16);
    let ctx = ProcessorContext::new(down_tx, up_tx, 1, "test".into());

    ctx.broadcast_interruption().await.unwrap();

    let down = down_rx.recv().await.unwrap();
    let up = up_rx.recv().await.unwrap();
    assert!(matches!(down.frame, Frame::Interruption(_)));
    assert!(matches!(up.frame, Frame::Interruption(_)));
    assert!(down.header.broadcast_sibling_id.is_some());
    assert!(up.header.broadcast_sibling_id.is_some());
}

// ---------------------------------------------------------------------------
// FrameHeader broadcast_sibling_id set by broadcast()
// ---------------------------------------------------------------------------

#[tokio::test]
async fn frame_header_broadcast_sibling_id() {
    let (down_tx, mut down_rx) = tokio::sync::mpsc::channel(16);
    let (up_tx, mut up_rx) = tokio::sync::mpsc::channel(16);
    let ctx = ProcessorContext::new(down_tx, up_tx, 1, "test".into());

    ctx.broadcast(Frame::Stop(StopFrame)).await.unwrap();

    let env_down = down_rx.recv().await.unwrap();
    let env_up = up_rx.recv().await.unwrap();

    // They are different envelopes with different IDs
    assert_ne!(env_down.header.id, env_up.header.id);

    // Each points to the other as its sibling
    assert_eq!(
        env_down.header.broadcast_sibling_id.unwrap(),
        env_up.header.id
    );
    assert_eq!(
        env_up.header.broadcast_sibling_id.unwrap(),
        env_down.header.id
    );
}

// ---------------------------------------------------------------------------
// on_before_process / on_after_process hooks
// ---------------------------------------------------------------------------

struct HookTracker {
    base: ProcessorBase,
    before_count: Arc<std::sync::atomic::AtomicU32>,
    after_count: Arc<std::sync::atomic::AtomicU32>,
    before_frame_names: Arc<Mutex<Vec<String>>>,
}

impl HookTracker {
    #[allow(clippy::type_complexity)]
    fn new() -> (
        Self,
        Arc<std::sync::atomic::AtomicU32>,
        Arc<std::sync::atomic::AtomicU32>,
        Arc<Mutex<Vec<String>>>,
    ) {
        let before = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let after = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let names = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                base: ProcessorBase::new("HookTracker"),
                before_count: before.clone(),
                after_count: after.clone(),
                before_frame_names: names.clone(),
            },
            before,
            after,
            names,
        )
    }
}

#[async_trait]
impl FrameProcessor for HookTracker {
    fn name(&self) -> &str {
        self.base.name()
    }
    fn id(&self) -> u64 {
        self.base.id()
    }
    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) {
        ctx.push_frame(envelope, direction).await.ok();
    }
    async fn on_before_process(&mut self, frame: &Frame, _direction: Direction) {
        self.before_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.before_frame_names
            .lock()
            .unwrap()
            .push(format!("{frame}"));
    }
    async fn on_after_process(&mut self) {
        self.after_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[tokio::test]
async fn hooks_fire_before_and_after() {
    let (tracker, before, after, names) = HookTracker::new();
    let (node, handle, _down_rx, _up_rx) = make_node(Box::new(tracker));

    let run = tokio::spawn(async move { node.run().await });

    send_and_settle(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;
    send_and_settle(
        &handle,
        Frame::Text(TextFrame::new("hooked")),
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

    // Before and after fire for each frame: Start, Text, Cancel = 3
    assert_eq!(before.load(Ordering::SeqCst), 3);
    assert_eq!(after.load(Ordering::SeqCst), 3);

    let frame_names = names.lock().unwrap();
    assert!(frame_names.contains(&"Start".to_string()));
    assert!(frame_names.contains(&"Text".to_string()));
    assert!(frame_names.contains(&"Cancel".to_string()));
}

// ---------------------------------------------------------------------------
// push_ttfb generates and sends MetricsFrame
// ---------------------------------------------------------------------------

#[tokio::test]
async fn push_ttfb_sends_metrics_frame() {
    let (down_tx, mut down_rx) = tokio::sync::mpsc::channel(16);
    let (up_tx, _up_rx) = tokio::sync::mpsc::channel(16);
    let ctx = ProcessorContext::new(down_tx, up_tx, 1, "test".into());

    let mut metrics = ProcessorMetrics::new("test_svc", Some("model-1".into()));
    metrics.start_ttfb();
    ctx.push_ttfb(&mut metrics).await.unwrap();

    let env = down_rx.recv().await.unwrap();
    match &env.frame {
        Frame::Metrics(m) => {
            assert_eq!(m.data.len(), 1);
            assert!(
                matches!(&m.data[0], MetricsData::Ttfb { processor, .. } if processor == "test_svc")
            );
        }
        other => panic!("expected MetricsFrame, got {other}"),
    }
}

#[tokio::test]
async fn push_ttfb_noop_when_not_started() {
    let (down_tx, mut down_rx) = tokio::sync::mpsc::channel(16);
    let (up_tx, _up_rx) = tokio::sync::mpsc::channel(16);
    let ctx = ProcessorContext::new(down_tx, up_tx, 1, "test".into());

    let mut metrics = ProcessorMetrics::new("test_svc", None);
    // Don't call start_ttfb
    ctx.push_ttfb(&mut metrics).await.unwrap();

    // Channel should be empty
    drop(ctx);
    assert!(down_rx.try_recv().is_err());
}

// ---------------------------------------------------------------------------
// observer sees destination_name when set
// ---------------------------------------------------------------------------

struct DestObserver {
    dest_names: Mutex<Vec<Option<String>>>,
}

impl DestObserver {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            dest_names: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl PipelineObserver for DestObserver {
    async fn on_push_frame(&self, event: FramePushedEvent<'_>) {
        self.dest_names
            .lock()
            .unwrap()
            .push(event.destination_name.map(|s| s.to_string()));
    }
}

#[tokio::test]
async fn observer_sees_destination_name() {
    let obs = DestObserver::new();
    let (down_tx, _down_rx) = tokio::sync::mpsc::channel(16);
    let (up_tx, _up_rx) = tokio::sync::mpsc::channel(16);
    let mut ctx = ProcessorContext::with_observer(down_tx, up_tx, 1, "source".into(), obs.clone());
    ctx.set_downstream_name("next_proc".into());
    ctx.set_upstream_name("prev_proc".into());

    ctx.send_downstream(Frame::Interruption(InterruptionFrame))
        .await
        .unwrap();
    ctx.send_upstream(Frame::Interruption(InterruptionFrame))
        .await
        .unwrap();

    let dests = obs.dest_names.lock().unwrap();
    assert_eq!(dests.len(), 2);
    assert_eq!(dests[0], Some("next_proc".to_string()));
    assert_eq!(dests[1], Some("prev_proc".to_string()));
}

#[tokio::test]
async fn observer_destination_none_by_default() {
    let obs = DestObserver::new();
    let (down_tx, _down_rx) = tokio::sync::mpsc::channel(16);
    let (up_tx, _up_rx) = tokio::sync::mpsc::channel(16);
    let ctx = ProcessorContext::with_observer(down_tx, up_tx, 1, "source".into(), obs.clone());

    ctx.send_downstream(Frame::Interruption(InterruptionFrame))
        .await
        .unwrap();

    let dests = obs.dest_names.lock().unwrap();
    assert_eq!(dests[0], None);
}

// ---------------------------------------------------------------------------
// push_processing_metrics / push_llm_usage / push_tts_usage
// ---------------------------------------------------------------------------

#[tokio::test]
async fn push_processing_metrics_sends_frame() {
    let (down_tx, mut down_rx) = tokio::sync::mpsc::channel(16);
    let (up_tx, _up_rx) = tokio::sync::mpsc::channel(16);
    let ctx = ProcessorContext::new(down_tx, up_tx, 1, "test".into());

    let mut metrics = ProcessorMetrics::new("proc_svc", Some("model-x".into()));
    metrics.start_processing();
    ctx.push_processing_metrics(&mut metrics).await.unwrap();

    let env = down_rx.recv().await.unwrap();
    match &env.frame {
        Frame::Metrics(m) => {
            assert_eq!(m.data.len(), 1);
            assert!(
                matches!(&m.data[0], MetricsData::Processing { processor, .. } if processor == "proc_svc")
            );
        }
        other => panic!("expected MetricsFrame, got {other}"),
    }
}

#[tokio::test]
async fn push_llm_usage_sends_frame() {
    use pipecat_core::metrics::LlmTokenUsage;

    let (down_tx, mut down_rx) = tokio::sync::mpsc::channel(16);
    let (up_tx, _up_rx) = tokio::sync::mpsc::channel(16);
    let ctx = ProcessorContext::new(down_tx, up_tx, 1, "test".into());

    let metrics = ProcessorMetrics::new("llm_svc", Some("gpt-4".into()));
    let usage = LlmTokenUsage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        ..Default::default()
    };
    ctx.push_llm_usage(&metrics, usage).await.unwrap();

    let env = down_rx.recv().await.unwrap();
    match &env.frame {
        Frame::Metrics(m) => {
            assert_eq!(m.data.len(), 1);
            match &m.data[0] {
                MetricsData::LlmUsage {
                    processor,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    ..
                } => {
                    assert_eq!(processor, "llm_svc");
                    assert_eq!(*prompt_tokens, 100);
                    assert_eq!(*completion_tokens, 50);
                    assert_eq!(*total_tokens, 150);
                }
                other => panic!("expected LlmUsage, got {other:?}"),
            }
        }
        other => panic!("expected MetricsFrame, got {other}"),
    }
}

#[tokio::test]
async fn push_tts_usage_sends_frame() {
    let (down_tx, mut down_rx) = tokio::sync::mpsc::channel(16);
    let (up_tx, _up_rx) = tokio::sync::mpsc::channel(16);
    let ctx = ProcessorContext::new(down_tx, up_tx, 1, "test".into());

    let metrics = ProcessorMetrics::new("tts_svc", Some("voice-1".into()));
    ctx.push_tts_usage(&metrics, "Hello, world!").await.unwrap();

    let env = down_rx.recv().await.unwrap();
    match &env.frame {
        Frame::Metrics(m) => {
            assert_eq!(m.data.len(), 1);
            match &m.data[0] {
                MetricsData::TtsUsage {
                    processor,
                    characters,
                    ..
                } => {
                    assert_eq!(processor, "tts_svc");
                    assert_eq!(*characters, 13);
                }
                other => panic!("expected TtsUsage, got {other:?}"),
            }
        }
        other => panic!("expected MetricsFrame, got {other}"),
    }
}
