use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use pipecat_context::{LLMContext, LLMContextAggregatorPair, LLMUserAggregatorParams};
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use pipecat_core::test_utils::*;
use pipecat_integration_tests::helpers::*;
use pipecat_integration_tests::mock_services::*;
use pipecat_pipeline::Pipeline;
use pipecat_services::settings::STTSettings;
use pipecat_services::stt::{STTService, STTServiceState, stt_process_frame};
use pipecat_turns::{
    SpeechTimeoutUserTurnStopStrategy, UserTurnStrategies, VadUserTurnStartStrategy,
};
use serde_json::json;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// SequentialFakeSTTService: returns different canned text for each turn
// ---------------------------------------------------------------------------

/// Fake STT that cycles through a list of canned responses, emitting the next
/// one each time it has received enough audio chunks.
#[derive(Debug)]
struct SequentialFakeSTTService {
    state: STTServiceState,
    responses: Vec<String>,
    call_index: Arc<Mutex<usize>>,
    chunks_before_emit: usize,
    audio_chunk_count: usize,
}

impl SequentialFakeSTTService {
    fn new(responses: Vec<&str>, chunks_before_emit: usize) -> Self {
        Self {
            state: STTServiceState::new("SequentialFakeSTT", STTSettings::default()),
            responses: responses.into_iter().map(String::from).collect(),
            call_index: Arc::new(Mutex::new(0)),
            chunks_before_emit,
            audio_chunk_count: 0,
        }
    }
}

#[async_trait]
impl STTService for SequentialFakeSTTService {
    async fn run_stt(&mut self, _audio: Bytes, ctx: &ProcessorContext) -> Result<()> {
        self.audio_chunk_count += 1;
        if self.audio_chunk_count >= self.chunks_before_emit {
            self.audio_chunk_count = 0;
            let text = {
                let mut idx = self.call_index.lock().unwrap();
                let t = self.responses[*idx % self.responses.len()].clone();
                *idx += 1;
                t
            };

            ctx.send_downstream(Frame::Transcription(TranscriptionFrame {
                text,
                user_id: "user".to_string(),
                timestamp: None,
                language: None,
                finalized: true,
                result: None,
            }))
            .await?;
        }
        Ok(())
    }

    fn stt_service_state(&self) -> &STTServiceState {
        &self.state
    }
    fn stt_service_state_mut(&mut self) -> &mut STTServiceState {
        &mut self.state
    }
}

#[async_trait]
impl FrameProcessor for SequentialFakeSTTService {
    fn name(&self) -> &str {
        self.state.base.processor.name()
    }
    fn id(&self) -> u64 {
        self.state.base.processor.id()
    }
    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        stt_process_frame(self, envelope, direction, ctx).await
    }
}

// ---------------------------------------------------------------------------
// Helper: run one full user turn (VAD start -> audio -> VAD stop -> wait)
// ---------------------------------------------------------------------------

async fn run_turn(handle: &pipecat_core::node::ProcessorNodeHandle, down: &FrameCollector) {
    // Clear previously collected frames so we can detect fresh output
    down.take_frames();

    // Start speaking
    handle
        .send(make_vad_started(), Direction::Downstream)
        .await
        .unwrap();

    // Send audio (STT emits transcription after 1 chunk)
    handle
        .send(
            make_input_audio_frame(&[0i16; 160], 16000),
            Direction::Downstream,
        )
        .await
        .unwrap();

    // Stop speaking — speech timeout (50ms) will trigger turn completion
    handle
        .send(make_vad_stopped(), Direction::Downstream)
        .await
        .unwrap();

    // Wait for speech timeout (50ms) to fire — real time-based behavior
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Wait for the turn's TTS output to fully propagate through the pipeline
    down.wait_for_frame("TTSStopped").await;
}

// ---------------------------------------------------------------------------
// Two complete user turns with context accumulation and content verification
// ---------------------------------------------------------------------------

#[tokio::test]
async fn two_turns_accumulate_context() {
    let context = LLMContext::new(vec![
        json!({"role": "system", "content": "You are helpful."}),
    ]);
    let context_ref = context.clone();

    let params = LLMUserAggregatorParams {
        user_turn_strategies: UserTurnStrategies {
            start: vec![Box::new(VadUserTurnStartStrategy::new())],
            stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.05))],
        },
        user_turn_stop_timeout: Duration::from_secs(5),
    };
    let pair = LLMContextAggregatorPair::new(context, params);
    let (user_agg, assistant_agg) = pair.into_processors();

    // Use sequential STT: "hello" for turn 1, "goodbye" for turn 2
    let stt = SequentialFakeSTTService::new(vec!["hello", "goodbye"], 1);
    let llm = FakeLLMService::new(vec!["Hi there.".to_string()]);
    let tts = FakeTTSService::new();

    let pipeline = Pipeline::new(vec![
        Box::new(stt),
        user_agg,
        Box::new(llm),
        Box::new(tts),
        assistant_agg,
    ]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // --- Turn 1 ---
    run_turn(&handle, &down).await;

    // Verify Turn 1 output
    let names_turn1 = down.frame_names();
    assert!(
        names_turn1.contains(&"TTSAudioRaw".to_string()),
        "Turn 1 should produce TTS audio: {names_turn1:?}"
    );

    // Synchronization gate: verify Turn 1 fully completed before starting Turn 2.
    // Context should have exactly system + user + assistant = 3 messages.
    let msg_count_after_turn1 = context_ref.message_count();
    assert_eq!(
        msg_count_after_turn1, 3,
        "after turn 1: expected system + user + assistant = 3 messages, got {msg_count_after_turn1}"
    );

    // Verify Turn 1 content before proceeding
    {
        let messages = context_ref.get_messages();
        assert_eq!(
            messages[1]["content"], "hello",
            "Turn 1 user message should be 'hello'"
        );
        let assistant_content = messages[2]["content"].as_str().unwrap_or("");
        assert!(
            !assistant_content.is_empty(),
            "Turn 1 assistant message should be non-empty"
        );
    }

    // --- Turn 2 ---
    run_turn(&handle, &down).await;

    // Verify Turn 2 output
    let names_turn2 = down.frame_names();
    assert!(
        names_turn2.contains(&"TTSAudioRaw".to_string()),
        "Turn 2 should produce TTS audio: {names_turn2:?}"
    );

    // Shut down
    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    // --- Verify final context: 5 messages total ---
    // system + user1("hello") + assistant1 + user2("goodbye") + assistant2
    let messages = context_ref.get_messages();
    assert_eq!(
        messages.len(),
        5,
        "expected 5 messages (system + user1 + assistant1 + user2 + assistant2), got {}",
        messages.len()
    );

    // Verify message roles in order
    assert_eq!(messages[0]["role"], "system", "message 0 should be system");
    assert_eq!(messages[1]["role"], "user", "message 1 should be user");
    assert_eq!(
        messages[2]["role"], "assistant",
        "message 2 should be assistant"
    );
    assert_eq!(messages[3]["role"], "user", "message 3 should be user");
    assert_eq!(
        messages[4]["role"], "assistant",
        "message 4 should be assistant"
    );

    // Verify content — each turn's STT output and assistant response
    assert_eq!(
        messages[0]["content"], "You are helpful.",
        "system message should have original content"
    );
    assert_eq!(
        messages[1]["content"], "hello",
        "Turn 1 user message should match STT output 'hello'"
    );
    let turn1_assistant = messages[2]["content"].as_str().unwrap_or("");
    assert!(
        !turn1_assistant.is_empty(),
        "Turn 1 assistant should have non-empty content, got {:?}",
        messages[2]
    );
    assert_eq!(
        messages[3]["content"], "goodbye",
        "Turn 2 user message should match STT output 'goodbye'"
    );
    let turn2_assistant = messages[4]["content"].as_str().unwrap_or("");
    assert!(
        !turn2_assistant.is_empty(),
        "Turn 2 assistant should have non-empty content, got {:?}",
        messages[4]
    );

    // Verify content isolation: turns produced different user messages
    assert_ne!(
        messages[1]["content"], messages[3]["content"],
        "Turn 1 and Turn 2 user messages should differ"
    );
}
