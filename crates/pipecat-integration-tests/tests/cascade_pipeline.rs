use std::time::Duration;

use pipecat_core::frame::*;
use pipecat_core::test_utils::*;
use pipecat_integration_tests::helpers::*;
use pipecat_integration_tests::mock_services::*;
use pipecat_pipeline::Pipeline;
use serde_json::json;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// 5a-1: STT produces transcription from audio
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stt_transcribes_audio() {
    let stt = FakeSTTService::new("hello world", 3);
    let pipeline = Pipeline::new(vec![Box::new(stt)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send 3 audio chunks to trigger transcription
    let samples = vec![0i16; 160];
    for _ in 0..3 {
        send_frame(
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
    }

    down.wait_for_frame("Transcription").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = down.frames();
    let names = down.frame_names();

    assert!(
        names.contains(&"Transcription".to_string()),
        "STT should produce a transcription: {names:?}"
    );

    // Verify the transcription text
    let transcription = frames
        .iter()
        .find_map(|f| match &f.frame {
            Frame::Transcription(t) => Some(t),
            _ => None,
        })
        .expect("should have TranscriptionFrame");
    assert_eq!(transcription.text, "hello world");
    assert!(transcription.finalized);
}

// ---------------------------------------------------------------------------
// 5a-2: LLM generates response from context
// ---------------------------------------------------------------------------

#[tokio::test]
async fn llm_generates_from_context() {
    let llm = FakeLLMService::new(vec![
        "Hello".to_string(),
        ", ".to_string(),
        "world!".to_string(),
    ]);
    let pipeline = Pipeline::new(vec![Box::new(llm)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send LLMContext frame
    let ctx_frame = make_llm_context_frame(vec![
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "Hi"}),
    ]);
    send_frame(&handle, ctx_frame.frame.clone(), Direction::Downstream).await;

    down.wait_for_frame("LLMFullResponseEnd").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = down.frames();
    let names = down.frame_names();

    assert!(
        names.contains(&"LLMFullResponseStart".to_string()),
        "LLM should emit LLMFullResponseStart: {names:?}"
    );
    assert!(
        names.contains(&"LLMFullResponseEnd".to_string()),
        "LLM should emit LLMFullResponseEnd: {names:?}"
    );

    // Collect all text tokens
    let text_frames: Vec<&str> = frames
        .iter()
        .filter_map(|f| match &f.frame {
            Frame::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text_frames, vec!["Hello", ", ", "world!"]);
}

// ---------------------------------------------------------------------------
// 5a-3: TTS synthesizes audio from LLM output
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tts_synthesizes_text() {
    let tts = FakeTTSService::new();
    let pipeline = Pipeline::new(vec![Box::new(tts)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Simulate LLM output sequence
    send_frame(
        &handle,
        Frame::LLMFullResponseStart(LLMFullResponseStartFrame { skip_tts: None }),
        Direction::Downstream,
    )
    .await;
    send_frame(
        &handle,
        Frame::Text(TextFrame::new("Hello world.")),
        Direction::Downstream,
    )
    .await;
    send_frame(
        &handle,
        Frame::LLMFullResponseEnd(LLMFullResponseEndFrame { skip_tts: None }),
        Direction::Downstream,
    )
    .await;

    down.wait_for_frame("TTSStopped").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();

    assert!(
        names.contains(&"TTSStarted".to_string()),
        "TTS should emit TTSStarted: {names:?}"
    );
    assert!(
        names.contains(&"TTSAudioRaw".to_string()),
        "TTS should emit audio: {names:?}"
    );
    assert!(
        names.contains(&"TTSStopped".to_string()),
        "TTS should emit TTSStopped: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// 5a-4: Full cascade STT → LLM → TTS
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_cascade_stt_llm_tts() {
    let stt = FakeSTTService::new("Hello", 2);
    let llm = FakeLLMService::new(vec!["Hi there.".to_string()]);
    let tts = FakeTTSService::new();

    let pipeline = Pipeline::new(vec![Box::new(stt), Box::new(llm), Box::new(tts)]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send audio. FakeSTT emits transcription after 2 chunks.
    // But in cascade without UserAggregator, the transcription just flows through.
    // The LLM only responds to LLMContextFrame, not Transcription.
    // So for this test, send audio → get transcription, then also send LLMContext directly.
    let samples = vec![0i16; 160];
    for _ in 0..2 {
        send_frame(
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
    }

    // Wait for STT to produce transcription before sending LLMContext
    down.wait_for_frame("Transcription").await;

    // Send LLMContext to trigger LLM response → TTS
    let ctx_frame = make_llm_context_frame(vec![json!({"role": "user", "content": "Hello"})]);
    send_frame(&handle, ctx_frame.frame.clone(), Direction::Downstream).await;

    // Wait for full cascade to complete
    down.wait_for_frame("TTSAudioRaw").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();

    // STT output
    assert!(
        names.contains(&"Transcription".to_string()),
        "cascade should include STT transcription: {names:?}"
    );

    // TTS output (proves LLM → TTS worked)
    assert!(
        names.contains(&"TTSAudioRaw".to_string()),
        "cascade should include TTS audio: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// 5a-5: Full cascade with aggregators and turn detection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_cascade_with_aggregators() {
    use pipecat_context::LLMContext;
    use pipecat_context::LLMContextAggregatorPair;
    use pipecat_context::LLMUserAggregatorParams;
    use pipecat_turns::{
        SpeechTimeoutUserTurnStopStrategy, UserTurnStrategies, VadUserTurnStartStrategy,
    };

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

    let stt = FakeSTTService::new("hello", 1);
    let llm = FakeLLMService::new(vec!["Hi there.".to_string()]);
    let tts = FakeTTSService::new();

    // STT must come before UserAggregator so transcriptions reach the aggregator.
    // UserAggregator consumes Transcription frames and pushes LLMContextFrame on turn stop.
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

    // Start a turn
    handle
        .send(make_vad_started(), Direction::Downstream)
        .await
        .unwrap();

    // Send audio (STT will emit transcription after 1 chunk)
    handle
        .send(
            make_input_audio_frame(&[0i16; 160], 16000),
            Direction::Downstream,
        )
        .await
        .unwrap();

    // Stop speaking — speech timeout (50ms) will trigger turn stop
    handle
        .send(make_vad_stopped(), Direction::Downstream)
        .await
        .unwrap();

    // Wait for speech timeout (50ms) + pipeline propagation
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Wait for TTS audio to arrive at the output
    down.wait_for_frame("TTSAudioRaw").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();

    // The full flow should produce TTS audio
    assert!(
        names.contains(&"TTSAudioRaw".to_string()),
        "full cascade with aggregators should produce TTS audio: {names:?}"
    );

    // Context should have system + user + assistant messages.
    // TTS pushes TTSText downstream, which AssistantAggregator accumulates.
    let msg_count = context_ref.message_count();
    assert!(
        msg_count >= 3,
        "context should have system + user + assistant messages, got {msg_count}"
    );
}
