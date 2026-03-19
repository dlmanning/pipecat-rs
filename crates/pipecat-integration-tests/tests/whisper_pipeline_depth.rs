//! Test whether pipeline depth affects WhisperSTTService transcription.
//!
//! Runs the same audio through pipelines with increasing numbers of
//! passthrough processors to detect if first-word cutting correlates
//! with pipeline depth.
//!
//! ```text
//! cargo test -p pipecat-integration-tests --features whisper,silero --test whisper_pipeline_depth -- --nocapture
//! ```

use std::sync::{Arc, Mutex as StdMutex};

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use pipecat_audio::vad::{SileroVadAnalyzer, VadController, VadProcessor};
use pipecat_context::{LLMContext, LLMContextAggregatorPair, LLMUserAggregatorParams};
use pipecat_core::VadParams;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_pipeline::{Pipeline, PipelineParams, PipelineTask};
use pipecat_services::settings::STTSettings;
use pipecat_services::whisper::WhisperSTTService;
use pipecat_transport::TransportParams;
use pipecat_transport::local::*;
use pipecat_turns::{
    SpeechTimeoutUserTurnStopStrategy, UserTurnStrategies, VadUserTurnStartStrategy,
};

const SAMPLE_RATE: u32 = 16000;
const TEST_WAV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/test.wav"
);

// ---------------------------------------------------------------------------
// Passthrough — a no-op processor
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Passthrough(ProcessorBase);

impl Passthrough {
    fn new(name: &str) -> Self {
        Self(ProcessorBase::new(name))
    }
}

#[async_trait]
impl FrameProcessor for Passthrough {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn id(&self) -> u64 {
        self.0.id()
    }
    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        ctx.push_frame(envelope, direction).await
    }
}

// ---------------------------------------------------------------------------
// TranscriptionCollector — collects transcriptions
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TranscriptionCollector {
    base: ProcessorBase,
    transcriptions: Arc<StdMutex<Vec<String>>>,
}

impl TranscriptionCollector {
    fn new(transcriptions: Arc<StdMutex<Vec<String>>>) -> Self {
        Self {
            base: ProcessorBase::new("TranscriptionCollector"),
            transcriptions,
        }
    }
}

#[async_trait]
impl FrameProcessor for TranscriptionCollector {
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
    ) -> Result<()> {
        if let Frame::Transcription(t) = &envelope.frame {
            if !t.text.is_empty() {
                self.transcriptions.lock().unwrap().push(t.text.clone());
            }
        }
        ctx.push_frame(envelope, direction).await
    }
}

// ---------------------------------------------------------------------------
// Helper: run pipeline with N passthrough processors after STT
// ---------------------------------------------------------------------------

async fn run_with_depth(extra_processors: usize) -> Vec<String> {
    let audio_data = std::fs::read(TEST_WAV).expect("failed to read test.wav");

    let params = TransportParams {
        audio_in_enabled: true,
        audio_in_resampler: Some(Box::new(pipecat_audio::resampler::LinearResampler::new())),
        ..Default::default()
    };

    let input = LocalAudioInputTransport::new(
        params,
        AudioInputSource::Buffer(Bytes::from(audio_data)),
    )
    .with_format(AudioFormat::Encoded);

    let vad = VadProcessor::new(VadController::with_params(
        SileroVadAnalyzer::new(SAMPLE_RATE).expect("failed to create VAD"),
        SAMPLE_RATE,
        VadParams {
            min_volume: 0.0,
            ..Default::default()
        },
    ));

    let model_path = {
        let home = std::env::var("HOME").expect("HOME not set");
        let cache_dir = std::path::PathBuf::from(home).join(".cache/pipecat-rs/whisper");
        pipecat_services::whisper::model::ensure_model("tiny.en", &cache_dir)
            .expect("failed to find Whisper model")
    };

    let mut stt = WhisperSTTService::new(
        &model_path,
        STTSettings {
            language: Some("en".to_string()),
            ..Default::default()
        },
    )
    .expect("failed to create WhisperSTTService");
    stt.set_audio_passthrough(false);

    let transcriptions = Arc::new(StdMutex::new(Vec::new()));
    let collector = TranscriptionCollector::new(transcriptions.clone());

    let mut processors: Vec<Box<dyn FrameProcessor>> = Vec::new();
    processors.push(Box::new(input));
    processors.push(Box::new(vad));
    processors.push(Box::new(stt));

    // Add passthrough processors between STT and collector.
    for i in 0..extra_processors {
        processors.push(Box::new(Passthrough::new(&format!("pass-{i}"))));
    }

    processors.push(Box::new(collector));

    let pipeline = Pipeline::new(processors);
    let mut task = PipelineTask::new(
        Box::new(pipeline),
        PipelineParams {
            audio_in_sample_rate: SAMPLE_RATE,
            idle_timeout: None,
            ..Default::default()
        },
    );

    task.run().await.unwrap();

    let result = transcriptions.lock().unwrap().clone();
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn whisper_pipeline_depth_0() {
    pipecat_services::whisper::suppress_stderr_logging();
    let transcriptions = run_with_depth(0).await;
    eprintln!("depth=0: {} segments", transcriptions.len());
    for t in &transcriptions {
        eprintln!("  \"{t}\"");
    }
    assert!(!transcriptions.is_empty(), "should produce transcriptions");
    assert!(
        transcriptions[0].contains("Dublin"),
        "first segment should contain 'Dublin', got: {:?}",
        transcriptions[0]
    );
}

#[tokio::test]
async fn whisper_pipeline_depth_7() {
    pipecat_services::whisper::suppress_stderr_logging();
    let transcriptions = run_with_depth(7).await;
    eprintln!("depth=7: {} segments", transcriptions.len());
    for t in &transcriptions {
        eprintln!("  \"{t}\"");
    }
    assert!(!transcriptions.is_empty(), "should produce transcriptions");
    assert!(
        transcriptions[0].contains("Dublin"),
        "first segment should contain 'Dublin', got: {:?}",
        transcriptions[0]
    );
}

/// Test with the same pipeline structure as listen-and-respond:
/// Input → VAD → STT → UserAgg → Passthrough(LLM) → AssistantAgg → Collector
#[tokio::test]
async fn whisper_with_aggregators() {
    pipecat_services::whisper::suppress_stderr_logging();

    let audio_data = std::fs::read(TEST_WAV).expect("failed to read test.wav");

    let params = TransportParams {
        audio_in_enabled: true,
        audio_in_resampler: Some(Box::new(pipecat_audio::resampler::LinearResampler::new())),
        ..Default::default()
    };

    let input = LocalAudioInputTransport::new(
        params,
        AudioInputSource::Buffer(Bytes::from(audio_data)),
    )
    .with_format(AudioFormat::Encoded);

    let vad = VadProcessor::new(VadController::with_params(
        SileroVadAnalyzer::new(SAMPLE_RATE).expect("failed to create VAD"),
        SAMPLE_RATE,
        VadParams {
            min_volume: 0.0,
            ..Default::default()
        },
    ));

    let model_path = {
        let home = std::env::var("HOME").expect("HOME not set");
        let cache_dir = std::path::PathBuf::from(home).join(".cache/pipecat-rs/whisper");
        pipecat_services::whisper::model::ensure_model("tiny.en", &cache_dir)
            .expect("failed to find Whisper model")
    };

    let mut stt = WhisperSTTService::new(
        &model_path,
        STTSettings {
            language: Some("en".to_string()),
            ..Default::default()
        },
    )
    .expect("failed to create WhisperSTTService");
    stt.set_audio_passthrough(false);

    // Context aggregators — same config as listen-and-respond
    let context = LLMContext::new(vec![]);
    let pair = LLMContextAggregatorPair::new(
        context,
        LLMUserAggregatorParams {
            user_turn_strategies: UserTurnStrategies {
                start: vec![Box::new(VadUserTurnStartStrategy::new())],
                stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.5))],
            },
            user_turn_stop_timeout: Duration::from_secs(5),
        },
    );
    let (user_agg, assistant_agg) = pair.into_processors();

    let transcriptions = Arc::new(StdMutex::new(Vec::new()));
    let collector = TranscriptionCollector::new(transcriptions.clone());

    let pipeline = Pipeline::new(vec![
        Box::new(input),
        Box::new(vad),
        Box::new(stt),
        Box::new(collector),
        user_agg,
        Box::new(Passthrough::new("mock-llm")),
        assistant_agg,
    ]);

    let mut task = PipelineTask::new(
        Box::new(pipeline),
        PipelineParams {
            audio_in_sample_rate: SAMPLE_RATE,
            idle_timeout: None,
            ..Default::default()
        },
    );

    task.run().await.unwrap();

    let result = transcriptions.lock().unwrap().clone();
    eprintln!("with_aggregators: {} segments", result.len());
    for t in &result {
        eprintln!("  \"{t}\"");
    }
    assert!(!result.is_empty(), "should produce transcriptions");
    assert!(
        result[0].contains("Dublin"),
        "first segment should contain 'Dublin', got: {:?}",
        result[0]
    );
}

/// Test with ONLY UserAgg (no AssistantAgg)
#[tokio::test]
async fn whisper_with_user_agg_only() {
    pipecat_services::whisper::suppress_stderr_logging();

    let audio_data = std::fs::read(TEST_WAV).expect("failed to read test.wav");

    let params = TransportParams {
        audio_in_enabled: true,
        audio_in_resampler: Some(Box::new(pipecat_audio::resampler::LinearResampler::new())),
        ..Default::default()
    };

    let input = LocalAudioInputTransport::new(
        params,
        AudioInputSource::Buffer(Bytes::from(audio_data)),
    )
    .with_format(AudioFormat::Encoded);

    let vad = VadProcessor::new(VadController::with_params(
        SileroVadAnalyzer::new(SAMPLE_RATE).expect("failed to create VAD"),
        SAMPLE_RATE,
        VadParams {
            min_volume: 0.0,
            ..Default::default()
        },
    ));

    let model_path = {
        let home = std::env::var("HOME").expect("HOME not set");
        let cache_dir = std::path::PathBuf::from(home).join(".cache/pipecat-rs/whisper");
        pipecat_services::whisper::model::ensure_model("tiny.en", &cache_dir)
            .expect("failed to find Whisper model")
    };

    let mut stt = WhisperSTTService::new(
        &model_path,
        STTSettings {
            language: Some("en".to_string()),
            ..Default::default()
        },
    )
    .expect("failed to create WhisperSTTService");
    stt.set_audio_passthrough(false);

    let context = LLMContext::new(vec![]);
    let pair = LLMContextAggregatorPair::new(
        context,
        LLMUserAggregatorParams {
            user_turn_strategies: UserTurnStrategies {
                start: vec![Box::new(VadUserTurnStartStrategy::new())],
                stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.5))],
            },
            user_turn_stop_timeout: Duration::from_secs(5),
        },
    );
    let (user_agg, _assistant_agg) = pair.into_processors();

    let transcriptions = Arc::new(StdMutex::new(Vec::new()));
    let collector = TranscriptionCollector::new(transcriptions.clone());

    let pipeline = Pipeline::new(vec![
        Box::new(input),
        Box::new(vad),
        Box::new(stt),
        Box::new(collector),
        user_agg,
    ]);

    let mut task = PipelineTask::new(
        Box::new(pipeline),
        PipelineParams {
            audio_in_sample_rate: SAMPLE_RATE,
            idle_timeout: None,
            ..Default::default()
        },
    );

    task.run().await.unwrap();

    let result = transcriptions.lock().unwrap().clone();
    eprintln!("user_agg_only: {} segments", result.len());
    for t in &result {
        eprintln!("  \"{t}\"");
    }
    assert!(!result.is_empty(), "should produce transcriptions");
    eprintln!(
        "first segment contains Dublin: {}",
        result[0].contains("Dublin")
    );
}
