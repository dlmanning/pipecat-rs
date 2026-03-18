//! Listen and respond: a voice conversational agent using local services.
//!
//! Mic → VAD → Whisper STT → User Aggregator → Claude Code LLM → macOS Say TTS → Speaker → Assistant Aggregator
//!
//! Entirely local — no API keys or network needed. Requires macOS for `say` TTS
//! and the `claude` CLI installed and authenticated.
//!
//! ```text
//! cargo run -p pipecat-examples --bin listen-and-respond
//! cargo run -p pipecat-examples --bin listen-and-respond -- --voice Samantha
//! cargo run -p pipecat-examples --bin listen-and-respond -- --model sonnet --list-devices
//! ```

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use pipecat_audio::vad::{SileroVadAnalyzer, VadController, VadProcessor};
use pipecat_context::{LLMContext, LLMContextAggregatorPair, LLMUserAggregatorParams};
use pipecat_core::VadParams;
use pipecat_pipeline::{Pipeline, PipelineParams, PipelineTask};
use pipecat_services::claude_code::{ClaudeCodeLLMService, ClaudeCodeSettings};
use pipecat_services::macos_say::{MacOSSaySettings, MacOSSayTTSService};
use pipecat_services::settings::{LLMSettings, STTSettings, TTSSettings};
use pipecat_services::whisper::{self, WhisperSTTService};
use pipecat_transport::{
    AudioPlayer, AudioPlayerConfig, MicInput, MicInputConfig, TransportParams, list_input_devices,
};
use pipecat_turns::{
    SpeechTimeoutUserTurnStopStrategy, UserTurnStrategies, VadUserTurnStartStrategy,
};
use serde_json::json;

const SAMPLE_RATE: u32 = 16000;

/// Listen and respond: a local voice conversational agent.
#[derive(Parser)]
struct Args {
    /// Claude Code model (sonnet, opus, haiku, or full model ID)
    #[arg(long, default_value = "sonnet")]
    model: String,

    /// macOS Say voice (use `say -v '?'` to list)
    #[arg(long, default_value = "Samantha")]
    voice: String,

    /// Speech rate for TTS in words per minute
    #[arg(long)]
    speech_rate: Option<u32>,

    /// Whisper GGML model name
    #[arg(long, default_value = "tiny.en")]
    whisper_model: String,

    /// Language code for Whisper
    #[arg(long, default_value = "en")]
    language: String,

    /// Select a specific input device by name
    #[arg(long)]
    device: Option<String>,

    /// List available audio input devices and exit
    #[arg(long, exclusive = true)]
    list_devices: bool,

    /// Seconds of silence before speech is considered stopped
    #[arg(long, default_value = "0.5")]
    stop_secs: f64,

    /// System instruction for the LLM
    #[arg(long)]
    system: Option<String>,
}

fn main() {
    tracing_subscriber::fmt().init();
    whisper_rs::install_logging_hooks();
    let args = Args::parse();

    // --list-devices: print and exit
    if args.list_devices {
        let devices = list_input_devices();
        if devices.is_empty() {
            eprintln!("No audio input devices found.");
        } else {
            for d in &devices {
                let tag = if d.is_default { " (default)" } else { "" };
                eprintln!("  {} — {}Hz {}ch{}", d.name, d.sample_rate, d.channels, tag);
            }
        }
        return;
    }

    // Load Whisper model
    let home = std::env::var("HOME").expect("HOME not set");
    let cache_dir = PathBuf::from(home).join(".cache/pipecat-rs/whisper");
    let model_path = whisper::model::ensure_model(&args.whisper_model, &cache_dir)
        .expect("failed to download/find Whisper model");
    eprintln!("Whisper model: {}", model_path.display());

    // Print device info
    let devices = list_input_devices();
    let selected = if let Some(ref name) = args.device {
        devices.iter().find(|d| d.name == *name)
    } else {
        devices.iter().find(|d| d.is_default)
    };
    if let Some(d) = selected {
        eprintln!(
            "Input device: {} ({}Hz {}ch)",
            d.name, d.sample_rate, d.channels
        );
    }

    let system_instruction = args.system.unwrap_or_else(|| {
        "You are a helpful voice assistant. Your output will be spoken aloud, \
         so keep responses concise and conversational. Avoid special characters, \
         markdown formatting, or bullet points."
            .to_string()
    });

    eprintln!("Model: {}", args.model);
    eprintln!("Voice: {}", args.voice);
    eprintln!("System: {system_instruction}");
    eprintln!("\nListening (Ctrl+C to stop)...\n");

    // --- Build pipeline components ---

    // Input: microphone
    let mic = MicInput::new(
        TransportParams {
            audio_in_enabled: true,
            ..Default::default()
        },
        MicInputConfig {
            device_name: args.device,
        },
    );

    // VAD
    let vad = VadProcessor::new(VadController::with_params(
        SileroVadAnalyzer::new(SAMPLE_RATE).expect("failed to create VAD"),
        SAMPLE_RATE,
        VadParams {
            min_volume: 0.0,
            stop_secs: args.stop_secs,
            ..Default::default()
        },
    ));

    // STT: Whisper (local)
    let stt = WhisperSTTService::new(
        &model_path,
        STTSettings {
            language: Some(args.language),
            ..Default::default()
        },
    )
    .expect("failed to load Whisper model");

    // Context aggregators
    let context = LLMContext::new(vec![
        json!({"role": "system", "content": system_instruction}),
    ]);
    let pair = LLMContextAggregatorPair::new(
        context,
        LLMUserAggregatorParams {
            user_turn_strategies: UserTurnStrategies {
                start: vec![Box::new(VadUserTurnStartStrategy::new())],
                stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(
                    args.stop_secs,
                ))],
            },
            user_turn_stop_timeout: Duration::from_secs(5),
        },
    );
    let (user_agg, assistant_agg) = pair.into_processors();

    // LLM: Claude Code
    let llm_settings = ClaudeCodeSettings::new(LLMSettings {
        model: Some(args.model),
        ..Default::default()
    });
    let llm = ClaudeCodeLLMService::new(llm_settings);

    // TTS: macOS Say
    let mut tts_settings = MacOSSaySettings::new(TTSSettings {
        voice: Some(args.voice),
        ..Default::default()
    });
    tts_settings.rate = args.speech_rate;
    let tts = MacOSSayTTSService::with_sample_rate(tts_settings, SAMPLE_RATE);

    // Output: speaker
    let speaker = AudioPlayer::new(AudioPlayerConfig::default());

    // --- Assemble pipeline ---
    // Mic → VAD → STT → UserAgg → LLM → TTS → Speaker → AssistantAgg
    let pipeline = Pipeline::new(vec![
        Box::new(mic),
        Box::new(vad),
        Box::new(stt),
        user_agg,
        Box::new(llm),
        Box::new(tts),
        Box::new(speaker),
        assistant_agg,
    ]);

    let mut task = PipelineTask::new(
        Box::new(pipeline),
        PipelineParams {
            audio_in_sample_rate: SAMPLE_RATE,
            audio_out_sample_rate: SAMPLE_RATE,
            idle_timeout: None,
            ..Default::default()
        },
    );

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async { task.run().await.unwrap() });
}
