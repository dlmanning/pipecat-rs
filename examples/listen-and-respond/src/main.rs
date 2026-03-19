//! Listen and respond: a voice conversational agent using local services.
//!
//! Mic (AEC3 filter) → VAD → Whisper STT → Filter → User Aggregator → Claude Code LLM → macOS Say TTS → Speaker (AEC3 sink) → Assistant Aggregator
//!
//! Uses only local services — no third-party API keys needed. Requires:
//! - macOS (for `say` TTS)
//! - `claude` CLI installed and authenticated (makes API calls to Anthropic)
//!
//! ```text
//! cargo run -p listen-and-respond
//! cargo run -p listen-and-respond -- --voice Samantha
//! cargo run -p listen-and-respond -- --model sonnet --list-devices
//! ```

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_trait::async_trait;
use clap::Parser;
use pipecat_audio::aec3::{Aec3Config, Aec3Filter};
use pipecat_audio::resampler::LinearResampler;
use pipecat_audio::vad::{SileroVadAnalyzer, VadController, VadProcessor};
use pipecat_context::{LLMContext, LLMContextAggregatorPair, LLMUserAggregatorParams};
use pipecat_core::VadParams;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_pipeline::{Pipeline, PipelineParams, PipelineTask};
use pipecat_services::claude_code::{ClaudeCodeLLMService, ClaudeCodeSettings};
use pipecat_services::settings::{LLMSettings, STTSettings};
use pipecat_services::whisper::{self, WhisperSTTService};
use pipecat_transport::{
    AudioPlayer, AudioPlayerConfig, MicInput, MicInputConfig, TransportParams, list_input_devices,
};
use pipecat_turns::{
    SpeechTimeoutUserTurnStopStrategy, UserTurnStrategies, VadUserTurnStartStrategy,
};

const SAMPLE_RATE: u32 = 16000;

/// Whisper artifacts that should not be sent to the LLM.
const WHISPER_ARTIFACTS: &[&str] = &[
    "[BLANK_AUDIO]",
    "[SILENCE]",
    "(silence)",
    "[MUSIC]",
    "(music)",
    "[NOISE]",
    "[LAUGHTER]",
    "[APPLAUSE]",
];

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

// ---------------------------------------------------------------------------
// TranscriptionFilter — drops Whisper artifacts
// ---------------------------------------------------------------------------

/// Filters out Whisper artifacts like `[BLANK_AUDIO]` that would confuse the LLM.
#[derive(Debug)]
struct TranscriptionFilter {
    base: ProcessorBase,
}

impl TranscriptionFilter {
    fn new() -> Self {
        Self {
            base: ProcessorBase::new("TranscriptionFilter"),
        }
    }

    fn is_artifact(text: &str) -> bool {
        let trimmed = text.trim();
        WHISPER_ARTIFACTS
            .iter()
            .any(|a| trimmed.eq_ignore_ascii_case(a))
    }
}

#[async_trait]
impl FrameProcessor for TranscriptionFilter {
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
        match &envelope.frame {
            Frame::Transcription(t) if Self::is_artifact(&t.text) => {
                // Drop artifact transcriptions.
            }
            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SayTTS — simple inline TTS that accumulates text and calls `say` once
// ---------------------------------------------------------------------------

/// Accumulates streaming `LLMText` into a buffer, then on
/// `LLMFullResponseEnd` synthesizes the full response with a single `say`
/// invocation and pushes `TTSAudioRaw` frames. Much simpler than routing
/// through the TTS service framework, which does sentence-by-sentence
/// synthesis that blocks the pipeline.
#[derive(Debug)]
struct SayTTS {
    base: ProcessorBase,
    buffer: String,
    voice: String,
    rate: Option<u32>,
    sample_rate: u32,
}

impl SayTTS {
    fn new(voice: String, rate: Option<u32>, sample_rate: u32) -> Self {
        Self {
            base: ProcessorBase::new("SayTTS"),
            buffer: String::new(),
            voice,
            rate,
            sample_rate,
        }
    }

    async fn synthesize(&self, text: &str, ctx: &ProcessorContext) -> Result<()> {
        use std::process::Stdio;

        use tokio::io::AsyncWriteExt;
        use tokio::process::Command;

        let invocation = SYNTH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = std::env::temp_dir().join(format!(
            "pipecat_say_{}_{invocation}.wav",
            std::process::id()
        ));

        let data_format = format!("LEI16@{}", self.sample_rate);
        let mut cmd = Command::new("say");
        cmd.arg("-f").arg("-");
        cmd.arg("-o").arg(&tmp_path);
        cmd.arg("--file-format=WAVE");
        cmd.arg(format!("--data-format={data_format}"));
        cmd.arg("-v").arg(&self.voice);
        if let Some(rate) = self.rate {
            cmd.arg("-r").arg(rate.to_string());
        }
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| PipecatError::ProcessorError(format!("Failed to spawn say: {e}")))?;

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(text.as_bytes()).await;
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| PipecatError::ProcessorError(format!("Failed to wait for say: {e}")))?;

        let wav_data = match tokio::fs::read(&tmp_path).await {
            Ok(data) => data,
            Err(_) => {
                let _ = tokio::fs::remove_file(&tmp_path).await;
                return Ok(());
            }
        };
        let _ = tokio::fs::remove_file(&tmp_path).await;

        if !output.status.success() {
            return Ok(());
        }

        // Find data chunk
        if let Some(offset) = find_wav_data_offset(&wav_data) {
            let pcm = &wav_data[offset..];
            if !pcm.is_empty() {
                // Push in chunks (~100ms each)
                for chunk in pcm.chunks(3200) {
                    ctx.send_downstream(Frame::TTSAudioRaw(TTSAudioRawFrame {
                        audio: bytes::Bytes::copy_from_slice(chunk),
                        sample_rate: self.sample_rate,
                        num_channels: 1,
                        context_id: None,
                    }))
                    .await?;
                }
            }
        }

        Ok(())
    }
}

use std::sync::atomic::AtomicU64;
static SYNTH_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Find the byte offset where PCM data begins in a WAV file.
fn find_wav_data_offset(wav: &[u8]) -> Option<usize> {
    if wav.len() < 12 {
        return None;
    }
    let mut pos = 12;
    while pos + 8 <= wav.len() {
        let fourcc = &wav[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([wav[pos + 4], wav[pos + 5], wav[pos + 6], wav[pos + 7]]) as usize;
        if fourcc == b"data" {
            return Some(pos + 8);
        }
        pos += 8 + chunk_size + (chunk_size & 1);
    }
    None
}

use pipecat_core::error::PipecatError;

#[async_trait]
impl FrameProcessor for SayTTS {
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
        match &envelope.frame {
            Frame::LLMFullResponseStart(_) => {
                self.buffer.clear();
                ctx.push_frame(envelope, direction).await?;
            }
            Frame::LLMText(t) => {
                self.buffer.push_str(&t.text);
            }
            Frame::LLMFullResponseEnd(_) => {
                if !self.buffer.is_empty() {
                    let text = std::mem::take(&mut self.buffer);
                    self.synthesize(&text, ctx).await?;
                }
                ctx.push_frame(envelope, direction).await?;
            }
            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConversationLogger — prints transcription and LLM responses
// ---------------------------------------------------------------------------

/// Prints user transcriptions and assistant responses to stderr.
#[derive(Debug)]
struct ConversationLogger {
    base: ProcessorBase,
}

impl ConversationLogger {
    fn new() -> Self {
        Self {
            base: ProcessorBase::new("ConversationLogger"),
        }
    }
}

#[async_trait]
impl FrameProcessor for ConversationLogger {
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
        match &envelope.frame {
            Frame::Transcription(t) => {
                eprintln!("\n  You: {}", t.text);
            }
            Frame::LLMFullResponseStart(_) => {
                eprint!("\n  Bot: ");
            }
            Frame::LLMText(t) => {
                eprint!("{}", t.text);
            }
            Frame::LLMFullResponseEnd(_) => {
                eprintln!();
            }
            _ => {}
        }

        ctx.push_frame(envelope, direction).await
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
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

    // Suppress whisper.cpp's verbose stderr logging
    whisper::suppress_stderr_logging();

    // Load Whisper model
    let home = std::env::var("HOME").expect("HOME not set");
    let cache_dir = PathBuf::from(home).join(".cache/pipecat-rs/whisper");
    let model_path = whisper::model::ensure_model(&args.whisper_model, &cache_dir)
        .expect("failed to download/find Whisper model");

    // Print device info
    let devices = list_input_devices();
    let selected = if let Some(ref name) = args.device {
        devices.iter().find(|d| d.name == *name)
    } else {
        devices.iter().find(|d| d.is_default)
    };
    if let Some(d) = selected {
        eprintln!("Input: {} ({}Hz {}ch)", d.name, d.sample_rate, d.channels);
    }

    let system_instruction = args.system.unwrap_or_else(|| {
        "You are a helpful voice assistant. Your output will be spoken aloud, \
         so keep responses concise and conversational. Avoid special characters, \
         markdown formatting, or bullet points."
            .to_string()
    });

    eprintln!(
        "Model: {} | Voice: {} | Whisper: {}",
        args.model, args.voice, args.whisper_model
    );
    eprintln!("\nListening (Ctrl+C to stop)...");

    // --- Build pipeline components ---

    // AEC3: acoustic echo cancellation
    let (aec_filter, aec_sink) = Aec3Filter::create(Aec3Config {
        sample_rate: SAMPLE_RATE,
    })
    .expect("Failed to create AEC3 filter");

    // Input: microphone (with AEC filter and resampler to match AEC sample rate)
    let mic = MicInput::new(
        TransportParams {
            audio_in_enabled: true,
            audio_in_filter: Some(aec_filter),
            audio_in_resampler: Some(Box::new(LinearResampler::new())),
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

    // STT: Whisper (local) — disable audio passthrough so mic audio
    // doesn't flow to the speaker (we only want TTS audio played back).
    let mut stt = WhisperSTTService::new(
        &model_path,
        STTSettings {
            language: Some(args.language),
            ..Default::default()
        },
    )
    .expect("failed to load Whisper model");
    stt.set_audio_passthrough(false);

    // Filter out Whisper artifacts
    let filter = TranscriptionFilter::new();

    // Context aggregators — system instruction goes in LLMSettings, not the
    // messages array, so it's passed as --system-prompt to Claude Code
    // (replacing its built-in system prompt).
    let context = LLMContext::new(vec![]);
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
    let llm = ClaudeCodeLLMService::new(ClaudeCodeSettings::new(LLMSettings {
        model: Some(args.model),
        system_instruction: Some(system_instruction),
        ..Default::default()
    }));

    // TTS: inline say (bypasses TTS framework's sentence-by-sentence synthesis)
    let say_tts = SayTTS::new(args.voice, args.speech_rate, SAMPLE_RATE);

    // Conversation loggers
    let user_logger = ConversationLogger::new();
    let bot_logger = ConversationLogger::new();

    // Output: speaker (with AEC echo reference sink)
    let speaker = AudioPlayer::new(AudioPlayerConfig {
        echo_reference_sink: Some(aec_sink),
        ..Default::default()
    });

    // --- Assemble pipeline ---
    // Mic (AEC filter) → VAD → STT → Filter → UserLogger → UserAgg → LLM → BotLogger → SayTTS → Speaker (AEC sink) → AssistantAgg
    let pipeline = Pipeline::new(vec![
        Box::new(mic),
        Box::new(vad),
        Box::new(stt),
        Box::new(filter),
        Box::new(user_logger),
        user_agg,
        Box::new(llm),
        Box::new(bot_logger),
        Box::new(say_tts),
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
