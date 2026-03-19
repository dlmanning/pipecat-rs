//! Transcribe audio using Silero VAD + Whisper STT.
//!
//! Supports file input (WAV, MP3, FLAC, OGG/Vorbis, AAC) or live microphone capture.
//!
//! ```text
//! cargo run -p transcribe -- recording.wav
//! cargo run -p transcribe -- recording.mp3 --play
//! cargo run -p transcribe -- --mic
//! cargo run -p transcribe -- --mic --device "MacBook Pro Microphone"
//! cargo run -p transcribe -- --list-devices
//! ```

use std::path::PathBuf;
use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use clap::Parser;
use pipecat_audio::vad::{SileroVadAnalyzer, VadController, VadProcessor};
use pipecat_core::VadParams;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_pipeline::{Pipeline, PipelineParams, PipelineTask};
use pipecat_services::settings::STTSettings;
use pipecat_services::whisper::WhisperSTTService;
use pipecat_transport::local::*;
use pipecat_transport::{
    AudioPlayer, AudioPlayerConfig, MicInput, MicInputConfig, TransportParams, list_input_devices,
};

const SAMPLE_RATE: u32 = 16000;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Transcribe audio using Silero VAD + Whisper STT.
#[derive(Parser)]
struct Args {
    /// Audio file to transcribe (WAV, MP3, FLAC, OGG, AAC, etc.)
    #[arg(
        required_unless_present = "mic",
        required_unless_present = "list_devices"
    )]
    audio_file: Option<PathBuf>,

    /// Use system microphone as input (Ctrl+C to stop)
    #[arg(long, conflicts_with_all = ["play", "realtime", "audio_file"])]
    mic: bool,

    /// Select a specific input device by name (use --list-devices to see options)
    #[arg(long, requires = "mic")]
    device: Option<String>,

    /// List available audio input devices and exit
    #[arg(long, exclusive = true)]
    list_devices: bool,

    /// Process audio at real-time pace (default: as fast as possible)
    #[arg(long)]
    realtime: bool,

    /// Play audio through default output device (implies --realtime)
    #[arg(long)]
    play: bool,

    /// Whisper GGML model name
    #[arg(long, default_value = "tiny.en")]
    model: String,

    /// Language code
    #[arg(long, default_value = "en")]
    language: String,

    /// Seconds of silence before speech is considered stopped
    #[arg(long, default_value = "0.2")]
    stop_secs: f64,
}

// ---------------------------------------------------------------------------
// TranscriptionPrinter — prints TranscriptionFrame text from WhisperSTTService
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TranscriptionPrinter {
    base: ProcessorBase,
    segments: usize,
}

impl TranscriptionPrinter {
    fn new() -> Self {
        Self {
            base: ProcessorBase::new("TranscriptionPrinter"),
            segments: 0,
        }
    }
}

#[async_trait]
impl FrameProcessor for TranscriptionPrinter {
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
        if let Frame::Transcription(t) = &envelope.frame
            && !t.text.is_empty()
        {
            self.segments += 1;
            println!("\"{}\"", t.text);
        }
        ctx.push_frame(envelope, direction).await
    }
}

// ---------------------------------------------------------------------------
// Input source builders
// ---------------------------------------------------------------------------

fn build_mic_input(device: Option<String>) -> Box<dyn FrameProcessor> {
    let config = MicInputConfig {
        device_name: device.clone(),
    };

    let devices = list_input_devices();
    let selected = if let Some(ref name) = device {
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
    eprintln!("Listening (Ctrl+C to stop)...\n");

    Box::new(MicInput::new(TransportParams::default(), config))
}

fn build_file_input(path: &PathBuf, realtime: bool) -> Box<dyn FrameProcessor> {
    let audio_data = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", path.display());
        std::process::exit(1);
    });

    let pacing = if realtime {
        AudioPacing::RealTime
    } else {
        AudioPacing::AsFastAsPossible
    };

    let params = TransportParams {
        audio_in_enabled: true,
        audio_in_resampler: Some(Box::new(pipecat_audio::resampler::LinearResampler::new())),
        ..Default::default()
    };

    Box::new(
        LocalAudioInputTransport::new(params, AudioInputSource::Buffer(Bytes::from(audio_data)))
            .with_format(AudioFormat::Encoded)
            .with_pacing(pacing),
    )
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    pipecat_services::whisper::suppress_stderr_logging();
    let args = Args::parse();

    // --list-devices: print and exit (no model load needed).
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

    // Load Whisper model.
    let home = std::env::var("HOME").expect("HOME not set");
    let cache_dir = PathBuf::from(home).join(".cache/pipecat-rs/whisper");
    let model_path = pipecat_services::whisper::model::ensure_model(&args.model, &cache_dir)
        .expect("failed to download/find Whisper model");

    eprintln!("Loading Whisper model: {}", model_path.display());

    let mut stt = WhisperSTTService::new(
        &model_path,
        STTSettings {
            language: Some(args.language),
            ..Default::default()
        },
    )
    .expect("failed to create WhisperSTTService");
    stt.set_audio_passthrough(false);

    eprintln!("Model loaded.\n");

    // Build pipeline: input → VAD → [optional playback] → STT → printer.
    let realtime = args.realtime || args.play;
    if args.play && !args.realtime {
        eprintln!("Note: --play forces real-time pacing");
    }

    let input: Box<dyn FrameProcessor> = if args.mic {
        build_mic_input(args.device)
    } else {
        build_file_input(args.audio_file.as_ref().unwrap(), realtime)
    };

    let vad = VadProcessor::new(VadController::with_params(
        SileroVadAnalyzer::new(SAMPLE_RATE).expect("failed to create VAD"),
        SAMPLE_RATE,
        VadParams {
            min_volume: 0.0,
            stop_secs: args.stop_secs,
            ..Default::default()
        },
    ));

    let printer = TranscriptionPrinter::new();

    let mut processors: Vec<Box<dyn FrameProcessor>> = vec![input, Box::new(vad)];
    if args.play {
        processors.push(Box::new(AudioPlayer::new(AudioPlayerConfig::default())));
    }
    processors.push(Box::new(stt));
    processors.push(Box::new(printer));

    let pipeline = Pipeline::new(processors);
    let mut task = PipelineTask::new(
        Box::new(pipeline),
        PipelineParams {
            audio_in_sample_rate: SAMPLE_RATE,
            idle_timeout: None,
            ..Default::default()
        },
    );

    let start = Instant::now();

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async { task.run().await.unwrap() });

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("\nDone in {elapsed:.2}s");
}
