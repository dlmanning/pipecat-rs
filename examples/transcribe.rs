//! Transcribe audio using Silero VAD + Whisper STT.
//!
//! Supports file input (WAV, MP3, FLAC, OGG/Vorbis, AAC) or live microphone capture.
//!
//! ```text
//! cargo run -p pipecat-examples --bin transcribe -- recording.wav
//! cargo run -p pipecat-examples --bin transcribe -- recording.mp3 --play
//! cargo run -p pipecat-examples --bin transcribe -- --mic
//! cargo run -p pipecat-examples --bin transcribe -- --mic --device "MacBook Pro Microphone"
//! cargo run -p pipecat-examples --bin transcribe -- --list-devices
//! ```

use std::path::PathBuf;
use std::sync::Arc;
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
use pipecat_transport::local::*;
use pipecat_transport::{
    AudioPlayer, AudioPlayerConfig, MicInput, MicInputConfig, TransportParams, list_input_devices,
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const SAMPLE_RATE: u32 = 16000;

/// Pre-speech audio buffer: 1 second at SAMPLE_RATE, mono, 16-bit PCM.
const PRE_SPEECH_BYTES: usize = SAMPLE_RATE as usize * 2;

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
// WhisperTranscribeProcessor
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct WhisperTranscribeProcessor {
    base: ProcessorBase,
    whisper_ctx: Arc<WhisperContext>,
    language: String,
    /// Rolling 1s pre-speech window; accumulates during speech.
    audio_buf: Vec<u8>,
    /// Total audio bytes seen (for timestamp computation).
    total_bytes: usize,
    speaking: bool,
    /// Byte offset where the current segment's buffer begins.
    segment_start: usize,
    segments: usize,
}

impl WhisperTranscribeProcessor {
    fn new(whisper_ctx: Arc<WhisperContext>, language: String) -> Self {
        Self {
            base: ProcessorBase::new("WhisperTranscribe"),
            whisper_ctx,
            language,
            audio_buf: Vec::new(),
            total_bytes: 0,
            speaking: false,
            segment_start: 0,
            segments: 0,
        }
    }

    fn segments(&self) -> usize {
        self.segments
    }
}

fn transcribe(ctx: &WhisperContext, audio: &[u8], language: &str) -> String {
    let samples: Vec<f32> = audio
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();

    let mut state = ctx.create_state().expect("failed to create Whisper state");
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(language));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_no_timestamps(true);

    state
        .full(params, &samples)
        .expect("Whisper inference failed");

    let mut text = String::new();
    for seg in state.as_iter() {
        if let Ok(t) = seg.to_str() {
            text.push_str(t);
        }
    }
    text.trim().to_string()
}

#[async_trait]
impl FrameProcessor for WhisperTranscribeProcessor {
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
            Frame::InputAudioRaw(audio) => {
                self.total_bytes += audio.audio.len();
                self.audio_buf.extend_from_slice(&audio.audio);

                // Keep only the last 1s when not speaking.
                if !self.speaking && self.audio_buf.len() > PRE_SPEECH_BYTES {
                    let excess = self.audio_buf.len() - PRE_SPEECH_BYTES;
                    self.audio_buf.drain(..excess);
                }
            }

            Frame::VADUserStartedSpeaking(_) => {
                self.speaking = true;
                self.segment_start = self.total_bytes - self.audio_buf.len();
            }

            Frame::VADUserStoppedSpeaking(_) => {
                self.speaking = false;
                let start_secs = self.segment_start as f64 / (SAMPLE_RATE as f64 * 2.0);
                let audio_data = std::mem::take(&mut self.audio_buf);
                let wctx = self.whisper_ctx.clone();
                let lang = self.language.clone();

                let text =
                    tokio::task::spawn_blocking(move || transcribe(&wctx, &audio_data, &lang))
                        .await
                        .unwrap();

                if !text.is_empty() {
                    self.segments += 1;
                    println!("[{start_secs:6.1}s] \"{text}\"");
                }
            }

            _ => {}
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
    whisper_rs::install_logging_hooks();
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
    let whisper_ctx = Arc::new(
        WhisperContext::new_with_params(
            model_path.to_str().unwrap(),
            WhisperContextParameters::default(),
        )
        .expect("failed to load Whisper model"),
    );
    eprintln!("Model loaded.\n");

    // Build pipeline: input → VAD → [optional playback] → whisper.
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

    let whisper = WhisperTranscribeProcessor::new(whisper_ctx, args.language);

    // We need the segment count after the pipeline runs, but the processor is
    // moved into the pipeline. Use a shared pointer to read it back out.
    let segment_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let segment_count_ref = segment_count.clone();

    let mut processors: Vec<Box<dyn FrameProcessor>> = vec![input, Box::new(vad)];
    if args.play {
        processors.push(Box::new(AudioPlayer::new(AudioPlayerConfig::default())));
    }
    processors.push(Box::new(SegmentCounter(whisper, segment_count)));

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
    let count = segment_count_ref.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("\n{count} segments in {elapsed:.2}s");
}

// ---------------------------------------------------------------------------
// SegmentCounter — thin wrapper to expose segment count after pipeline ends
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct SegmentCounter(
    WhisperTranscribeProcessor,
    Arc<std::sync::atomic::AtomicUsize>,
);

#[async_trait]
impl FrameProcessor for SegmentCounter {
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
        self.0.process_frame(envelope, direction, ctx).await?;
        self.1
            .store(self.0.segments(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}
