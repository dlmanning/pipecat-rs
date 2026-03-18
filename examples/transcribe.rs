//! Transcribe a WAV file using Silero VAD + Whisper STT.
//!
//! ```text
//! cargo run -p pipecat-examples --bin transcribe -- <audio.wav> [--fast|--realtime] [--play]
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use pipecat_audio::vad::{SileroVadAnalyzer, VadController, VadProcessor};
use pipecat_core::VadParams;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_pipeline::{Pipeline, PipelineParams, PipelineTask};
use pipecat_transport::local::*;
use pipecat_transport::{DeviceConfig, TransportParams};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Pre-speech audio buffer: 1 second at 16 kHz mono 16-bit PCM.
const PRE_SPEECH_BYTES: usize = 16000 * 2;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    audio_file: PathBuf,
    mode: Mode,
    model: String,
    language: String,
    play: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Fast,
    Realtime,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "Usage: transcribe <audio.wav> [--fast|--realtime] [--play] [--model <name>] [--language <lang>]"
        );
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --fast       Transcribe as fast as possible (default)");
        eprintln!("  --realtime   Process audio at real-time pace");
        eprintln!("  --play       Play audio through default output device");
        eprintln!("  --model      Whisper GGML model name (default: tiny.en)");
        eprintln!("  --language   Language code (default: en)");
        std::process::exit(if args.len() < 2 { 1 } else { 0 });
    }

    let audio_file = PathBuf::from(&args[1]);
    let mut mode = Mode::Fast;
    let mut model = "tiny.en".to_string();
    let mut language = "en".to_string();
    let mut play = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--fast" => mode = Mode::Fast,
            "--realtime" => mode = Mode::Realtime,
            "--play" => play = true,
            "--model" => {
                i += 1;
                model = args.get(i).expect("--model requires a value").clone();
            }
            "--language" => {
                i += 1;
                language = args.get(i).expect("--language requires a value").clone();
            }
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        audio_file,
        mode,
        model,
        language,
        play,
    }
}

// ---------------------------------------------------------------------------
// WhisperTranscribeProcessor — prints each segment as it's transcribed
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct WhisperTranscribeProcessor {
    base: ProcessorBase,
    whisper_ctx: Arc<WhisperContext>,
    language: String,
    /// Audio buffer: rolling 1s pre-speech window, accumulates during speech.
    audio_buf: Vec<u8>,
    /// Total audio bytes seen so far (for timestamp computation).
    total_audio_bytes: usize,
    user_speaking: bool,
    /// Byte offset where current segment's buffered audio begins.
    segment_start_bytes: usize,
    segment_count: Arc<AtomicUsize>,
}

impl WhisperTranscribeProcessor {
    fn new(
        whisper_ctx: Arc<WhisperContext>,
        language: String,
        segment_count: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            base: ProcessorBase::new("WhisperTranscribe"),
            whisper_ctx,
            language,
            audio_buf: Vec::new(),
            total_audio_bytes: 0,
            user_speaking: false,
            segment_start_bytes: 0,
            segment_count,
        }
    }
}

fn byte_offset_to_secs(offset: usize) -> f64 {
    offset as f64 / (16000.0 * 2.0)
}

fn whisper_transcribe(ctx: &WhisperContext, audio: &[u8], language: &str) -> String {
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
                self.total_audio_bytes += audio.audio.len();
                self.audio_buf.extend_from_slice(&audio.audio);

                if !self.user_speaking && self.audio_buf.len() > PRE_SPEECH_BYTES {
                    let excess = self.audio_buf.len() - PRE_SPEECH_BYTES;
                    self.audio_buf.drain(..excess);
                }
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::VADUserStartedSpeaking(_) => {
                self.user_speaking = true;
                self.segment_start_bytes = self.total_audio_bytes - self.audio_buf.len();
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::VADUserStoppedSpeaking(_) => {
                self.user_speaking = false;
                let start_secs = byte_offset_to_secs(self.segment_start_bytes);
                let audio_data = std::mem::take(&mut self.audio_buf);
                let wctx = self.whisper_ctx.clone();
                let lang = self.language.clone();

                let text = tokio::task::spawn_blocking(move || {
                    whisper_transcribe(&wctx, &audio_data, &lang)
                })
                .await
                .unwrap();

                if !text.is_empty() {
                    self.segment_count.fetch_add(1, Ordering::Relaxed);
                    println!("[{start_secs:6.1}s] \"{text}\"");
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
// InputToOutputAudio — re-emits InputAudioRaw as OutputAudioRaw
// ---------------------------------------------------------------------------

/// Converts `InputAudioRaw` frames to `OutputAudioRaw` so they can be played
/// by a `LocalAudioOutputTransport`. All other frames pass through unchanged.
#[derive(Debug)]
struct InputToOutputAudio {
    base: ProcessorBase,
}

impl InputToOutputAudio {
    fn new() -> Self {
        Self {
            base: ProcessorBase::new("InputToOutputAudio"),
        }
    }
}

#[async_trait]
impl FrameProcessor for InputToOutputAudio {
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
        if let Frame::InputAudioRaw(ref audio) = envelope.frame {
            let out = FrameEnvelope::new(Frame::OutputAudioRaw(audio.clone()));
            ctx.push_frame(out, direction).await?;
        }
        ctx.push_frame(envelope, direction).await
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    whisper_rs::install_logging_hooks();

    let args = parse_args();

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

    let pacing = if args.play || args.mode == Mode::Realtime {
        if args.play && args.mode == Mode::Fast {
            eprintln!("Note: --play forces real-time pacing");
        }
        AudioPacing::RealTime
    } else {
        AudioPacing::AsFastAsPossible
    };

    let segment_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let wav_data = std::fs::read(&args.audio_file).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", args.audio_file.display());
        std::process::exit(1);
    });

    let in_params = TransportParams {
        audio_in_enabled: true,
        ..Default::default()
    };
    let input_transport =
        LocalAudioInputTransport::new(in_params, AudioInputSource::Buffer(Bytes::from(wav_data)))
            .with_format(AudioFormat::Wav)
            .with_pacing(pacing);

    let vad_processor = VadProcessor::new(VadController::with_params(
        SileroVadAnalyzer::new(16000).expect("failed to create VAD"),
        16000,
        VadParams {
            min_volume: 0.0,
            ..Default::default()
        },
    ));

    let whisper_processor =
        WhisperTranscribeProcessor::new(whisper_ctx, args.language.clone(), segment_count.clone());

    let mut processors: Vec<Box<dyn FrameProcessor>> =
        vec![Box::new(input_transport), Box::new(vad_processor)];

    // When playing audio, insert the converter and output transport BEFORE
    // whisper so that audio flows to the speakers without being blocked by
    // whisper inference stalls.
    if args.play {
        processors.push(Box::new(InputToOutputAudio::new()));

        let out_params = TransportParams {
            audio_out_enabled: true,
            audio_out_sample_rate: Some(16000),
            ..Default::default()
        };
        let output_transport = LocalAudioOutputTransport::new(
            out_params,
            AudioOutputSink::Device(DeviceConfig::default()),
        );
        processors.push(Box::new(output_transport));
    }

    processors.push(Box::new(whisper_processor));

    let pipeline = Pipeline::new(processors);

    let mut task = PipelineTask::new(
        Box::new(pipeline),
        PipelineParams {
            audio_in_sample_rate: 16000,
            idle_timeout: None,
            ..Default::default()
        },
    );

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async { task.run().await.unwrap() });

    let elapsed = start.elapsed().as_secs_f64();
    let count = segment_count.load(Ordering::Relaxed);
    eprintln!("\n{count} segments in {elapsed:.2}s");
}
