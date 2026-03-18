//! Transcribe a WAV file using Silero VAD + Whisper STT.
//!
//! ```text
//! cargo run -p pipecat-examples --bin transcribe -- <audio.wav> [--fast|--realtime]
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use pipecat_audio::vad::{SileroVadAnalyzer, VadAnalyzerBase, VadController, VadControllerEvent};
use pipecat_core::VadParams;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Silero at 16 kHz: 512 samples per VAD chunk = 1024 bytes of int16 PCM.
const VAD_CHUNK_BYTES: usize = 512 * 2;

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
            "Usage: transcribe <audio.wav> [--fast|--realtime] [--model <name>] [--language <lang>]"
        );
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --fast       Pre-scan VAD, transcribe as fast as possible (default)");
        eprintln!("  --realtime   Process audio at real-time pace");
        eprintln!("  --model      Whisper GGML model name (default: tiny.en)");
        eprintln!("  --language   Language code (default: en)");
        std::process::exit(if args.len() < 2 { 1 } else { 0 });
    }

    let audio_file = PathBuf::from(&args[1]);
    let mut mode = Mode::Fast;
    let mut model = "tiny.en".to_string();
    let mut language = "en".to_string();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--fast" => mode = Mode::Fast,
            "--realtime" => mode = Mode::Realtime,
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
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

struct SpeechSegment {
    start_byte: usize,
    end_byte: usize,
}

fn wav_to_pcm(path: &Path) -> Vec<u8> {
    let reader = hound::WavReader::open(path).unwrap_or_else(|e| {
        eprintln!("Failed to open {}: {e}", path.display());
        std::process::exit(1);
    });
    let spec = reader.spec();
    assert_eq!(
        spec.channels, 1,
        "expected mono audio, got {} channels",
        spec.channels
    );
    assert_eq!(
        spec.sample_rate, 16000,
        "expected 16 kHz, got {} Hz",
        spec.sample_rate
    );
    reader
        .into_samples::<i16>()
        .map(|s| s.unwrap())
        .flat_map(|s| s.to_le_bytes())
        .collect()
}

fn prescan_vad(pcm: &[u8]) -> Vec<SpeechSegment> {
    let analyzer = SileroVadAnalyzer::new(16000).expect("failed to create VAD");
    let params = VadParams {
        min_volume: 0.0,
        ..Default::default()
    };
    let base = VadAnalyzerBase::new(analyzer, Some(16000), Some(params));
    let mut controller = VadController::new(base);
    controller.handle_start(16000);

    let mut segments = Vec::new();
    let mut current_start: Option<usize> = None;

    for (i, chunk) in pcm.chunks_exact(VAD_CHUNK_BYTES).enumerate() {
        let byte_offset = i * VAD_CHUNK_BYTES;
        for event in controller.handle_audio(chunk) {
            match event {
                VadControllerEvent::SpeechStarted => {
                    current_start = Some(byte_offset);
                }
                VadControllerEvent::SpeechStopped => {
                    if let Some(start) = current_start.take() {
                        segments.push(SpeechSegment {
                            start_byte: start,
                            end_byte: byte_offset + VAD_CHUNK_BYTES,
                        });
                    }
                }
                _ => {}
            }
        }
    }
    if let Some(start) = current_start {
        segments.push(SpeechSegment {
            start_byte: start,
            end_byte: pcm.len(),
        });
    }
    segments
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

fn print_results(header: &str, results: &[(f64, String)]) {
    println!("{header}\n");
    for (timestamp, text) in results {
        println!("[{timestamp:6.1}s] \"{text}\"");
    }
}

// ---------------------------------------------------------------------------
// Fast mode: pre-scan VAD, transcribe segments at full speed
// ---------------------------------------------------------------------------

fn run_fast(pcm: &[u8], whisper_ctx: &WhisperContext, language: &str) {
    let start = Instant::now();
    let segments = prescan_vad(pcm);

    let mut results = Vec::new();
    for segment in &segments {
        let audio = &pcm[segment.start_byte..segment.end_byte];
        let text = whisper_transcribe(whisper_ctx, audio, language);
        if !text.is_empty() {
            results.push((byte_offset_to_secs(segment.start_byte), text));
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let audio_dur = byte_offset_to_secs(pcm.len());
    print_results(
        &format!(
            "Fast pipeline ({elapsed:.2}s for {audio_dur:.0}s audio, {} segments)",
            results.len()
        ),
        &results,
    );
}

// ---------------------------------------------------------------------------
// Real-time mode: LocalTransport → VAD + Whisper processor
// ---------------------------------------------------------------------------

fn run_realtime(audio_file: &Path, whisper_ctx: Arc<WhisperContext>, language: &str) {
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(run_realtime_inner(audio_file, whisper_ctx, language));
}

async fn run_realtime_inner(audio_file: &Path, whisper_ctx: Arc<WhisperContext>, language: &str) {
    use std::time::Duration;

    use async_trait::async_trait;
    use bytes::Bytes;
    use pipecat_core::error::Result;
    use pipecat_core::frame::*;
    use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
    use pipecat_core::test_utils::*;
    use pipecat_pipeline::Pipeline;
    use pipecat_transport::TransportParams;
    use pipecat_transport::local::*;
    use tokio::time::timeout;

    let wav_data = std::fs::read(audio_file).expect("failed to read audio file");
    let audio_duration_s = {
        let reader = hound::WavReader::new(std::io::Cursor::new(&wav_data)).unwrap();
        reader.len() as f64 / reader.spec().sample_rate as f64
    };

    let results: Arc<std::sync::Mutex<Vec<(f64, String)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));

    // Combined VAD + Whisper processor: detects speech segments in the audio
    // stream, buffers audio during speech, and transcribes on speech stop.
    #[derive(Debug)]
    struct VadWhisperProcessor {
        base: ProcessorBase,
        controller: Option<VadController<SileroVadAnalyzer>>,
        whisper_ctx: Arc<WhisperContext>,
        language: String,
        /// Audio buffer: rolling 1s pre-speech window, accumulates during speech.
        audio_buf: Vec<u8>,
        /// Partial VAD-chunk buffer (transport chunks != VAD chunks).
        vad_buf: Vec<u8>,
        /// Total audio bytes seen so far (for timestamp computation).
        total_audio_bytes: usize,
        user_speaking: bool,
        /// Byte offset where current segment's buffered audio begins.
        segment_start_bytes: usize,
        results: Arc<std::sync::Mutex<Vec<(f64, String)>>>,
    }

    #[async_trait]
    impl FrameProcessor for VadWhisperProcessor {
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
                Frame::Start(_) => {
                    let analyzer = SileroVadAnalyzer::new(16000).unwrap();
                    let params = VadParams {
                        min_volume: 0.0,
                        ..Default::default()
                    };
                    let base = VadAnalyzerBase::new(analyzer, Some(16000), Some(params));
                    let mut controller = VadController::new(base);
                    controller.handle_start(16000);
                    self.controller = Some(controller);
                    ctx.push_frame(envelope, direction).await?;
                }

                Frame::InputAudioRaw(audio) => {
                    self.total_audio_bytes += audio.audio.len();
                    self.audio_buf.extend_from_slice(&audio.audio);
                    self.vad_buf.extend_from_slice(&audio.audio);

                    // Trim pre-speech buffer when not speaking.
                    if !self.user_speaking && self.audio_buf.len() > PRE_SPEECH_BYTES {
                        let excess = self.audio_buf.len() - PRE_SPEECH_BYTES;
                        self.audio_buf.drain(..excess);
                    }

                    // Feed VAD in chunk-sized blocks.
                    if let Some(controller) = &mut self.controller {
                        while self.vad_buf.len() >= VAD_CHUNK_BYTES {
                            let chunk: Vec<u8> = self.vad_buf.drain(..VAD_CHUNK_BYTES).collect();
                            for event in controller.handle_audio(&chunk) {
                                match event {
                                    VadControllerEvent::SpeechStarted => {
                                        self.user_speaking = true;
                                        self.segment_start_bytes =
                                            self.total_audio_bytes - self.audio_buf.len();
                                    }
                                    VadControllerEvent::SpeechStopped => {
                                        self.user_speaking = false;
                                        let start_secs =
                                            byte_offset_to_secs(self.segment_start_bytes);
                                        let audio_data = std::mem::take(&mut self.audio_buf);
                                        let wctx = self.whisper_ctx.clone();
                                        let lang = self.language.clone();

                                        let text = tokio::task::spawn_blocking(move || {
                                            whisper_transcribe(&wctx, &audio_data, &lang)
                                        })
                                        .await
                                        .unwrap();

                                        if !text.is_empty() {
                                            eprint!(".");
                                            self.results.lock().unwrap().push((start_secs, text));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
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

    let processor = VadWhisperProcessor {
        base: ProcessorBase::new("VadWhisper"),
        controller: None,
        whisper_ctx,
        language: language.to_string(),
        audio_buf: Vec::new(),
        vad_buf: Vec::new(),
        total_audio_bytes: 0,
        user_speaking: false,
        segment_start_bytes: 0,
        results: results.clone(),
    };

    let in_params = TransportParams {
        audio_in_enabled: true,
        ..Default::default()
    };
    let input_transport =
        LocalAudioInputTransport::new(in_params, AudioInputSource::Buffer(Bytes::from(wav_data)))
            .with_format(AudioFormat::Wav)
            .with_pacing(AudioPacing::RealTime);

    let pipeline = Pipeline::new(vec![Box::new(input_transport), Box::new(processor)]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let _down = FrameCollector::spawn(down_rx);
    let run = tokio::spawn(async move { node.run().await });

    let start = Instant::now();

    send_frame(
        &handle,
        Frame::Start(StartFrame {
            audio_in_sample_rate: 16000,
            ..Default::default()
        }),
        Direction::Downstream,
    )
    .await;

    // Wait for real-time audio playback to complete.
    let wait_secs = audio_duration_s as u64 + 5;
    eprintln!(
        "Processing {audio_duration_s:.0}s of audio at real-time pace ({wait_secs}s timeout)"
    );
    tokio::time::sleep(Duration::from_secs(wait_secs)).await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;
    timeout(Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap();
    eprintln!();

    let elapsed = start.elapsed().as_secs_f64();
    let results = results.lock().unwrap();
    print_results(
        &format!(
            "Real-time pipeline ({elapsed:.2}s for {audio_duration_s:.0}s audio, {} segments)",
            results.len()
        ),
        &results,
    );
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    // Redirect whisper.cpp C library logging through Rust's `log` crate.
    // With no log subscriber registered, all C output is silently dropped.
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

    match args.mode {
        Mode::Fast => {
            let pcm = wav_to_pcm(&args.audio_file);
            run_fast(&pcm, &whisper_ctx, &args.language);
        }
        Mode::Realtime => {
            run_realtime(&args.audio_file, whisper_ctx, &args.language);
        }
    }
}
