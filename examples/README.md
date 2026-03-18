# Examples

## transcribe

Transcribe audio using Silero VAD + local Whisper STT. Supports file input (WAV, MP3, FLAC, OGG/Vorbis, AAC via symphonia) or live microphone capture. Demonstrates building a pipeline with `PipelineTask`, `LocalAudioInputTransport`/`MicInput`, `VadProcessor`, and a custom `WhisperTranscribeProcessor`.

```bash
# Transcribe a file as fast as possible (default)
cargo run -p pipecat-examples --bin transcribe -- audio.wav

# Transcribe an MP3 file
cargo run -p pipecat-examples --bin transcribe -- recording.mp3

# Transcribe at real-time pace
cargo run -p pipecat-examples --bin transcribe -- audio.wav --realtime

# Transcribe and play audio through speakers
cargo run -p pipecat-examples --bin transcribe -- audio.wav --play

# Transcribe from microphone
cargo run -p pipecat-examples --bin transcribe -- --mic

# Transcribe from a specific microphone
cargo run -p pipecat-examples --bin transcribe -- --mic --device "MacBook Pro Microphone"

# List available audio input devices
cargo run -p pipecat-examples --bin transcribe -- --list-devices
```

The Whisper model (`tiny.en` by default) is downloaded automatically to `~/.cache/pipecat-rs/whisper/` on first run.

### Pipeline

```
Input (file or mic) → VadProcessor → [AudioPlayer] → WhisperTranscribe
```

- **Input** reads audio from a file or captures from the microphone, feeding 20ms chunks into the pipeline
- **VadProcessor** detects speech start/stop using Silero VAD
- **AudioPlayer** (optional, `--play`, file mode only) plays audio through the default system output device via cpal
- **WhisperTranscribe** buffers audio during speech segments and transcribes on speech stop

### Options

| Flag                   | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| `--mic`                | Use system microphone as input (Ctrl+C to stop)                |
| `--device <name>`      | Select a specific input device (requires `--mic`)              |
| `--list-devices`       | List available audio input devices and exit                    |
| `--realtime`           | Process audio at real-time pace instead of as fast as possible |
| `--play`               | Play audio through speakers (implies `--realtime`, file only)  |
| `--model <name>`       | Whisper GGML model name (default: `tiny.en`)                   |
| `--language <code>`    | Language code (default: `en`)                                  |
| `--stop-secs <f64>`   | Silence duration before speech stop (default: `0.2`)           |
