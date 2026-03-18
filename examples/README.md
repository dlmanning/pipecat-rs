# Examples

## transcribe

Transcribe a WAV file using Silero VAD + local Whisper STT. Demonstrates building a pipeline with `PipelineTask`, `LocalAudioInputTransport`, `VadProcessor`, and a custom `WhisperTranscribeProcessor`.

```bash
# Transcribe as fast as possible (default)
cargo run -p pipecat-examples --bin transcribe -- audio.wav

# Transcribe at real-time pace
cargo run -p pipecat-examples --bin transcribe -- audio.wav --realtime

# Transcribe and play audio through speakers
cargo run -p pipecat-examples --bin transcribe -- audio.wav --play
```

The Whisper model (`tiny.en` by default) is downloaded automatically to `~/.cache/pipecat-rs/whisper/` on first run.

### Pipeline

```
LocalAudioInput → VadProcessor → [AudioPlayer] → WhisperTranscribe
```

- **LocalAudioInput** reads the WAV file and feeds 20ms audio chunks into the pipeline
- **VadProcessor** detects speech start/stop using Silero VAD
- **AudioPlayer** (optional, `--play`) plays audio through the default system output device via cpal
- **WhisperTranscribe** buffers audio during speech segments and transcribes on speech stop

### Options

| Flag                | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| `--realtime`        | Process audio at real-time pace instead of as fast as possible |
| `--play`            | Play audio through speakers (implies `--realtime`)             |
| `--model <name>`    | Whisper GGML model name (default: `tiny.en`)                   |
| `--language <code>` | Language code (default: `en`)                                  |
