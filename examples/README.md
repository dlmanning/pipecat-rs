# Examples

Examples are organized into `foundational/` for core pipeline demonstrations.

## foundational/listen-and-respond

A voice conversational agent running entirely locally — no API keys or network needed.

Mic → VAD → Whisper STT → User Aggregator → Claude Code LLM → macOS Say TTS → Speaker → Assistant Aggregator

Requires macOS (for `say` TTS) and the `claude` CLI installed and authenticated.

```bash
# Default: sonnet model, Samantha voice
cargo run -p pipecat-examples --bin listen-and-respond

# Choose a different model and voice
cargo run -p pipecat-examples --bin listen-and-respond -- --model opus --voice Alex

# Custom system prompt
cargo run -p pipecat-examples --bin listen-and-respond -- --system "You are a pirate. Respond in pirate speak."

# List audio devices
cargo run -p pipecat-examples --bin listen-and-respond -- --list-devices

# Use a specific microphone
cargo run -p pipecat-examples --bin listen-and-respond -- --device "MacBook Pro Microphone"
```

### Pipeline

```
MicInput → VadProcessor → WhisperSTT → UserAggregator → ClaudeCodeLLM → MacOSSayTTS → AudioPlayer → AssistantAggregator
```

- **MicInput** captures audio from the system microphone
- **VadProcessor** detects speech start/stop using Silero VAD
- **WhisperSTT** transcribes speech segments locally using whisper.cpp
- **UserAggregator** accumulates transcriptions into user messages and manages turn lifecycle
- **ClaudeCodeLLM** generates responses via the `claude` CLI (built-in tools disabled, LLM mode)
- **MacOSSayTTS** synthesizes speech using the macOS `say` command
- **AudioPlayer** plays audio through the default system output device
- **AssistantAggregator** records assistant responses back into the conversation context

### Options

| Flag                 | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `--model <name>`     | Claude model: sonnet, opus, haiku (default: sonnet)    |
| `--voice <name>`     | macOS Say voice (default: Samantha)                    |
| `--speech-rate <n>`  | TTS words per minute                                   |
| `--whisper-model <n>`| Whisper GGML model name (default: tiny.en)             |
| `--language <code>`  | Language code for Whisper (default: en)                 |
| `--device <name>`    | Select a specific input device                         |
| `--list-devices`     | List available audio input devices and exit            |
| `--stop-secs <f64>` | Silence duration before speech stop (default: 0.5)     |
| `--system <text>`    | Custom system instruction for the LLM                  |

---

## foundational/transcribe

Transcribe audio using Silero VAD + local Whisper STT. Supports file input (WAV, MP3, FLAC, OGG/Vorbis, AAC) or live microphone capture.

```bash
# Transcribe a file
cargo run -p pipecat-examples --bin transcribe -- audio.wav

# Transcribe and play audio
cargo run -p pipecat-examples --bin transcribe -- audio.wav --play

# Transcribe from microphone
cargo run -p pipecat-examples --bin transcribe -- --mic

# List available audio input devices
cargo run -p pipecat-examples --bin transcribe -- --list-devices
```

### Pipeline

```
Input (file or mic) → VadProcessor → [AudioPlayer] → WhisperTranscribe
```

### Options

| Flag                 | Description                                                    |
| -------------------- | -------------------------------------------------------------- |
| `--mic`              | Use system microphone as input (Ctrl+C to stop)                |
| `--device <name>`    | Select a specific input device (requires `--mic`)              |
| `--list-devices`     | List available audio input devices and exit                    |
| `--realtime`         | Process audio at real-time pace instead of as fast as possible |
| `--play`             | Play audio through speakers (implies `--realtime`, file only)  |
| `--model <name>`     | Whisper GGML model name (default: `tiny.en`)                   |
| `--language <code>`  | Language code (default: `en`)                                  |
| `--stop-secs <f64>` | Silence duration before speech stop (default: `0.2`)           |
