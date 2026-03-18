# Examples

Each example is its own crate with only the dependencies it needs.

## listen-and-respond

A voice conversational agent using local services. No third-party API keys needed — requires macOS (for `say` TTS) and the `claude` CLI installed and authenticated.

Mic → VAD → Whisper STT → User Aggregator → Claude Code LLM → macOS Say TTS → Speaker → Assistant Aggregator

```bash
# Default: sonnet model, Samantha voice
cargo run -p listen-and-respond

# Choose a different model and voice
cargo run -p listen-and-respond -- --model opus --voice Alex

# Custom system prompt
cargo run -p listen-and-respond -- --system "You are a pirate. Respond in pirate speak."

# List audio devices
cargo run -p listen-and-respond -- --list-devices
```

### Pipeline

```
MicInput → VadProcessor → WhisperSTT → ConversationLogger → UserAggregator → ClaudeCodeLLM → MacOSSayTTS → AudioPlayer → AssistantAggregator
```

### Options

| Flag                  | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `--model <name>`      | Claude model: sonnet, opus, haiku (default: sonnet)    |
| `--voice <name>`      | macOS Say voice (default: Samantha)                    |
| `--speech-rate <n>`   | TTS words per minute                                   |
| `--whisper-model <n>` | Whisper GGML model name (default: tiny.en)             |
| `--language <code>`   | Language code for Whisper (default: en)                 |
| `--device <name>`     | Select a specific input device                         |
| `--list-devices`      | List available audio input devices and exit            |
| `--stop-secs <f64>`  | Silence duration before speech stop (default: 0.5)     |
| `--system <text>`     | Custom system instruction for the LLM                  |

---

## transcribe

Transcribe audio using Silero VAD + local Whisper STT. Supports file input (WAV, MP3, FLAC, OGG/Vorbis, AAC) or live microphone capture.

```bash
cargo run -p transcribe -- audio.wav
cargo run -p transcribe -- audio.wav --play
cargo run -p transcribe -- --mic
cargo run -p transcribe -- --list-devices
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
