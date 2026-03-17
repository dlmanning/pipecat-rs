# pipecat-services

AI service integrations for pipecat-rs. Provides base traits for LLM, STT, and TTS services, along with concrete implementations for popular providers.

## Service Traits

- **`LLMService`** — Base trait for large language model services. Manages settings, function call registries, and frame routing.
- **`STTService`** — Base trait for speech-to-text services. Routes audio frames and emits transcription frames.
- **`TTSService`** — Base trait for text-to-speech services. Synthesizes text into audio frames with context ID tracking for word timestamps.
- **`ServiceBase`** — Shared foundation combining processor identity with metrics (TTFB, processing duration).

## Providers

| Provider   | Feature Flag | Capabilities                                                |
| ---------- | ------------ | ----------------------------------------------------------- |
| OpenAI     | `openai`     | LLM (Chat Completions), Realtime API (WebSocket multimodal) |
| Deepgram   | `deepgram`   | STT (WebSocket streaming)                                   |
| ElevenLabs | `elevenlabs` | TTS (WebSocket streaming)                                   |

## Additional Components

- **`SimpleTextAggregator`** — Accumulates text tokens into sentences with configurable `TextAggregationMode`
- **`FunctionCallRegistry`** — Registers and dispatches function call handlers for LLM tool use
- **`LatencyObserver`** — Pipeline observer that measures service TTFB and total processing time

## Usage

```toml
[dependencies]
pipecat-services = { path = "crates/pipecat-services", features = ["openai", "deepgram", "elevenlabs"] }
```

Enable only the provider features you need to keep dependencies minimal.

## License

BSD-2-Clause
