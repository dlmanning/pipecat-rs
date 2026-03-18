# pipecat-rs

A Rust implementation of [pipecat](https://github.com/pipecat-ai/pipecat), the open-source framework for building real-time voice and multimodal conversational AI agents.

This is a from-scratch Rust implementation — not a binding. The Python framework serves as the behavioral specification, reimplemented using idiomatic Rust patterns with `tokio` for async, channels for processor communication, and enums for type-safe frame handling.

## Architecture

Frames flow through a pipeline of processors connected by `tokio::mpsc` channels:

```
Transport In → [Processor] → [Processor] → ... → Transport Out
                    ↑                                  ↓
                 upstream                          downstream
```

- **Frames** are a flat enum (~99 variants) wrapped in an envelope carrying metadata
- **Processors** implement a minimal trait (`process_frame`) and are wrapped in `ProcessorNode` for priority queuing and channel management
- **System frames** (lifecycle, interruptions) are prioritized over normal frames via `select!`
- **Pipelines** wire processors together linearly or in parallel, managed by a task runner

## Crates

| Crate                                            | Description                                                      |
| ------------------------------------------------ | ---------------------------------------------------------------- |
| [`pipecat-core`](crates/pipecat-core/)           | Frame types, processor trait, pipeline observer, metrics, errors |
| [`pipecat-pipeline`](crates/pipecat-pipeline/)   | Pipeline, ParallelPipeline, PipelineTask, PipelineRunner         |
| [`pipecat-transport`](crates/pipecat-transport/) | Base input/output transports for external I/O                    |
| [`pipecat-audio`](crates/pipecat-audio/)         | VAD, audio mixing, codecs, resampling, filtering                 |
| [`pipecat-turns`](crates/pipecat-turns/)         | User turn detection with pluggable start/stop strategies         |
| [`pipecat-context`](crates/pipecat-context/)     | LLM conversation context and message aggregation                 |
| [`pipecat-services`](crates/pipecat-services/)   | LLM, STT, TTS service traits and provider integrations           |

### Dependency Graph

```
pipecat-core
├── pipecat-audio
│   ├── pipecat-transport
│   └── pipecat-turns
│       └── pipecat-context
│           └── pipecat-services
└── pipecat-pipeline
```

## Supported Providers

| Provider        | Type          | Feature Flag |
| --------------- | ------------- | ------------ |
| OpenAI          | LLM, Realtime | `openai`     |
| Deepgram        | STT           | `deepgram`   |
| ElevenLabs      | STT, TTS      | `elevenlabs` |
| Azure Speech    | STT, TTS      | `azure`      |
| AWS Transcribe  | STT           | `aws`        |
| Whisper (local) | STT           | `whisper`    |

## Getting Started

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
pipecat-core = { path = "crates/pipecat-core" }
pipecat-pipeline = { path = "crates/pipecat-pipeline" }
pipecat-services = { path = "crates/pipecat-services", features = ["openai", "deepgram", "elevenlabs", "azure", "aws"] }
pipecat-context = { path = "crates/pipecat-context" }
```

### Implementing a Processor

Processors are the building blocks of a pipeline. Implement the `FrameProcessor` trait to handle or transform frames:

```rust
use async_trait::async_trait;
use pipecat_core::frame::*;
use pipecat_core::processor::*;

struct UpperCaseProcessor {
    base: ProcessorBase,
}

impl UpperCaseProcessor {
    fn new() -> Self {
        Self { base: ProcessorBase::new("UpperCase") }
    }
}

#[async_trait]
impl FrameProcessor for UpperCaseProcessor {
    fn name(&self) -> &str { self.base.name() }
    fn id(&self) -> u64 { self.base.id() }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        match &envelope.frame {
            Frame::Text(t) => {
                ctx.send_downstream(Frame::Text(TextFrame::new(t.text.to_uppercase()))).await?;
            }
            _ => ctx.push_frame(envelope, direction).await?,
        }
        Ok(())
    }
}
```

### Building a Pipeline

Chain processors together into a pipeline and run it:

```rust
use pipecat_pipeline::Pipeline;

let pipeline = Pipeline::new(vec![
    Box::new(stt_service),
    Box::new(user_aggregator),
    Box::new(llm_service),
    Box::new(tts_service),
    Box::new(assistant_aggregator),
]);
```

A typical voice agent pipeline flows: **Transport In → STT → User Aggregator → LLM → TTS → Assistant Aggregator → Transport Out**.

## Feature Matrix

Comparison with the [Python pipecat](https://github.com/pipecat-ai/pipecat) framework.

### Core

| Feature                |     Python      |           Rust            |
| ---------------------- | :-------------: | :-----------------------: |
| Frame types            | 122 frame types |  99 variants (flat enum)  |
| FrameProcessor base    |       Yes       |            Yes            |
| Pipeline               |       Yes       |            Yes            |
| ParallelPipeline       |       Yes       |            Yes            |
| PipelineTask / Runner  |       Yes       |            Yes            |
| Observer pattern       |       Yes       |            Yes            |
| Metrics (TTFB, tokens) |       Yes       |            Yes            |
| Error handling         |       Yes       | Yes (Result + ErrorFrame) |
| Function call registry |       Yes       |            Yes            |

### Transports

| Transport                 | Python |    Rust    |
| ------------------------- | :----: | :--------: |
| Base input/output         |  Yes   |    Yes     |
| Local audio               |  Yes   |    Yes     |
| Audio device playback     |   —    | Yes (cpal) |
| Daily                     |  Yes   |     —      |
| LiveKit                   |  Yes   |     —      |
| WebSocket (client/server) |  Yes   |     —      |
| SmallWebRTC               |  Yes   |     —      |
| HeyGen                    |  Yes   |     —      |
| Tavus                     |  Yes   |     —      |
| WhatsApp                  |  Yes   |     —      |

### Audio

| Feature                      |              Python              |        Rust         |
| ---------------------------- | :------------------------------: | :-----------------: |
| VAD analyzer (state machine) |               Yes                |         Yes         |
| Silero VAD backend           |               Yes                | Yes (feature-gated) |
| Audio mixer                  |         SoundFile mixer          |     Trait only      |
| Audio filter                 | 6 filters (Krisp, RNNoise, etc.) |     Trait only      |
| Resampler (linear)           |                —                 |         Yes         |
| Resampler (sinc)             |                —                 | Yes (feature-gated) |
| Resampler (SoX)              |               Yes                |          —          |
| Opus codec                   |                —                 | Yes (feature-gated) |
| DTMF                         |               Yes                |          —          |

### Turn Management

| Feature                      |    Python    | Rust |
| ---------------------------- | :----------: | :--: |
| User turn controller         |     Yes      | Yes  |
| VAD start strategy           |     Yes      | Yes  |
| External start strategy      |     Yes      | Yes  |
| Transcription start strategy |     Yes      | Yes  |
| Min words start strategy     |     Yes      | Yes  |
| Speech timeout stop strategy |     Yes      | Yes  |
| External stop strategy       |     Yes      | Yes  |
| Turn analyzer stop strategy  |     Yes      | Yes  |
| Transcription stop strategy  |     Yes      |  —   |
| User mute strategies         | 4 strategies |  —   |
| User idle controller         |     Yes      |  —   |

### Context & Aggregation

| Feature                  | Python | Rust |
| ------------------------ | :----: | :--: |
| LLM context              |  Yes   | Yes  |
| User aggregator          |  Yes   | Yes  |
| Assistant aggregator     |  Yes   | Yes  |
| Aggregator pair          |  Yes   | Yes  |
| Context summarization    |  Yes   |  —   |
| Gated context            |  Yes   |  —   |
| Vision/image aggregation |  Yes   |  —   |

### Services — LLM

| Provider                  | Python | Rust |
| ------------------------- | :----: | :--: |
| OpenAI (Chat Completions) |  Yes   | Yes  |
| OpenAI Realtime           |  Yes   | Yes  |
| Anthropic Claude          |  Yes   |  —   |
| Google Gemini             |  Yes   |  —   |
| AWS Bedrock               |  Yes   |  —   |
| Azure OpenAI              |  Yes   |  —   |
| Groq                      |  Yes   |  —   |
| Fireworks                 |  Yes   |  —   |
| Together AI               |  Yes   |  —   |
| Cerebras                  |  Yes   |  —   |
| DeepSeek                  |  Yes   |  —   |
| Mistral                   |  Yes   |  —   |
| Ollama                    |  Yes   |  —   |
| Others (8+)               |  Yes   |  —   |

### Services — STT

| Provider        | Python | Rust |
| --------------- | :----: | :--: |
| Deepgram        |  Yes   | Yes  |
| ElevenLabs      |  Yes   | Yes  |
| Azure           |  Yes   | Yes  |
| AWS Transcribe  |  Yes   | Yes  |
| AssemblyAI      |  Yes   |  —   |
| Google          |  Yes   |  —   |
| Gladia          |  Yes   |  —   |
| Speechmatics    |  Yes   |  —   |
| Whisper (local) |  Yes   | Yes  |
| Others (10+)    |  Yes   |  —   |

### Services — TTS

| Provider     | Python | Rust |
| ------------ | :----: | :--: |
| ElevenLabs   |  Yes   | Yes  |
| Azure        |  Yes   | Yes  |
| Cartesia     |  Yes   |  —   |
| Google       |  Yes   |  —   |
| AWS Polly    |  Yes   |  —   |
| Deepgram     |  Yes   |  —   |
| LMNT         |  Yes   |  —   |
| Kokoro       |  Yes   |  —   |
| Others (15+) |  Yes   |  —   |

### Observers

| Observer                            | Python | Rust |
| ----------------------------------- | :----: | :--: |
| User-bot latency                    |  Yes   | Yes  |
| Startup timing                      |  Yes   |  —   |
| Turn tracking                       |  Yes   |  —   |
| Debug/metrics/transcription loggers |  Yes   |  —   |

### Other

| Feature                              |  Python   |  Rust   |
| ------------------------------------ | :-------: | :-----: |
| Serializers (Protobuf, Twilio, etc.) | 7 formats |    —    |
| Frame filters                        |  7 types  | 4 types |
| LLM/service switching                |    Yes    |    —    |
| IVR / voicemail extensions           |    Yes    |    —    |
| OpenTelemetry tracing                |    Yes    |    —    |

## Examples

See [`examples/`](examples/) for runnable demos.

- **[transcribe](examples/)** — Transcribe a WAV file using Silero VAD + local Whisper STT, with optional speaker playback (`--play`)

```bash
cargo run -p pipecat-examples --bin transcribe -- audio.wav --play
```

## Building

```bash
# Check everything compiles
cargo build

# Run all tests
cargo test

# Run tests for a single crate
cargo test -p pipecat-core

# Lint
cargo clippy -- -D warnings

# Format check
cargo fmt --check
```

## License

BSD-2-Clause
