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

- **Frames** are a flat enum (~155 variants) wrapped in an envelope carrying metadata
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

| Provider   | Type          | Feature Flag |
| ---------- | ------------- | ------------ |
| OpenAI     | LLM, Realtime | `openai`     |
| Deepgram   | STT           | `deepgram`   |
| ElevenLabs | TTS           | `elevenlabs` |

## Getting Started

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
pipecat-core = { path = "crates/pipecat-core" }
pipecat-pipeline = { path = "crates/pipecat-pipeline" }
pipecat-services = { path = "crates/pipecat-services", features = ["openai", "deepgram", "elevenlabs"] }
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
