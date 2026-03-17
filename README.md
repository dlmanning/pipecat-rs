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
