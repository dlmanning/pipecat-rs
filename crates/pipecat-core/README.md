# pipecat-core

Core foundation for the pipecat-rs framework. Defines the frame types, processor trait, pipeline observer interface, metrics, and error handling that all other crates build on.

## Key Types

- **`Frame`** — Flat enum with ~155 variants covering system lifecycle, data, and control frames
- **`FrameEnvelope`** — Wraps a `Frame` with a `FrameHeader` carrying metadata (ID, PTS, transport routing, etc.)
- **`Direction`** — `Downstream` or `Upstream`, determines frame routing
- **`FrameProcessor`** — Minimal trait: `async fn process_frame(FrameEnvelope, Direction, ProcessorContext)`
- **`ProcessorContext`** — Carries channel senders for pushing frames downstream/upstream, plus optional observer notifications
- **`ProcessorNode`** / **`ProcessorNodeHandle`** — Runtime wrapper that manages priority queuing via `select!` over system and normal channels
- **`PipelineObserver`** — Trait for observing frame processing and push events across the pipeline
- **`ProcessorMetrics`** — TTFB, processing duration, and text aggregation latency tracking

## Architecture

Frames flow between processors through `tokio::mpsc` channels. Each processor is wrapped in a `ProcessorNode` that provides:

- **Priority handling**: System frames (lifecycle, interruptions) are processed ahead of normal frames using `select!`
- **Bidirectional routing**: Frames can flow downstream or upstream through separate channel pairs
- **Observer integration**: Frame processing and push events are reported to an optional `PipelineObserver`

## Features

- **`test-utils`** — Enables test helpers (`make_text_frame`, `make_audio_frame`, channel wiring utilities) for writing processor tests without mocks

## Usage

```toml
[dependencies]
pipecat-core = { path = "crates/pipecat-core" }
```

## License

BSD-2-Clause
