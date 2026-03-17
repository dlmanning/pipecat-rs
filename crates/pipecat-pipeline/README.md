# pipecat-pipeline

Pipeline orchestration for pipecat-rs. Connects processors into linear or parallel pipelines, manages task lifecycle, and runs pipelines to completion.

## Key Types

- **`Pipeline`** — Linear chain of processors wired together via channels
- **`ParallelPipeline`** — Branches a pipeline into parallel paths, using `tokio::sync::Barrier` to synchronize lifecycle frames (Start, End, Cancel) across branches
- **`PipelineTask`** — Wraps a pipeline with task-level lifecycle: idle timeouts, heartbeats, cancellation, and error/event callbacks
- **`PipelineParams`** — Configuration for audio rates, interruption behavior, metrics, heartbeat intervals, idle/cancel timeouts
- **`PipelineRunner`** — Manages task lifecycle, spawns tasks as `JoinHandle`s, and routes frames between tasks

## Architecture

A typical setup:

1. Create processors and add them to a `Pipeline`
2. Wrap the pipeline in a `PipelineTask` with desired parameters and callbacks
3. Run the task with `PipelineRunner`

The runner handles spawning, frame routing between the transport and pipeline internals, and clean shutdown.

## Usage

```toml
[dependencies]
pipecat-pipeline = { path = "crates/pipecat-pipeline" }
```

## License

BSD-2-Clause
