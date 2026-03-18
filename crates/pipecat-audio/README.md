# pipecat-audio

Audio processing utilities for pipecat-rs. Provides voice activity detection (VAD), audio mixing, codecs, resampling, filtering, and interruption handling.

## Key Types

- **`VadAnalyzer`** — Trait for voice activity detection implementations
- **`VadAnalyzerBase`** — State machine implementation with configurable thresholds, tracking `VadState` (Quiet, Starting, Speaking, Stopping) transitions
- **`VadController`** — Wraps an analyzer and emits `VadControllerEvent`s for speech start/stop
- **`AudioFilter`** — Trait for audio input/output filtering with dynamic configuration via `FilterControlFrame`
- **`Resampler`** — Trait for audio resampling, with linear and optional sinc implementations

## Features

- **`silero`** — Silero VAD backend via the `ort` crate (ONNX Runtime), bundles the ~2.3MB model
- **`opus`** — Opus codec support via the `audiopus` crate
- **`sinc-resampler`** — High-quality sinc resampling via the `rubato` crate

## Modules

| Module         | Description                                             |
| -------------- | ------------------------------------------------------- |
| `vad`          | Voice activity detection analyzer and controller        |
| `mixer`        | Audio stream mixing                                     |
| `codec`        | Audio codec support (PCM, raw audio)                    |
| `opus`         | Opus codec (feature-gated)                              |
| `filter`       | Audio filtering trait and control frames                |
| `resampler`    | Linear and sinc resampling                              |
| `interruption` | Queue draining and frame filtering during interruptions |
| `turn`         | Turn-taking audio utilities                             |

## Usage

```toml
[dependencies]
pipecat-audio = { path = "crates/pipecat-audio" }

# With optional features
pipecat-audio = { path = "crates/pipecat-audio", features = ["silero", "opus", "sinc-resampler"] }
```

## License

BSD-2-Clause
