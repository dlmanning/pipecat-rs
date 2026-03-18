# pipecat-transport

Base transport layer for pipecat-rs. Provides abstract input and output transport processors that concrete implementations (WebSocket, HTTP, WebRTC, etc.) extend, plus local transports for file/buffer I/O and audio device playback.

## Key Types

- **`BaseInputTransport`** — Processor that receives audio/video from an external source and pushes frames into the pipeline. Supports pause/resume, configurable audio filtering, and respects `TransportParams` for enabling/disabling audio/video channels.
- **`BaseOutputTransport`** — Processor that receives frames from the pipeline and sends them to an external sink via async callbacks. Handles audio/video output and transport lifecycle events (connected, disconnected, transmit).
- **`LocalAudioInputTransport`** — Reads audio from a buffer, file, or channel and feeds it into the pipeline. Supports raw PCM and multi-format decoding (WAV, MP3, FLAC, OGG/Vorbis, AAC via symphonia). Supports real-time pacing.
- **`LocalAudioOutputTransport`** — Writes pipeline audio to a buffer, file, or discard sink. Useful for tests and offline processing.
- **`MicInput`** — Captures audio from the system microphone via cpal and feeds it into the pipeline. Automatically resamples from device sample rate to the pipeline's target rate. Includes `list_input_devices()` for device enumeration. Requires the `cpal` feature.
- **`AudioPlayer`** — Standalone `FrameProcessor` that plays `InputAudioRaw` frames through a system audio device via cpal. Handles resampling and channel mapping to the device's native format. Requires the `cpal` feature.
- **`TransportParams`** — Configuration for audio/video enable flags, sample rate, channel count, and optional audio filters.

## Features

- **`cpal`** — Enables `AudioPlayer` for system audio device playback and `MicInput` for microphone capture via the [cpal](https://crates.io/crates/cpal) crate.

Multi-format audio decoding (WAV, MP3, FLAC, OGG/Vorbis, AAC) is always available via [symphonia](https://crates.io/crates/symphonia). Use `AudioFormat::Encoded` for auto-detection of any supported format.

## Architecture

Transports sit at the edges of a pipeline. The input transport pushes externally-received media into the processor chain, while the output transport captures processed frames and sends them out. Both are `FrameProcessor` implementations that integrate with the standard pipeline infrastructure.

Concrete transport implementations (e.g., a WebSocket transport) compose the base types and provide the actual I/O logic through callbacks. Local transports provide file and buffer-based I/O for testing and CLI tools.

`AudioPlayer` is a lightweight processor (not a transport) that taps into the audio stream and plays it through speakers. It passes all frames through unchanged, so downstream processors still receive the audio.

`MicInput` wraps `BaseInputTransport` to capture live audio from the system microphone. A cpal input stream runs on a dedicated thread, writing to a shared buffer that a tokio task drains into 20ms pipeline-ready chunks.

## Usage

```toml
[dependencies]
pipecat-transport = { path = "crates/pipecat-transport" }

# With audio device playback
pipecat-transport = { path = "crates/pipecat-transport", features = ["cpal"] }
```

## License

BSD-2-Clause
