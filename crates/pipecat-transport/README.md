# pipecat-transport

Base transport layer for pipecat-rs. Provides abstract input and output transport processors that concrete implementations (WebSocket, HTTP, WebRTC, etc.) extend.

## Key Types

- **`BaseInputTransport`** — Processor that receives audio/video from an external source and pushes frames into the pipeline. Supports pause/resume, configurable audio filtering, and respects `TransportParams` for enabling/disabling audio/video channels.
- **`BaseOutputTransport`** — Processor that receives frames from the pipeline and sends them to an external sink via async callbacks. Handles audio/video output and transport lifecycle events (connected, disconnected, transmit).
- **`TransportParams`** — Configuration for audio/video enable flags, sample rate, channel count, and optional audio filters.
- **`OutputTransportCallbacks`** — Async callbacks for transport events: `on_connected`, `on_disconnected`, `on_transmit`.

## Architecture

Transports sit at the edges of a pipeline. The input transport pushes externally-received media into the processor chain, while the output transport captures processed frames and sends them out. Both are `FrameProcessor` implementations that integrate with the standard pipeline infrastructure.

Concrete transport implementations (e.g., a WebSocket transport) compose these base types and provide the actual I/O logic through callbacks and the `push_audio_frame` / `push_video_frame` methods.

## Usage

```toml
[dependencies]
pipecat-transport = { path = "crates/pipecat-transport" }
```

## License

BSD-2-Clause
