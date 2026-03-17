# pipecat-turns

User turn detection and management for pipecat-rs. Orchestrates configurable start and stop strategies to determine when a user is speaking and when their turn is complete.

## Key Types

- **`UserTurnController`** — Central controller that routes frames to start/stop strategies and manages turn state. Not a `FrameProcessor` itself — it's owned by the input transport and returns `Vec<TurnAction>` for the caller to execute.
- **`TurnAction`** — Enum: `UserTurnStarted`, `UserTurnStopped` (with associated params)
- **`UserTurnStrategies`** — Holds vectors of start and stop strategies to evaluate

### Start Strategies

| Strategy                             | Trigger                    |
| ------------------------------------ | -------------------------- |
| `VadUserTurnStartStrategy`           | Voice activity detected    |
| `TranscriptionUserTurnStartStrategy` | Transcription received     |
| `MinWordsUserTurnStartStrategy`      | Minimum word count reached |
| `ExternalUserTurnStartStrategy`      | External signal            |

### Stop Strategies

| Strategy                            | Trigger                           |
| ----------------------------------- | --------------------------------- |
| `SpeechTimeoutUserTurnStopStrategy` | Silence timeout after speech      |
| `TurnAnalyzerUserTurnStopStrategy`  | Semantic turn completion analysis |
| `ExternalUserTurnStopStrategy`      | External signal                   |

## Architecture

The controller evaluates start strategies when no turn is active and stop strategies when the user is speaking. A global inactivity timeout can also end a turn. Strategies are composable — multiple start and stop strategies can be combined for nuanced turn detection.

## Usage

```toml
[dependencies]
pipecat-turns = { path = "crates/pipecat-turns" }
```

## License

BSD-2-Clause
