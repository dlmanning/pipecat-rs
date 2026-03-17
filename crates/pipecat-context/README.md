# pipecat-context

LLM conversation context management for pipecat-rs. Maintains the shared message history and provides aggregators that accumulate streaming text into complete messages.

## Key Types

- **`LLMContext`** — Thread-safe (`Arc<Mutex<>>`) conversation state. Stores messages, tools, and tool choice. Builds OpenAI-format context JSON. Never holds the lock across an await point.
- **`LLMUserAggregator`** — Processor that accumulates user transcription text parts into complete messages and appends them to the context.
- **`LLMAssistantAggregator`** — Processor that accumulates assistant response text parts into complete messages and appends them to the context.
- **`LLMContextAggregatorPair`** — Convenience wrapper holding both user and assistant aggregators sharing the same `LLMContext`.
- **`TextPart`** — Text fragment with spacing metadata, used during aggregation.

## Architecture

The aggregators sit in the pipeline between the transport and the LLM service. The user aggregator collects transcription frames into a user message, while the assistant aggregator collects LLM response tokens into an assistant message. Both push completed messages to the shared `LLMContext`, which the LLM service reads when generating the next completion.

## Usage

```toml
[dependencies]
pipecat-context = { path = "crates/pipecat-context" }
```

## License

BSD-2-Clause
