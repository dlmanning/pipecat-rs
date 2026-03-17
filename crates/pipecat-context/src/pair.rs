use pipecat_core::processor::FrameProcessor;

use crate::assistant_aggregator::LLMAssistantAggregator;
use crate::context::LLMContext;
use crate::user_aggregator::{LLMUserAggregator, LLMUserAggregatorParams};

/// Paired user and assistant aggregators sharing the same LLM context.
///
/// In a typical pipeline:
/// ```text
/// [TransportInput] → [UserAggregator] → [LLM] → [TTS] → [TransportOutput] → [AssistantAggregator]
/// ```
#[derive(Debug)]
pub struct LLMContextAggregatorPair {
    user: LLMUserAggregator,
    assistant: LLMAssistantAggregator,
}

impl LLMContextAggregatorPair {
    /// Create a paired set of aggregators sharing the given context.
    pub fn new(context: LLMContext, user_params: LLMUserAggregatorParams) -> Self {
        let user = LLMUserAggregator::new(context.clone(), user_params);
        let assistant = LLMAssistantAggregator::new(context);
        Self { user, assistant }
    }

    pub fn user(&self) -> &LLMUserAggregator {
        &self.user
    }

    pub fn user_mut(&mut self) -> &mut LLMUserAggregator {
        &mut self.user
    }

    pub fn assistant(&self) -> &LLMAssistantAggregator {
        &self.assistant
    }

    pub fn assistant_mut(&mut self) -> &mut LLMAssistantAggregator {
        &mut self.assistant
    }

    /// Consume self and return the aggregators as boxed FrameProcessors
    /// for pipeline wiring.
    pub fn into_processors(self) -> (Box<dyn FrameProcessor>, Box<dyn FrameProcessor>) {
        (Box::new(self.user), Box::new(self.assistant))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipecat_turns::{
        SpeechTimeoutUserTurnStopStrategy, UserTurnStrategies, VadUserTurnStartStrategy,
    };
    use serde_json::json;
    use std::time::Duration;

    fn make_pair() -> LLMContextAggregatorPair {
        let context = LLMContext::new(vec![json!({"role": "system", "content": "test"})]);
        let params = LLMUserAggregatorParams {
            user_turn_strategies: UserTurnStrategies {
                start: vec![Box::new(VadUserTurnStartStrategy::new())],
                stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.1))],
            },
            user_turn_stop_timeout: Duration::from_secs(5),
        };
        LLMContextAggregatorPair::new(context, params)
    }

    #[test]
    fn shared_context() {
        let pair = make_pair();
        // Both should see the same initial context
        assert_eq!(pair.user().context().message_count(), 1);
        assert_eq!(pair.assistant().context().message_count(), 1);

        // Adding via one should be visible from the other
        pair.user()
            .context()
            .add_message(json!({"role": "user", "content": "hi"}));
        assert_eq!(pair.assistant().context().message_count(), 2);
    }

    #[test]
    fn into_processors() {
        let pair = make_pair();
        let (user, assistant) = pair.into_processors();
        assert_eq!(user.name(), "LLMUserAggregator");
        assert_eq!(assistant.name(), "LLMAssistantAggregator");
    }
}
