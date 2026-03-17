use crate::context::LLMContext;
use crate::text::{TextPart, concatenate_aggregated_text};

/// Shared aggregation logic for both user and assistant aggregators.
///
/// This is a helper struct (not a trait) embedded in both aggregator types.
/// It manages text accumulation and context message creation.
#[derive(Debug)]
pub struct LLMContextAggregatorBase {
    context: LLMContext,
    role: String,
    aggregation: Vec<TextPart>,
}

impl LLMContextAggregatorBase {
    pub fn new(context: LLMContext, role: impl Into<String>) -> Self {
        Self {
            context,
            role: role.into(),
            aggregation: Vec::new(),
        }
    }

    /// Get the concatenated aggregation string.
    pub fn aggregation_string(&self) -> String {
        concatenate_aggregated_text(&self.aggregation)
    }

    /// Whether there is any accumulated text.
    pub fn has_aggregation(&self) -> bool {
        !self.aggregation.is_empty()
    }

    /// Reset the aggregation buffer.
    pub fn reset(&mut self) {
        self.aggregation.clear();
    }

    /// Add a text part to the aggregation.
    pub fn add_text_part(&mut self, text: String, includes_inter_part_spaces: bool) {
        self.aggregation.push(TextPart {
            text,
            includes_inter_part_spaces,
        });
    }

    /// Push the aggregated text as a message to the context.
    /// Returns the aggregated text, or empty string if nothing was aggregated.
    pub fn push_aggregation(&mut self) -> String {
        let text = self.aggregation_string();
        if !text.is_empty() {
            self.context.add_message(serde_json::json!({
                "role": self.role,
                "content": text,
            }));
        }
        self.aggregation.clear();
        text
    }

    /// Access the shared context.
    pub fn context(&self) -> &LLMContext {
        &self.context
    }

    /// Get the role.
    pub fn role(&self) -> &str {
        &self.role
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_base() -> LLMContextAggregatorBase {
        let ctx = LLMContext::new(vec![json!({"role": "system", "content": "test"})]);
        LLMContextAggregatorBase::new(ctx, "user")
    }

    #[test]
    fn empty_aggregation() {
        let base = make_base();
        assert!(!base.has_aggregation());
        assert_eq!(base.aggregation_string(), "");
    }

    #[test]
    fn add_and_get_aggregation() {
        let mut base = make_base();
        base.add_text_part("hello".into(), false);
        base.add_text_part("world".into(), false);
        assert!(base.has_aggregation());
        assert_eq!(base.aggregation_string(), "hello world");
    }

    #[test]
    fn push_aggregation_adds_message() {
        let mut base = make_base();
        base.add_text_part("hello world".into(), false);

        let text = base.push_aggregation();
        assert_eq!(text, "hello world");
        assert!(!base.has_aggregation());

        let msgs = base.context().get_messages();
        assert_eq!(msgs.len(), 2); // system + user
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "hello world");
    }

    #[test]
    fn push_empty_aggregation_does_not_add_message() {
        let mut base = make_base();
        let text = base.push_aggregation();
        assert_eq!(text, "");
        assert_eq!(base.context().message_count(), 1); // just system
    }

    #[test]
    fn reset_clears_aggregation() {
        let mut base = make_base();
        base.add_text_part("hello".into(), false);
        base.reset();
        assert!(!base.has_aggregation());
        assert_eq!(base.aggregation_string(), "");
    }

    #[test]
    fn multiple_push_cycles() {
        let mut base = make_base();

        base.add_text_part("first".into(), false);
        base.push_aggregation();

        base.add_text_part("second".into(), false);
        base.push_aggregation();

        let msgs = base.context().get_messages();
        assert_eq!(msgs.len(), 3); // system + 2 user messages
    }
}
