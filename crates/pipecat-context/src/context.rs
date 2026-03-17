use std::sync::{Arc, Mutex};

/// Shared LLM conversation state.
///
/// Wraps `Arc<Mutex<Inner>>` so user and assistant aggregators can share the
/// same context. The lock is never held across await points.
#[derive(Debug, Clone)]
pub struct LLMContext {
    inner: Arc<Mutex<LLMContextInner>>,
}

#[derive(Debug)]
struct LLMContextInner {
    messages: Vec<serde_json::Value>,
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
}

impl LLMContext {
    /// Create a new context with initial messages.
    pub fn new(messages: Vec<serde_json::Value>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LLMContextInner {
                messages,
                tools: None,
                tool_choice: None,
            })),
        }
    }

    /// Add a single message to the context.
    pub fn add_message(&self, message: serde_json::Value) {
        self.inner.lock().unwrap().messages.push(message);
    }

    /// Add multiple messages to the context.
    pub fn add_messages(&self, messages: &[serde_json::Value]) {
        self.inner
            .lock()
            .unwrap()
            .messages
            .extend_from_slice(messages);
    }

    /// Replace all messages.
    pub fn set_messages(&self, messages: Vec<serde_json::Value>) {
        self.inner.lock().unwrap().messages = messages;
    }

    /// Get a clone of all messages.
    pub fn get_messages(&self) -> Vec<serde_json::Value> {
        self.inner.lock().unwrap().messages.clone()
    }

    /// Get the number of messages.
    pub fn message_count(&self) -> usize {
        self.inner.lock().unwrap().messages.len()
    }

    /// Set the tools configuration.
    pub fn set_tools(&self, tools: serde_json::Value) {
        self.inner.lock().unwrap().tools = Some(tools);
    }

    /// Get the tools configuration.
    pub fn tools(&self) -> Option<serde_json::Value> {
        self.inner.lock().unwrap().tools.clone()
    }

    /// Set the tool choice configuration.
    pub fn set_tool_choice(&self, tool_choice: serde_json::Value) {
        self.inner.lock().unwrap().tool_choice = Some(tool_choice);
    }

    /// Get the tool choice configuration.
    pub fn tool_choice(&self) -> Option<serde_json::Value> {
        self.inner.lock().unwrap().tool_choice.clone()
    }

    /// Build an OpenAI-format context value containing messages, tools, and tool_choice.
    pub fn to_context_value(&self) -> serde_json::Value {
        let inner = self.inner.lock().unwrap();
        let mut ctx = serde_json::json!({
            "messages": inner.messages,
        });
        if let Some(ref tools) = inner.tools {
            ctx["tools"] = tools.clone();
        }
        if let Some(ref tool_choice) = inner.tool_choice {
            ctx["tool_choice"] = tool_choice.clone();
        }
        ctx
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn new_context_with_messages() {
        let ctx = LLMContext::new(vec![
            json!({"role": "system", "content": "You are helpful."}),
        ]);
        let msgs = ctx.get_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "system");
    }

    #[test]
    fn add_message() {
        let ctx = LLMContext::new(vec![]);
        ctx.add_message(json!({"role": "user", "content": "hello"}));
        assert_eq!(ctx.message_count(), 1);
        assert_eq!(ctx.get_messages()[0]["content"], "hello");
    }

    #[test]
    fn add_messages() {
        let ctx = LLMContext::new(vec![]);
        ctx.add_messages(&[
            json!({"role": "user", "content": "a"}),
            json!({"role": "assistant", "content": "b"}),
        ]);
        assert_eq!(ctx.message_count(), 2);
    }

    #[test]
    fn set_messages_replaces() {
        let ctx = LLMContext::new(vec![json!({"role": "user", "content": "old"})]);
        ctx.set_messages(vec![json!({"role": "user", "content": "new"})]);
        assert_eq!(ctx.message_count(), 1);
        assert_eq!(ctx.get_messages()[0]["content"], "new");
    }

    #[test]
    fn shared_context_via_clone() {
        let ctx1 = LLMContext::new(vec![]);
        let ctx2 = ctx1.clone();
        ctx1.add_message(json!({"role": "user", "content": "from ctx1"}));
        assert_eq!(ctx2.message_count(), 1);
        assert_eq!(ctx2.get_messages()[0]["content"], "from ctx1");
    }

    #[test]
    fn tools_management() {
        let ctx = LLMContext::new(vec![]);
        assert!(ctx.tools().is_none());

        let tools = json!([{"type": "function", "function": {"name": "get_weather"}}]);
        ctx.set_tools(tools.clone());
        assert_eq!(ctx.tools(), Some(tools));
    }

    #[test]
    fn tool_choice_management() {
        let ctx = LLMContext::new(vec![]);
        assert!(ctx.tool_choice().is_none());

        ctx.set_tool_choice(json!("auto"));
        assert_eq!(ctx.tool_choice(), Some(json!("auto")));
    }

    #[test]
    fn to_context_value() {
        let ctx = LLMContext::new(vec![json!({"role": "user", "content": "hi"})]);
        ctx.set_tools(json!([{"type": "function"}]));
        ctx.set_tool_choice(json!("auto"));

        let val = ctx.to_context_value();
        assert_eq!(val["messages"][0]["content"], "hi");
        assert!(val["tools"].is_array());
        assert_eq!(val["tool_choice"], "auto");
    }

    #[test]
    fn to_context_value_without_tools() {
        let ctx = LLMContext::new(vec![json!({"role": "user", "content": "hi"})]);
        let val = ctx.to_context_value();
        assert!(val.get("tools").is_none());
        assert!(val.get("tool_choice").is_none());
    }
}
