use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use pipecat_context::LLMContext;

/// Parameters passed to a function call handler.
#[derive(Debug, Clone)]
pub struct FunctionCallParams {
    pub function_name: String,
    pub tool_call_id: String,
    pub arguments: serde_json::Value,
    pub context: LLMContext,
}

/// The result type returned by function call handlers.
pub type FunctionCallResult = serde_json::Value;

/// A function call handler: async function that takes params and returns a JSON result.
pub type FunctionCallHandler = Arc<
    dyn Fn(FunctionCallParams) -> Pin<Box<dyn Future<Output = FunctionCallResult> + Send>>
        + Send
        + Sync,
>;

/// Registration entry for a single function call handler.
#[derive(Clone)]
pub struct FunctionCallRegistryItem {
    /// The function name this handler matches. `None` is a catch-all handler.
    pub function_name: Option<String>,
    pub handler: FunctionCallHandler,
    pub cancel_on_interruption: bool,
    pub timeout: Duration,
}

impl std::fmt::Debug for FunctionCallRegistryItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionCallRegistryItem")
            .field("function_name", &self.function_name)
            .field("cancel_on_interruption", &self.cancel_on_interruption)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Registry of function call handlers, keyed by function name.
///
/// Supports exact-match lookup by function name and a catch-all handler
/// registered under `None`. Exact matches take precedence over the catch-all.
#[derive(Debug, Default, Clone)]
pub struct FunctionCallRegistry {
    functions: HashMap<Option<String>, FunctionCallRegistryItem>,
}

impl FunctionCallRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a handler for a specific function name.
    pub fn register(
        &mut self,
        function_name: impl Into<String>,
        handler: FunctionCallHandler,
        cancel_on_interruption: bool,
        timeout: Duration,
    ) {
        let name = function_name.into();
        self.functions.insert(
            Some(name.clone()),
            FunctionCallRegistryItem {
                function_name: Some(name),
                handler,
                cancel_on_interruption,
                timeout,
            },
        );
    }

    /// Register a catch-all handler that matches any unregistered function name.
    pub fn register_catch_all(
        &mut self,
        handler: FunctionCallHandler,
        cancel_on_interruption: bool,
        timeout: Duration,
    ) {
        self.functions.insert(
            None,
            FunctionCallRegistryItem {
                function_name: None,
                handler,
                cancel_on_interruption,
                timeout,
            },
        );
    }

    /// Unregister a handler for a specific function name.
    pub fn unregister(&mut self, function_name: &str) {
        self.functions.remove(&Some(function_name.to_string()));
    }

    /// Unregister the catch-all handler.
    pub fn unregister_catch_all(&mut self) {
        self.functions.remove(&None);
    }

    /// Look up a handler for the given function name.
    /// Returns the exact match if found, otherwise the catch-all.
    pub fn lookup(&self, function_name: &str) -> Option<&FunctionCallRegistryItem> {
        self.functions
            .get(&Some(function_name.to_string()))
            .or_else(|| self.functions.get(&None))
    }

    /// Check whether a handler is registered for the given function name
    /// (either exact match or catch-all).
    pub fn has_function(&self, function_name: &str) -> bool {
        self.lookup(function_name).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_handler() -> FunctionCallHandler {
        Arc::new(|_| Box::pin(async { serde_json::json!({"result": "ok"}) }))
    }

    #[test]
    fn register_and_lookup() {
        let mut reg = FunctionCallRegistry::new();
        reg.register(
            "get_weather",
            dummy_handler(),
            false,
            Duration::from_secs(10),
        );

        let item = reg.lookup("get_weather");
        assert!(item.is_some());
        assert_eq!(item.unwrap().function_name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn lookup_missing_returns_none() {
        let reg = FunctionCallRegistry::new();
        assert!(reg.lookup("nonexistent").is_none());
    }

    #[test]
    fn unregister() {
        let mut reg = FunctionCallRegistry::new();
        reg.register(
            "get_weather",
            dummy_handler(),
            false,
            Duration::from_secs(10),
        );
        reg.unregister("get_weather");
        assert!(!reg.has_function("get_weather"));
    }

    #[test]
    fn catch_all_handler() {
        let mut reg = FunctionCallRegistry::new();
        reg.register_catch_all(dummy_handler(), true, Duration::from_secs(5));

        let item = reg.lookup("any_function");
        assert!(item.is_some());
        assert!(item.unwrap().function_name.is_none());
        assert!(item.unwrap().cancel_on_interruption);
    }

    #[test]
    fn exact_match_takes_precedence_over_catch_all() {
        let mut reg = FunctionCallRegistry::new();
        reg.register_catch_all(dummy_handler(), true, Duration::from_secs(5));
        reg.register("specific", dummy_handler(), false, Duration::from_secs(10));

        let item = reg.lookup("specific").unwrap();
        assert_eq!(item.function_name.as_deref(), Some("specific"));
        assert!(!item.cancel_on_interruption);

        // Unknown name falls through to catch-all
        let item = reg.lookup("other").unwrap();
        assert!(item.function_name.is_none());
        assert!(item.cancel_on_interruption);
    }

    #[test]
    fn has_function_with_catch_all() {
        let mut reg = FunctionCallRegistry::new();
        assert!(!reg.has_function("anything"));

        reg.register_catch_all(dummy_handler(), false, Duration::from_secs(5));
        assert!(reg.has_function("anything"));
    }

    #[tokio::test]
    async fn handler_invocation() {
        let handler: FunctionCallHandler = Arc::new(|params| {
            Box::pin(async move { serde_json::json!({"echo": params.function_name}) })
        });

        let mut reg = FunctionCallRegistry::new();
        reg.register("echo", handler, false, Duration::from_secs(10));

        let item = reg.lookup("echo").unwrap();
        let params = FunctionCallParams {
            function_name: "echo".into(),
            tool_call_id: "call_1".into(),
            arguments: serde_json::json!({}),
            context: pipecat_context::LLMContext::new(vec![]),
        };
        let result = (item.handler)(params).await;
        assert_eq!(result["echo"], "echo");
    }
}
