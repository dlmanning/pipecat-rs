use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use pipecat_core::test_utils::*;
use pipecat_integration_tests::helpers::*;
use pipecat_pipeline::Pipeline;
use pipecat_services::function_call::{
    FunctionCallHandler, FunctionCallParams, FunctionCallRegistry,
};
use pipecat_services::llm::{LLMService, LLMServiceState, llm_process_frame};
use pipecat_services::settings::LLMSettings;
use serde_json::json;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// FunctionCallingLLMService — mock LLM that emits function calls
// ---------------------------------------------------------------------------

/// Mock LLM that, upon receiving an LLMContextFrame, emits function call
/// frames following the same protocol as a real LLM service: broadcasts
/// FunctionCallsStarted, FunctionCallInProgress per call, invokes the
/// registered handler, and broadcasts FunctionCallResult.
#[derive(Debug)]
struct FunctionCallingLLMService {
    state: LLMServiceState,
    /// Canned function calls to emit when an LLMContextFrame arrives.
    pending_calls: Vec<PendingFunctionCall>,
}

#[derive(Debug, Clone)]
struct PendingFunctionCall {
    function_name: String,
    tool_call_id: String,
    arguments: serde_json::Value,
}

impl FunctionCallingLLMService {
    fn new(pending_calls: Vec<PendingFunctionCall>, registry: FunctionCallRegistry) -> Self {
        let mut state = LLMServiceState::new("FunctionCallingLLM", LLMSettings::default());
        state.function_registry = registry;
        Self {
            state,
            pending_calls,
        }
    }
}

#[async_trait]
impl LLMService for FunctionCallingLLMService {
    fn llm_service_state(&self) -> &LLMServiceState {
        &self.state
    }
    fn llm_service_state_mut(&mut self) -> &mut LLMServiceState {
        &mut self.state
    }
}

#[async_trait]
impl FrameProcessor for FunctionCallingLLMService {
    fn name(&self) -> &str {
        self.state.base.processor.name()
    }
    fn id(&self) -> u64 {
        self.state.base.processor.id()
    }
    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        match &envelope.frame {
            Frame::LLMContext(context_frame) => {
                // Emit LLMFullResponseStart
                ctx.send_downstream(Frame::LLMFullResponseStart(LLMFullResponseStartFrame {
                    skip_tts: None,
                }))
                .await?;

                // Emit LLMFullResponseEnd (the LLM "response" is just function calls)
                ctx.send_downstream(Frame::LLMFullResponseEnd(LLMFullResponseEndFrame {
                    skip_tts: None,
                }))
                .await?;

                // Build FunctionCallFromLLM list
                let function_calls: Vec<FunctionCallFromLLM> = self
                    .pending_calls
                    .iter()
                    .map(|pc| FunctionCallFromLLM {
                        function_name: pc.function_name.clone(),
                        tool_call_id: pc.tool_call_id.clone(),
                        arguments: pc.arguments.clone(),
                    })
                    .collect();

                // Broadcast FunctionCallsStarted
                ctx.broadcast(Frame::FunctionCallsStarted(FunctionCallsStartedFrame {
                    function_calls: function_calls.clone(),
                }))
                .await?;

                // Build LLMContext for the handler (matching real OpenAI LLM)
                let messages = context_frame
                    .context
                    .get("messages")
                    .cloned()
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();
                let llm_context = pipecat_context::LLMContext::new(messages);
                if let Some(tools) = context_frame.context.get("tools").cloned() {
                    llm_context.set_tools(tools);
                }
                if let Some(tool_choice) = context_frame.context.get("tool_choice").cloned() {
                    llm_context.set_tool_choice(tool_choice);
                }

                // Execute each function call through the registry
                for pc in &self.pending_calls {
                    let registry_item = self.state.function_registry.lookup(&pc.function_name);

                    ctx.broadcast(Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
                        function_name: pc.function_name.clone(),
                        tool_call_id: pc.tool_call_id.clone(),
                        arguments: pc.arguments.clone(),
                        cancel_on_interruption: registry_item
                            .map(|item| item.cancel_on_interruption)
                            .unwrap_or(false),
                    }))
                    .await?;

                    let result_value = if let Some(item) = registry_item {
                        let params = FunctionCallParams {
                            function_name: pc.function_name.clone(),
                            tool_call_id: pc.tool_call_id.clone(),
                            arguments: pc.arguments.clone(),
                            context: llm_context.clone(),
                        };
                        match tokio::time::timeout(item.timeout, (item.handler)(params)).await {
                            Ok(result) => result,
                            Err(_) => serde_json::Value::Null,
                        }
                    } else {
                        json!({"error": format!("No handler for {}", pc.function_name)})
                    };

                    ctx.broadcast(Frame::FunctionCallResult(FunctionCallResultFrame {
                        function_name: pc.function_name.clone(),
                        tool_call_id: pc.tool_call_id.clone(),
                        arguments: pc.arguments.clone(),
                        result: result_value,
                        run_llm: Some(true),
                        properties: Some(FunctionCallResultProperties {
                            run_llm: Some(true),
                        }),
                    }))
                    .await?;
                }

                Ok(())
            }
            _ => llm_process_frame(self, envelope, direction, ctx).await,
        }
    }
}

// ---------------------------------------------------------------------------
// Test: function call round-trip through pipeline
// ---------------------------------------------------------------------------

/// Verifies the full function call round-trip:
/// 1. Send LLMContextFrame to a mock LLM with a registered function
/// 2. LLM emits FunctionCallsStarted, FunctionCallInProgress
/// 3. Registry handler is invoked and returns a result
/// 4. LLM emits FunctionCallResult with the handler's output
#[tokio::test]
async fn function_call_round_trip() {
    // Set up a function handler that returns weather data
    let handler: FunctionCallHandler = Arc::new(|params| {
        Box::pin(async move {
            let city = params
                .arguments
                .get("city")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            json!({
                "temperature": 72,
                "condition": "sunny",
                "city": city
            })
        })
    });

    let mut registry = FunctionCallRegistry::new();
    registry.register("get_weather", handler, false, Duration::from_secs(10));

    let llm = FunctionCallingLLMService::new(
        vec![PendingFunctionCall {
            function_name: "get_weather".to_string(),
            tool_call_id: "call_001".to_string(),
            arguments: json!({"city": "San Francisco"}),
        }],
        registry,
    );

    let pipeline = Pipeline::new(vec![Box::new(llm)]);
    let (node, handle, down_rx, up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);
    let up = FrameCollector::spawn(up_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    // Send LLMContext to trigger function calling
    let ctx_frame = make_llm_context_frame(vec![
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "What's the weather in SF?"}),
    ]);
    send_frame(&handle, ctx_frame.frame.clone(), Direction::Downstream).await;

    down.wait_for_frame("FunctionCallResult").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let downstream_frames = down.frames();
    let down_names = down.frame_names();
    let up_names = up.frame_names();

    // Verify downstream frame sequence
    assert!(
        down_names.contains(&"LLMFullResponseStart".to_string()),
        "should emit LLMFullResponseStart: {down_names:?}"
    );
    assert!(
        down_names.contains(&"LLMFullResponseEnd".to_string()),
        "should emit LLMFullResponseEnd: {down_names:?}"
    );
    assert!(
        down_names.contains(&"FunctionCallsStarted".to_string()),
        "should emit FunctionCallsStarted downstream: {down_names:?}"
    );
    assert!(
        down_names.contains(&"FunctionCallInProgress".to_string()),
        "should emit FunctionCallInProgress downstream: {down_names:?}"
    );
    assert!(
        down_names.contains(&"FunctionCallResult".to_string()),
        "should emit FunctionCallResult downstream: {down_names:?}"
    );

    // broadcast() sends upstream too
    assert!(
        up_names.contains(&"FunctionCallsStarted".to_string()),
        "should emit FunctionCallsStarted upstream: {up_names:?}"
    );
    assert!(
        up_names.contains(&"FunctionCallInProgress".to_string()),
        "should emit FunctionCallInProgress upstream: {up_names:?}"
    );
    assert!(
        up_names.contains(&"FunctionCallResult".to_string()),
        "should emit FunctionCallResult upstream: {up_names:?}"
    );

    // Verify the FunctionCallResult contains the handler's output
    let result_frame = downstream_frames
        .iter()
        .find_map(|f| match &f.frame {
            Frame::FunctionCallResult(r) => Some(r),
            _ => None,
        })
        .expect("should have a FunctionCallResultFrame");

    assert_eq!(result_frame.function_name, "get_weather");
    assert_eq!(result_frame.tool_call_id, "call_001");
    assert_eq!(result_frame.result["temperature"], 72);
    assert_eq!(result_frame.result["condition"], "sunny");
    assert_eq!(result_frame.result["city"], "San Francisco");
    assert_eq!(result_frame.run_llm, Some(true));

    // Verify FunctionCallInProgress has the right metadata
    let in_progress_frame = downstream_frames
        .iter()
        .find_map(|f| match &f.frame {
            Frame::FunctionCallInProgress(p) => Some(p),
            _ => None,
        })
        .expect("should have a FunctionCallInProgressFrame");

    assert_eq!(in_progress_frame.function_name, "get_weather");
    assert_eq!(in_progress_frame.tool_call_id, "call_001");
    assert!(!in_progress_frame.cancel_on_interruption);

    // Verify ordering: FunctionCallsStarted before FunctionCallInProgress before FunctionCallResult
    let started_pos = down_names
        .iter()
        .position(|n| n == "FunctionCallsStarted")
        .unwrap();
    let in_progress_pos = down_names
        .iter()
        .position(|n| n == "FunctionCallInProgress")
        .unwrap();
    let result_pos = down_names
        .iter()
        .position(|n| n == "FunctionCallResult")
        .unwrap();
    assert!(
        started_pos < in_progress_pos,
        "FunctionCallsStarted ({started_pos}) should come before FunctionCallInProgress ({in_progress_pos})"
    );
    assert!(
        in_progress_pos < result_pos,
        "FunctionCallInProgress ({in_progress_pos}) should come before FunctionCallResult ({result_pos})"
    );
}

// ---------------------------------------------------------------------------
// Test: multiple function calls in sequence
// ---------------------------------------------------------------------------

/// Verifies that multiple function calls are executed in order, each producing
/// its own FunctionCallInProgress + FunctionCallResult pair.
#[tokio::test]
async fn multiple_function_calls() {
    let handler: FunctionCallHandler = Arc::new(|params| {
        Box::pin(async move { json!({"called": params.function_name, "id": params.tool_call_id}) })
    });

    let mut registry = FunctionCallRegistry::new();
    registry.register("func_a", handler.clone(), false, Duration::from_secs(10));
    registry.register("func_b", handler, true, Duration::from_secs(10));

    let llm = FunctionCallingLLMService::new(
        vec![
            PendingFunctionCall {
                function_name: "func_a".to_string(),
                tool_call_id: "call_a".to_string(),
                arguments: json!({}),
            },
            PendingFunctionCall {
                function_name: "func_b".to_string(),
                tool_call_id: "call_b".to_string(),
                arguments: json!({"x": 42}),
            },
        ],
        registry,
    );

    let pipeline = Pipeline::new(vec![Box::new(llm)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    let ctx_frame = make_llm_context_frame(vec![json!({"role": "user", "content": "do both"})]);
    send_frame(&handle, ctx_frame.frame.clone(), Direction::Downstream).await;

    // Wait for second FunctionCallResult (both calls complete)
    down.wait_for(|f| matches!(f, Frame::FunctionCallResult(r) if r.function_name == "func_b"))
        .await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = down.frames();
    let names = down.frame_names();

    // Should have exactly one FunctionCallsStarted
    let started_count = names
        .iter()
        .filter(|n| n.as_str() == "FunctionCallsStarted")
        .count();
    assert_eq!(
        started_count, 1,
        "should have exactly one FunctionCallsStarted"
    );

    // Should have two FunctionCallInProgress and two FunctionCallResult
    let in_progress_count = names
        .iter()
        .filter(|n| n.as_str() == "FunctionCallInProgress")
        .count();
    let result_count = names
        .iter()
        .filter(|n| n.as_str() == "FunctionCallResult")
        .count();
    assert_eq!(
        in_progress_count, 2,
        "should have two FunctionCallInProgress frames"
    );
    assert_eq!(result_count, 2, "should have two FunctionCallResult frames");

    // Collect results in order
    let results: Vec<&FunctionCallResultFrame> = frames
        .iter()
        .filter_map(|f| match &f.frame {
            Frame::FunctionCallResult(r) => Some(r),
            _ => None,
        })
        .collect();

    assert_eq!(results[0].function_name, "func_a");
    assert_eq!(results[0].tool_call_id, "call_a");
    assert_eq!(results[0].result["called"], "func_a");

    assert_eq!(results[1].function_name, "func_b");
    assert_eq!(results[1].tool_call_id, "call_b");
    assert_eq!(results[1].result["called"], "func_b");

    // Verify cancel_on_interruption is correctly propagated
    let in_progress_frames: Vec<&FunctionCallInProgressFrame> = frames
        .iter()
        .filter_map(|f| match &f.frame {
            Frame::FunctionCallInProgress(p) => Some(p),
            _ => None,
        })
        .collect();

    assert!(
        !in_progress_frames[0].cancel_on_interruption,
        "func_a should not cancel on interruption"
    );
    assert!(
        in_progress_frames[1].cancel_on_interruption,
        "func_b should cancel on interruption"
    );
}

// ---------------------------------------------------------------------------
// Test: catch-all handler matches unknown functions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn catch_all_handler() {
    let catch_all: FunctionCallHandler = Arc::new(|params| {
        Box::pin(async move {
            json!({
                "handled_by": "catch_all",
                "function": params.function_name,
            })
        })
    });

    let mut registry = FunctionCallRegistry::new();
    registry.register_catch_all(catch_all, false, Duration::from_secs(10));

    let llm = FunctionCallingLLMService::new(
        vec![PendingFunctionCall {
            function_name: "unknown_function".to_string(),
            tool_call_id: "call_x".to_string(),
            arguments: json!({}),
        }],
        registry,
    );

    let pipeline = Pipeline::new(vec![Box::new(llm)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    let ctx_frame = make_llm_context_frame(vec![json!({"role": "user", "content": "call it"})]);
    send_frame(&handle, ctx_frame.frame.clone(), Direction::Downstream).await;

    down.wait_for_frame("FunctionCallResult").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = down.frames();

    let result = frames
        .iter()
        .find_map(|f| match &f.frame {
            Frame::FunctionCallResult(r) => Some(r),
            _ => None,
        })
        .expect("catch-all should produce a FunctionCallResult");

    assert_eq!(result.function_name, "unknown_function");
    assert_eq!(result.result["handled_by"], "catch_all");
    assert_eq!(result.result["function"], "unknown_function");
}

// ---------------------------------------------------------------------------
// Test: unregistered function produces error result
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unregistered_function_produces_error() {
    // Empty registry — no handlers
    let registry = FunctionCallRegistry::new();

    let llm = FunctionCallingLLMService::new(
        vec![PendingFunctionCall {
            function_name: "nonexistent".to_string(),
            tool_call_id: "call_missing".to_string(),
            arguments: json!({}),
        }],
        registry,
    );

    let pipeline = Pipeline::new(vec![Box::new(llm)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    let ctx_frame = make_llm_context_frame(vec![json!({"role": "user", "content": "try it"})]);
    send_frame(&handle, ctx_frame.frame.clone(), Direction::Downstream).await;

    down.wait_for_frame("FunctionCallResult").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = down.frames();

    let result = frames
        .iter()
        .find_map(|f| match &f.frame {
            Frame::FunctionCallResult(r) => Some(r),
            _ => None,
        })
        .expect("unregistered function should still produce a FunctionCallResult");

    assert_eq!(result.function_name, "nonexistent");
    assert!(
        result.result.get("error").is_some(),
        "result should contain an error field: {:?}",
        result.result
    );
}

// ---------------------------------------------------------------------------
// Test: function call handler receives correct context
// ---------------------------------------------------------------------------

/// Verifies that the handler receives the LLMContext messages from the
/// LLMContextFrame, and the correct function name/arguments/tool_call_id.
#[tokio::test]
async fn handler_receives_correct_params() {
    let received_params = Arc::new(tokio::sync::Mutex::new(None::<FunctionCallParams>));
    let params_capture = received_params.clone();

    let handler: FunctionCallHandler = Arc::new(move |params| {
        let capture = params_capture.clone();
        Box::pin(async move {
            *capture.lock().await = Some(params);
            json!({"ok": true})
        })
    });

    let mut registry = FunctionCallRegistry::new();
    registry.register("check_params", handler, false, Duration::from_secs(10));

    let llm = FunctionCallingLLMService::new(
        vec![PendingFunctionCall {
            function_name: "check_params".to_string(),
            tool_call_id: "call_verify".to_string(),
            arguments: json!({"key": "value", "num": 99}),
        }],
        registry,
    );

    let pipeline = Pipeline::new(vec![Box::new(llm)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    let messages = vec![
        json!({"role": "system", "content": "System prompt."}),
        json!({"role": "user", "content": "User message."}),
    ];
    let tools = json!([
        {"type": "function", "function": {"name": "check_params", "parameters": {"type": "object"}}},
    ]);
    let tool_choice = json!("auto");
    let ctx_frame =
        make_llm_context_frame_with_tools(messages.clone(), tools.clone(), tool_choice.clone());
    send_frame(&handle, ctx_frame.frame.clone(), Direction::Downstream).await;

    down.wait_for_frame("FunctionCallResult").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let captured = received_params.lock().await;
    let params = captured
        .as_ref()
        .expect("handler should have been called with params");

    assert_eq!(params.function_name, "check_params");
    assert_eq!(params.tool_call_id, "call_verify");
    assert_eq!(params.arguments["key"], "value");
    assert_eq!(params.arguments["num"], 99);

    // Verify the LLMContext was constructed from the LLMContextFrame messages
    let ctx_messages = params.context.get_messages();
    assert_eq!(ctx_messages.len(), 2);
    assert_eq!(ctx_messages[0]["role"], "system");
    assert_eq!(ctx_messages[1]["role"], "user");

    // Verify the LLMContext includes tools and tool_choice from the context frame
    let ctx_tools = params.context.tools();
    assert!(
        ctx_tools.is_some(),
        "LLMContext should have tools from the context frame"
    );
    assert_eq!(
        ctx_tools.unwrap()[0]["function"]["name"],
        "check_params",
        "tools should be propagated to the handler's context"
    );
    let ctx_tool_choice = params.context.tool_choice();
    assert_eq!(
        ctx_tool_choice,
        Some(json!("auto")),
        "tool_choice should be propagated to the handler's context"
    );
}

// ---------------------------------------------------------------------------
// Test: function call timeout produces null result
// ---------------------------------------------------------------------------

#[tokio::test]
async fn function_call_timeout() {
    let timeout_ms = 50;
    let slow_handler: FunctionCallHandler = Arc::new(|_| {
        Box::pin(async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            json!({"should": "never_arrive"})
        })
    });

    let mut registry = FunctionCallRegistry::new();
    // Very short timeout — handler will not complete in time
    registry.register(
        "slow_fn",
        slow_handler,
        false,
        Duration::from_millis(timeout_ms),
    );

    let llm = FunctionCallingLLMService::new(
        vec![PendingFunctionCall {
            function_name: "slow_fn".to_string(),
            tool_call_id: "call_slow".to_string(),
            arguments: json!({}),
        }],
        registry,
    );

    let pipeline = Pipeline::new(vec![Box::new(llm)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    let ctx_frame = make_llm_context_frame(vec![json!({"role": "user", "content": "slow call"})]);

    // Time the context processing to verify the timeout actually fired
    let before = Instant::now();
    send_frame(&handle, ctx_frame.frame.clone(), Direction::Downstream).await;

    // Wait for the result (the timeout fires at ~50ms)
    down.wait_for_frame("FunctionCallResult").await;
    let elapsed = before.elapsed();

    // The result should have arrived after at least the timeout duration,
    // proving the timeout actually fired (not an instant error).
    assert!(
        elapsed >= Duration::from_millis(timeout_ms),
        "elapsed {:?} should be >= timeout {:?} — timeout must actually fire",
        elapsed,
        Duration::from_millis(timeout_ms),
    );
    // But it should not have waited for the full handler duration (10s)
    assert!(
        elapsed < Duration::from_secs(2),
        "elapsed {:?} should be well under the handler's 10s sleep — timeout should have cut it short",
        elapsed,
    );

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = down.frames();

    let result = frames
        .iter()
        .find_map(|f| match &f.frame {
            Frame::FunctionCallResult(r) => Some(r),
            _ => None,
        })
        .expect("timed-out function should still emit FunctionCallResult");

    assert_eq!(result.function_name, "slow_fn");
    assert_eq!(result.tool_call_id, "call_slow");
    // Timed-out handler should produce null — matching the pattern used by
    // the real OpenAI LLM service (Value::Null on timeout).
    assert!(
        result.result.is_null(),
        "timed-out handler should produce null result (same as real LLM service), got: {:?}",
        result.result
    );
}

// ---------------------------------------------------------------------------
// Test: FunctionCallsStarted frame contains all call metadata
// ---------------------------------------------------------------------------

#[tokio::test]
async fn function_calls_started_contains_all_calls() {
    let handler: FunctionCallHandler = Arc::new(|_| Box::pin(async { json!({"ok": true}) }));

    let mut registry = FunctionCallRegistry::new();
    registry.register("alpha", handler.clone(), false, Duration::from_secs(10));
    registry.register("beta", handler, false, Duration::from_secs(10));

    let llm = FunctionCallingLLMService::new(
        vec![
            PendingFunctionCall {
                function_name: "alpha".to_string(),
                tool_call_id: "id_alpha".to_string(),
                arguments: json!({"a": 1}),
            },
            PendingFunctionCall {
                function_name: "beta".to_string(),
                tool_call_id: "id_beta".to_string(),
                arguments: json!({"b": 2}),
            },
        ],
        registry,
    );

    let pipeline = Pipeline::new(vec![Box::new(llm)]);
    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame::default()),
        Direction::Downstream,
    )
    .await;

    let ctx_frame = make_llm_context_frame(vec![json!({"role": "user", "content": "go"})]);
    send_frame(&handle, ctx_frame.frame.clone(), Direction::Downstream).await;

    // Wait for second result to ensure all function calls are complete
    down.wait_for(|f| matches!(f, Frame::FunctionCallResult(r) if r.function_name == "beta"))
        .await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let frames = down.frames();

    let started = frames
        .iter()
        .find_map(|f| match &f.frame {
            Frame::FunctionCallsStarted(s) => Some(s),
            _ => None,
        })
        .expect("should have FunctionCallsStartedFrame");

    assert_eq!(started.function_calls.len(), 2);
    assert_eq!(started.function_calls[0].function_name, "alpha");
    assert_eq!(started.function_calls[0].tool_call_id, "id_alpha");
    assert_eq!(started.function_calls[0].arguments["a"], 1);
    assert_eq!(started.function_calls[1].function_name, "beta");
    assert_eq!(started.function_calls[1].tool_call_id, "id_beta");
    assert_eq!(started.function_calls[1].arguments["b"], 2);
}
