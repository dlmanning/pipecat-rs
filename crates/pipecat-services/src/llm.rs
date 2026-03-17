use async_trait::async_trait;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};

use crate::function_call::FunctionCallRegistry;
use crate::service_base::ServiceBase;
use crate::settings::LLMSettings;

/// State held by LLM service implementations.
#[derive(Debug)]
pub struct LLMServiceState {
    pub base: ServiceBase,
    pub settings: LLMSettings,
    pub function_registry: FunctionCallRegistry,
    pub skip_tts: Option<bool>,
}

impl LLMServiceState {
    pub fn new(name: impl Into<String>, settings: LLMSettings) -> Self {
        let model = settings.model.clone();
        Self {
            base: ServiceBase::new(name, model),
            settings,
            function_registry: FunctionCallRegistry::new(),
            skip_tts: None,
        }
    }
}

/// Trait for LLM service implementations.
///
/// Implementors handle `LLMContextFrame` to run inference and push
/// text/function-call frames. Frame routing is handled by the shared
/// `llm_process_frame` helper.
#[async_trait]
pub trait LLMService: FrameProcessor {
    /// Access the LLM service state.
    fn llm_service_state(&self) -> &LLMServiceState;
    fn llm_service_state_mut(&mut self) -> &mut LLMServiceState;
}

/// Common frame routing for LLM services. Call from `process_frame`.
pub async fn llm_process_frame(
    llm: &mut dyn LLMService,
    envelope: FrameEnvelope,
    direction: Direction,
    ctx: &ProcessorContext,
) -> Result<()> {
    match &envelope.frame {
        Frame::Start(f) => {
            let state = llm.llm_service_state_mut();
            state.base.handle_start_frame(f);
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::LLMUpdateSettings(f) => {
            let state = llm.llm_service_state_mut();
            let changed = state.settings.apply_delta(&f.settings);
            if changed.contains(&"model".to_string())
                && let Some(ref model) = state.settings.model
            {
                state.base.set_model(model.clone());
            }
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::LLMConfigureOutput(f) => {
            llm.llm_service_state_mut().skip_tts = Some(f.skip_tts);
            ctx.push_frame(envelope, direction).await?;
        }

        Frame::Interruption(_) => {
            llm.llm_service_state_mut().base.metrics.reset();
            ctx.push_frame(envelope, direction).await?;
        }

        _ => {
            ctx.push_frame(envelope, direction).await?;
        }
    }

    Ok(())
}
