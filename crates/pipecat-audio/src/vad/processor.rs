use async_trait::async_trait;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};

use super::analyzer::VadAnalyzer;
use super::controller::{VadController, VadControllerEvent};

/// A `FrameProcessor` that wraps a [`VadController`] and emits VAD frames.
///
/// Converts `InputAudioRaw` frames into `VADUserStartedSpeaking`,
/// `VADUserStoppedSpeaking`, and `UserSpeaking` frames. Audio is forwarded
/// downstream unchanged.
///
/// Handles internal audio buffering — no need to match the VAD chunk size.
///
/// # Example
///
/// ```ignore
/// use pipecat_audio::vad::{VadController, VadProcessor, SileroVadAnalyzer};
///
/// let controller = VadController::new(SileroVadAnalyzer::new(16000)?, 16000);
/// let processor = VadProcessor::new(controller);
/// ```
#[derive(Debug)]
pub struct VadProcessor<A: VadAnalyzer> {
    base: ProcessorBase,
    controller: VadController<A>,
}

impl<A: VadAnalyzer> VadProcessor<A> {
    pub fn new(controller: VadController<A>) -> Self {
        Self {
            base: ProcessorBase::new("VadProcessor"),
            controller,
        }
    }

    pub fn controller(&self) -> &VadController<A> {
        &self.controller
    }

    pub fn controller_mut(&mut self) -> &mut VadController<A> {
        &mut self.controller
    }
}

#[async_trait]
impl<A: VadAnalyzer + 'static> FrameProcessor for VadProcessor<A> {
    fn name(&self) -> &str {
        self.base.name()
    }

    fn id(&self) -> u64 {
        self.base.id()
    }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        match &envelope.frame {
            Frame::Start(start) => {
                self.controller.handle_start(start.audio_in_sample_rate);
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::InputAudioRaw(audio) => {
                let events = self.controller.handle_audio(&audio.audio);
                for event in &events {
                    match event {
                        VadControllerEvent::SpeechStarted => {
                            let params = self.controller.analyzer().params();
                            ctx.send_downstream(Frame::VADUserStartedSpeaking(
                                VADUserStartedSpeakingFrame {
                                    start_secs: params.start_secs,
                                    timestamp: 0.0,
                                },
                            ))
                            .await?;
                        }
                        VadControllerEvent::SpeechStopped => {
                            let params = self.controller.analyzer().params();
                            ctx.send_downstream(Frame::VADUserStoppedSpeaking(
                                VADUserStoppedSpeakingFrame {
                                    stop_secs: params.stop_secs,
                                    timestamp: 0.0,
                                },
                            ))
                            .await?;
                        }
                        VadControllerEvent::SpeechActivity => {
                            ctx.send_downstream(Frame::UserSpeaking(UserSpeakingFrame))
                                .await?;
                        }
                    }
                }
                ctx.push_frame(envelope, direction).await?;
            }

            Frame::VadParamsUpdate(update) => {
                self.controller.handle_params_update(update.params.clone());
                ctx.push_frame(envelope, direction).await?;
            }

            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }
        Ok(())
    }
}
