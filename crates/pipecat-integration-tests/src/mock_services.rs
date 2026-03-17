use async_trait::async_trait;
use bytes::Bytes;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_services::llm::{LLMService, LLMServiceState, llm_process_frame};
use pipecat_services::settings::{LLMSettings, STTSettings, TTSSettings};
use pipecat_services::stt::{STTService, STTServiceState, stt_process_frame};
use pipecat_services::text_aggregator::TextAggregationMode;
use pipecat_services::tts::{TTSService, TTSServiceState, tts_process_frame};

// ---------------------------------------------------------------------------
// FakeSTTService
// ---------------------------------------------------------------------------

/// Fake STT that emits a canned transcription after receiving N audio chunks.
#[derive(Debug)]
pub struct FakeSTTService {
    state: STTServiceState,
    canned_response: String,
    chunks_before_emit: usize,
    audio_chunk_count: usize,
}

impl FakeSTTService {
    pub fn new(canned_text: &str, chunks_before_emit: usize) -> Self {
        Self {
            state: STTServiceState::new("FakeSTT", STTSettings::default()),
            canned_response: canned_text.to_string(),
            chunks_before_emit,
            audio_chunk_count: 0,
        }
    }
}

#[async_trait]
impl STTService for FakeSTTService {
    async fn run_stt(&mut self, _audio: Bytes, ctx: &ProcessorContext) -> Result<()> {
        self.audio_chunk_count += 1;
        if self.audio_chunk_count >= self.chunks_before_emit {
            self.audio_chunk_count = 0;
            ctx.send_downstream(Frame::Transcription(TranscriptionFrame {
                text: self.canned_response.clone(),
                user_id: "user".to_string(),
                timestamp: None,
                language: None,
                finalized: true,
                result: None,
            }))
            .await?;
        }
        Ok(())
    }

    fn stt_service_state(&self) -> &STTServiceState {
        &self.state
    }
    fn stt_service_state_mut(&mut self) -> &mut STTServiceState {
        &mut self.state
    }
}

#[async_trait]
impl FrameProcessor for FakeSTTService {
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
        stt_process_frame(self, envelope, direction, ctx).await
    }
}

// ---------------------------------------------------------------------------
// FakeLLMService
// ---------------------------------------------------------------------------

/// Fake LLM that emits canned response tokens when it receives an LLMContextFrame.
#[derive(Debug)]
pub struct FakeLLMService {
    state: LLMServiceState,
    response_tokens: Vec<String>,
}

impl FakeLLMService {
    pub fn new(response_tokens: Vec<String>) -> Self {
        Self {
            state: LLMServiceState::new("FakeLLM", LLMSettings::default()),
            response_tokens,
        }
    }
}

#[async_trait]
impl LLMService for FakeLLMService {
    fn llm_service_state(&self) -> &LLMServiceState {
        &self.state
    }
    fn llm_service_state_mut(&mut self) -> &mut LLMServiceState {
        &mut self.state
    }
}

#[async_trait]
impl FrameProcessor for FakeLLMService {
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
            Frame::LLMContext(_) => {
                // Emit canned response
                ctx.send_downstream(Frame::LLMFullResponseStart(LLMFullResponseStartFrame {
                    skip_tts: None,
                }))
                .await?;

                for token in &self.response_tokens {
                    ctx.send_downstream(Frame::Text(TextFrame::new(token)))
                        .await?;
                }

                ctx.send_downstream(Frame::LLMFullResponseEnd(LLMFullResponseEndFrame {
                    skip_tts: None,
                }))
                .await?;

                Ok(())
            }
            _ => llm_process_frame(self, envelope, direction, ctx).await,
        }
    }
}

// ---------------------------------------------------------------------------
// FakeTTSService
// ---------------------------------------------------------------------------

/// Fake TTS that emits dummy audio bytes proportional to text length.
#[derive(Debug)]
pub struct FakeTTSService {
    state: TTSServiceState,
}

impl Default for FakeTTSService {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeTTSService {
    pub fn new() -> Self {
        Self {
            state: TTSServiceState::new(
                "FakeTTS",
                TTSSettings::default(),
                TextAggregationMode::Sentence,
                16000,
            ),
        }
    }
}

#[async_trait]
impl TTSService for FakeTTSService {
    async fn run_tts(
        &mut self,
        text: &str,
        context_id: &str,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        ctx.send_downstream(Frame::TTSStarted(TTSStartedFrame {
            context_id: Some(context_id.to_string()),
        }))
        .await?;

        // Generate dummy audio: 2 bytes per character (one i16 sample)
        let audio_len = text.len() * 2;
        let audio = vec![0u8; audio_len];
        ctx.send_downstream(Frame::TTSAudioRaw(TTSAudioRawFrame {
            audio: Bytes::from(audio),
            sample_rate: 16000,
            num_channels: 1,
            context_id: Some(context_id.to_string()),
        }))
        .await?;

        ctx.send_downstream(Frame::TTSStopped(TTSStoppedFrame {
            context_id: Some(context_id.to_string()),
        }))
        .await?;

        Ok(())
    }

    fn tts_service_state(&self) -> &TTSServiceState {
        &self.state
    }
    fn tts_service_state_mut(&mut self) -> &mut TTSServiceState {
        &mut self.state
    }
}

#[async_trait]
impl FrameProcessor for FakeTTSService {
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
        tts_process_frame(self, envelope, direction, ctx).await
    }
}

// ---------------------------------------------------------------------------
// FakeRealtimeProcessor
// ---------------------------------------------------------------------------

/// Fake speech-to-speech processor that consumes InputAudioRaw and emits
/// TTSAudioRaw after accumulating N chunks.
#[derive(Debug)]
pub struct FakeRealtimeProcessor {
    base: ProcessorBase,
    chunks_before_response: usize,
    chunk_count: usize,
}

impl FakeRealtimeProcessor {
    pub fn new(chunks_before_response: usize) -> Self {
        Self {
            base: ProcessorBase::new("FakeRealtime"),
            chunks_before_response,
            chunk_count: 0,
        }
    }
}

#[async_trait]
impl FrameProcessor for FakeRealtimeProcessor {
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
            Frame::InputAudioRaw(_) => {
                // Consume audio (do not forward)
                self.chunk_count += 1;
                if self.chunk_count >= self.chunks_before_response {
                    self.chunk_count = 0;
                    ctx.send_downstream(Frame::TTSAudioRaw(TTSAudioRawFrame {
                        audio: Bytes::from(vec![0u8; 320]),
                        sample_rate: 16000,
                        num_channels: 1,
                        context_id: None,
                    }))
                    .await?;
                }
            }
            Frame::Interruption(_) => {
                self.chunk_count = 0;
                ctx.push_frame(envelope, direction).await?;
            }
            _ => {
                ctx.push_frame(envelope, direction).await?;
            }
        }
        Ok(())
    }
}
