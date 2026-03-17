pub mod common;
pub mod control;
pub mod data;
pub mod system;

// Re-export everything so callers use `frame::StartFrame` not `frame::system::StartFrame`.
use std::fmt;

pub use common::*;
pub use control::*;
pub use data::*;
pub use system::*;

// ---------------------------------------------------------------------------
// Frame enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Frame {
    // ==== System frames (high priority) ====
    Start(StartFrame),
    Cancel(CancelFrame),
    Error(ErrorFrame),
    Interruption(InterruptionFrame),

    // Speaking events
    UserStartedSpeaking(UserStartedSpeakingFrame),
    UserStoppedSpeaking(UserStoppedSpeakingFrame),
    UserMuteStarted(UserMuteStartedFrame),
    UserMuteStopped(UserMuteStoppedFrame),
    UserSpeaking(UserSpeakingFrame),
    VADUserStartedSpeaking(VADUserStartedSpeakingFrame),
    VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame),
    BotStartedSpeaking(BotStartedSpeakingFrame),
    BotStoppedSpeaking(BotStoppedSpeakingFrame),
    BotSpeaking(BotSpeakingFrame),

    // System-priority input
    InputAudioRaw(AudioRawFrame),
    InputImageRaw(ImageRawFrame),
    InputTextRaw(InputTextRawFrame),
    UserAudioRaw(UserAudioRawFrame),
    UserImageRaw(UserImageRawFrame),
    UserImageRequest(UserImageRequestFrame),

    // System-priority control
    ProcessorPauseUrgent(ProcessorPauseUrgentFrame),
    ProcessorResumeUrgent(ProcessorResumeUrgentFrame),
    STTMute(STTMuteFrame),
    SpeechControlParams(SpeechControlParamsFrame),
    BotConnected(BotConnectedFrame),
    ClientConnected(ClientConnectedFrame),
    UserIdleTimeoutUpdate(UserIdleTimeoutUpdateFrame),

    // System-priority transport
    InputTransportMessage(InputTransportMessageFrame),
    OutputTransportMessageUrgent(OutputTransportMessageUrgentFrame),

    // System-priority function call events
    FunctionCallsStarted(FunctionCallsStartedFrame),
    FunctionCallCancel(FunctionCallCancelFrame),

    // System-priority service metadata
    ServiceMetadata(ServiceMetadataFrame),
    STTMetadata(STTMetadataFrame),

    // System-priority metrics
    Metrics(MetricsFrame),

    // System-priority task control
    EndTask(EndTaskFrame),
    CancelTask(CancelTaskFrame),
    StopTask(StopTaskFrame),
    InterruptionTask(InterruptionTaskFrame),

    // System-priority DTMF
    OutputDTMFUrgent(OutputDTMFUrgentFrame),
    InputDTMF(InputDTMFFrame),

    // ==== Data frames (normal priority, interruptible) ====
    OutputAudioRaw(AudioRawFrame),
    TTSAudioRaw(TTSAudioRawFrame),
    SpeechOutputAudioRaw(AudioRawFrame),
    OutputImageRaw(ImageRawFrame),
    URLImageRaw(URLImageRawFrame),
    AssistantImageRaw(AssistantImageRawFrame),
    Sprite(SpriteFrame),

    Text(TextFrame),
    LLMText(TextFrame),
    TTSText(AggregatedTextFrame),
    Transcription(TranscriptionFrame),
    InterimTranscription(InterimTranscriptionFrame),
    Translation(TranslationFrame),
    TTSSpeak(TTSSpeakFrame),

    LLMThoughtText(LLMThoughtTextFrame),
    LLMContextAssistantTimestamp(LLMContextAssistantTimestampFrame),

    LLMRun(LLMRunFrame),
    LLMMessagesAppend(LLMMessagesAppendFrame),
    LLMMessagesUpdate(LLMMessagesUpdateFrame),
    LLMSetTools(LLMSetToolsFrame),
    LLMSetToolChoice(LLMSetToolChoiceFrame),
    LLMEnablePromptCaching(LLMEnablePromptCachingFrame),
    LLMConfigureOutput(LLMConfigureOutputFrame),
    LLMContext(LLMContextFrame),

    FunctionCallResult(FunctionCallResultFrame),

    OutputTransportMessage(OutputTransportMessageFrame),

    OutputDTMF(OutputDTMFFrame),

    AggregatedText(AggregatedTextFrame),

    // ==== Control frames (normal priority) ====
    End(EndFrame),                                         // uninterruptible
    Stop(StopFrame),                                       // uninterruptible
    FunctionCallInProgress(FunctionCallInProgressFrame),   // uninterruptible
    ServiceUpdateSettings(ServiceUpdateSettingsFrame),     // uninterruptible
    LLMUpdateSettings(ServiceUpdateSettingsFrame),         // uninterruptible
    TTSUpdateSettings(ServiceUpdateSettingsFrame),         // uninterruptible
    STTUpdateSettings(ServiceUpdateSettingsFrame),         // uninterruptible
    LLMContextSummaryResult(LLMContextSummaryResultFrame), // uninterruptible

    TTSStarted(TTSStartedFrame),
    TTSStopped(TTSStoppedFrame),
    LLMFullResponseStart(LLMFullResponseStartFrame),
    LLMFullResponseEnd(LLMFullResponseEndFrame),
    LLMThoughtStart(LLMThoughtStartFrame),
    LLMThoughtEnd(LLMThoughtEndFrame),
    LLMAssistantPushAggregation(LLMAssistantPushAggregationFrame),
    LLMSummarizeContext(LLMSummarizeContextFrame),
    LLMContextSummaryRequest(LLMContextSummaryRequestFrame),

    VadParamsUpdate(VadParamsUpdateFrame),
    VisionFullResponseStart(VisionFullResponseStartFrame),
    VisionFullResponseEnd(VisionFullResponseEndFrame),
    OutputTransportReady(OutputTransportReadyFrame),
    Heartbeat(HeartbeatFrame),

    ProcessorPause(ProcessorPauseFrame),
    ProcessorResume(ProcessorResumeFrame),

    FilterUpdateSettings(FilterUpdateSettingsFrame),
    FilterEnable(FilterEnableFrame),
    MixerUpdateSettings(MixerUpdateSettingsFrame),
    MixerEnable(MixerEnableFrame),

    ServiceSwitcherRequestMetadata(ServiceSwitcherRequestMetadataFrame),
    ManuallySwitchService(ManuallySwitchServiceFrame),

    Wakeup(WakeupFrame), // uninterruptible
}

// ---------------------------------------------------------------------------
// Display — variant name for logging
// ---------------------------------------------------------------------------

impl fmt::Display for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Produces the variant name without inner data, e.g. "Start", "TTSAudioRaw".
        // Using a match instead of Debug to avoid printing payload in logs.
        let name = match self {
            Frame::Start(_) => "Start",
            Frame::Cancel(_) => "Cancel",
            Frame::Error(_) => "Error",
            Frame::Interruption(_) => "Interruption",
            Frame::UserStartedSpeaking(_) => "UserStartedSpeaking",
            Frame::UserStoppedSpeaking(_) => "UserStoppedSpeaking",
            Frame::UserMuteStarted(_) => "UserMuteStarted",
            Frame::UserMuteStopped(_) => "UserMuteStopped",
            Frame::UserSpeaking(_) => "UserSpeaking",
            Frame::VADUserStartedSpeaking(_) => "VADUserStartedSpeaking",
            Frame::VADUserStoppedSpeaking(_) => "VADUserStoppedSpeaking",
            Frame::BotStartedSpeaking(_) => "BotStartedSpeaking",
            Frame::BotStoppedSpeaking(_) => "BotStoppedSpeaking",
            Frame::BotSpeaking(_) => "BotSpeaking",
            Frame::InputAudioRaw(_) => "InputAudioRaw",
            Frame::InputImageRaw(_) => "InputImageRaw",
            Frame::InputTextRaw(_) => "InputTextRaw",
            Frame::UserAudioRaw(_) => "UserAudioRaw",
            Frame::UserImageRaw(_) => "UserImageRaw",
            Frame::UserImageRequest(_) => "UserImageRequest",
            Frame::ProcessorPauseUrgent(_) => "ProcessorPauseUrgent",
            Frame::ProcessorResumeUrgent(_) => "ProcessorResumeUrgent",
            Frame::STTMute(_) => "STTMute",
            Frame::SpeechControlParams(_) => "SpeechControlParams",
            Frame::BotConnected(_) => "BotConnected",
            Frame::ClientConnected(_) => "ClientConnected",
            Frame::UserIdleTimeoutUpdate(_) => "UserIdleTimeoutUpdate",
            Frame::InputTransportMessage(_) => "InputTransportMessage",
            Frame::OutputTransportMessageUrgent(_) => "OutputTransportMessageUrgent",
            Frame::FunctionCallsStarted(_) => "FunctionCallsStarted",
            Frame::FunctionCallCancel(_) => "FunctionCallCancel",
            Frame::ServiceMetadata(_) => "ServiceMetadata",
            Frame::STTMetadata(_) => "STTMetadata",
            Frame::Metrics(_) => "Metrics",
            Frame::EndTask(_) => "EndTask",
            Frame::CancelTask(_) => "CancelTask",
            Frame::StopTask(_) => "StopTask",
            Frame::InterruptionTask(_) => "InterruptionTask",
            Frame::OutputDTMFUrgent(_) => "OutputDTMFUrgent",
            Frame::InputDTMF(_) => "InputDTMF",
            Frame::OutputAudioRaw(_) => "OutputAudioRaw",
            Frame::TTSAudioRaw(_) => "TTSAudioRaw",
            Frame::SpeechOutputAudioRaw(_) => "SpeechOutputAudioRaw",
            Frame::OutputImageRaw(_) => "OutputImageRaw",
            Frame::URLImageRaw(_) => "URLImageRaw",
            Frame::AssistantImageRaw(_) => "AssistantImageRaw",
            Frame::Sprite(_) => "Sprite",
            Frame::Text(_) => "Text",
            Frame::LLMText(_) => "LLMText",
            Frame::TTSText(_) => "TTSText",
            Frame::Transcription(_) => "Transcription",
            Frame::InterimTranscription(_) => "InterimTranscription",
            Frame::Translation(_) => "Translation",
            Frame::TTSSpeak(_) => "TTSSpeak",
            Frame::LLMThoughtText(_) => "LLMThoughtText",
            Frame::LLMContextAssistantTimestamp(_) => "LLMContextAssistantTimestamp",
            Frame::LLMRun(_) => "LLMRun",
            Frame::LLMMessagesAppend(_) => "LLMMessagesAppend",
            Frame::LLMMessagesUpdate(_) => "LLMMessagesUpdate",
            Frame::LLMSetTools(_) => "LLMSetTools",
            Frame::LLMSetToolChoice(_) => "LLMSetToolChoice",
            Frame::LLMEnablePromptCaching(_) => "LLMEnablePromptCaching",
            Frame::LLMConfigureOutput(_) => "LLMConfigureOutput",
            Frame::LLMContext(_) => "LLMContext",
            Frame::FunctionCallResult(_) => "FunctionCallResult",
            Frame::OutputTransportMessage(_) => "OutputTransportMessage",
            Frame::OutputDTMF(_) => "OutputDTMF",
            Frame::AggregatedText(_) => "AggregatedText",
            Frame::End(_) => "End",
            Frame::Stop(_) => "Stop",
            Frame::FunctionCallInProgress(_) => "FunctionCallInProgress",
            Frame::ServiceUpdateSettings(_) => "ServiceUpdateSettings",
            Frame::LLMUpdateSettings(_) => "LLMUpdateSettings",
            Frame::TTSUpdateSettings(_) => "TTSUpdateSettings",
            Frame::STTUpdateSettings(_) => "STTUpdateSettings",
            Frame::LLMContextSummaryResult(_) => "LLMContextSummaryResult",
            Frame::TTSStarted(_) => "TTSStarted",
            Frame::TTSStopped(_) => "TTSStopped",
            Frame::LLMFullResponseStart(_) => "LLMFullResponseStart",
            Frame::LLMFullResponseEnd(_) => "LLMFullResponseEnd",
            Frame::LLMThoughtStart(_) => "LLMThoughtStart",
            Frame::LLMThoughtEnd(_) => "LLMThoughtEnd",
            Frame::LLMAssistantPushAggregation(_) => "LLMAssistantPushAggregation",
            Frame::LLMSummarizeContext(_) => "LLMSummarizeContext",
            Frame::LLMContextSummaryRequest(_) => "LLMContextSummaryRequest",
            Frame::VadParamsUpdate(_) => "VadParamsUpdate",
            Frame::VisionFullResponseStart(_) => "VisionFullResponseStart",
            Frame::VisionFullResponseEnd(_) => "VisionFullResponseEnd",
            Frame::OutputTransportReady(_) => "OutputTransportReady",
            Frame::Heartbeat(_) => "Heartbeat",
            Frame::ProcessorPause(_) => "ProcessorPause",
            Frame::ProcessorResume(_) => "ProcessorResume",
            Frame::FilterUpdateSettings(_) => "FilterUpdateSettings",
            Frame::FilterEnable(_) => "FilterEnable",
            Frame::MixerUpdateSettings(_) => "MixerUpdateSettings",
            Frame::MixerEnable(_) => "MixerEnable",
            Frame::ServiceSwitcherRequestMetadata(_) => "ServiceSwitcherRequestMetadata",
            Frame::ManuallySwitchService(_) => "ManuallySwitchService",
            Frame::Wakeup(_) => "Wakeup",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// Categorization
// ---------------------------------------------------------------------------

impl Frame {
    pub fn is_system(&self) -> bool {
        matches!(
            self,
            Frame::Start(_)
                | Frame::Cancel(_)
                | Frame::Error(_)
                | Frame::Interruption(_)
                | Frame::UserStartedSpeaking(_)
                | Frame::UserStoppedSpeaking(_)
                | Frame::UserMuteStarted(_)
                | Frame::UserMuteStopped(_)
                | Frame::UserSpeaking(_)
                | Frame::VADUserStartedSpeaking(_)
                | Frame::VADUserStoppedSpeaking(_)
                | Frame::BotStartedSpeaking(_)
                | Frame::BotStoppedSpeaking(_)
                | Frame::BotSpeaking(_)
                | Frame::InputAudioRaw(_)
                | Frame::InputImageRaw(_)
                | Frame::InputTextRaw(_)
                | Frame::UserAudioRaw(_)
                | Frame::UserImageRaw(_)
                | Frame::UserImageRequest(_)
                | Frame::ProcessorPauseUrgent(_)
                | Frame::ProcessorResumeUrgent(_)
                | Frame::STTMute(_)
                | Frame::SpeechControlParams(_)
                | Frame::BotConnected(_)
                | Frame::ClientConnected(_)
                | Frame::UserIdleTimeoutUpdate(_)
                | Frame::InputTransportMessage(_)
                | Frame::OutputTransportMessageUrgent(_)
                | Frame::FunctionCallsStarted(_)
                | Frame::FunctionCallCancel(_)
                | Frame::ServiceMetadata(_)
                | Frame::STTMetadata(_)
                | Frame::Metrics(_)
                | Frame::EndTask(_)
                | Frame::CancelTask(_)
                | Frame::StopTask(_)
                | Frame::InterruptionTask(_)
                | Frame::OutputDTMFUrgent(_)
                | Frame::InputDTMF(_)
        )
    }

    pub fn is_data(&self) -> bool {
        matches!(
            self,
            Frame::OutputAudioRaw(_)
                | Frame::TTSAudioRaw(_)
                | Frame::SpeechOutputAudioRaw(_)
                | Frame::OutputImageRaw(_)
                | Frame::URLImageRaw(_)
                | Frame::AssistantImageRaw(_)
                | Frame::Sprite(_)
                | Frame::Text(_)
                | Frame::LLMText(_)
                | Frame::TTSText(_)
                | Frame::Transcription(_)
                | Frame::InterimTranscription(_)
                | Frame::Translation(_)
                | Frame::TTSSpeak(_)
                | Frame::LLMThoughtText(_)
                | Frame::LLMContextAssistantTimestamp(_)
                | Frame::LLMRun(_)
                | Frame::LLMMessagesAppend(_)
                | Frame::LLMMessagesUpdate(_)
                | Frame::LLMSetTools(_)
                | Frame::LLMSetToolChoice(_)
                | Frame::LLMEnablePromptCaching(_)
                | Frame::LLMConfigureOutput(_)
                | Frame::LLMContext(_)
                | Frame::FunctionCallResult(_)
                | Frame::OutputTransportMessage(_)
                | Frame::OutputDTMF(_)
                | Frame::AggregatedText(_)
        )
    }

    pub fn is_control(&self) -> bool {
        matches!(
            self,
            Frame::End(_)
                | Frame::Stop(_)
                | Frame::FunctionCallInProgress(_)
                | Frame::ServiceUpdateSettings(_)
                | Frame::LLMUpdateSettings(_)
                | Frame::TTSUpdateSettings(_)
                | Frame::STTUpdateSettings(_)
                | Frame::LLMContextSummaryResult(_)
                | Frame::TTSStarted(_)
                | Frame::TTSStopped(_)
                | Frame::LLMFullResponseStart(_)
                | Frame::LLMFullResponseEnd(_)
                | Frame::LLMThoughtStart(_)
                | Frame::LLMThoughtEnd(_)
                | Frame::LLMAssistantPushAggregation(_)
                | Frame::LLMSummarizeContext(_)
                | Frame::LLMContextSummaryRequest(_)
                | Frame::VadParamsUpdate(_)
                | Frame::VisionFullResponseStart(_)
                | Frame::VisionFullResponseEnd(_)
                | Frame::OutputTransportReady(_)
                | Frame::Heartbeat(_)
                | Frame::ProcessorPause(_)
                | Frame::ProcessorResume(_)
                | Frame::FilterUpdateSettings(_)
                | Frame::FilterEnable(_)
                | Frame::MixerUpdateSettings(_)
                | Frame::MixerEnable(_)
                | Frame::ServiceSwitcherRequestMetadata(_)
                | Frame::ManuallySwitchService(_)
                | Frame::Wakeup(_)
        )
    }

    /// Lifecycle frames that ParallelPipeline synchronizes across branches.
    pub fn is_lifecycle(&self) -> bool {
        matches!(self, Frame::Start(_) | Frame::End(_) | Frame::Cancel(_))
    }

    pub fn is_uninterruptible(&self) -> bool {
        matches!(
            self,
            Frame::End(_)
                | Frame::Stop(_)
                | Frame::FunctionCallResult(_)
                | Frame::FunctionCallInProgress(_)
                | Frame::ServiceUpdateSettings(_)
                | Frame::LLMUpdateSettings(_)
                | Frame::TTSUpdateSettings(_)
                | Frame::STTUpdateSettings(_)
                | Frame::LLMContextSummaryResult(_)
                | Frame::Wakeup(_)
        )
    }

    /// Compile-time exhaustiveness check. Not called at runtime — exists solely
    /// so that adding a new Frame variant without updating the categorization
    /// methods produces a compiler error here.
    #[allow(dead_code, unreachable_code)]
    fn exhaustiveness_check(frame: &Frame) {
        // If you add a variant and get a "non-exhaustive patterns" error here,
        // update is_system/is_data/is_control, Display, and the test all_frames().
        match frame {
            // System
            Frame::Start(_) => {}
            Frame::Cancel(_) => {}
            Frame::Error(_) => {}
            Frame::Interruption(_) => {}
            Frame::UserStartedSpeaking(_) => {}
            Frame::UserStoppedSpeaking(_) => {}
            Frame::UserMuteStarted(_) => {}
            Frame::UserMuteStopped(_) => {}
            Frame::UserSpeaking(_) => {}
            Frame::VADUserStartedSpeaking(_) => {}
            Frame::VADUserStoppedSpeaking(_) => {}
            Frame::BotStartedSpeaking(_) => {}
            Frame::BotStoppedSpeaking(_) => {}
            Frame::BotSpeaking(_) => {}
            Frame::InputAudioRaw(_) => {}
            Frame::InputImageRaw(_) => {}
            Frame::InputTextRaw(_) => {}
            Frame::UserAudioRaw(_) => {}
            Frame::UserImageRaw(_) => {}
            Frame::UserImageRequest(_) => {}
            Frame::ProcessorPauseUrgent(_) => {}
            Frame::ProcessorResumeUrgent(_) => {}
            Frame::STTMute(_) => {}
            Frame::SpeechControlParams(_) => {}
            Frame::BotConnected(_) => {}
            Frame::ClientConnected(_) => {}
            Frame::UserIdleTimeoutUpdate(_) => {}
            Frame::InputTransportMessage(_) => {}
            Frame::OutputTransportMessageUrgent(_) => {}
            Frame::FunctionCallsStarted(_) => {}
            Frame::FunctionCallCancel(_) => {}
            Frame::ServiceMetadata(_) => {}
            Frame::STTMetadata(_) => {}
            Frame::Metrics(_) => {}
            Frame::EndTask(_) => {}
            Frame::CancelTask(_) => {}
            Frame::StopTask(_) => {}
            Frame::InterruptionTask(_) => {}
            Frame::OutputDTMFUrgent(_) => {}
            Frame::InputDTMF(_) => {}
            // Data
            Frame::OutputAudioRaw(_) => {}
            Frame::TTSAudioRaw(_) => {}
            Frame::SpeechOutputAudioRaw(_) => {}
            Frame::OutputImageRaw(_) => {}
            Frame::URLImageRaw(_) => {}
            Frame::AssistantImageRaw(_) => {}
            Frame::Sprite(_) => {}
            Frame::Text(_) => {}
            Frame::LLMText(_) => {}
            Frame::TTSText(_) => {}
            Frame::Transcription(_) => {}
            Frame::InterimTranscription(_) => {}
            Frame::Translation(_) => {}
            Frame::TTSSpeak(_) => {}
            Frame::LLMThoughtText(_) => {}
            Frame::LLMContextAssistantTimestamp(_) => {}
            Frame::LLMRun(_) => {}
            Frame::LLMMessagesAppend(_) => {}
            Frame::LLMMessagesUpdate(_) => {}
            Frame::LLMSetTools(_) => {}
            Frame::LLMSetToolChoice(_) => {}
            Frame::LLMEnablePromptCaching(_) => {}
            Frame::LLMConfigureOutput(_) => {}
            Frame::LLMContext(_) => {}
            Frame::FunctionCallResult(_) => {}
            Frame::OutputTransportMessage(_) => {}
            Frame::OutputDTMF(_) => {}
            Frame::AggregatedText(_) => {}
            // Control
            Frame::End(_) => {}
            Frame::Stop(_) => {}
            Frame::FunctionCallInProgress(_) => {}
            Frame::ServiceUpdateSettings(_) => {}
            Frame::LLMUpdateSettings(_) => {}
            Frame::TTSUpdateSettings(_) => {}
            Frame::STTUpdateSettings(_) => {}
            Frame::LLMContextSummaryResult(_) => {}
            Frame::TTSStarted(_) => {}
            Frame::TTSStopped(_) => {}
            Frame::LLMFullResponseStart(_) => {}
            Frame::LLMFullResponseEnd(_) => {}
            Frame::LLMThoughtStart(_) => {}
            Frame::LLMThoughtEnd(_) => {}
            Frame::LLMAssistantPushAggregation(_) => {}
            Frame::LLMSummarizeContext(_) => {}
            Frame::LLMContextSummaryRequest(_) => {}
            Frame::VadParamsUpdate(_) => {}
            Frame::VisionFullResponseStart(_) => {}
            Frame::VisionFullResponseEnd(_) => {}
            Frame::OutputTransportReady(_) => {}
            Frame::Heartbeat(_) => {}
            Frame::ProcessorPause(_) => {}
            Frame::ProcessorResume(_) => {}
            Frame::FilterUpdateSettings(_) => {}
            Frame::FilterEnable(_) => {}
            Frame::MixerUpdateSettings(_) => {}
            Frame::MixerEnable(_) => {}
            Frame::ServiceSwitcherRequestMetadata(_) => {}
            Frame::ManuallySwitchService(_) => {}
            Frame::Wakeup(_) => {}
        }
    }
}

#[cfg(test)]
mod tests;
