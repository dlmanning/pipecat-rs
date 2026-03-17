use std::collections::HashMap;

use bytes::Bytes;

use super::*;

fn audio() -> AudioRawFrame {
    AudioRawFrame {
        audio: Bytes::new(),
        sample_rate: 16000,
        num_channels: 1,
    }
}

fn image() -> ImageRawFrame {
    ImageRawFrame {
        image: Bytes::new(),
        size: (0, 0),
        format: None,
    }
}

/// Every Frame variant, for exhaustive categorization testing.
/// Keep in sync with Frame enum — the exhaustiveness_check() function
/// will produce a compile error if a variant is missing from there,
/// and every_frame_is_exactly_one_category will catch if one is missing here.
fn all_frames() -> Vec<(&'static str, Frame)> {
    vec![
        // System
        ("Start", Frame::Start(StartFrame::default())),
        ("Cancel", Frame::Cancel(CancelFrame::default())),
        (
            "Error",
            Frame::Error(ErrorFrame {
                error: "e".into(),
                fatal: false,
                source_processor: "p".into(),
            }),
        ),
        ("Interruption", Frame::Interruption(InterruptionFrame)),
        (
            "UserStartedSpeaking",
            Frame::UserStartedSpeaking(UserStartedSpeakingFrame),
        ),
        (
            "UserStoppedSpeaking",
            Frame::UserStoppedSpeaking(UserStoppedSpeakingFrame),
        ),
        (
            "UserMuteStarted",
            Frame::UserMuteStarted(UserMuteStartedFrame),
        ),
        (
            "UserMuteStopped",
            Frame::UserMuteStopped(UserMuteStoppedFrame),
        ),
        ("UserSpeaking", Frame::UserSpeaking(UserSpeakingFrame)),
        (
            "VADUserStartedSpeaking",
            Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
                start_secs: 0.0,
                timestamp: 0.0,
            }),
        ),
        (
            "VADUserStoppedSpeaking",
            Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
                stop_secs: 0.0,
                timestamp: 0.0,
            }),
        ),
        (
            "BotStartedSpeaking",
            Frame::BotStartedSpeaking(BotStartedSpeakingFrame),
        ),
        (
            "BotStoppedSpeaking",
            Frame::BotStoppedSpeaking(BotStoppedSpeakingFrame),
        ),
        ("BotSpeaking", Frame::BotSpeaking(BotSpeakingFrame)),
        ("InputAudioRaw", Frame::InputAudioRaw(audio())),
        ("InputImageRaw", Frame::InputImageRaw(image())),
        (
            "InputTextRaw",
            Frame::InputTextRaw(InputTextRawFrame { text: "hi".into() }),
        ),
        (
            "UserAudioRaw",
            Frame::UserAudioRaw(UserAudioRawFrame {
                audio: Bytes::new(),
                sample_rate: 16000,
                num_channels: 1,
                user_id: String::new(),
            }),
        ),
        (
            "UserImageRaw",
            Frame::UserImageRaw(UserImageRawFrame {
                image: Bytes::new(),
                size: (0, 0),
                format: None,
                user_id: String::new(),
                text: None,
                append_to_context: None,
            }),
        ),
        (
            "UserImageRequest",
            Frame::UserImageRequest(UserImageRequestFrame {
                user_id: String::new(),
                text: None,
                append_to_context: None,
                video_source: None,
                function_name: None,
                tool_call_id: None,
            }),
        ),
        (
            "ProcessorPauseUrgent",
            Frame::ProcessorPauseUrgent(ProcessorPauseUrgentFrame {
                processor_name: "p".into(),
            }),
        ),
        (
            "ProcessorResumeUrgent",
            Frame::ProcessorResumeUrgent(ProcessorResumeUrgentFrame {
                processor_name: "p".into(),
            }),
        ),
        ("STTMute", Frame::STTMute(STTMuteFrame { mute: true })),
        (
            "SpeechControlParams",
            Frame::SpeechControlParams(SpeechControlParamsFrame {
                vad_params: None,
                turn_params: None,
            }),
        ),
        ("BotConnected", Frame::BotConnected(BotConnectedFrame)),
        (
            "ClientConnected",
            Frame::ClientConnected(ClientConnectedFrame),
        ),
        (
            "UserIdleTimeoutUpdate",
            Frame::UserIdleTimeoutUpdate(UserIdleTimeoutUpdateFrame { timeout: 5.0 }),
        ),
        (
            "InputTransportMessage",
            Frame::InputTransportMessage(InputTransportMessageFrame {
                message: serde_json::Value::Null,
            }),
        ),
        (
            "OutputTransportMessageUrgent",
            Frame::OutputTransportMessageUrgent(OutputTransportMessageUrgentFrame {
                message: serde_json::Value::Null,
            }),
        ),
        (
            "FunctionCallsStarted",
            Frame::FunctionCallsStarted(FunctionCallsStartedFrame {
                function_calls: vec![],
            }),
        ),
        (
            "FunctionCallCancel",
            Frame::FunctionCallCancel(FunctionCallCancelFrame {
                function_name: "f".into(),
                tool_call_id: "t".into(),
            }),
        ),
        (
            "ServiceMetadata",
            Frame::ServiceMetadata(ServiceMetadataFrame {
                service_name: "s".into(),
            }),
        ),
        (
            "STTMetadata",
            Frame::STTMetadata(STTMetadataFrame {
                service_name: "s".into(),
                ttfs_p99_latency: 0.35,
            }),
        ),
        ("Metrics", Frame::Metrics(MetricsFrame { data: vec![] })),
        (
            "EndTask",
            Frame::EndTask(EndTaskFrame {
                task_id: "t".into(),
                handler_id: "h".into(),
                reason: None,
            }),
        ),
        (
            "CancelTask",
            Frame::CancelTask(CancelTaskFrame {
                task_id: "t".into(),
                handler_id: "h".into(),
                reason: None,
            }),
        ),
        (
            "StopTask",
            Frame::StopTask(StopTaskFrame {
                task_id: "t".into(),
                handler_id: "h".into(),
            }),
        ),
        (
            "InterruptionTask",
            Frame::InterruptionTask(InterruptionTaskFrame {
                task_id: "t".into(),
                handler_id: "h".into(),
            }),
        ),
        // Data
        ("OutputAudioRaw", Frame::OutputAudioRaw(audio())),
        (
            "TTSAudioRaw",
            Frame::TTSAudioRaw(TTSAudioRawFrame {
                audio: Bytes::new(),
                sample_rate: 24000,
                num_channels: 1,
                context_id: None,
            }),
        ),
        ("SpeechOutputAudioRaw", Frame::SpeechOutputAudioRaw(audio())),
        ("OutputImageRaw", Frame::OutputImageRaw(image())),
        (
            "URLImageRaw",
            Frame::URLImageRaw(URLImageRawFrame {
                image: Bytes::new(),
                size: (0, 0),
                format: None,
                url: None,
            }),
        ),
        (
            "AssistantImageRaw",
            Frame::AssistantImageRaw(AssistantImageRawFrame {
                image: Bytes::new(),
                size: (0, 0),
                format: None,
                original_data: None,
                original_mime_type: None,
            }),
        ),
        ("Sprite", Frame::Sprite(SpriteFrame { images: vec![] })),
        ("Text", Frame::Text(TextFrame::new("t"))),
        ("LLMText", Frame::LLMText(TextFrame::new("t"))),
        ("TTSText", Frame::TTSText(TextFrame::new("t"))),
        (
            "Transcription",
            Frame::Transcription(TranscriptionFrame {
                text: "t".into(),
                user_id: "u".into(),
                timestamp: None,
                language: None,
                finalized: false,
                result: None,
            }),
        ),
        (
            "InterimTranscription",
            Frame::InterimTranscription(InterimTranscriptionFrame {
                text: "t".into(),
                user_id: "u".into(),
                timestamp: None,
                language: None,
                result: None,
            }),
        ),
        (
            "Translation",
            Frame::Translation(TranslationFrame {
                text: "t".into(),
                user_id: "u".into(),
                timestamp: None,
                language: None,
            }),
        ),
        (
            "TTSSpeak",
            Frame::TTSSpeak(TTSSpeakFrame {
                text: "t".into(),
                append_to_context: None,
            }),
        ),
        (
            "LLMThoughtText",
            Frame::LLMThoughtText(LLMThoughtTextFrame {
                text: "t".into(),
                includes_inter_frame_spaces: true,
            }),
        ),
        (
            "LLMContextAssistantTimestamp",
            Frame::LLMContextAssistantTimestamp(LLMContextAssistantTimestampFrame {
                timestamp: "ts".into(),
            }),
        ),
        ("LLMRun", Frame::LLMRun(LLMRunFrame)),
        (
            "LLMMessagesAppend",
            Frame::LLMMessagesAppend(LLMMessagesAppendFrame {
                messages: vec![],
                run_llm: None,
            }),
        ),
        (
            "LLMMessagesUpdate",
            Frame::LLMMessagesUpdate(LLMMessagesUpdateFrame {
                messages: vec![],
                run_llm: None,
            }),
        ),
        (
            "LLMSetTools",
            Frame::LLMSetTools(LLMSetToolsFrame {
                tools: serde_json::Value::Null,
            }),
        ),
        (
            "LLMSetToolChoice",
            Frame::LLMSetToolChoice(LLMSetToolChoiceFrame {
                tool_choice: serde_json::Value::Null,
            }),
        ),
        (
            "LLMEnablePromptCaching",
            Frame::LLMEnablePromptCaching(LLMEnablePromptCachingFrame { enable: true }),
        ),
        (
            "LLMConfigureOutput",
            Frame::LLMConfigureOutput(LLMConfigureOutputFrame { skip_tts: false }),
        ),
        (
            "LLMContext",
            Frame::LLMContext(LLMContextFrame {
                context: serde_json::Value::Null,
            }),
        ),
        (
            "FunctionCallResult",
            Frame::FunctionCallResult(FunctionCallResultFrame {
                function_name: "f".into(),
                tool_call_id: "t".into(),
                arguments: serde_json::Value::Null,
                result: serde_json::Value::Null,
                run_llm: None,
            }),
        ),
        (
            "OutputTransportMessage",
            Frame::OutputTransportMessage(OutputTransportMessageFrame {
                message: serde_json::Value::Null,
            }),
        ),
        // Control
        ("End", Frame::End(EndFrame::default())),
        ("Stop", Frame::Stop(StopFrame)),
        (
            "FunctionCallInProgress",
            Frame::FunctionCallInProgress(FunctionCallInProgressFrame {
                function_name: "f".into(),
                tool_call_id: "t".into(),
                arguments: serde_json::Value::Null,
                cancel_on_interruption: false,
            }),
        ),
        (
            "ServiceUpdateSettings",
            Frame::ServiceUpdateSettings(ServiceUpdateSettingsFrame::default()),
        ),
        (
            "LLMUpdateSettings",
            Frame::LLMUpdateSettings(ServiceUpdateSettingsFrame::default()),
        ),
        (
            "TTSUpdateSettings",
            Frame::TTSUpdateSettings(ServiceUpdateSettingsFrame::default()),
        ),
        (
            "STTUpdateSettings",
            Frame::STTUpdateSettings(ServiceUpdateSettingsFrame::default()),
        ),
        (
            "LLMContextSummaryResult",
            Frame::LLMContextSummaryResult(LLMContextSummaryResultFrame {
                request_id: "r".into(),
                summary: "s".into(),
                last_summarized_index: 0,
                error: None,
            }),
        ),
        ("TTSStarted", Frame::TTSStarted(TTSStartedFrame::default())),
        ("TTSStopped", Frame::TTSStopped(TTSStoppedFrame::default())),
        (
            "LLMFullResponseStart",
            Frame::LLMFullResponseStart(LLMFullResponseStartFrame { skip_tts: None }),
        ),
        (
            "LLMFullResponseEnd",
            Frame::LLMFullResponseEnd(LLMFullResponseEndFrame { skip_tts: None }),
        ),
        (
            "LLMThoughtStart",
            Frame::LLMThoughtStart(LLMThoughtStartFrame {
                append_to_context: false,
                llm: None,
            }),
        ),
        (
            "LLMThoughtEnd",
            Frame::LLMThoughtEnd(LLMThoughtEndFrame { signature: None }),
        ),
        (
            "LLMAssistantPushAggregation",
            Frame::LLMAssistantPushAggregation(LLMAssistantPushAggregationFrame),
        ),
        (
            "LLMSummarizeContext",
            Frame::LLMSummarizeContext(LLMSummarizeContextFrame { config: None }),
        ),
        (
            "LLMContextSummaryRequest",
            Frame::LLMContextSummaryRequest(LLMContextSummaryRequestFrame {
                request_id: "r".into(),
                context: serde_json::Value::Null,
                min_messages_to_keep: 5,
                target_context_tokens: 1000,
                summarization_prompt: "s".into(),
                summarization_timeout: None,
            }),
        ),
        (
            "VadParamsUpdate",
            Frame::VadParamsUpdate(VadParamsUpdateFrame {
                params: VadParams::default(),
            }),
        ),
        (
            "VisionFullResponseStart",
            Frame::VisionFullResponseStart(VisionFullResponseStartFrame { skip_tts: None }),
        ),
        (
            "VisionFullResponseEnd",
            Frame::VisionFullResponseEnd(VisionFullResponseEndFrame { skip_tts: None }),
        ),
        (
            "OutputTransportReady",
            Frame::OutputTransportReady(OutputTransportReadyFrame),
        ),
        (
            "Heartbeat",
            Frame::Heartbeat(HeartbeatFrame { timestamp: 0 }),
        ),
        (
            "ProcessorPause",
            Frame::ProcessorPause(ProcessorPauseFrame {
                processor_name: "p".into(),
            }),
        ),
        (
            "ProcessorResume",
            Frame::ProcessorResume(ProcessorResumeFrame {
                processor_name: "p".into(),
            }),
        ),
        (
            "FilterUpdateSettings",
            Frame::FilterUpdateSettings(FilterUpdateSettingsFrame {
                settings: HashMap::new(),
            }),
        ),
        (
            "FilterEnable",
            Frame::FilterEnable(FilterEnableFrame { enable: true }),
        ),
        (
            "MixerUpdateSettings",
            Frame::MixerUpdateSettings(MixerUpdateSettingsFrame {
                settings: HashMap::new(),
            }),
        ),
        (
            "MixerEnable",
            Frame::MixerEnable(MixerEnableFrame { enable: true }),
        ),
        (
            "ServiceSwitcherRequestMetadata",
            Frame::ServiceSwitcherRequestMetadata(ServiceSwitcherRequestMetadataFrame {
                service_name: "s".into(),
            }),
        ),
    ]
}

#[test]
fn frame_header_generates_unique_ids() {
    let h1 = FrameHeader::new();
    let h2 = FrameHeader::new();
    let h3 = FrameHeader::new();
    assert!(h2.id > h1.id);
    assert!(h3.id > h2.id);
}

#[test]
fn frame_envelope_wraps_frame() {
    let envelope = FrameEnvelope::new(Frame::Interruption(InterruptionFrame));
    assert!(envelope.header.id > 0);
    assert!(matches!(envelope.frame, Frame::Interruption(_)));
}

#[test]
fn every_frame_is_exactly_one_category() {
    for (name, frame) in all_frames() {
        let cats = [frame.is_system(), frame.is_data(), frame.is_control()];
        let count = cats.iter().filter(|&&c| c).count();
        assert_eq!(
            count, 1,
            "{name} belongs to {count} categories (expected 1): system={}, data={}, control={}",
            cats[0], cats[1], cats[2],
        );
    }
}

#[test]
fn system_frames_classified_correctly() {
    let system_names: Vec<&str> = all_frames()
        .iter()
        .filter(|(_, f)| f.is_system())
        .map(|(n, _)| *n)
        .collect();
    assert!(system_names.contains(&"Start"));
    assert!(system_names.contains(&"Interruption"));
    assert!(system_names.contains(&"InputAudioRaw"));
    assert!(system_names.contains(&"VADUserStartedSpeaking"));
    assert!(system_names.contains(&"Metrics"));
    assert!(system_names.contains(&"EndTask"));
}

#[test]
fn data_frames_classified_correctly() {
    let data_names: Vec<&str> = all_frames()
        .iter()
        .filter(|(_, f)| f.is_data())
        .map(|(n, _)| *n)
        .collect();
    assert!(data_names.contains(&"OutputAudioRaw"));
    assert!(data_names.contains(&"TTSAudioRaw"));
    assert!(data_names.contains(&"Text"));
    assert!(data_names.contains(&"Transcription"));
    assert!(data_names.contains(&"FunctionCallResult"));
    assert!(data_names.contains(&"LLMRun"));
}

#[test]
fn control_frames_classified_correctly() {
    let control_names: Vec<&str> = all_frames()
        .iter()
        .filter(|(_, f)| f.is_control())
        .map(|(n, _)| *n)
        .collect();
    assert!(control_names.contains(&"End"));
    assert!(control_names.contains(&"Stop"));
    assert!(control_names.contains(&"TTSStarted"));
    assert!(control_names.contains(&"VadParamsUpdate"));
    assert!(control_names.contains(&"LLMContextSummaryResult"));
    assert!(control_names.contains(&"VisionFullResponseStart"));
}

#[test]
fn uninterruptible_frames() {
    let uninterruptible: Vec<&str> = all_frames()
        .iter()
        .filter(|(_, f)| f.is_uninterruptible())
        .map(|(n, _)| *n)
        .collect();
    assert!(uninterruptible.contains(&"End"));
    assert!(uninterruptible.contains(&"Stop"));
    assert!(uninterruptible.contains(&"FunctionCallResult"));
    assert!(uninterruptible.contains(&"FunctionCallInProgress"));
    assert!(uninterruptible.contains(&"ServiceUpdateSettings"));
    assert!(uninterruptible.contains(&"LLMUpdateSettings"));
    assert!(uninterruptible.contains(&"TTSUpdateSettings"));
    assert!(uninterruptible.contains(&"STTUpdateSettings"));
    assert!(uninterruptible.contains(&"LLMContextSummaryResult"));

    // Regular data/control frames are interruptible
    assert!(!Frame::Text(TextFrame::new("t")).is_uninterruptible());
    assert!(!Frame::TTSStarted(TTSStartedFrame::default()).is_uninterruptible());
}

#[test]
fn display_shows_variant_name() {
    assert_eq!(format!("{}", Frame::Start(StartFrame::default())), "Start");
    assert_eq!(
        format!(
            "{}",
            Frame::TTSAudioRaw(TTSAudioRawFrame {
                audio: Bytes::new(),
                sample_rate: 24000,
                num_channels: 1,
                context_id: None,
            })
        ),
        "TTSAudioRaw"
    );
    assert_eq!(format!("{}", Frame::End(EndFrame::default())), "End");
}

#[test]
fn audio_frame_with_bytes() {
    let data = [0i16, 100, -100, 200];
    let bytes: Vec<u8> = data.iter().flat_map(|s| s.to_le_bytes()).collect();
    let frame = AudioRawFrame {
        audio: Bytes::from(bytes),
        sample_rate: 16000,
        num_channels: 1,
    };
    assert_eq!(frame.audio.len(), 8);
    assert_eq!(frame.sample_rate, 16000);
}

#[test]
fn start_frame_defaults() {
    let sf = StartFrame::default();
    assert_eq!(sf.audio_in_sample_rate, 16000);
    assert_eq!(sf.audio_out_sample_rate, 24000);
    assert!(!sf.allow_interruptions);
    assert!(!sf.enable_metrics);
}

#[test]
fn vad_params_defaults() {
    let vp = VadParams::default();
    assert_eq!(vp.confidence, 0.7);
    assert_eq!(vp.start_secs, 0.2);
    assert_eq!(vp.stop_secs, 0.8);
    assert_eq!(vp.min_volume, 0.6);
}

#[test]
fn settings_update_variants_use_same_struct() {
    // All settings update variants use ServiceUpdateSettingsFrame
    let s = ServiceUpdateSettingsFrame::default();
    let _ = Frame::ServiceUpdateSettings(s.clone());
    let _ = Frame::LLMUpdateSettings(s.clone());
    let _ = Frame::TTSUpdateSettings(s.clone());
    let _ = Frame::STTUpdateSettings(s);
}
