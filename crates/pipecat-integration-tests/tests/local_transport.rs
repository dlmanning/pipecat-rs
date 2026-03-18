use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::processor::{FrameProcessor, ProcessorBase, ProcessorContext};
use pipecat_core::test_utils::*;
use pipecat_pipeline::Pipeline;
use pipecat_transport::TransportParams;
use pipecat_transport::local::*;
use tokio::time::timeout;

const TEST_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// AudioLoopback — converts InputAudioRaw → OutputAudioRaw
// ---------------------------------------------------------------------------

/// Test processor that converts `InputAudioRaw` frames to `OutputAudioRaw`
/// frames, bridging the input and output transport frame domains.
struct AudioLoopback {
    base: ProcessorBase,
}

impl AudioLoopback {
    fn new() -> Self {
        Self {
            base: ProcessorBase::new("AudioLoopback"),
        }
    }
}

#[async_trait]
impl FrameProcessor for AudioLoopback {
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
            Frame::InputAudioRaw(audio) => {
                ctx.send_downstream(Frame::OutputAudioRaw(AudioRawFrame {
                    audio: audio.audio.clone(),
                    sample_rate: audio.sample_rate,
                    num_channels: audio.num_channels,
                }))
                .await?;
                Ok(())
            }
            _ => ctx.push_frame(envelope, direction).await,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_pcm_data(num_samples: usize) -> Bytes {
    let samples: Vec<i16> = (0..num_samples).map(|i| (i % 200) as i16).collect();
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    Bytes::from(bytes)
}

// ---------------------------------------------------------------------------
// Test: Loopback round-trip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn loopback_round_trip() {
    // 2000 samples at 16kHz = 125ms of audio, several 20ms chunks
    let pcm_data = make_pcm_data(2000);
    let output_buf = Arc::new(StdMutex::new(Vec::<u8>::new()));

    let in_params = TransportParams {
        audio_in_enabled: true,
        audio_in_passthrough: true,
        ..Default::default()
    };
    let out_params = TransportParams {
        audio_out_enabled: true,
        // Use 16kHz output to match input, avoiding resampling complexities
        audio_out_sample_rate: Some(16000),
        ..Default::default()
    };

    let input_transport =
        LocalAudioInputTransport::new(in_params, AudioInputSource::Buffer(pcm_data));
    let loopback = AudioLoopback::new();
    let output_transport =
        LocalAudioOutputTransport::new(out_params, AudioOutputSink::Buffer(output_buf.clone()));

    let pipeline = Pipeline::new(vec![
        Box::new(input_transport),
        Box::new(loopback),
        Box::new(output_transport),
    ]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    send_frame(
        &handle,
        Frame::Start(StartFrame {
            audio_in_sample_rate: 16000,
            audio_out_sample_rate: 16000,
            ..Default::default()
        }),
        Direction::Downstream,
    )
    .await;

    // Wait for audio to flow through the full pipeline:
    // InputTransport → Loopback → OutputTransport → downstream.
    // The output transport writes to the buffer BEFORE pushing downstream,
    // so by the time we see OutputAudioRaw, the buffer has data.
    down.wait_for_frame("OutputAudioRaw").await;

    // Shut down cleanly.
    send_frame(
        &handle,
        Frame::End(EndFrame::default()),
        Direction::Downstream,
    )
    .await;

    down.wait_for_frame("End").await;

    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let collected = output_buf.lock().unwrap();
    assert!(
        !collected.is_empty(),
        "output buffer should contain audio after loopback round-trip"
    );

    // Verify we got a reasonable amount of audio through (at least some chunks).
    // The output transport chunks in 40ms increments, so we should have at least
    // one chunk (1280 bytes at 16kHz, 1ch, 4x10ms).
    assert!(
        collected.len() >= 640,
        "expected at least 640 bytes of audio, got {}",
        collected.len()
    );
}

// ---------------------------------------------------------------------------
// Test: Lifecycle — Start → audio flows → End → clean shutdown
// ---------------------------------------------------------------------------

#[tokio::test]
async fn lifecycle_start_audio_end_shutdown() {
    let pcm_data = make_pcm_data(320); // 20ms at 16kHz
    let output_buf = Arc::new(StdMutex::new(Vec::<u8>::new()));

    let in_params = TransportParams {
        audio_in_enabled: true,
        audio_in_passthrough: true,
        ..Default::default()
    };
    let out_params = TransportParams {
        audio_out_enabled: true,
        audio_out_sample_rate: Some(16000),
        ..Default::default()
    };

    let input_transport =
        LocalAudioInputTransport::new(in_params, AudioInputSource::Buffer(pcm_data));
    let loopback = AudioLoopback::new();
    let output_transport =
        LocalAudioOutputTransport::new(out_params, AudioOutputSink::Buffer(output_buf.clone()));

    let pipeline = Pipeline::new(vec![
        Box::new(input_transport),
        Box::new(loopback),
        Box::new(output_transport),
    ]);

    let (node, handle, down_rx, _up_rx) = make_node(Box::new(pipeline));
    let down = FrameCollector::spawn(down_rx);

    let run = tokio::spawn(async move { node.run().await });

    // Start
    send_frame(
        &handle,
        Frame::Start(StartFrame {
            audio_in_sample_rate: 16000,
            audio_out_sample_rate: 16000,
            ..Default::default()
        }),
        Direction::Downstream,
    )
    .await;

    down.wait_for_frame("Start").await;

    // End
    send_frame(
        &handle,
        Frame::End(EndFrame::default()),
        Direction::Downstream,
    )
    .await;

    down.wait_for_frame("End").await;

    // Cancel to terminate the node
    send_frame(
        &handle,
        Frame::Cancel(CancelFrame::default()),
        Direction::Downstream,
    )
    .await;

    timeout(TEST_TIMEOUT, run).await.unwrap().unwrap();

    let names = down.frame_names();
    assert!(
        names.contains(&"Start".to_string()),
        "should have Start frame: {names:?}"
    );
    assert!(
        names.contains(&"End".to_string()),
        "should have End frame: {names:?}"
    );
}
