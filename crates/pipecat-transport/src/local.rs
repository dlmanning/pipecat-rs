use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use bytes::Bytes;
use pipecat_core::error::Result;
use pipecat_core::frame::*;
use pipecat_core::node::ProcessorNodeHandle;
use pipecat_core::processor::{FrameProcessor, ProcessorContext};
use tokio::sync::mpsc;
use tracing;

use crate::input::BaseInputTransport;
use crate::output::{BaseOutputTransport, OutputTransportCallbacks};
use crate::params::TransportParams;

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Audio file format for reading/writing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AudioFormat {
    /// Raw PCM: signed 16-bit little-endian interleaved samples.
    #[default]
    RawPcm,
    /// Auto-detect format via symphonia.
    /// Supports WAV, MP3, FLAC, OGG/Vorbis, AAC, and more.
    Encoded,
}

/// Source of audio data for the local input transport.
#[derive(Debug)]
pub enum AudioInputSource {
    /// Raw PCM bytes (useful for tests).
    Buffer(Bytes),
    /// Audio file on disk (interpreted according to `AudioFormat`).
    File(PathBuf),
    /// Programmatic injection via a channel (always raw PCM frames).
    Channel(mpsc::Receiver<AudioRawFrame>),
}

/// Destination for audio data from the local output transport.
pub enum AudioOutputSink {
    /// Collect raw PCM bytes for assertions.
    Buffer(Arc<StdMutex<Vec<u8>>>),
    /// Write raw PCM to a file on disk.
    File(PathBuf),
    /// Discard all output audio.
    Discard,
}

impl std::fmt::Debug for AudioOutputSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buffer(_) => f.debug_tuple("Buffer").finish(),
            Self::File(path) => f.debug_tuple("File").field(path).finish(),
            Self::Discard => write!(f, "Discard"),
        }
    }
}

/// Controls the pacing of audio chunk delivery.
#[derive(Debug, Default)]
pub enum AudioPacing {
    /// Deliver chunks as fast as possible (for tests).
    #[default]
    AsFastAsPossible,
    /// Deliver chunks at real-time rate (~20ms per chunk).
    RealTime,
}

// ---------------------------------------------------------------------------
// LocalAudioInputTransport
// ---------------------------------------------------------------------------

/// Local audio input transport that reads from a buffer, file, or channel
/// and feeds audio into the pipeline via `BaseInputTransport`.
pub struct LocalAudioInputTransport {
    inner: BaseInputTransport,
    source: Option<AudioInputSource>,
    pacing: AudioPacing,
    format: AudioFormat,
    num_channels: u16,
    source_task: Option<tokio::task::JoinHandle<()>>,
    node_handle: Option<ProcessorNodeHandle>,
}

impl LocalAudioInputTransport {
    pub fn new(params: TransportParams, source: AudioInputSource) -> Self {
        let num_channels = params.audio_in_channels;
        Self {
            inner: BaseInputTransport::with_name("LocalAudioInput", params),
            source: Some(source),
            pacing: AudioPacing::default(),
            format: AudioFormat::default(),
            num_channels,
            source_task: None,
            node_handle: None,
        }
    }

    pub fn with_pacing(mut self, pacing: AudioPacing) -> Self {
        self.pacing = pacing;
        self
    }

    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = format;
        self
    }

    fn spawn_source_task(&mut self) {
        let Some(source) = self.source.take() else {
            tracing::warn!("LocalAudioInput: source already consumed");
            return;
        };

        let Some(tx) = self.inner.take_audio_in_sender() else {
            tracing::warn!("LocalAudioInput: audio task not ready, cannot spawn source");
            return;
        };

        let sample_rate = self.inner.sample_rate();
        let num_channels = self.num_channels;
        let realtime = matches!(self.pacing, AudioPacing::RealTime);
        let format = self.format;
        let node_handle = self.node_handle.clone();
        let audio_drained = self.inner.audio_drained_notify();

        self.source_task = Some(tokio::spawn(async move {
            let is_finite = !matches!(&source, AudioInputSource::Channel(_));

            match (source, format) {
                (AudioInputSource::Buffer(data), AudioFormat::RawPcm) => {
                    let chunk_bytes = compute_chunk_bytes(sample_rate, num_channels);
                    feed_chunks(&tx, &data, chunk_bytes, sample_rate, num_channels, realtime).await;
                }
                (AudioInputSource::File(path), AudioFormat::RawPcm) => match std::fs::read(&path) {
                    Ok(data) => {
                        let chunk_bytes = compute_chunk_bytes(sample_rate, num_channels);
                        feed_chunks(&tx, &data, chunk_bytes, sample_rate, num_channels, realtime)
                            .await;
                    }
                    Err(e) => {
                        tracing::error!("LocalAudioInput: failed to read file {:?}: {}", path, e);
                    }
                },

                (AudioInputSource::Buffer(data), AudioFormat::Encoded) => {
                    let cursor = std::io::Cursor::new(data);
                    decode_and_feed(Box::new(cursor), &tx, realtime).await;
                }
                (AudioInputSource::File(path), AudioFormat::Encoded) => {
                    match std::fs::File::open(&path) {
                        Ok(file) => {
                            decode_and_feed(Box::new(file), &tx, realtime).await;
                        }
                        Err(e) => {
                            tracing::error!(
                                "LocalAudioInput: failed to open file {:?}: {}",
                                path,
                                e
                            );
                        }
                    }
                }
                (AudioInputSource::Channel(mut rx), _) => {
                    // Channel is always raw PCM frames, format ignored.
                    let mut interval = realtime
                        .then(|| tokio::time::interval(std::time::Duration::from_millis(20)));
                    while let Some(frame) = rx.recv().await {
                        if let Some(ref mut iv) = interval {
                            iv.tick().await;
                        }
                        if tx.send(frame).await.is_err() {
                            break;
                        }
                    }
                }
            }

            // Finite sources: drop the audio sender to close the channel,
            // wait for the audio task to drain all queued frames, then send
            // EndFrame so the pipeline shuts down cleanly.
            if is_finite {
                drop(tx);
                audio_drained.notified().await;
                if let Some(handle) = node_handle {
                    handle
                        .send(
                            FrameEnvelope::new(Frame::End(EndFrame::default())),
                            Direction::Downstream,
                        )
                        .await
                        .ok();
                }
            }
        }));
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute 20ms chunk size in bytes for i16 PCM.
fn compute_chunk_bytes(sample_rate: u32, num_channels: u16) -> usize {
    (sample_rate as usize / 50) * (num_channels as usize) * 2
}

/// Feed raw PCM data as chunked AudioRawFrames through the sender.
async fn feed_chunks(
    tx: &mpsc::Sender<AudioRawFrame>,
    data: &[u8],
    chunk_bytes: usize,
    sample_rate: u32,
    num_channels: u16,
    realtime: bool,
) {
    // Use interval (not sleep) for real-time pacing to avoid cumulative
    // timer drift that would slowly drain the output buffer.
    let mut interval =
        realtime.then(|| tokio::time::interval(std::time::Duration::from_millis(20)));

    for chunk in data.chunks(chunk_bytes) {
        if let Some(ref mut iv) = interval {
            iv.tick().await;
        }
        let frame = AudioRawFrame {
            audio: Bytes::copy_from_slice(chunk),
            sample_rate,
            num_channels,
        };
        if tx.send(frame).await.is_err() {
            break;
        }
    }
}

/// Decode audio from any supported format and stream 20ms chunks through the sender.
/// Decodes incrementally so playback starts immediately without buffering the entire file.
async fn decode_and_feed(
    source: Box<dyn symphonia::core::io::MediaSource>,
    tx: &mpsc::Sender<AudioRawFrame>,
    realtime: bool,
) {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let mss = MediaSourceStream::new(source, Default::default());
    let hint = Hint::new();

    let probed = match symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    ) {
        Ok(probed) => probed,
        Err(e) => {
            tracing::error!("LocalAudioInput: failed to probe audio format: {}", e);
            return;
        }
    };

    let mut format_reader = probed.format;
    let track = match format_reader.default_track() {
        Some(track) => track,
        None => {
            tracing::error!("LocalAudioInput: no audio track found");
            return;
        }
    };

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let num_channels = track.codec_params.channels.map_or(0, |c| c.count()) as u16;

    if sample_rate == 0 || num_channels == 0 {
        tracing::error!(
            "LocalAudioInput: invalid audio params: sr={}, ch={}",
            sample_rate,
            num_channels
        );
        return;
    }

    let mut decoder = match symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
    {
        Ok(decoder) => decoder,
        Err(e) => {
            tracing::error!("LocalAudioInput: failed to create decoder: {}", e);
            return;
        }
    };

    let chunk_bytes = compute_chunk_bytes(sample_rate, num_channels);
    let mut buffer = Vec::new();
    let mut interval =
        realtime.then(|| tokio::time::interval(std::time::Duration::from_millis(20)));

    loop {
        let packet = match format_reader.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                tracing::error!("LocalAudioInput: error reading packet: {}", e);
                break;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(e) => {
                tracing::warn!("LocalAudioInput: decode error (skipping packet): {}", e);
                continue;
            }
        };

        let spec = *decoded.spec();
        let duration = decoded.capacity();
        let mut sample_buf = SampleBuffer::<i16>::new(duration as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);

        for &sample in sample_buf.samples() {
            buffer.extend_from_slice(&sample.to_le_bytes());
        }

        // Send complete 20ms chunks as they become available.
        while buffer.len() >= chunk_bytes {
            if let Some(ref mut iv) = interval {
                iv.tick().await;
            }
            let chunk: Vec<u8> = buffer.drain(..chunk_bytes).collect();
            let frame = AudioRawFrame {
                audio: Bytes::from(chunk),
                sample_rate,
                num_channels,
            };
            if tx.send(frame).await.is_err() {
                return;
            }
        }
    }

    // Send any remaining samples as a final (possibly short) chunk.
    if !buffer.is_empty() {
        if let Some(ref mut iv) = interval {
            iv.tick().await;
        }
        let frame = AudioRawFrame {
            audio: Bytes::from(buffer),
            sample_rate,
            num_channels,
        };
        tx.send(frame).await.ok();
    }
}

/// Cancel the source task if it's running.
async fn cancel_source_task(source_task: &mut Option<tokio::task::JoinHandle<()>>) {
    if let Some(handle) = source_task.take() {
        handle.abort();
        let _ = handle.await;
    }
}

#[async_trait]
impl FrameProcessor for LocalAudioInputTransport {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn id(&self) -> u64 {
        self.inner.id()
    }

    fn set_node_handle(&mut self, handle: ProcessorNodeHandle) {
        self.node_handle = Some(handle);
    }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let is_start = matches!(&envelope.frame, Frame::Start(_));
        let is_end = matches!(&envelope.frame, Frame::End(_));
        let is_cancel = matches!(&envelope.frame, Frame::Cancel(_));

        // Delegate to inner transport for all frame handling.
        self.inner.process_frame(envelope, direction, ctx).await?;

        if is_start {
            // After inner has initialized, mark transport ready and spawn source.
            self.inner.set_transport_ready().await;
            self.spawn_source_task();
        } else if is_end || is_cancel {
            // Explicitly cancel the source task on shutdown.
            cancel_source_task(&mut self.source_task).await;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LocalOutputCallbacks
// ---------------------------------------------------------------------------

/// The resolved sink used by the callbacks at runtime.
enum ResolvedSink {
    Buffer(Arc<StdMutex<Vec<u8>>>),
    File(StdMutex<std::io::BufWriter<std::fs::File>>),
    Discard,
}

/// Callbacks for the local output transport that write audio to a sink.
struct LocalOutputCallbacks {
    sink: ResolvedSink,
}

impl LocalOutputCallbacks {
    fn new(sink: AudioOutputSink) -> std::io::Result<Self> {
        let resolved = match sink {
            AudioOutputSink::Buffer(buf) => ResolvedSink::Buffer(buf),
            AudioOutputSink::File(path) => {
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)?;
                ResolvedSink::File(StdMutex::new(std::io::BufWriter::new(file)))
            }
            AudioOutputSink::Discard => ResolvedSink::Discard,
        };
        Ok(Self { sink: resolved })
    }
}

#[async_trait]
impl OutputTransportCallbacks for LocalOutputCallbacks {
    async fn write_audio_frame(&self, frame: &AudioRawFrame) -> bool {
        use std::io::Write;
        match &self.sink {
            ResolvedSink::Buffer(buf) => {
                buf.lock().unwrap().extend_from_slice(&frame.audio);
                true
            }
            ResolvedSink::File(writer) => {
                let mut w = writer.lock().unwrap();
                if let Err(e) = w.write_all(&frame.audio) {
                    tracing::error!("LocalAudioOutput: write error: {}", e);
                    return false;
                }
                true
            }
            ResolvedSink::Discard => true,
        }
    }
}

// ---------------------------------------------------------------------------
// LocalAudioOutputTransport
// ---------------------------------------------------------------------------

/// Local audio output transport that writes raw PCM audio to a buffer, file, or discards it.
pub struct LocalAudioOutputTransport {
    inner: BaseOutputTransport,
    output_buffer: Option<Arc<StdMutex<Vec<u8>>>>,
}

impl LocalAudioOutputTransport {
    /// Create a new local audio output transport.
    ///
    /// # Panics
    ///
    /// Panics if `sink` is `AudioOutputSink::File` and the file cannot be opened.
    pub fn new(params: TransportParams, sink: AudioOutputSink) -> Self {
        let output_buffer = match &sink {
            AudioOutputSink::Buffer(buf) => Some(buf.clone()),
            _ => None,
        };
        let callbacks = Arc::new(
            LocalOutputCallbacks::new(sink).expect("LocalAudioOutput: failed to open output sink"),
        );
        Self {
            inner: BaseOutputTransport::with_name("LocalAudioOutput", params, callbacks),
            output_buffer,
        }
    }

    /// Get the output buffer if this transport was configured with `AudioOutputSink::Buffer`.
    pub fn output_buffer(&self) -> Option<Arc<StdMutex<Vec<u8>>>> {
        self.output_buffer.clone()
    }
}

#[async_trait]
impl FrameProcessor for LocalAudioOutputTransport {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn id(&self) -> u64 {
        self.inner.id()
    }

    async fn process_frame(
        &mut self,
        envelope: FrameEnvelope,
        direction: Direction,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        let is_start = matches!(&envelope.frame, Frame::Start(_));

        // Delegate to inner transport for all frame handling.
        self.inner.process_frame(envelope, direction, ctx).await?;

        if is_start {
            self.inner.set_transport_ready().await;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LocalAudioTransport — convenience builder
// ---------------------------------------------------------------------------

/// Convenience builder that wraps both input and output local transports.
///
/// Since `TransportParams` contains non-Clone trait objects, the builder
/// takes separate params for input and output transports.
pub struct LocalAudioTransport;

impl LocalAudioTransport {
    /// Build an input transport with the given params and source.
    pub fn input(
        mut params: TransportParams,
        source: AudioInputSource,
    ) -> LocalAudioInputTransport {
        params.audio_in_enabled = true;
        params.audio_in_passthrough = true;
        LocalAudioInputTransport::new(params, source)
    }

    /// Build an output transport with the given params and sink.
    pub fn output(mut params: TransportParams, sink: AudioOutputSink) -> LocalAudioOutputTransport {
        params.audio_out_enabled = true;
        LocalAudioOutputTransport::new(params, sink)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use pipecat_core::test_utils::*;

    use super::*;

    fn make_pcm_bytes(num_samples: usize) -> Bytes {
        let samples: Vec<i16> = (0..num_samples).map(|i| (i % 200) as i16).collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        Bytes::from(bytes)
    }

    /// Create a WAV buffer in memory with the given samples.
    fn make_wav_bytes(samples: &[i16], sample_rate: u32, num_channels: u16) -> Bytes {
        use std::io::Write;
        let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let data_size = pcm.len() as u32;
        let byte_rate = sample_rate * num_channels as u32 * 2;
        let block_align = num_channels * 2;
        let mut buf = Vec::with_capacity(44 + pcm.len());
        buf.write_all(b"RIFF").unwrap();
        buf.write_all(&(36 + data_size).to_le_bytes()).unwrap();
        buf.write_all(b"WAVEfmt ").unwrap();
        buf.write_all(&16u32.to_le_bytes()).unwrap();
        buf.write_all(&1u16.to_le_bytes()).unwrap();
        buf.write_all(&num_channels.to_le_bytes()).unwrap();
        buf.write_all(&sample_rate.to_le_bytes()).unwrap();
        buf.write_all(&byte_rate.to_le_bytes()).unwrap();
        buf.write_all(&block_align.to_le_bytes()).unwrap();
        buf.write_all(&16u16.to_le_bytes()).unwrap();
        buf.write_all(b"data").unwrap();
        buf.write_all(&data_size.to_le_bytes()).unwrap();
        buf.extend_from_slice(&pcm);
        Bytes::from(buf)
    }

    #[tokio::test]
    async fn input_transport_feeds_audio_from_buffer() {
        let pcm_data = make_pcm_bytes(640); // 40ms at 16kHz = 2 chunks
        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_passthrough: true,
            ..Default::default()
        };

        let mut transport =
            LocalAudioInputTransport::new(params, AudioInputSource::Buffer(pcm_data.clone()));

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_in_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Wait for source task to deliver frames.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }

        // Should have Start + at least one InputAudioRaw
        assert!(
            frames.len() >= 2,
            "got {} frames: {:?}",
            frames.len(),
            frame_names(&frames)
        );
        assert!(matches!(&frames[0].frame, Frame::Start(_)));

        let audio_count = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::InputAudioRaw(_)))
            .count();
        assert!(audio_count >= 1, "expected audio frames, got {audio_count}");
    }

    #[tokio::test]
    async fn output_transport_collects_audio() {
        let output_buf = Arc::new(StdMutex::new(Vec::new()));
        let params = TransportParams {
            audio_out_enabled: true,
            ..Default::default()
        };

        let mut transport =
            LocalAudioOutputTransport::new(params, AudioOutputSink::Buffer(output_buf.clone()));

        let (down_tx, _down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        // Start the transport.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame::default())),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Default chunk size at 24kHz, 1 channel, 4x10ms = 1920 bytes.
        // Send enough audio data to fill at least one chunk.
        let audio_data = make_pcm_bytes(1920); // 3840 bytes, well over one chunk
        transport
            .process_frame(
                FrameEnvelope::new(Frame::OutputAudioRaw(AudioRawFrame {
                    audio: audio_data.clone(),
                    sample_rate: 24000,
                    num_channels: 1,
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Give the audio task time to process.
        tokio::time::sleep(Duration::from_millis(200)).await;

        let collected = output_buf.lock().unwrap();
        assert!(
            !collected.is_empty(),
            "output buffer should have received audio data"
        );
    }

    #[tokio::test]
    async fn input_transport_channel_source() {
        let (audio_tx, audio_rx) = mpsc::channel(16);
        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_passthrough: true,
            ..Default::default()
        };

        let mut transport =
            LocalAudioInputTransport::new(params, AudioInputSource::Channel(audio_rx));

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_in_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Send a frame through the channel.
        audio_tx
            .send(AudioRawFrame {
                audio: make_pcm_bytes(160),
                sample_rate: 16000,
                num_channels: 1,
            })
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }

        let audio_count = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::InputAudioRaw(_)))
            .count();
        assert!(audio_count >= 1, "expected audio from channel source");
    }

    #[tokio::test]
    async fn output_discard_sink() {
        let params = TransportParams {
            audio_out_enabled: true,
            ..Default::default()
        };

        let mut transport = LocalAudioOutputTransport::new(params, AudioOutputSink::Discard);

        let (down_tx, _down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame::default())),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        // Should not panic.
        transport
            .process_frame(
                FrameEnvelope::new(Frame::OutputAudioRaw(AudioRawFrame {
                    audio: make_pcm_bytes(320),
                    sample_rate: 24000,
                    num_channels: 1,
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn wav_input_from_buffer() {
        let samples: Vec<i16> = (0..640).map(|i| (i % 200) as i16).collect();
        let wav_data = make_wav_bytes(&samples, 16000, 1);

        let params = TransportParams {
            audio_in_enabled: true,
            audio_in_passthrough: true,
            ..Default::default()
        };

        let mut transport =
            LocalAudioInputTransport::new(params, AudioInputSource::Buffer(wav_data))
                .with_format(AudioFormat::Encoded);

        let (down_tx, mut down_rx) = mpsc::channel(64);
        let (up_tx, _up_rx) = mpsc::channel(64);
        let ctx =
            ProcessorContext::new(down_tx, up_tx, transport.id(), transport.name().to_string());

        transport
            .process_frame(
                FrameEnvelope::new(Frame::Start(StartFrame {
                    audio_in_sample_rate: 16000,
                    ..Default::default()
                })),
                Direction::Downstream,
                &ctx,
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut frames = Vec::new();
        while let Ok(env) = down_rx.try_recv() {
            frames.push(env);
        }

        let audio_count = frames
            .iter()
            .filter(|f| matches!(&f.frame, Frame::InputAudioRaw(_)))
            .count();
        assert!(
            audio_count >= 1,
            "WAV input should produce audio frames, got {audio_count}"
        );

        // Verify the total PCM data matches the original samples.
        let total_pcm: Vec<u8> = frames
            .iter()
            .filter_map(|f| match &f.frame {
                Frame::InputAudioRaw(a) => Some(a.audio.to_vec()),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(
            total_pcm.len(),
            samples.len() * 2,
            "total PCM bytes should match input sample count * 2"
        );
    }
}
