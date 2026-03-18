//! Frame filter processors for controlling which frames pass through a pipeline.
//!
//! Filters are processors that selectively allow or block frames based on
//! criteria. System and lifecycle frames are always passed through by default
//! to maintain pipeline integrity.
//!
//! # Filters
//!
//! - [`IdentityFilter`] — forwards all frames unchanged (useful for testing)
//! - [`NullFilter`] — blocks everything except system + End frames
//! - [`FrameFilter`] — allows frames matching a predicate; system + End always pass
//! - [`FunctionFilter`] — like `FrameFilter` with direction-awareness and
//!   configurable system frame handling

use async_trait::async_trait;

use crate::error::Result;
use crate::frame::*;
use crate::processor::{FrameProcessor, ProcessorBase, ProcessorContext};

// ---------------------------------------------------------------------------
// IdentityFilter
// ---------------------------------------------------------------------------

/// Forwards every frame unchanged. Useful for testing `ParallelPipeline` to
/// ensure frames pass through without duplication.
#[derive(Debug)]
pub struct IdentityFilter {
    base: ProcessorBase,
}

impl IdentityFilter {
    pub fn new() -> Self {
        Self {
            base: ProcessorBase::new("IdentityFilter"),
        }
    }
}

impl Default for IdentityFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FrameProcessor for IdentityFilter {
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
        ctx.push_frame(envelope, direction).await
    }
}

// ---------------------------------------------------------------------------
// NullFilter
// ---------------------------------------------------------------------------

/// Blocks all frames except system frames and `End`. Useful for temporarily
/// stopping frame flow while keeping the pipeline alive.
#[derive(Debug)]
pub struct NullFilter {
    base: ProcessorBase,
}

impl NullFilter {
    pub fn new() -> Self {
        Self {
            base: ProcessorBase::new("NullFilter"),
        }
    }
}

impl Default for NullFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FrameProcessor for NullFilter {
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
        if envelope.frame.is_system() || matches!(envelope.frame, Frame::End(_)) {
            ctx.push_frame(envelope, direction).await?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FrameFilter
// ---------------------------------------------------------------------------

/// Selectively passes frames that match a predicate. System frames and `End`
/// always pass through regardless of the predicate.
///
/// # Example
///
/// ```ignore
/// // Only allow text and transcription frames through
/// let filter = FrameFilter::new(|f| matches!(f, Frame::Text(_) | Frame::Transcription(_)));
/// ```
pub struct FrameFilter {
    base: ProcessorBase,
    filter: Box<dyn Fn(&Frame) -> bool + Send + Sync>,
}

impl std::fmt::Debug for FrameFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameFilter")
            .field("base", &self.base)
            .finish_non_exhaustive()
    }
}

impl FrameFilter {
    pub fn new(filter: impl Fn(&Frame) -> bool + Send + Sync + 'static) -> Self {
        Self {
            base: ProcessorBase::new("FrameFilter"),
            filter: Box::new(filter),
        }
    }
}

#[async_trait]
impl FrameProcessor for FrameFilter {
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
        let pass = envelope.frame.is_system()
            || matches!(envelope.frame, Frame::End(_))
            || (self.filter)(&envelope.frame);
        if pass {
            ctx.push_frame(envelope, direction).await?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FunctionFilter
// ---------------------------------------------------------------------------

/// Filters frames using a predicate with direction-awareness and configurable
/// system frame handling.
///
/// By default:
/// - Lifecycle frames (`Start`, `End`, `Cancel`) always pass through.
/// - System frames pass through (unless `filter_system_frames(true)` is set).
/// - Frames traveling in a direction other than the configured one pass through.
///
/// # Example
///
/// ```ignore
/// // Only filter downstream text frames; upstream frames pass untouched
/// let filter = FunctionFilter::new(|f| matches!(f, Frame::Text(_)))
///     .with_direction(Direction::Downstream);
/// ```
pub struct FunctionFilter {
    base: ProcessorBase,
    filter: Box<dyn Fn(&Frame) -> bool + Send + Sync>,
    direction: Option<Direction>,
    filter_system: bool,
}

impl std::fmt::Debug for FunctionFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionFilter")
            .field("base", &self.base)
            .field("direction", &self.direction)
            .field("filter_system", &self.filter_system)
            .finish_non_exhaustive()
    }
}

impl FunctionFilter {
    pub fn new(filter: impl Fn(&Frame) -> bool + Send + Sync + 'static) -> Self {
        Self {
            base: ProcessorBase::new("FunctionFilter"),
            filter: Box::new(filter),
            direction: None,
            filter_system: false,
        }
    }

    /// Only filter frames traveling in the given direction. Frames in the other
    /// direction pass through unfiltered.
    pub fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = Some(direction);
        self
    }

    /// When `true`, system frames are also evaluated by the filter predicate.
    /// By default system frames always pass through.
    pub fn filter_system_frames(mut self, filter: bool) -> Self {
        self.filter_system = filter;
        self
    }
}

#[async_trait]
impl FrameProcessor for FunctionFilter {
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
        // Lifecycle frames always pass.
        if envelope.frame.is_lifecycle() {
            return ctx.push_frame(envelope, direction).await;
        }

        // If direction is configured and frame is going the other way, pass through.
        if let Some(filter_dir) = self.direction
            && direction != filter_dir
        {
            return ctx.push_frame(envelope, direction).await;
        }

        // System frames pass by default unless filter_system is set.
        if !self.filter_system && envelope.frame.is_system() {
            return ctx.push_frame(envelope, direction).await;
        }

        // Apply the filter predicate.
        if (self.filter)(&envelope.frame) {
            ctx.push_frame(envelope, direction).await?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::run_processor;

    #[tokio::test]
    async fn identity_passes_all_frames() {
        let mut filter = IdentityFilter::new();
        let (down, _up) = run_processor(
            &mut filter,
            vec![
                (Frame::Start(StartFrame::default()), Direction::Downstream),
                (Frame::Text(TextFrame::new("hello")), Direction::Downstream),
                (Frame::End(EndFrame::default()), Direction::Downstream),
            ],
        )
        .await;

        let names: Vec<_> = down.iter().map(|e| format!("{}", e.frame)).collect();
        assert_eq!(names, vec!["Start", "Text", "End"]);
    }

    #[tokio::test]
    async fn identity_passes_upstream() {
        let mut filter = IdentityFilter::new();
        let (_down, up) = run_processor(
            &mut filter,
            vec![(Frame::Text(TextFrame::new("hello")), Direction::Upstream)],
        )
        .await;

        assert_eq!(up.len(), 1);
        assert!(matches!(&up[0].frame, Frame::Text(_)));
    }

    #[tokio::test]
    async fn null_passes_system_and_end() {
        let mut filter = NullFilter::new();
        let (down, _up) = run_processor(
            &mut filter,
            vec![
                (Frame::Start(StartFrame::default()), Direction::Downstream),
                (
                    Frame::Text(TextFrame::new("blocked")),
                    Direction::Downstream,
                ),
                (
                    Frame::Interruption(InterruptionFrame),
                    Direction::Downstream,
                ),
                (Frame::Stop(StopFrame), Direction::Downstream),
                (Frame::End(EndFrame::default()), Direction::Downstream),
            ],
        )
        .await;

        let names: Vec<_> = down.iter().map(|e| format!("{}", e.frame)).collect();
        assert_eq!(names, vec!["Start", "Interruption", "End"]);
    }

    #[tokio::test]
    async fn frame_filter_allows_matching_and_system() {
        let mut filter = FrameFilter::new(|f| matches!(f, Frame::Text(_)));
        let (down, _up) = run_processor(
            &mut filter,
            vec![
                (Frame::Start(StartFrame::default()), Direction::Downstream),
                (
                    Frame::Text(TextFrame::new("allowed")),
                    Direction::Downstream,
                ),
                (Frame::Stop(StopFrame), Direction::Downstream),
                (Frame::End(EndFrame::default()), Direction::Downstream),
            ],
        )
        .await;

        let names: Vec<_> = down.iter().map(|e| format!("{}", e.frame)).collect();
        assert_eq!(names, vec!["Start", "Text", "End"]);
    }

    #[tokio::test]
    async fn function_filter_direction_downstream_only() {
        let mut filter = FunctionFilter::new(|f| matches!(f, Frame::Text(_)))
            .with_direction(Direction::Downstream);

        let (down, up) = run_processor(
            &mut filter,
            vec![
                // Downstream text: filtered (allowed by predicate)
                (Frame::Text(TextFrame::new("down")), Direction::Downstream),
                // Downstream Stop: filtered (blocked by predicate)
                (Frame::Stop(StopFrame), Direction::Downstream),
                // Upstream Stop: not filtered (wrong direction), passes through
                (Frame::Stop(StopFrame), Direction::Upstream),
            ],
        )
        .await;

        let down_names: Vec<_> = down.iter().map(|e| format!("{}", e.frame)).collect();
        assert_eq!(down_names, vec!["Text"]);
        assert_eq!(up.len(), 1);
        assert!(matches!(&up[0].frame, Frame::Stop(_)));
    }

    #[tokio::test]
    async fn function_filter_direction_upstream_only() {
        let mut filter = FunctionFilter::new(|f| matches!(f, Frame::Text(_)))
            .with_direction(Direction::Upstream);

        let (down, up) = run_processor(
            &mut filter,
            vec![
                // Downstream Stop: not filtered (wrong direction), passes through
                (Frame::Stop(StopFrame), Direction::Downstream),
                // Upstream text: filtered, allowed by predicate
                (Frame::Text(TextFrame::new("up")), Direction::Upstream),
                // Upstream Stop: filtered, blocked by predicate
                (Frame::Stop(StopFrame), Direction::Upstream),
            ],
        )
        .await;

        assert_eq!(down.len(), 1);
        assert!(matches!(&down[0].frame, Frame::Stop(_)));
        assert_eq!(up.len(), 1);
        assert!(matches!(&up[0].frame, Frame::Text(_)));
    }

    #[tokio::test]
    async fn function_filter_no_direction_filters_both() {
        // No with_direction — should filter in both directions
        let mut filter = FunctionFilter::new(|f| matches!(f, Frame::Text(_)));

        let (down, up) = run_processor(
            &mut filter,
            vec![
                (Frame::Text(TextFrame::new("down")), Direction::Downstream),
                (Frame::Stop(StopFrame), Direction::Downstream),
                (Frame::Text(TextFrame::new("up")), Direction::Upstream),
                (Frame::Stop(StopFrame), Direction::Upstream),
            ],
        )
        .await;

        let down_names: Vec<_> = down.iter().map(|e| format!("{}", e.frame)).collect();
        assert_eq!(down_names, vec!["Text"]);
        let up_names: Vec<_> = up.iter().map(|e| format!("{}", e.frame)).collect();
        assert_eq!(up_names, vec!["Text"]);
    }

    #[tokio::test]
    async fn function_filter_lifecycle_always_passes() {
        // A filter that blocks everything
        let mut filter = FunctionFilter::new(|_| false).filter_system_frames(true);

        let (down, _up) = run_processor(
            &mut filter,
            vec![
                (Frame::Start(StartFrame::default()), Direction::Downstream),
                (
                    Frame::Text(TextFrame::new("blocked")),
                    Direction::Downstream,
                ),
                (Frame::Cancel(CancelFrame::default()), Direction::Downstream),
                (Frame::End(EndFrame::default()), Direction::Downstream),
            ],
        )
        .await;

        let names: Vec<_> = down.iter().map(|e| format!("{}", e.frame)).collect();
        assert_eq!(names, vec!["Start", "Cancel", "End"]);
    }

    #[tokio::test]
    async fn function_filter_system_frames_pass_by_default() {
        // Blocks everything via predicate, but system frames should still pass
        let mut filter = FunctionFilter::new(|_| false);

        let (down, _up) = run_processor(
            &mut filter,
            vec![
                (
                    Frame::Interruption(InterruptionFrame),
                    Direction::Downstream,
                ),
                (
                    Frame::Text(TextFrame::new("blocked")),
                    Direction::Downstream,
                ),
            ],
        )
        .await;

        let names: Vec<_> = down.iter().map(|e| format!("{}", e.frame)).collect();
        assert_eq!(names, vec!["Interruption"]);
    }

    #[tokio::test]
    async fn function_filter_system_frames_filtered_when_enabled() {
        // filter_system_frames(true) means system frames ARE evaluated by predicate
        let mut filter =
            FunctionFilter::new(|f| matches!(f, Frame::Interruption(_))).filter_system_frames(true);

        let (down, _up) = run_processor(
            &mut filter,
            vec![
                // Interruption: system frame, passes predicate
                (
                    Frame::Interruption(InterruptionFrame),
                    Direction::Downstream,
                ),
                // BotStartedSpeaking: system frame, blocked by predicate
                (
                    Frame::BotStartedSpeaking(BotStartedSpeakingFrame),
                    Direction::Downstream,
                ),
                // Start: lifecycle, always passes
                (Frame::Start(StartFrame::default()), Direction::Downstream),
            ],
        )
        .await;

        let names: Vec<_> = down.iter().map(|e| format!("{}", e.frame)).collect();
        assert_eq!(names, vec!["Interruption", "Start"]);
    }
}
