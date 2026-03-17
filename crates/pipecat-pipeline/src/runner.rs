use tracing::{debug, warn};

use pipecat_core::error::Result;

use crate::task::PipelineTask;

// ---------------------------------------------------------------------------
// PipelineRunner
// ---------------------------------------------------------------------------

/// Top-level entry point for running a pipeline task with signal handling.
///
/// Handles SIGINT (Ctrl-C) for graceful cancellation. When the signal fires,
/// the runner cancels the task and waits for it to finish.
///
/// # Usage
///
/// ```ignore
/// let mut runner = PipelineRunner::new();
/// runner.run(task).await?;
/// ```
#[derive(Debug)]
pub struct PipelineRunner {
    handle_sigint: bool,
}

impl Default for PipelineRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineRunner {
    pub fn new() -> Self {
        Self {
            handle_sigint: true,
        }
    }

    /// Create a runner with explicit signal handling configuration.
    pub fn with_signals(handle_sigint: bool) -> Self {
        Self { handle_sigint }
    }

    /// Run a pipeline task to completion.
    ///
    /// If `handle_sigint` is true, Ctrl-C will trigger task cancellation
    /// instead of immediate process termination.
    pub async fn run(&mut self, mut task: PipelineTask) -> Result<()> {
        debug!("PipelineRunner: starting");

        if self.handle_sigint {
            tokio::select! {
                result = task.run() => {
                    debug!("PipelineRunner: task completed normally");
                    result
                }
                _ = tokio::signal::ctrl_c() => {
                    warn!("PipelineRunner: SIGINT received, cancelling");
                    task.cancel().await;
                    // task.run() in the other select arm will observe the
                    // CancelFrame and shut down. The select drops that future
                    // when this arm wins, so we just return Ok.
                    Ok(())
                }
            }
        } else {
            task.run().await
        }
    }
}
