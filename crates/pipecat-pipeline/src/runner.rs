use pipecat_core::error::{PipecatError, Result};
use pipecat_core::frame::{CancelFrame, Frame};
use tracing::{debug, warn};

use crate::task::PipelineTask;

// ---------------------------------------------------------------------------
// PipelineRunner
// ---------------------------------------------------------------------------

/// Top-level entry point for running a pipeline task with signal handling.
///
/// Handles SIGINT (Ctrl-C) and optionally SIGTERM for graceful cancellation.
/// When a signal fires, the runner sends a CancelFrame into the running task
/// and waits for it to shut down cleanly.
///
/// # Usage
///
/// ```ignore
/// let mut runner = PipelineRunner::new();
/// runner.run(task).await?;
/// ```
pub struct PipelineRunner {
    handle_sigint: bool,
    #[cfg(unix)]
    handle_sigterm: bool,
}

impl std::fmt::Debug for PipelineRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineRunner")
            .field("handle_sigint", &self.handle_sigint)
            .finish()
    }
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
            #[cfg(unix)]
            handle_sigterm: false,
        }
    }

    /// Create a runner with explicit SIGINT handling configuration.
    pub fn with_signals(handle_sigint: bool) -> Self {
        Self {
            handle_sigint,
            #[cfg(unix)]
            handle_sigterm: false,
        }
    }

    /// Enable or disable SIGTERM handling (Unix only).
    #[cfg(unix)]
    pub fn with_sigterm(mut self, handle_sigterm: bool) -> Self {
        self.handle_sigterm = handle_sigterm;
        self
    }

    /// Run a pipeline task to completion.
    ///
    /// If signal handling is enabled, the corresponding signals will trigger
    /// graceful task cancellation instead of immediate process termination.
    pub async fn run(&mut self, mut task: PipelineTask) -> Result<()> {
        debug!("PipelineRunner: starting");

        let needs_signals = self.needs_signal_handling();

        if !needs_signals {
            return task.run().await;
        }

        // Clone the push sender so we can inject CancelFrame after task is spawned.
        let cancel_tx = task.push_tx.clone();

        let mut join = tokio::spawn(async move { task.run().await });

        tokio::select! {
            result = &mut join => {
                debug!("PipelineRunner: task completed normally");
                match result {
                    Ok(r) => r,
                    Err(e) => Err(PipecatError::PipelineError(format!("task panicked: {e}"))),
                }
            }
            signal_name = Self::wait_for_signal(self.handle_sigint, self.wants_sigterm()) => {
                warn!("PipelineRunner: {signal_name} received, cancelling");
                cancel_tx.send(Frame::Cancel(CancelFrame::default())).await.ok();
                match join.await {
                    Ok(r) => r,
                    Err(e) => Err(PipecatError::PipelineError(format!("task panicked: {e}"))),
                }
            }
        }
    }

    fn needs_signal_handling(&self) -> bool {
        self.handle_sigint || self.wants_sigterm()
    }

    #[cfg(unix)]
    fn wants_sigterm(&self) -> bool {
        self.handle_sigterm
    }

    #[cfg(not(unix))]
    fn wants_sigterm(&self) -> bool {
        false
    }

    /// Wait for a termination signal. Returns the signal name.
    ///
    /// At least one of the signal flags must be true, otherwise this will
    /// panic (no arms in the select). Callers must check `needs_signal_handling()`.
    async fn wait_for_signal(handle_sigint: bool, _handle_sigterm: bool) -> &'static str {
        debug_assert!(
            handle_sigint || _handle_sigterm,
            "wait_for_signal called with no signals enabled"
        );
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};

            let mut sigterm_stream = if _handle_sigterm {
                Some(signal(SignalKind::terminate()).expect("failed to install SIGTERM handler"))
            } else {
                None
            };

            tokio::select! {
                r = tokio::signal::ctrl_c(), if handle_sigint => {
                    let _ = r;
                    "SIGINT"
                }
                _ = async {
                    match sigterm_stream {
                        Some(ref mut s) => { s.recv().await; }
                        None => std::future::pending::<()>().await,
                    }
                }, if _handle_sigterm => {
                    "SIGTERM"
                }
            }
        }

        #[cfg(not(unix))]
        {
            if handle_sigint {
                tokio::signal::ctrl_c().await.ok();
                "SIGINT"
            } else {
                std::future::pending::<&str>().await
            }
        }
    }
}
