use std::time::Duration;

use pipecat_core::frame::{self, Frame};
use pipecat_core::node::ProcessorNodeHandle;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, trace};

use crate::action::TurnAction;
use crate::params::{UserTurnStartedParams, UserTurnStoppedParams};
use crate::strategies::UserTurnStrategies;

/// Orchestrates user turn detection using pluggable start/stop strategies.
///
/// The controller routes frames to all strategies, manages turn state
/// (start/stop with deduplication), and runs a global inactivity timeout.
/// It returns `Vec<TurnAction>` from `process_frame` — the caller (typically
/// an input transport) is responsible for executing those actions.
///
/// The controller is **not** a `FrameProcessor`. It is owned by the input
/// transport which calls `process_frame` on each incoming frame.
#[derive(Debug)]
pub struct UserTurnController {
    strategies: UserTurnStrategies,
    user_turn_stop_timeout: Duration,

    /// Whether the user is currently speaking (from VAD or external events).
    user_speaking: bool,
    /// Whether a user turn is currently active.
    user_turn: bool,

    // Global stop timeout: a background loop that fires when no activity
    // is detected for `user_turn_stop_timeout` duration.
    timeout_reset_tx: mpsc::UnboundedSender<()>,
    timeout_action_rx: mpsc::UnboundedReceiver<GlobalTimeoutSignal>,
    timeout_task: Option<JoinHandle<()>>,
    node_handle: Option<ProcessorNodeHandle>,
}

#[derive(Debug)]
enum GlobalTimeoutSignal {
    Elapsed,
}

impl UserTurnController {
    /// Create a new controller. Call `setup()` before use.
    pub fn new(strategies: UserTurnStrategies, user_turn_stop_timeout: Duration) -> Self {
        let (timeout_reset_tx, _reset_rx) = mpsc::unbounded_channel();
        let (_action_tx, timeout_action_rx) = mpsc::unbounded_channel();
        Self {
            strategies,
            user_turn_stop_timeout,
            user_speaking: false,
            user_turn: false,
            timeout_reset_tx,
            timeout_action_rx,
            timeout_task: None,
            node_handle: None,
        }
    }

    /// Set the node handle for self-notification. Background timeout tasks
    /// will send Wakeup frames through this handle to trigger processing.
    pub fn set_node_handle(&mut self, handle: ProcessorNodeHandle) {
        self.node_handle = Some(handle.clone());
        for strategy in &mut self.strategies.stop {
            strategy.set_node_handle(handle.clone());
        }
    }

    /// Initialize the controller: start the global timeout task and set up strategies.
    pub async fn setup(&mut self) {
        // Create fresh channels for the timeout loop
        let (timeout_reset_tx, reset_rx) = mpsc::unbounded_channel();
        let (action_tx, timeout_action_rx) = mpsc::unbounded_channel();
        self.timeout_reset_tx = timeout_reset_tx;
        self.timeout_action_rx = timeout_action_rx;

        let timeout = self.user_turn_stop_timeout;
        let node_handle = self.node_handle.clone();
        self.timeout_task = Some(tokio::spawn(async move {
            global_timeout_loop(reset_rx, action_tx, timeout, node_handle).await;
        }));

        for strategy in &mut self.strategies.start {
            strategy.setup().await;
        }
        for strategy in &mut self.strategies.stop {
            strategy.setup().await;
        }

        debug!("UserTurnController setup complete");
    }

    /// Shut down the controller: cancel the timeout task and clean up strategies.
    pub async fn cleanup(&mut self) {
        if let Some(handle) = self.timeout_task.take() {
            handle.abort();
        }

        for strategy in &mut self.strategies.start {
            strategy.cleanup().await;
        }
        for strategy in &mut self.strategies.stop {
            strategy.cleanup().await;
        }

        debug!("UserTurnController cleanup complete");
    }

    /// Replace the current strategies with new ones.
    pub async fn update_strategies(&mut self, mut strategies: UserTurnStrategies) {
        // Cleanup old strategies
        for strategy in &mut self.strategies.start {
            strategy.cleanup().await;
        }
        for strategy in &mut self.strategies.stop {
            strategy.cleanup().await;
        }

        // Setup new strategies
        for strategy in &mut strategies.start {
            strategy.setup().await;
        }
        for strategy in &mut strategies.stop {
            if let Some(ref handle) = self.node_handle {
                strategy.set_node_handle(handle.clone());
            }
            strategy.setup().await;
        }

        self.strategies = strategies;
        debug!("UserTurnController strategies updated");
    }

    /// Whether a user turn is currently active.
    pub fn is_user_turn(&self) -> bool {
        self.user_turn
    }

    /// Whether the user is currently speaking.
    pub fn is_user_speaking(&self) -> bool {
        self.user_speaking
    }

    /// Process a frame through all strategies and return actions for the caller.
    pub async fn process_frame(&mut self, frame: &Frame) -> Vec<TurnAction> {
        // Track speaking state and reset the global timeout on activity
        match frame {
            Frame::UserStartedSpeaking(_) | Frame::VADUserStartedSpeaking(_) => {
                self.user_speaking = true;
                self.reset_timeout();
            }
            Frame::UserStoppedSpeaking(_) | Frame::VADUserStoppedSpeaking(_) => {
                self.user_speaking = false;
                self.reset_timeout();
            }
            Frame::Transcription(_) | Frame::InterimTranscription(_) => {
                self.reset_timeout();
            }
            _ => {}
        }

        // Collect all actions, processing turn start/stop through the controller
        let mut output_actions = Vec::new();

        // First, drain pending async actions from stop strategies and the
        // global timeout. These are from previous frame cycles (timeouts that
        // fired between frames). Processing them first ensures a turn can
        // stop before a new one starts in the same cycle.
        let mut pending_stop_actions = Vec::new();
        for strategy in &mut self.strategies.stop {
            pending_stop_actions.extend(strategy.drain_pending_actions());
        }
        for action in pending_stop_actions {
            match action {
                TurnAction::UserTurnStopped(params) => {
                    output_actions.extend(self.trigger_user_turn_stop(params).await);
                }
                other => output_actions.push(other),
            }
        }
        output_actions.extend(self.drain_global_timeout_actions().await);

        // Run frame through start strategies
        let mut start_actions = Vec::new();
        for strategy in &mut self.strategies.start {
            start_actions.extend(strategy.process_frame(frame).await);
        }
        for action in start_actions {
            match action {
                TurnAction::UserTurnStarted(params) => {
                    output_actions.extend(self.trigger_user_turn_start(params).await);
                }
                other => output_actions.push(other),
            }
        }

        // Run frame through stop strategies (synchronous actions from this frame)
        let mut stop_actions = Vec::new();
        for strategy in &mut self.strategies.stop {
            stop_actions.extend(strategy.process_frame(frame).await);
        }
        for action in stop_actions {
            match action {
                TurnAction::UserTurnStopped(params) => {
                    output_actions.extend(self.trigger_user_turn_stop(params).await);
                }
                other => output_actions.push(other),
            }
        }

        output_actions
    }

    /// Handle a turn start trigger with double-start prevention.
    async fn trigger_user_turn_start(&mut self, params: UserTurnStartedParams) -> Vec<TurnAction> {
        if self.user_turn {
            trace!("UserTurnController: ignoring duplicate turn start");
            return vec![];
        }

        self.user_turn = true;
        self.reset_timeout();

        // Reset all start strategies so they don't re-fire for this same turn.
        for strategy in &mut self.strategies.start {
            strategy.reset().await;
        }

        debug!("User turn started");
        vec![TurnAction::UserTurnStarted(params)]
    }

    /// Handle a turn stop trigger with double-stop prevention.
    async fn trigger_user_turn_stop(&mut self, params: UserTurnStoppedParams) -> Vec<TurnAction> {
        if !self.user_turn {
            trace!("UserTurnController: ignoring duplicate turn stop");
            return vec![];
        }

        self.user_turn = false;
        self.reset_timeout();

        // Reset all stop strategies so they don't re-fire for this same gap.
        for strategy in &mut self.strategies.stop {
            strategy.reset().await;
        }

        debug!("User turn stopped");
        vec![TurnAction::UserTurnStopped(params)]
    }

    /// Send a reset signal to the global timeout loop.
    fn reset_timeout(&self) {
        let _ = self.timeout_reset_tx.send(());
    }

    /// Drain and process global timeout signals.
    async fn drain_global_timeout_actions(&mut self) -> Vec<TurnAction> {
        let mut actions = Vec::new();
        while let Ok(GlobalTimeoutSignal::Elapsed) = self.timeout_action_rx.try_recv() {
            if self.user_turn && !self.user_speaking {
                debug!("User turn stop timeout fired");
                actions.push(TurnAction::UserTurnStopTimeout);

                let stop_params = UserTurnStoppedParams {
                    enable_user_speaking_frames: true,
                };
                actions.extend(self.trigger_user_turn_stop(stop_params).await);
            }
        }
        actions
    }
}

/// Background task that monitors for inactivity.
///
/// Loops waiting for either a reset signal (activity detected) or a timeout.
/// When timeout fires, sends `GlobalTimeoutSignal::Elapsed` on the action
/// channel. The controller checks conditions (turn active, user not speaking)
/// before acting on it.
async fn global_timeout_loop(
    mut reset_rx: mpsc::UnboundedReceiver<()>,
    action_tx: mpsc::UnboundedSender<GlobalTimeoutSignal>,
    timeout: Duration,
    node_handle: Option<ProcessorNodeHandle>,
) {
    loop {
        match tokio::time::timeout(timeout, reset_rx.recv()).await {
            Ok(Some(())) => {
                // Activity received — drain any extra signals and restart the timer
                while reset_rx.try_recv().is_ok() {}
            }
            Ok(None) => {
                // Channel closed — controller was dropped
                break;
            }
            Err(_) => {
                // Timeout elapsed with no activity
                if action_tx.send(GlobalTimeoutSignal::Elapsed).is_err() {
                    break;
                }
                // Wake up the processor node so drain_pending_actions runs
                if let Some(ref handle) = node_handle {
                    handle.try_send_normal(
                        frame::FrameEnvelope::new(frame::Frame::Wakeup(frame::WakeupFrame)),
                        frame::Direction::Downstream,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pipecat_core::frame::{
        InterimTranscriptionFrame, TranscriptionFrame, UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
    };

    use super::*;
    use crate::start::{TranscriptionUserTurnStartStrategy, VadUserTurnStartStrategy};
    use crate::stop::SpeechTimeoutUserTurnStopStrategy;

    fn make_controller(stop_timeout_secs: f64) -> UserTurnController {
        let strategies = UserTurnStrategies {
            start: vec![
                Box::new(VadUserTurnStartStrategy::new()),
                Box::new(TranscriptionUserTurnStartStrategy::new()),
            ],
            stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.1))],
        };
        UserTurnController::new(strategies, Duration::from_secs_f64(stop_timeout_secs))
    }

    fn vad_started() -> Frame {
        Frame::VADUserStartedSpeaking(VADUserStartedSpeakingFrame {
            start_secs: 0.2,
            timestamp: 1.0,
        })
    }

    fn vad_stopped() -> Frame {
        Frame::VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame {
            stop_secs: 0.2,
            timestamp: 2.0,
        })
    }

    fn transcription(text: &str) -> Frame {
        Frame::Transcription(TranscriptionFrame {
            text: text.into(),
            user_id: String::new(),
            timestamp: None,
            language: None,
            finalized: true,
            result: None,
        })
    }

    fn interim(text: &str) -> Frame {
        Frame::InterimTranscription(InterimTranscriptionFrame {
            text: text.into(),
            user_id: String::new(),
            timestamp: None,
            language: None,
            result: None,
        })
    }

    #[tokio::test]
    async fn turn_start_on_vad() {
        let mut ctrl = make_controller(5.0);
        ctrl.setup().await;

        let actions = ctrl.process_frame(&vad_started()).await;
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, TurnAction::UserTurnStarted(_)))
        );
        assert!(ctrl.is_user_turn());
        assert!(ctrl.is_user_speaking());

        ctrl.cleanup().await;
    }

    #[tokio::test]
    async fn double_start_prevention() {
        let mut ctrl = make_controller(5.0);
        ctrl.setup().await;

        let actions1 = ctrl.process_frame(&vad_started()).await;
        let start_count1 = actions1
            .iter()
            .filter(|a| matches!(a, TurnAction::UserTurnStarted(_)))
            .count();
        assert_eq!(start_count1, 1);

        // Second VAD start should not produce another UserTurnStarted
        let actions2 = ctrl.process_frame(&vad_started()).await;
        let start_count2 = actions2
            .iter()
            .filter(|a| matches!(a, TurnAction::UserTurnStarted(_)))
            .count();
        assert_eq!(start_count2, 0);

        ctrl.cleanup().await;
    }

    #[tokio::test]
    async fn double_stop_prevention() {
        let mut ctrl = make_controller(5.0);
        ctrl.setup().await;

        // Start a turn
        ctrl.process_frame(&vad_started()).await;
        ctrl.process_frame(&transcription("hello")).await;
        ctrl.process_frame(&vad_stopped()).await;

        // Wait for speech timeout to fire
        tokio::time::sleep(Duration::from_millis(200)).await;

        let actions = ctrl.process_frame(&vad_started()).await;
        // Should get the stopped action from the previous timeout
        let stop_count = actions
            .iter()
            .filter(|a| matches!(a, TurnAction::UserTurnStopped(_)))
            .count();
        // At most 1 stop
        assert!(stop_count <= 1);

        ctrl.cleanup().await;
    }

    #[tokio::test]
    async fn transcription_also_starts_turn() {
        let mut ctrl = make_controller(5.0);
        ctrl.setup().await;

        let actions = ctrl.process_frame(&transcription("hello")).await;
        // TranscriptionUserTurnStartStrategy should trigger
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, TurnAction::UserTurnStarted(_)))
        );

        ctrl.cleanup().await;
    }

    #[tokio::test]
    async fn global_timeout_fires_when_turn_active_and_not_speaking() {
        let mut ctrl = make_controller(0.2);
        ctrl.setup().await;

        // Start turn
        ctrl.process_frame(&vad_started()).await;
        ctrl.process_frame(&vad_stopped()).await;
        assert!(ctrl.is_user_turn());
        assert!(!ctrl.is_user_speaking());

        // Wait for global timeout
        tokio::time::sleep(Duration::from_millis(400)).await;

        // Process any frame to trigger drain
        let actions = ctrl
            .process_frame(&Frame::UserStoppedSpeaking(UserStoppedSpeakingFrame))
            .await;
        let has_timeout = actions
            .iter()
            .any(|a| matches!(a, TurnAction::UserTurnStopTimeout));
        assert!(has_timeout);

        ctrl.cleanup().await;
    }

    #[tokio::test]
    async fn global_timeout_does_not_fire_when_not_in_turn() {
        let mut ctrl = make_controller(0.1);
        ctrl.setup().await;

        // No turn started — just wait
        tokio::time::sleep(Duration::from_millis(200)).await;

        let actions = ctrl
            .process_frame(&Frame::UserStartedSpeaking(UserStartedSpeakingFrame))
            .await;
        let has_timeout = actions
            .iter()
            .any(|a| matches!(a, TurnAction::UserTurnStopTimeout));
        assert!(!has_timeout);

        ctrl.cleanup().await;
    }

    #[tokio::test]
    async fn global_timeout_resets_on_activity() {
        let mut ctrl = make_controller(0.2);
        ctrl.setup().await;

        // Start turn
        ctrl.process_frame(&vad_started()).await;
        ctrl.process_frame(&vad_stopped()).await;

        // Send activity before timeout
        tokio::time::sleep(Duration::from_millis(100)).await;
        ctrl.process_frame(&interim("hel")).await;

        // Wait past original timeout but not past reset
        tokio::time::sleep(Duration::from_millis(150)).await;

        let actions = ctrl
            .process_frame(&Frame::UserStoppedSpeaking(UserStoppedSpeakingFrame))
            .await;
        let has_timeout = actions
            .iter()
            .any(|a| matches!(a, TurnAction::UserTurnStopTimeout));
        assert!(!has_timeout);

        ctrl.cleanup().await;
    }

    #[tokio::test]
    async fn full_turn_lifecycle() {
        let mut ctrl = make_controller(5.0);
        ctrl.setup().await;

        // 1. VAD start → turn starts
        let actions = ctrl.process_frame(&vad_started()).await;
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, TurnAction::UserTurnStarted(_)))
        );
        assert!(ctrl.is_user_turn());

        // 2. Transcription arrives
        ctrl.process_frame(&transcription("hello world")).await;

        // 3. VAD stop → speech timeout starts
        ctrl.process_frame(&vad_stopped()).await;

        // 4. Wait for speech timeout → turn stops
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Trigger drain with a neutral frame
        let actions = ctrl.process_frame(&vad_started()).await;
        let has_stop = actions
            .iter()
            .any(|a| matches!(a, TurnAction::UserTurnStopped(_)));
        assert!(has_stop);

        ctrl.cleanup().await;
    }

    #[tokio::test]
    async fn update_strategies() {
        let mut ctrl = make_controller(5.0);
        ctrl.setup().await;

        let new_strategies = UserTurnStrategies {
            start: vec![Box::new(VadUserTurnStartStrategy::new())],
            stop: vec![Box::new(SpeechTimeoutUserTurnStopStrategy::new(0.5))],
        };
        ctrl.update_strategies(new_strategies).await;

        // Should still work with new strategies
        let actions = ctrl.process_frame(&vad_started()).await;
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, TurnAction::UserTurnStarted(_)))
        );

        ctrl.cleanup().await;
    }
}
