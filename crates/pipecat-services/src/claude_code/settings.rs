use std::path::PathBuf;
use std::time::Duration;

use crate::settings::LLMSettings;

/// Effort level for Claude Code CLI (`--effort`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    Low,
    Medium,
    High,
    Max,
}

impl Effort {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

/// Permission mode for Claude Code CLI (`--permission-mode`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionMode {
    Default,
    AcceptEdits,
    BypassPermissions,
    DontAsk,
    Plan,
}

impl PermissionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AcceptEdits => "acceptEdits",
            Self::BypassPermissions => "bypassPermissions",
            Self::DontAsk => "dontAsk",
            Self::Plan => "plan",
        }
    }
}

/// Settings for the Claude Code LLM service.
///
/// Controls the CLI flags passed to `claude -p`. Fields map directly to
/// CLI arguments. `None` values use the CLI's own defaults.
#[derive(Debug, Clone)]
pub struct ClaudeCodeSettings {
    /// Base LLM settings. `model` maps to `--model` (accepts "sonnet", "opus",
    /// "haiku", or full model IDs). `system_instruction` maps to `--system-prompt`.
    pub base: LLMSettings,

    /// Effort level (`--effort`).
    pub effort: Option<Effort>,

    /// Restrict available tools (`--tools`). An empty vec disables all tools.
    pub tools: Option<Vec<String>>,

    /// Auto-approve specific tools (`--allowedTools`).
    pub allowed_tools: Option<Vec<String>>,

    /// Block specific tools (`--disallowedTools`).
    pub disallowed_tools: Option<Vec<String>>,

    /// Append to system prompt (`--append-system-prompt`), in addition to
    /// `base.system_instruction` which sets `--system-prompt`.
    pub append_system_prompt: Option<String>,

    /// Maximum agentic turns (`--max-turns`).
    pub max_turns: Option<u32>,

    /// Permission mode (`--permission-mode`).
    pub permission_mode: Option<PermissionMode>,

    /// Skip all permission checks (`--dangerously-skip-permissions`).
    pub dangerously_skip_permissions: bool,

    /// Path to MCP config file (`--mcp-config`).
    pub mcp_config: Option<PathBuf>,

    /// JSON Schema for structured output (`--json-schema`).
    pub json_schema: Option<serde_json::Value>,

    /// Path to the `claude` CLI binary. Defaults to `"claude"` (found via PATH).
    pub claude_path: PathBuf,

    /// Extra CLI arguments appended after all other flags.
    pub extra_args: Vec<String>,

    /// Use `--resume` for conversation continuity across calls. Default: `true`.
    pub enable_session_resumption: bool,

    /// Don't save sessions to disk (`--no-session-persistence`). Default: `false`.
    /// Useful for headless pipeline usage to avoid accumulating session data.
    pub no_session_persistence: bool,

    /// Overall timeout for each CLI invocation. Default: `None` (no timeout).
    /// If the subprocess doesn't complete within this duration, it is abandoned
    /// and an error is returned.
    pub timeout: Option<Duration>,
}

impl ClaudeCodeSettings {
    pub fn new(base: LLMSettings) -> Self {
        Self {
            base,
            effort: None,
            tools: None,
            allowed_tools: None,
            disallowed_tools: None,
            append_system_prompt: None,
            max_turns: None,
            permission_mode: None,
            dangerously_skip_permissions: false,
            mcp_config: None,
            json_schema: None,
            claude_path: PathBuf::from("claude"),
            extra_args: Vec::new(),
            enable_session_resumption: true,
            no_session_persistence: false,
            timeout: None,
        }
    }
}

impl Default for ClaudeCodeSettings {
    fn default() -> Self {
        Self::new(LLMSettings::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings() {
        let settings = ClaudeCodeSettings::default();
        assert_eq!(settings.claude_path, PathBuf::from("claude"));
        assert!(settings.enable_session_resumption);
        assert!(!settings.dangerously_skip_permissions);
        assert!(settings.effort.is_none());
        assert!(settings.tools.is_none());
        assert!(settings.max_turns.is_none());
    }

    #[test]
    fn effort_as_str() {
        assert_eq!(Effort::Low.as_str(), "low");
        assert_eq!(Effort::Medium.as_str(), "medium");
        assert_eq!(Effort::High.as_str(), "high");
        assert_eq!(Effort::Max.as_str(), "max");
    }

    #[test]
    fn permission_mode_as_str() {
        assert_eq!(PermissionMode::Default.as_str(), "default");
        assert_eq!(PermissionMode::AcceptEdits.as_str(), "acceptEdits");
        assert_eq!(
            PermissionMode::BypassPermissions.as_str(),
            "bypassPermissions"
        );
        assert_eq!(PermissionMode::DontAsk.as_str(), "dontAsk");
        assert_eq!(PermissionMode::Plan.as_str(), "plan");
    }
}
