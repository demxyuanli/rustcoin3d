//! Terminal UI for command entry (ratatui). Runs on a dedicated thread alongside the GPU window.

mod shell;

pub use shell::{run_tui_thread, AppEvent};
