use super::command::CliCommand;
use super::parse;
use crate::engine::execute;
use crate::engine::state::{EngineEvent, EngineState};

pub enum CliSubmitResult {
    Continue { lines: Vec<String> },
    Quit { lines: Vec<String> },
}

pub fn submit(trimmed_line: &str, state: &mut EngineState) -> CliSubmitResult {
    let mut lines: Vec<String> = Vec::new();
    if trimmed_line.is_empty() {
        return CliSubmitResult::Continue { lines };
    }

    match parse(trimmed_line) {
        Ok(cmd) => {
            if matches!(&cmd, CliCommand::Quit) {
                execute(cmd, state);
                lines.extend(format_taken_events(state));
                return CliSubmitResult::Quit { lines };
            }
            execute(cmd, state);
            lines.extend(format_taken_events(state));
            CliSubmitResult::Continue { lines }
        }
        Err(e) => {
            lines.push(format!("Error: {e}"));
            CliSubmitResult::Continue { lines }
        }
    }
}

fn format_taken_events(state: &mut EngineState) -> Vec<String> {
    let events: Vec<_> = state.take_events();
    events
        .into_iter()
        .map(|event| format_one_event(event, state))
        .collect()
}

fn format_one_event(event: EngineEvent, state: &EngineState) -> String {
    match event {
        EngineEvent::SceneLoaded => "Scene loaded.".into(),
        EngineEvent::SceneReset => "Scene reset.".into(),
        EngineEvent::TestStarted(name) => format!("Test started: {name}"),
        EngineEvent::TestPassed(name) => format!("PASS: {name}"),
        EngineEvent::TestFailed(name) => format!("FAIL: {name}"),
        EngineEvent::TestStopped => "Test stopped.".into(),
        EngineEvent::SelectionChanged => format!(
            "Selection: {:?}",
            state.selection.iter().collect::<Vec<_>>()
        ),
        EngineEvent::PropertyChanged { node, field } => {
            format!("Property changed: {node:?}.{field}")
        }
        EngineEvent::LogAppended => state
            .log_entries
            .back()
            .map(|e| format!("[{}] {}", e.level, e.message))
            .unwrap_or_default(),
        EngineEvent::LogCleared => "Log cleared.".into(),
        EngineEvent::DisplayModeChanged(mode) => format!("Display mode: {mode}"),
        EngineEvent::Quit => "Quitting...".into(),
    }
}
