use std::collections::{HashSet, VecDeque};
use rc3d_core::NodeId;
use rc3d_scene::SceneGraph;

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct TestRun {
    pub name: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub running: bool,
}

#[derive(Debug, Clone)]
pub enum EngineEvent {
    SceneLoaded,
    SceneReset,
    TestStarted(String),
    TestPassed(String),
    TestFailed(String),
    TestStopped,
    SelectionChanged,
    PropertyChanged { node: NodeId, field: String },
    LogAppended,
    LogCleared,
    DisplayModeChanged(String),
    Quit,
}

pub struct EngineState {
    pub scene: SceneGraph,
    pub selection: HashSet<NodeId>,
    pub active_test: Option<TestRun>,
    pub log_entries: VecDeque<LogEntry>,
    pub display_mode: String,
    pub events: VecDeque<EngineEvent>,
}

impl EngineState {
    pub fn new() -> Self {
        Self {
            scene: SceneGraph::new(),
            selection: HashSet::new(),
            active_test: None,
            log_entries: VecDeque::with_capacity(256),
            display_mode: "shaded".into(),
            events: VecDeque::new(),
        }
    }

    pub fn push_event(&mut self, event: EngineEvent) {
        self.events.push_back(event);
    }

    pub fn take_events(&mut self) -> Vec<EngineEvent> {
        std::mem::take(&mut self.events).into_iter().collect()
    }

    pub fn add_log(&mut self, level: &str, message: &str) {
        if self.log_entries.len() >= 256 {
            self.log_entries.pop_front();
        }
        self.log_entries.push_back(LogEntry {
            level: level.to_string(),
            message: message.to_string(),
        });
        self.push_event(EngineEvent::LogAppended);
    }
}
