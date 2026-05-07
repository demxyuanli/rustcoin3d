use std::sync::RwLock;

use crate::engine::state::EngineState;

/// Shared engine state between the GPU window thread and the terminal CLI thread.
pub struct EditorSession {
    pub state: RwLock<EngineState>,
}

impl EditorSession {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(EngineState::new()),
        }
    }

    pub fn load_initial_demo(&self) {
        let mut guard = self.state.write().expect("session lock poisoned");
        crate::engine::execute(crate::cli::CliCommand::SceneLoad("demo".into()), &mut guard);
    }
}
