use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

/// A reversible operation on the scene graph.
pub trait Command: std::fmt::Debug + Send + Sync {
    fn execute(&mut self, graph: &mut SceneGraph);
    fn undo(&mut self, graph: &mut SceneGraph);
    fn description(&self) -> &str;
}

/// Bounded undo/redo history.
pub struct CommandHistory {
    undo_stack: Vec<Box<dyn Command>>,
    redo_stack: Vec<Box<dyn Command>>,
    max_depth: usize,
}

impl CommandHistory {
    pub fn new(max_depth: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_depth: max_depth.clamp(1, 1024),
        }
    }

    pub fn execute(&mut self, mut cmd: Box<dyn Command>, graph: &mut SceneGraph) {
        cmd.execute(graph);
        self.redo_stack.clear();
        self.undo_stack.push(cmd);
        if self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    pub fn undo(&mut self, graph: &mut SceneGraph) -> bool {
        let mut cmd = match self.undo_stack.pop() {
            Some(c) => c,
            None => return false,
        };
        cmd.undo(graph);
        self.redo_stack.push(cmd);
        true
    }

    pub fn redo(&mut self, graph: &mut SceneGraph) -> bool {
        let mut cmd = match self.redo_stack.pop() {
            Some(c) => c,
            None => return false,
        };
        cmd.execute(graph);
        self.undo_stack.push(cmd);
        true
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}

/// Command: change a Transform node's translation.
#[derive(Debug)]
pub struct SetTranslationCommand {
    pub node: NodeId,
    pub old_value: Vec3,
    pub new_value: Vec3,
}

impl Command for SetTranslationCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        if let Some(e) = graph.get_mut(self.node) {
            if let NodeData::Transform(t) = &mut e.data {
                t.translation = self.new_value;
            }
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if let Some(e) = graph.get_mut(self.node) {
            if let NodeData::Transform(t) = &mut e.data {
                t.translation = self.old_value;
            }
        }
    }
    fn description(&self) -> &str {
        "SetTranslation"
    }
}

/// Command: change a Transform node's rotation.
#[derive(Debug)]
pub struct SetRotationCommand {
    pub node: NodeId,
    pub old_value: Mat4,
    pub new_value: Mat4,
}

impl Command for SetRotationCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        if let Some(e) = graph.get_mut(self.node) {
            if let NodeData::Transform(t) = &mut e.data {
                t.rotation = self.new_value;
            }
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if let Some(e) = graph.get_mut(self.node) {
            if let NodeData::Transform(t) = &mut e.data {
                t.rotation = self.old_value;
            }
        }
    }
    fn description(&self) -> &str {
        "SetRotation"
    }
}

/// Command: change a Transform node's scale.
#[derive(Debug)]
pub struct SetScaleCommand {
    pub node: NodeId,
    pub old_value: Vec3,
    pub new_value: Vec3,
}

impl Command for SetScaleCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        if let Some(e) = graph.get_mut(self.node) {
            if let NodeData::Transform(t) = &mut e.data {
                t.scale = self.new_value;
            }
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if let Some(e) = graph.get_mut(self.node) {
            if let NodeData::Transform(t) = &mut e.data {
                t.scale = self.old_value;
            }
        }
    }
    fn description(&self) -> &str {
        "SetScale"
    }
}
