use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, NodeEntry, SceneGraph};

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

/// Generic field mutation command — covers material, light, camera, and section-plane properties.
pub struct SetFieldCommand<T: Clone + std::fmt::Debug + Send + Sync + 'static> {
    pub node: NodeId,
    pub old_value: T,
    pub new_value: T,
    apply: Box<dyn Fn(&mut NodeEntry, T) + Send + Sync>,
    desc: String,
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> std::fmt::Debug for SetFieldCommand<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SetFieldCommand")
            .field("node", &self.node)
            .field("desc", &self.desc)
            .finish()
    }
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> SetFieldCommand<T> {
    pub fn new(
        node: NodeId,
        old_value: T,
        new_value: T,
        desc: impl Into<String>,
        apply: impl Fn(&mut NodeEntry, T) + Send + Sync + 'static,
    ) -> Self {
        Self {
            node,
            old_value,
            new_value,
            apply: Box::new(apply),
            desc: desc.into(),
        }
    }
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> Command for SetFieldCommand<T> {
    fn execute(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.node) {
            (self.apply)(entry, self.new_value.clone());
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.node) {
            (self.apply)(entry, self.old_value.clone());
        }
    }
    fn description(&self) -> &str {
        &self.desc
    }
}

/// Command: add a child node to a parent.
#[derive(Debug)]
pub struct AddChildCommand {
    pub parent: NodeId,
    pub child: NodeId,
}

impl AddChildCommand {
    pub fn new(parent: NodeId, child: NodeId) -> Self {
        Self { parent, child }
    }
}

impl Command for AddChildCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.parent) {
            if !entry.children.contains(&self.child) {
                entry.children.push(self.child);
            }
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.parent) {
            entry.children.retain(|c| *c != self.child);
        }
    }
    fn description(&self) -> &str {
        "AddChild"
    }
}

/// Command: remove a child node from its parent (node stays in graph as orphan).
/// Undo simply re-attaches the child to the parent at the original index.
#[derive(Debug)]
pub struct RemoveChildCommand {
    pub parent: NodeId,
    pub child: NodeId,
    child_index: usize,
}

impl RemoveChildCommand {
    pub fn new(parent: NodeId, child: NodeId, graph: &SceneGraph) -> Self {
        let child_index = graph
            .get(parent)
            .map(|e| e.children.iter().position(|c| *c == child).unwrap_or(0))
            .unwrap_or(0);
        Self { parent, child, child_index }
    }
}

impl Command for RemoveChildCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        self.child_index = graph
            .get(self.parent)
            .map(|e| e.children.iter().position(|c| *c == self.child).unwrap_or(0))
            .unwrap_or(0);
        if let Some(entry) = graph.get_mut(self.parent) {
            entry.children.retain(|c| *c != self.child);
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if let Some(entry) = graph.get_mut(self.parent) {
            let idx = self.child_index.min(entry.children.len());
            if !entry.children.contains(&self.child) {
                entry.children.insert(idx, self.child);
            }
        }
    }
    fn description(&self) -> &str {
        "RemoveChild"
    }
}

/// Compound command: bundles multiple commands into one atomic transaction.
#[derive(Debug)]
pub struct CompoundCommand {
    pub commands: Vec<Box<dyn Command>>,
    desc: String,
}

impl CompoundCommand {
    pub fn new(commands: Vec<Box<dyn Command>>, desc: impl Into<String>) -> Self {
        Self { commands, desc: desc.into() }
    }
}

impl Command for CompoundCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        for cmd in &mut self.commands {
            cmd.execute(graph);
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        for cmd in self.commands.iter_mut().rev() {
            cmd.undo(graph);
        }
    }
    fn description(&self) -> &str {
        &self.desc
    }
}
