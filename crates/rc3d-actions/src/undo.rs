use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, NodeEntry, SceneGraph};

/// A reversible operation on the scene graph.
///
/// Implementors store enough state to both apply (`execute`) and reverse (`undo`)
/// the mutation. Commands are `Send + Sync` so they can be moved across threads.
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

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    pub fn undo_len(&self) -> usize {
        self.undo_stack.len()
    }

    pub fn undo_log(&self) -> Vec<String> {
        self.undo_stack
            .iter()
            .map(|c| c.description().to_string())
            .collect()
    }

    pub fn redo_log(&self) -> Vec<String> {
        self.redo_stack
            .iter()
            .rev()
            .map(|c| c.description().to_string())
            .collect()
    }

    pub fn jump_to_undo_len(&mut self, len: usize, graph: &mut SceneGraph) {
        while self.undo_stack.len() > len {
            if !self.undo(graph) {
                break;
            }
        }
        while self.undo_stack.len() < len {
            if !self.redo(graph) {
                break;
            }
        }
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

type ApplyFn<T> = Box<dyn Fn(&mut NodeEntry, T) + Send + Sync>;

/// Generic field mutation command — covers material, light, camera, and section-plane properties.
pub struct SetFieldCommand<T: Clone + std::fmt::Debug + Send + Sync + 'static> {
    pub node: NodeId,
    pub old_value: T,
    pub new_value: T,
    apply: ApplyFn<T>,
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

/// Command: create a new node with specified data under a parent (or as root).
#[derive(Debug)]
pub struct CreateNodeCommand {
    pub node_id: NodeId,
    pub parent_id: Option<NodeId>,
    pub data: NodeData,
}

impl CreateNodeCommand {
    pub fn new(node_id: NodeId, parent_id: Option<NodeId>, data: NodeData) -> Self {
        Self { node_id, parent_id, data }
    }
}

impl Command for CreateNodeCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        let actual_id = if let Some(parent) = self.parent_id {
            graph.add_child(parent, self.data.clone())
        } else {
            graph.add_root(self.data.clone())
        };
        self.node_id = actual_id;
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        graph.remove(self.node_id);
    }
    fn description(&self) -> &str {
        "CreateNode"
    }
}

/// Command: move existing nodes to a new parent / sibling slot.
#[derive(Debug)]
pub struct ReparentNodesCommand {
    ids: Vec<NodeId>,
    new_parent: Option<NodeId>,
    index: usize,
    old: Vec<(NodeId, Option<NodeId>, usize)>,
}

impl ReparentNodesCommand {
    pub fn new(ids: Vec<NodeId>, new_parent: Option<NodeId>, index: usize) -> Self {
        Self {
            ids,
            new_parent,
            index,
            old: Vec::new(),
        }
    }
}

impl Command for ReparentNodesCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        self.old.clear();
        for &id in &self.ids {
            let parent = graph.parent(id);
            let index = match parent {
                Some(p) => graph
                    .children(p)
                    .and_then(|c| c.iter().position(|&x| x == id))
                    .unwrap_or(0),
                None => graph.roots().iter().position(|&x| x == id).unwrap_or(0),
            };
            self.old.push((id, parent, index));
        }
        let mut idx = self.index;
        for &id in &self.ids {
            if graph.reparent(id, self.new_parent, idx) {
                idx += 1;
            }
        }
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        for &(id, parent, index) in self.old.iter().rev() {
            graph.reparent(id, parent, index);
        }
    }
    fn description(&self) -> &str {
        "Reparent"
    }
}

/// Command: delete a node and its subtree, preserving data for undo.
#[derive(Debug)]
pub struct DeleteNodeCommand {
    pub node_id: NodeId,
    parent_id: Option<NodeId>,
    /// Serialized subtree snapshot (JSON) for undo reconstruction.
    snapshot: String,
}

impl DeleteNodeCommand {
    pub fn new(node_id: NodeId, graph: &SceneGraph) -> Self {
        let parent_id = graph.get(node_id).and_then(|e| e.parent);
        // Snapshot the subtree
        let mut sub = SceneGraph::new();
        if let Some(entry) = graph.get(node_id) {
            let copy = entry.data.clone();
            let root = sub.add_root(copy);
            clone_subtree(graph, node_id, &mut sub, root);
            let snapshot = serde_json::to_string(&sub).unwrap_or_default();
            Self { node_id, parent_id, snapshot }
        } else {
            Self { node_id, parent_id, snapshot: String::new() }
        }
    }
}

impl Command for DeleteNodeCommand {
    fn execute(&mut self, graph: &mut SceneGraph) {
        graph.remove(self.node_id);
    }
    fn undo(&mut self, graph: &mut SceneGraph) {
        if self.snapshot.is_empty() {
            return;
        }
        if let Ok(sub) = serde_json::from_str::<SceneGraph>(&self.snapshot) {
            if let Some(sub_root) = sub.roots().first().copied() {
                merge_subtree(&sub, sub_root, graph, self.parent_id);
            }
        }
    }
    fn description(&self) -> &str {
        "DeleteNode"
    }
}

/// Recursively clone children from source graph into destination graph.
fn clone_subtree(src: &SceneGraph, src_id: NodeId, dst: &mut SceneGraph, dst_parent: NodeId) {
    if let Some(entry) = src.get(src_id) {
        for &child_id in &entry.children {
            if let Some(child) = src.get(child_id) {
                let child_data = child.data.clone();
                let new_id = dst.add_child(dst_parent, child_data);
                clone_subtree(src, child_id, dst, new_id);
            }
        }
    }
}

/// Merge a subtree from `src` graph into `dst` graph under a given parent.
fn merge_subtree(src: &SceneGraph, src_root: NodeId, dst: &mut SceneGraph, parent: Option<NodeId>) {
    if let Some(entry) = src.get(src_root) {
        let data = entry.data.clone();
        let new_id = if let Some(p) = parent {
            dst.add_child(p, data)
        } else {
            dst.add_root(data)
        };
        for &child_id in &entry.children {
            merge_subtree(src, child_id, dst, Some(new_id));
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_scene::node_data::{MaterialNode, TransformNode};

    fn make_graph() -> SceneGraph {
        SceneGraph::new()
    }

    fn add_material(graph: &mut SceneGraph) -> NodeId {
        graph.add_root(NodeData::Material(MaterialNode::default()))
    }

    fn add_transform(graph: &mut SceneGraph) -> NodeId {
        graph.add_root(NodeData::Transform(TransformNode::default()))
    }

    // ── CommandHistory basics ──

    #[test]
    fn test_history_new_is_empty() {
        let h = CommandHistory::new(64);
        assert!(!h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn test_history_undo_redo_translation() {
        let mut g = make_graph();
        let mut h = CommandHistory::new(64);
        let n = add_transform(&mut g);

        h.execute(
            Box::new(SetTranslationCommand {
                node: n,
                old_value: rc3d_core::math::Vec3::ZERO,
                new_value: rc3d_core::math::Vec3::X,
            }),
            &mut g,
        );
        assert!(h.can_undo());
        assert!(!h.can_redo());

        assert!(h.undo(&mut g));
        if let Some(e) = g.get(n) {
            if let NodeData::Transform(t) = &e.data {
                assert!((t.translation - rc3d_core::math::Vec3::ZERO).length() < 1e-5);
            }
        }
        assert!(!h.can_undo());
        assert!(h.can_redo());

        assert!(h.redo(&mut g));
        if let Some(e) = g.get(n) {
            if let NodeData::Transform(t) = &e.data {
                assert!((t.translation - rc3d_core::math::Vec3::X).length() < 1e-5);
            }
        }
    }

    #[test]
    fn test_history_redo_cleared_by_new_execute() {
        let mut g = make_graph();
        let mut h = CommandHistory::new(64);
        let n = add_transform(&mut g);

        h.execute(
            Box::new(SetTranslationCommand {
                node: n,
                old_value: rc3d_core::math::Vec3::ZERO,
                new_value: rc3d_core::math::Vec3::X,
            }),
            &mut g,
        );
        h.undo(&mut g);
        assert!(h.can_redo());

        h.execute(
            Box::new(SetTranslationCommand {
                node: n,
                old_value: rc3d_core::math::Vec3::ZERO,
                new_value: rc3d_core::math::Vec3::Y,
            }),
            &mut g,
        );
        assert!(!h.can_redo());
    }

    #[test]
    fn test_history_max_depth() {
        let mut g = make_graph();
        let mut h = CommandHistory::new(3);
        let n = add_transform(&mut g);

        for i in 0..5 {
            h.execute(
                Box::new(SetTranslationCommand {
                    node: n,
                    old_value: rc3d_core::math::Vec3::splat(i as f32),
                    new_value: rc3d_core::math::Vec3::splat((i + 1) as f32),
                }),
                &mut g,
            );
        }
        let mut count = 0;
        while h.undo(&mut g) {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    // ── SetFieldCommand ──

    #[test]
    fn test_set_field_material_opacity() {
        let mut g = make_graph();
        let n = add_material(&mut g);

        let mut cmd = SetFieldCommand::new(n, 1.0f32, 0.5f32, "SetOpacity", |entry, v| {
            if let NodeData::Material(m) = &mut entry.data {
                m.opacity = v;
            }
        });
        cmd.execute(&mut g);
        if let Some(e) = g.get(n) {
            if let NodeData::Material(m) = &e.data {
                assert!((m.opacity - 0.5).abs() < 1e-5);
            }
        }
        cmd.undo(&mut g);
        if let Some(e) = g.get(n) {
            if let NodeData::Material(m) = &e.data {
                assert!((m.opacity - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_set_field_command_description() {
        let cmd = SetFieldCommand::new(
            rc3d_core::NodeId::default(),
            0u32,
            1u32,
            "TestDesc",
            |_, _| {},
        );
        assert_eq!(cmd.description(), "TestDesc");
    }

    // ── AddChildCommand ──

    #[test]
    fn test_add_child_execute_undo() {
        let mut g = make_graph();
        let parent = add_transform(&mut g);
        let child = add_material(&mut g);

        let mut cmd = AddChildCommand::new(parent, child);
        cmd.execute(&mut g);
        assert!(g.children(parent).unwrap().contains(&child));

        cmd.undo(&mut g);
        assert!(!g.children(parent).unwrap().contains(&child));
    }

    #[test]
    fn test_add_child_idempotent() {
        let mut g = make_graph();
        let parent = add_transform(&mut g);
        let child = add_material(&mut g);

        let mut cmd = AddChildCommand::new(parent, child);
        cmd.execute(&mut g);
        cmd.execute(&mut g);
        assert_eq!(
            g.children(parent).unwrap().iter().filter(|&&c| c == child).count(),
            1
        );
    }

    // ── RemoveChildCommand ──

    #[test]
    fn test_remove_child_execute_undo() {
        let mut g = make_graph();
        let parent = add_transform(&mut g);
        let child = g.add_child(parent, NodeData::Material(MaterialNode::default()));

        assert!(g.children(parent).unwrap().contains(&child));

        let mut cmd = RemoveChildCommand::new(parent, child, &g);
        cmd.execute(&mut g);
        assert!(!g.children(parent).unwrap().contains(&child));

        cmd.undo(&mut g);
        assert!(g.children(parent).unwrap().contains(&child));
    }

    #[test]
    fn test_remove_child_preserves_index_on_undo() {
        let mut g = make_graph();
        let parent = add_transform(&mut g);
        let a = g.add_child(parent, NodeData::Material(MaterialNode::default()));
        let b = g.add_child(parent, NodeData::Material(MaterialNode::default()));
        let c = g.add_child(parent, NodeData::Material(MaterialNode::default()));

        let mut cmd = RemoveChildCommand::new(parent, b, &g);
        cmd.execute(&mut g);
        cmd.undo(&mut g);

        let children = g.children(parent).unwrap();
        assert_eq!(children[0], a);
        assert_eq!(children[1], b);
        assert_eq!(children[2], c);
    }

    // ── CreateNodeCommand ──

    #[test]
    fn test_create_root_node_execute_undo() {
        let mut g = make_graph();
        let node_id = rc3d_core::NodeId::default(); // will be replaced by add_root
        let mut cmd = CreateNodeCommand::new(
            node_id,
            None,
            NodeData::Cube(rc3d_scene::node_data::CubeNode::default()),
        );
        cmd.execute(&mut g);
        assert_eq!(g.roots().len(), 1);

        cmd.undo(&mut g);
        assert!(g.roots().is_empty());
    }

    #[test]
    fn test_create_child_node_execute_undo() {
        let mut g = make_graph();
        let parent = add_transform(&mut g);
        let mut cmd = CreateNodeCommand::new(
            rc3d_core::NodeId::default(),
            Some(parent),
            NodeData::Sphere(rc3d_scene::node_data::SphereNode::default()),
        );
        cmd.execute(&mut g);
        assert_eq!(g.children(parent).unwrap().len(), 1);

        cmd.undo(&mut g);
        assert!(g.children(parent).unwrap().is_empty());
    }

    // ── DeleteNodeCommand ──

    #[test]
    fn test_delete_node_execute_undo() {
        let mut g = make_graph();
        let root = g.add_root(NodeData::Cube(rc3d_scene::node_data::CubeNode::default()));
        let root_count_before = g.roots().len();
        let mut cmd = DeleteNodeCommand::new(root, &g);
        cmd.execute(&mut g);
        assert!(g.get(root).is_none());
        assert_eq!(g.roots().len(), root_count_before - 1);

        cmd.undo(&mut g);
        // Undo restores the subtree (root gets a new ID from the snapshot)
        assert_eq!(g.roots().len(), root_count_before);
    }

    #[test]
    fn test_delete_subtree_undo_restores_children() {
        let mut g = make_graph();
        let root = g.add_root(NodeData::Group(rc3d_scene::node_data::GroupNode));
        let child = g.add_child(root, NodeData::Cube(rc3d_scene::node_data::CubeNode::default()));
        let grandchild = g.add_child(child, NodeData::Sphere(rc3d_scene::node_data::SphereNode::default()));

        let mut cmd = DeleteNodeCommand::new(root, &g);
        cmd.execute(&mut g);
        assert!(g.get(root).is_none());
        assert!(g.get(child).is_none());
        assert!(g.get(grandchild).is_none());

        cmd.undo(&mut g);
        // After undo, the subtree should be back
        let new_root_id = g.roots()[0]; // re-created root gets new ID from sub-graph
        assert!(g.get(new_root_id).is_some());
    }

    // ── CompoundCommand ──

    #[test]
    fn test_compound_execute_undo_atomic() {
        let mut g = make_graph();
        let n = add_transform(&mut g);

        let cmd1: Box<dyn Command> = Box::new(SetTranslationCommand {
            node: n,
            old_value: rc3d_core::math::Vec3::ZERO,
            new_value: rc3d_core::math::Vec3::X,
        });
        let cmd2: Box<dyn Command> = Box::new(SetScaleCommand {
            node: n,
            old_value: rc3d_core::math::Vec3::ONE,
            new_value: rc3d_core::math::Vec3::splat(2.0),
        });

        let mut compound = CompoundCommand::new(vec![cmd1, cmd2], "Move+Scale");
        assert_eq!(compound.description(), "Move+Scale");

        compound.execute(&mut g);
        if let Some(e) = g.get(n) {
            if let NodeData::Transform(t) = &e.data {
                assert!((t.translation - rc3d_core::math::Vec3::X).length() < 1e-5);
                assert!((t.scale - rc3d_core::math::Vec3::splat(2.0)).length() < 1e-5);
            }
        }

        compound.undo(&mut g);
        if let Some(e) = g.get(n) {
            if let NodeData::Transform(t) = &e.data {
                assert!((t.translation - rc3d_core::math::Vec3::ZERO).length() < 1e-5);
                assert!((t.scale - rc3d_core::math::Vec3::ONE).length() < 1e-5);
            }
        }
    }

    // ── SceneGraph structural ──

    #[test]
    fn test_graph_remove_subtree() {
        let mut g = make_graph();
        let root = add_transform(&mut g);
        let child = g.add_child(root, NodeData::Material(MaterialNode::default()));
        g.add_child(child, NodeData::Material(MaterialNode::default()));

        g.remove(root);
        assert!(g.get(root).is_none());
        assert!(g.get(child).is_none());
    }

    #[test]
    fn test_graph_remove_root_clears_roots_vec() {
        let mut g = make_graph();
        let n = add_transform(&mut g);
        assert_eq!(g.roots().len(), 1);
        g.remove(n);
        assert!(g.roots().is_empty());
    }

    #[test]
    fn test_selection_basic_ops() {
        let mut g = make_graph();
        let n = add_transform(&mut g);

        assert!(!g.is_selected(n));
        g.select(n);
        assert!(g.is_selected(n));
        g.deselect(n);
        assert!(!g.is_selected(n));

        g.toggle_selection(n);
        assert!(g.is_selected(n));
        g.toggle_selection(n);
        assert!(!g.is_selected(n));

        g.select(n);
        g.clear_selection();
        assert!(g.selected_nodes().is_empty());
    }

    #[test]
    fn test_select_many_multiple() {
        let mut g = make_graph();
        let a = add_transform(&mut g);
        let b = add_material(&mut g);

        g.select_many([a, b]);
        assert!(g.is_selected(a));
        assert!(g.is_selected(b));
        assert_eq!(g.selected_nodes().len(), 2);
    }

    #[test]
    fn test_mark_fields_dirty_subtree() {
        let mut g = make_graph();
        let root = add_transform(&mut g);
        if let Some(e) = g.get_mut(root) {
            e.fields.insert(root, 0, rc3d_fields::FieldValue::Float(1.0));
        }
        assert!(!g.get(root).unwrap().fields.any_dirty());

        g.mark_fields_dirty_subtree(root);
        assert!(g.get(root).unwrap().fields.any_dirty());
    }
}
