use rc3d_core::NodeId;
use rc3d_scene::SceneGraph;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionKind {
    GLRender,
    GetBoundingBox,
    RayPick,
    Search,
}

pub trait Action {
    fn kind(&self) -> ActionKind;
    fn apply(&mut self, graph: &SceneGraph, root: NodeId);
}

/// Helper to run an action against all roots of a scene graph.
pub fn apply_to_all_roots(action: &mut dyn Action, graph: &SceneGraph) {
    for &root in graph.roots() {
        action.apply(graph, root);
    }
}
