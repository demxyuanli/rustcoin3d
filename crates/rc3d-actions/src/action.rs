use rc3d_core::NodeId;
use rc3d_scene::SceneGraph;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionKind {
    GLRender,
    GetBoundingBox,
    RayPick,
    Search,
    HandleEvent,
    GetMatrix,
}

pub trait Action: Send {
    fn kind(&self) -> ActionKind;
    fn apply(&mut self, graph: &SceneGraph, root: NodeId);
}

/// Helper to run an action against all roots of a scene graph sequentially.
pub fn apply_to_all_roots(action: &mut dyn Action, graph: &SceneGraph) {
    for &root in graph.roots() {
        action.apply(graph, root);
    }
}

/// Parallel helper to run an action against all roots using rayon.
/// Each root subtree is processed in its own thread with a clone of the action.
pub fn par_apply_to_all_roots<T: Action + Clone + Sync>(
    action: &T,
    graph: &SceneGraph,
) {
    let roots = graph.roots().to_vec();
    rayon::scope(|s| {
        for &root in &roots {
            let mut a = action.clone();
            s.spawn(move |_| {
                a.apply(graph, root);
            });
        }
    });
}
