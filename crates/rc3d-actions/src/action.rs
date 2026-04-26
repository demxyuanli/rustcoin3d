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
