//! Bounding-box action — logic lives in `rc3d-scene`; this module adds the [`Action`] adapter.

use rc3d_core::NodeId;
use rc3d_scene::SceneGraph;

pub use rc3d_scene::GetBoundingBoxAction;

impl crate::Action for GetBoundingBoxAction {
    fn kind(&self) -> crate::ActionKind {
        crate::ActionKind::GetBoundingBox
    }

    fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        GetBoundingBoxAction::apply(self, graph, root);
    }
}
