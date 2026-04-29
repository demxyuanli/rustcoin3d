use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};
use crate::action::{Action, ActionKind};

/// Collects all enabled section planes from the scene graph.
pub struct SectionPlaneAction {
    pub planes: Vec<[f32; 4]>,
}

impl SectionPlaneAction {
    pub fn new() -> Self {
        Self { planes: Vec::new() }
    }
}

impl Action for SectionPlaneAction {
    fn kind(&self) -> ActionKind { ActionKind::Search }

    fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        Self::collect(graph, root, &mut self.planes);
    }
}

impl SectionPlaneAction {
    fn collect(graph: &SceneGraph, node: NodeId, planes: &mut Vec<[f32; 4]>) {
        let Some(entry) = graph.get(node) else { return };
        if let NodeData::SectionPlane(sp) = &entry.data {
            if sp.enabled {
                planes.push(sp.plane);
            }
        }
        for &child in &entry.children {
            Self::collect(graph, child, planes);
        }
    }
}
