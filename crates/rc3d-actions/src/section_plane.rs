use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};
use crate::action::{Action, ActionKind};

/// Collects all enabled section planes from the scene graph.
pub struct SectionPlaneAction {
    pub planes: Vec<[f32; 4]>,
    /// Per plane (same order as `planes`): cap tint when `cap_enabled` on the node.
    pub cap_tints: Vec<Option<[f32; 4]>>,
    pub has_caps: bool,
}

impl SectionPlaneAction {
    pub fn new() -> Self {
        Self { planes: Vec::new(), cap_tints: Vec::new(), has_caps: false }
    }
}

impl Action for SectionPlaneAction {
    fn kind(&self) -> ActionKind { ActionKind::Search }

    fn apply(&mut self, graph: &SceneGraph, root: NodeId) {
        self.planes.clear();
        self.cap_tints.clear();
        self.has_caps = false;
        Self::collect(graph, root, &mut self.planes, &mut self.cap_tints, &mut self.has_caps);
    }
}

impl SectionPlaneAction {
    fn collect(
        graph: &SceneGraph,
        node: NodeId,
        planes: &mut Vec<[f32; 4]>,
        cap_tints: &mut Vec<Option<[f32; 4]>>,
        has_caps: &mut bool,
    ) {
        let Some(entry) = graph.get(node) else { return };
        if let NodeData::SectionPlane(sp) = &entry.data {
            if sp.enabled {
                planes.push(sp.plane);
                if sp.cap_enabled {
                    cap_tints.push(Some(sp.cap_color));
                    *has_caps = true;
                } else {
                    cap_tints.push(None);
                }
            }
        }
        for &child in &entry.children {
            Self::collect(graph, child, planes, cap_tints, has_caps);
        }
    }
}
