//! Scene-graph transform manipulator (three.js TransformControls analog).

use rc3d_core::NodeId;
pub use rc3d_scene::node_data::{DraggerKind, ManipMode, ManipSpace};
use rc3d_scene::node_data::TransformManipNode;

use crate::scene::{NodeHandle, Scene};

/// Builder for [`TransformManipNode`] plus optional child dragger parts.
#[derive(Clone, Debug)]
pub struct TransformManip {
    pub mode: ManipMode,
    pub space: ManipSpace,
    pub size: f32,
    pub enabled: bool,
    pub target: Option<NodeId>,
    pub parts: Vec<DraggerKind>,
}

impl Default for TransformManip {
    fn default() -> Self {
        Self {
            mode: ManipMode::Translate,
            space: ManipSpace::World,
            size: 0.0,
            enabled: true,
            target: None,
            parts: Vec::new(),
        }
    }
}

impl TransformManip {
    pub fn mode(mut self, mode: ManipMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn space(mut self, space: ManipSpace) -> Self {
        self.space = space;
        self
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn target(mut self, target: NodeId) -> Self {
        self.target = Some(target);
        self
    }

    pub fn parts(mut self, parts: impl Into<Vec<DraggerKind>>) -> Self {
        self.parts = parts.into();
        self
    }
}

impl Scene {
    /// Add a transform manipulator under `parent` (usually a kit Separator).
    pub fn add_transform_manip(&mut self, parent: NodeHandle, manip: TransformManip) -> NodeHandle {
        let id = crate::kits::transform_manip_kit(
            self.graph_mut(),
            parent.0,
            TransformManipNode {
                mode: manip.mode,
                space: manip.space,
                enabled: manip.enabled,
                size: manip.size,
                target: manip.target,
            },
            &manip.parts,
        );
        NodeHandle(id)
    }
}
