//! Per-viewport camera management.
//!
//! Binds CameraController instances to Viewports in a ViewportLayout.

use rc3d_core::NodeId;
use rc3d_render::viewport::{ViewportId, ViewportLayout};
use rc3d_scene::SceneGraph;

use crate::camera::CameraController;

/// Associates a CameraController with a viewport.
pub struct ViewportCamera {
    pub viewport_id: ViewportId,
    pub controller: CameraController,
    pub camera_node: NodeId,
}

impl ViewportCamera {
    pub fn new(viewport_id: ViewportId, controller: CameraController, camera_node: NodeId) -> Self {
        Self {
            viewport_id,
            controller,
            camera_node,
        }
    }
}

/// Collection of viewport cameras with active-viewport tracking.
pub struct ViewportCameraSet {
    pub cameras: Vec<ViewportCamera>,
    pub active_viewport: ViewportId,
}

impl ViewportCameraSet {
    pub fn new() -> Self {
        Self {
            cameras: Vec::new(),
            active_viewport: ViewportId(0),
        }
    }

    pub fn add(&mut self, vc: ViewportCamera) {
        if self.cameras.is_empty() {
            self.active_viewport = vc.viewport_id;
        }
        self.cameras.push(vc);
    }

    pub fn active(&self) -> Option<&ViewportCamera> {
        self.cameras
            .iter()
            .find(|vc| vc.viewport_id == self.active_viewport)
    }

    pub fn active_mut(&mut self) -> Option<&mut ViewportCamera> {
        self.cameras
            .iter_mut()
            .find(|vc| vc.viewport_id == self.active_viewport)
    }

    pub fn find(&self, vp_id: ViewportId) -> Option<&ViewportCamera> {
        self.cameras.iter().find(|vc| vc.viewport_id == vp_id)
    }

    pub fn find_mut(&mut self, vp_id: ViewportId) -> Option<&mut ViewportCamera> {
        self.cameras.iter_mut().find(|vc| vc.viewport_id == vp_id)
    }

    /// Set the active viewport; also updates the ViewportLayout.
    pub fn set_active(&mut self, vp_id: ViewportId, layout: &mut ViewportLayout) {
        self.active_viewport = vp_id;
        layout.set_active(vp_id);
    }

    /// Update all camera nodes in the scene graph from their current states.
    pub fn update_all(&self, graph: &mut SceneGraph, layout: &ViewportLayout) {
        for vc in &self.cameras {
            let vp = match layout.viewports.iter().find(|v| v.id == vc.viewport_id) {
                Some(v) => v,
                None => continue,
            };
            let aspect = vp.rect.aspect();
            vc.controller
                .update_camera_node(graph, vc.camera_node, aspect);
        }
    }

    /// After [`ViewportLayout::rebuild`], viewport ids are re-allocated; sync stored ids by index.
    pub fn remap_viewport_ids_from_layout(&mut self, layout: &ViewportLayout) {
        for (i, vc) in self.cameras.iter_mut().enumerate() {
            if let Some(vp) = layout.viewports.get(i) {
                vc.viewport_id = vp.id;
            }
        }
        self.active_viewport = layout.active_id;
    }
}

impl Default for ViewportCameraSet {
    fn default() -> Self {
        Self::new()
    }
}
