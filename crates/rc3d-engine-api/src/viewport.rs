//! Per-viewport camera management.
//!
//! Binds CameraController instances to Viewports in a ViewportLayout.

use rc3d_core::aabb::Aabb;
use rc3d_core::NodeId;
use rc3d_render::viewport::{QuadViewEye, ViewportId, ViewportLayout};
use rc3d_scene::node_data::{
    NodeData, OrthographicCameraNode, PerspectiveCameraNode, SeparatorNode, StereoCameraNode,
    StereoMode,
};
use rc3d_scene::SceneGraph;

use crate::camera::{CameraController, ViewPreset};

/// Camera node name prefix for the HOOPS-style four-view pack.
pub const STANDARD_QUAD_NODE_PREFIX: &str = "vp.";

/// Default pack: Front / Right / Top ortho + Persp (Iso). Matched by viewport name.
pub fn standard_quad_preset(viewport_name: &str) -> ViewPreset {
    match viewport_name {
        "Top" => ViewPreset::Top,
        "Bottom" => ViewPreset::Bottom,
        "Front" => ViewPreset::Front,
        "Back" => ViewPreset::Back,
        "Right" => ViewPreset::Right,
        "Left" => ViewPreset::Left,
        _ => ViewPreset::Iso,
    }
}

fn quad_want_ortho(viewport_name: &str) -> bool {
    !matches!(viewport_name, "Persp" | "Perspective" | "Iso")
}

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

    /// Bind stored camera nodes onto layout viewports (by index).
    pub fn bind_camera_nodes(&self, layout: &mut ViewportLayout) {
        for (i, vp) in layout.viewports.iter_mut().enumerate() {
            vp.camera_node = self.cameras.get(i).map(|vc| vc.camera_node);
        }
    }

    /// Install Front/Right/Top/Persp cameras for the current Quad layout.
    pub fn install_standard_quad(
        &mut self,
        graph: &mut SceneGraph,
        layout: &mut ViewportLayout,
        aabb: Option<Aabb>,
    ) {
        let pack_root = graph.find_named("StandardViews").unwrap_or_else(|| {
            let id = graph.add_root(NodeData::Separator(SeparatorNode));
            graph.set_name(id, "StandardViews");
            id
        });
        let (target, distance) = match aabb {
            Some(ref box_) => {
                let size = box_.size().length().max(0.5);
                (box_.center(), (size / (std::f32::consts::FRAC_PI_4 * 0.5).sin()).max(1.0))
            }
            None => (rc3d_core::math::Vec3::ZERO, 8.0),
        };

        self.cameras.clear();
        for vp in &mut layout.viewports {
            let preset = standard_quad_preset(&vp.name);
            let node_name = format!("{STANDARD_QUAD_NODE_PREFIX}{}", vp.name);
            let want_ortho = quad_want_ortho(&vp.name);
            let cam_id = match graph.find_named(&node_name) {
                Some(id) => {
                    if let Some(entry) = graph.get_mut(id) {
                        match (want_ortho, &entry.data) {
                            (true, NodeData::OrthographicCamera(_)) => {}
                            (false, NodeData::PerspectiveCamera(_)) => {}
                            (true, _) => {
                                entry.data = NodeData::OrthographicCamera(OrthographicCameraNode::default());
                            }
                            (false, _) => {
                                entry.data = NodeData::PerspectiveCamera(PerspectiveCameraNode::default());
                            }
                        }
                    }
                    id
                }
                None => {
                    let data = if want_ortho {
                        NodeData::OrthographicCamera(OrthographicCameraNode::default())
                    } else {
                        NodeData::PerspectiveCamera(PerspectiveCameraNode::default())
                    };
                    let id = graph.add_child(pack_root, data);
                    graph.set_name(id, node_name);
                    id
                }
            };
            let mut controller = CameraController::new(target, distance);
            controller.set_view_preset(preset);
            if let Some(ref box_) = aabb {
                controller.fit_bounds(box_, std::f32::consts::FRAC_PI_4);
            }
            let aspect = vp.rect.aspect();
            controller.update_camera_node(graph, cam_id, aspect);
            vp.camera_node = Some(cam_id);
            self.add(ViewportCamera::new(vp.id, controller, cam_id));
        }
        if let Some(persp) = layout.viewports.iter().find(|v| v.name == "Persp") {
            self.set_active(persp.id, layout);
        } else if let Some(first) = layout.viewports.first() {
            self.set_active(first.id, layout);
        }
    }

    /// Read view/projection from bound camera nodes (layout order).
    pub fn collect_quad_eyes(&self, graph: &SceneGraph, layout: &ViewportLayout) -> Vec<QuadViewEye> {
        let mut eyes = Vec::with_capacity(layout.viewports.len());
        for vp in &layout.viewports {
            let Some(vc) = self.find(vp.id) else {
                continue;
            };
            let Some(entry) = graph.get(vc.camera_node) else {
                continue;
            };
            match &entry.data {
                NodeData::PerspectiveCamera(cam) => eyes.push(QuadViewEye {
                    view: cam.view_matrix(),
                    projection: cam.projection_matrix(),
                    camera_pos: cam.position,
                    orthographic: false,
                }),
                NodeData::OrthographicCamera(cam) => eyes.push(QuadViewEye {
                    view: cam.view_matrix(),
                    projection: cam.projection_matrix(),
                    camera_pos: cam.position,
                    orthographic: true,
                }),
                _ => {}
            }
        }
        eyes
    }
}

fn find_stereo_camera(graph: &SceneGraph) -> Option<StereoCameraNode> {
    fn walk(graph: &SceneGraph, node: NodeId) -> Option<StereoCameraNode> {
        let entry = graph.get(node)?;
        if let NodeData::StereoCamera(s) = &entry.data {
            return Some(s.clone());
        }
        for &child in &entry.children {
            if let Some(found) = walk(graph, child) {
                return Some(found);
            }
        }
        None
    }
    graph.roots().iter().copied().find_map(|r| walk(graph, r))
}

fn find_perspective_camera(
    graph: &SceneGraph,
    prefer: NodeId,
) -> Option<&PerspectiveCameraNode> {
    if let Some(entry) = graph.get(prefer) {
        if let NodeData::PerspectiveCamera(cam) = &entry.data {
            return Some(cam);
        }
    }
    fn walk(graph: &SceneGraph, node: NodeId) -> Option<&PerspectiveCameraNode> {
        let entry = graph.get(node)?;
        if let NodeData::PerspectiveCamera(cam) = &entry.data {
            return Some(cam);
        }
        for &child in &entry.children {
            if let Some(found) = walk(graph, child) {
                return Some(found);
            }
        }
        None
    }
    graph.roots().iter().copied().find_map(|r| walk(graph, r))
}

/// Left/right eye matrices from the first `StereoCamera` in the graph.
/// `None` when missing, IPD is ~0, or no perspective base camera is available.
pub fn collect_stereo_eyes(
    graph: &SceneGraph,
    surface_width: u32,
    surface_height: u32,
) -> Option<(StereoMode, Vec<QuadViewEye>)> {
    let stereo = find_stereo_camera(graph)?;
    let cam = find_perspective_camera(graph, stereo.base_camera)?;
    let aspect = stereo.mode.eye_aspect(surface_width, surface_height);
    let eyes = stereo.eyes_from_perspective(cam, aspect)?;
    Some((
        stereo.mode,
        eyes.into_iter()
            .map(|e| QuadViewEye {
                view: e.view,
                projection: e.projection,
                camera_pos: e.position,
                orthographic: false,
            })
            .collect(),
    ))
}

impl Default for ViewportCameraSet {
    fn default() -> Self {
        Self::new()
    }
}
