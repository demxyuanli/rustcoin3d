use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::PerspectiveCameraNode;

use crate::get_bounding_box::GetBoundingBoxAction;

/// Controls how `fit_camera_to_scene` positions cameras.
pub struct CameraFitConfig {
    /// Override FOV (radians). None = read from existing scene camera, default PI/4.
    pub fov: Option<f32>,
    /// Override eye position. None = auto-computed from bounding box.
    pub eye: Option<Vec3>,
    /// Override look-at target. None = bounding box center.
    pub target: Option<Vec3>,
    /// Extra distance multiplier. 1.0 = bounding sphere exactly fills viewport vertically.
    pub distance_scale: f32,
}

impl Default for CameraFitConfig {
    fn default() -> Self {
        Self {
            fov: None,
            eye: None,
            target: None,
            distance_scale: 1.0,
        }
    }
}

/// Compute scene bounding box and reposition all cameras to frame it.
///
/// Returns `(target, orbit_distance)` suitable for `CameraController::new`.
pub fn fit_camera_to_scene(
    graph: &mut SceneGraph,
    config: CameraFitConfig,
) -> (Vec3, f32) {
    let mut bbox_action = GetBoundingBoxAction::new();
    crate::apply_to_all_roots(&mut bbox_action, graph);
    let bbox = bbox_action.bounding_box;

    if !bbox.min.x.is_finite() || !bbox.max.x.is_finite() {
        return (Vec3::ZERO, 10.0);
    }

    let center = config.target.unwrap_or_else(|| bbox.center());
    let extent = bbox.size();
    let radius = extent.length().max(1.0) * 0.5;

    let fov = config.fov.unwrap_or_else(|| {
        read_first_camera_fov(graph).unwrap_or(std::f32::consts::FRAC_PI_4)
    });

    let distance = (radius / (fov * 0.5).sin()).max(radius * 0.5) * config.distance_scale;

    let eye = config.eye.unwrap_or_else(|| {
        let dir = Vec3::new(1.5, 1.1, 2.0).normalize();
        center + dir * distance
    });

    let near = (distance * 0.001).max(0.01);
    let far = (distance * 20.0).max(100.0);

    for &root in graph.roots().to_vec().iter() {
        apply_camera_fit_recursive(graph, root, eye, center, near, far);
    }

    (center, distance)
}

fn read_first_camera_fov(graph: &SceneGraph) -> Option<f32> {
    for &root in graph.roots() {
        if let Some(fov) = find_camera_fov(graph, root) {
            return Some(fov);
        }
    }
    None
}

fn find_camera_fov(graph: &SceneGraph, node: NodeId) -> Option<f32> {
    let entry = graph.get(node)?;
    if let NodeData::PerspectiveCamera(cam) = &entry.data {
        return Some(cam.fov);
    }
    for &child in &entry.children {
        if let Some(fov) = find_camera_fov(graph, child) {
            return Some(fov);
        }
    }
    None
}

fn apply_camera_fit_recursive(
    graph: &mut SceneGraph,
    node: NodeId,
    eye: Vec3,
    target: Vec3,
    near: f32,
    far: f32,
) {
    let children = graph.children(node).unwrap_or(&[]).to_vec();
    if let Some(entry) = graph.get_mut(node) {
        match &mut entry.data {
            NodeData::PerspectiveCamera(cam) => {
                let fov = cam.fov;
                let aspect = cam.aspect;
                *cam = PerspectiveCameraNode::look_at(eye, target, Vec3::Y, fov, aspect);
                cam.near = near;
                cam.far = far;
            }
            NodeData::OrthographicCamera(cam) => {
                cam.position = eye;
                cam.orientation = Mat4::look_at_rh(eye, target, Vec3::Y);
                cam.near = near;
                cam.far = far;
                cam.height = (target - eye).length().max(1.0);
            }
            _ => {}
        }
    }
    for child in children {
        apply_camera_fit_recursive(graph, child, eye, target, near, far);
    }
}
