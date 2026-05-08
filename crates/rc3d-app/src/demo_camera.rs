//! Orbit camera helpers for `rc3d-app` examples and simple viewers.
use rc3d_actions::{apply_to_all_roots, GetBoundingBoxAction};
use rc3d_core::math::Vec3;
use rc3d_scene::SceneGraph;

use crate::CameraController;

/// Build an orbit [`CameraController`] from world-space geometry bounds.
///
/// Falls back to `(fallback_target, fallback_distance)` when the bounding box is unusable.
pub fn camera_controller_from_scene_bounds(
    graph: &SceneGraph,
    fallback_target: Vec3,
    fallback_distance: f32,
) -> CameraController {
    let mut bbox_action = GetBoundingBoxAction::new();
    apply_to_all_roots(&mut bbox_action, graph);
    let bbox = bbox_action.bounding_box;
    let valid = bbox.min.x.is_finite()
        && bbox.max.x.is_finite()
        && bbox.min.y.is_finite()
        && bbox.max.y.is_finite()
        && bbox.min.z.is_finite()
        && bbox.max.z.is_finite();
    if !valid {
        return CameraController::new(fallback_target, fallback_distance);
    }
    let center = bbox.center();
    let extent = bbox.size();
    let radius = extent.length().max(1.0e-6) * 0.5;
    let distance = (radius * 2.2).max(fallback_distance.max(1.0));
    CameraController::new(center, distance.max(1.0))
}
