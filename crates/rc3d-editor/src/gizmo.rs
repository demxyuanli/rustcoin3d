//! Gizmo: resolve transform targets, pick rays, and apply translate deltas from `rc3d_gizmo`.

use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_gizmo::Gizmo;
#[cfg(test)]
use rc3d_gizmo::GizmoMode;
use rc3d_render::viewport::{ProjectionType, Viewport};
use rc3d_scene::node_data::NodeData;
use rc3d_scene::SceneGraph;

use rc3d_engine_api::ViewportCamera;

/// Pick the transform node to drive the gizmo: prefer a selected `Transform`, else a transform ancestor.
pub fn find_transform_for_selection(graph: &SceneGraph) -> Option<NodeId> {
    for &id in graph.selected_nodes() {
        if let Some(e) = graph.get(id) {
            if matches!(e.data, NodeData::Transform(_)) {
                return Some(id);
            }
        }
    }
    for &id in graph.selected_nodes() {
        let mut cur = id;
        while let Some(p) = graph.get(cur).and_then(|e| e.parent) {
            if let Some(e) = graph.get(p) {
                if matches!(e.data, NodeData::Transform(_)) {
                    return Some(p);
                }
            }
            cur = p;
        }
    }
    None
}

/// Syncs [`Gizmo::target_node`] and [`Gizmo::update_target`].
pub fn sync_gizmo_from_selection(gizmo: &mut Gizmo, graph: &SceneGraph) {
    gizmo.target_node = find_transform_for_selection(graph);
    gizmo.update_target(graph);
}

/// World view + projection for picking in the active viewport, preferring scene graph camera node data.
pub fn pick_view_proj(graph: &SceneGraph, vc: &ViewportCamera, vport: &Viewport) -> (Mat4, Mat4) {
    if let Some(e) = graph.get(vc.camera_node) {
        match &e.data {
            NodeData::PerspectiveCamera(c) => return (c.view_matrix(), c.projection_matrix()),
            NodeData::OrthographicCamera(c) => return (c.view_matrix(), c.projection_matrix()),
            _ => {}
        }
    }
    let aspect = vport.rect.aspect();
    let v = vc.controller.view_matrix();
    let p = match vport.projection_type {
        ProjectionType::Perspective => {
            Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0)
        }
        ProjectionType::Orthographic => {
            let height = vc.controller.distance * 1.2;
            let w = height * aspect;
            rc3d_render::shadow_map::orthographic_wgpu_rh(
                -w * 0.5,
                w * 0.5,
                -height * 0.5,
                height * 0.5,
                0.1,
                1000.0,
            )
        }
    };
    (v, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gizmo_mode_variants_are_distinct() {
        let t = GizmoMode::Translate;
        let r = GizmoMode::Rotate;
        let s = GizmoMode::Scale;

        assert_ne!(t, r);
        assert_ne!(t, s);
        assert_ne!(r, s);

        // Verify each is equal to itself
        assert_eq!(t, GizmoMode::Translate);
        assert_eq!(r, GizmoMode::Rotate);
        assert_eq!(s, GizmoMode::Scale);
    }
}
