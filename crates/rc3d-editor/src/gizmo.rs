//! Gizmo: re-export engine-owned overlay bind helpers.

#[cfg(test)]
use rc3d_gizmo::GizmoMode;

pub use rc3d_engine_api::{
    find_transform_for_selection, sync_gizmo_from_selection, viewport_pick_matrices as pick_view_proj,
};

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

        assert_eq!(t, GizmoMode::Translate);
        assert_eq!(r, GizmoMode::Rotate);
        assert_eq!(s, GizmoMode::Scale);
    }
}
