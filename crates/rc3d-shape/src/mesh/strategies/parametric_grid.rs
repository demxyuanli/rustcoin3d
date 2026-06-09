//! ParametricGrid strategy — trimmed UV grid meshing (fallback).
//!
//! Used when CDT constraint enforcement fails. Requires UV bounds
//! and grid resolution that are computed from the face's UV domain.

use crate::mesh::face_mesher::{
    FaceContext, FaceMeshMutContext, FaceMeshOutcome, FaceMeshStrategyId, FaceMesher,
};
use crate::mesh::face_dispatch::FaceMeshPlan;

#[derive(Debug, Clone, Default)]
pub struct ParametricGridMesher;

impl FaceMesher for ParametricGridMesher {
    fn id(&self) -> FaceMeshStrategyId {
        FaceMeshStrategyId::ParametricGrid
    }

    fn can_handle(&self, _plan: FaceMeshPlan, ctx: &FaceContext) -> bool {
        ctx.policy.fallback.parametric_grid && !ctx.loops.outer.boundary.is_empty()
    }

    fn mesh(&self, face_ctx: &FaceContext, mesh_ctx: &mut FaceMeshMutContext<'_>) -> FaceMeshOutcome {
        use crate::mesh::grid::mesh_trimmed_uv_grid;
        let loops = &face_ctx.loops;

        // Compute UV bounds from loop vertices.
        let (mut u_min, mut u_max, mut v_min, mut v_max) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for v in &loops.outer.boundary {
            u_min = u_min.min(v.uv.0); u_max = u_max.max(v.uv.0);
            v_min = v_min.min(v.uv.1); v_max = v_max.max(v.uv.1);
        }
        if u_max <= u_min || v_max <= v_min {
            return FaceMeshOutcome::Unavailable;
        }
        let uv_bounds = (u_min, u_max, v_min, v_max);
        // Grid resolution from config: use parameter_division_max_depth.
        let grid_segs = mesh_ctx.config.parameter_division_max_depth.max(2) as u32 * 8;

        mesh_trimmed_uv_grid(
            &face_ctx.face,
            loops,
            uv_bounds,
            Some(mesh_ctx.config),
            Some(grid_segs),
            mesh_ctx.global_vertices,
            mesh_ctx.global_normals,
            mesh_ctx.boundary_pos_to_idx,
            mesh_ctx.all_indices,
            mesh_ctx.shared_boundary,
        );

        let tris_after = mesh_ctx.all_indices.len() / 4;
        FaceMeshOutcome::Ok(crate::mesh::face_fill::FaceMeshRange {
            face_key: face_ctx.face_key,
            first_tri: 0, // caller tracks
            tri_count: tris_after,
            boundary_global: Default::default(),
            max_chord_error: 0.0,
            cdt_constraint_failures: 0,
        })
    }
}
