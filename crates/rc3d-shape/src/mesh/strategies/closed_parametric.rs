//! ClosedParametric strategy — full native parametric domain mesh.
//!
//! Handles VERTEX_LOOP faces (empty wire) and closed periodic surfaces
//! (sphere, torus without holes) using `mesh_closed_surface`.

use crate::mesh::face_mesher::{
    FaceContext, FaceMeshMutContext, FaceMeshOutcome, FaceMeshStrategyId, FaceMesher,
};

#[derive(Debug, Clone, Default)]
pub struct ClosedParametricMesher;

impl FaceMesher for ClosedParametricMesher {
    fn id(&self) -> FaceMeshStrategyId {
        FaceMeshStrategyId::ClosedParametric
    }

    fn can_handle(&self, plan: crate::mesh::face_dispatch::FaceMeshPlan, _ctx: &FaceContext) -> bool {
        matches!(plan, crate::mesh::face_dispatch::FaceMeshPlan::ClosedParametric)
    }

    fn mesh(&self, face_ctx: &FaceContext, mesh_ctx: &mut FaceMeshMutContext<'_>) -> FaceMeshOutcome {
        use crate::mesh::grid::mesh_closed_surface;
        let tris_before = mesh_ctx.all_indices.len() / 4;
        mesh_closed_surface(
            &face_ctx.face,
            mesh_ctx.config,
            mesh_ctx.global_vertices,
            mesh_ctx.global_normals,
            mesh_ctx.boundary_pos_to_idx,
            mesh_ctx.all_indices,
            mesh_ctx.shared_boundary,
        );
        let tris_after = mesh_ctx.all_indices.len() / 4;
        FaceMeshOutcome::Ok(crate::mesh::face_fill::FaceMeshRange {
            face_key: face_ctx.face_key,
            first_tri: tris_before,
            tri_count: tris_after.saturating_sub(tris_before),
            boundary_global: Default::default(),
            max_chord_error: 0.0,
            cdt_constraint_failures: 0,
        })
    }
}
