//! TrimmedCdt strategy — Constrained Delaunay Triangulation on trimmed UV domain.
//!
//! Primary meshing path for faces with valid UV loops (PCurve boundaries).
//! Handles inner wires (holes), degenerate edges, and Steiner refinement.

use crate::mesh::face_mesher::{
    FaceContext, FaceMeshMutContext, FaceMeshOutcome, FaceMeshStrategyId, FaceMesher,
};
use crate::mesh::face_dispatch::FaceMeshPlan;

#[derive(Debug, Clone, Default)]
pub struct TrimmedCdtMesher;

impl FaceMesher for TrimmedCdtMesher {
    fn id(&self) -> FaceMeshStrategyId {
        FaceMeshStrategyId::TrimmedCdt
    }

    fn can_handle(&self, plan: FaceMeshPlan, ctx: &FaceContext) -> bool {
        matches!(plan, FaceMeshPlan::TrimmedCdt)
            && ctx.loops.is_fillable()
    }

    fn mesh(&self, face_ctx: &FaceContext, mesh_ctx: &mut FaceMeshMutContext<'_>) -> FaceMeshOutcome {
        use crate::mesh::face_fill::fill_trimmed;
        // Count wire edges for the quality heuristic.
        let wire = match mesh_ctx.reg.wires.get(face_ctx.face.outer_wire) {
            Some(w) => w,
            None => return FaceMeshOutcome::Unavailable,
        };
        let wire_edge_count = wire.edges.len();

        let range = fill_trimmed(
            face_ctx.face_key,
            &face_ctx.loops,
            &face_ctx.face,
            mesh_ctx.reg,
            mesh_ctx.global_vertices,
            mesh_ctx.global_normals,
            mesh_ctx.all_indices,
            mesh_ctx.boundary_pos_to_idx,
            wire_edge_count,
            mesh_ctx.config,
            mesh_ctx.shared_boundary,
        );

        if range.cdt_constraint_failures > 0 {
            // Some constraints could not be enforced — try retry with subdivided edges.
            FaceMeshOutcome::Retry(crate::mesh::face_mesher::RetryHint::SubdivideBoundaryEdges)
        } else {
            FaceMeshOutcome::Ok(range)
        }
    }
}
