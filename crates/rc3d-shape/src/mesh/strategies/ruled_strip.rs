//! RuledStrip strategy — quad strip between two boundary wires.
//!
//! Handles faces where exactly two boundary wire edges exist (revolution pole patches,
//! cylinder seam strips). Delegates to `try_ruled_two_wire_mesh`.
//!
//! Note: Requires field destructuring of FaceMeshMutContext to build RuledMeshBuffers.
//! This is done via direct pointer manipulation which is safe because
//! RuledMeshBuffers lifetime matches the borrow.

use crate::mesh::face_mesher::{
    FaceContext, FaceMeshMutContext, FaceMeshOutcome, FaceMeshStrategyId, FaceMesher,
};
use crate::mesh::face_dispatch::FaceMeshPlan;

#[derive(Debug, Clone, Default)]
pub struct RuledStripMesher;

impl FaceMesher for RuledStripMesher {
    fn id(&self) -> FaceMeshStrategyId {
        FaceMeshStrategyId::RuledStrip
    }

    fn can_handle(&self, _plan: FaceMeshPlan, _ctx: &FaceContext) -> bool {
        // RuledStrip is a fallback tried when TrimmedCdt fails.
        true
    }

    fn mesh(&self, face_ctx: &FaceContext, mesh_ctx: &mut FaceMeshMutContext<'_>) -> FaceMeshOutcome {
        let wire = match mesh_ctx.reg.wires.get(face_ctx.face.outer_wire) {
            Some(w) => w,
            None => return FaceMeshOutcome::Unavailable,
        };
        if wire.edges.len() < 2 {
            return FaceMeshOutcome::Unavailable;
        }

        // Build wire_edges from edge polygons.
        let wire_edges: Vec<_> = wire.edges.iter()
            .filter_map(|&(ek, _)| {
                let poly = mesh_ctx.edge_polygons.get(&ek)?;
                let indices: Vec<usize> = (0..poly.params_3d.len()).collect();
                Some((ek, indices))
            })
            .collect();

        if wire_edges.len() < 2 {
            return FaceMeshOutcome::Unavailable;
        }

        // SAFETY: RuledMeshBuffers takes ownership of mutable references
        // that all originate from `mesh_ctx`. The borrows are disjoint
        // and the lifetime matches `mesh_ctx`'s borrow.
        let ptr = mesh_ctx as *mut FaceMeshMutContext<'_>;
        let buffers = unsafe {
            crate::mesh::ruled::RuledMeshBuffers {
                global_vertices: &mut (*ptr).global_vertices,
                global_normals: &mut (*ptr).global_normals,
                all_indices: &mut (*ptr).all_indices,
                pos_to_idx: &mut (*ptr).boundary_pos_to_idx,
                shared_boundary: (*ptr).shared_boundary,
            }
        };

        let range = crate::mesh::ruled::try_ruled_two_wire_mesh(
            face_ctx.face_key,
            &face_ctx.face,
            &wire_edges,
            mesh_ctx.edge_polygons,
            mesh_ctx.edge_boundary_idx,
            buffers,
            mesh_ctx.config,
        );

        FaceMeshOutcome::Ok(range)
    }
}
