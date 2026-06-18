use std::borrow::Cow;
use std::collections::HashMap;

use rc3d_core::math::PVec3;
use crate::topo::{BRepFace, EdgeKey, FaceKey};
use crate::mesh::boundary::BoundaryPosIndex;
use crate::mesh::face_fill::FaceMeshRange;
use crate::mesh::face_uv::FaceUvLoops;
use crate::mesh::report::ShellMeshReport;

/// Shell orientation XOR face.same_sense for mesh normals and winding.
pub(crate) fn mesh_face_view(face: &BRepFace, effective_same_sense: bool) -> Cow<'_, BRepFace> {
    if effective_same_sense == face.same_sense {
        Cow::Borrowed(face)
    } else {
        let mut f = face.clone();
        f.same_sense = effective_same_sense;
        Cow::Owned(f)
    }
}

pub(crate) struct FaceWireInfo {
    pub face_key: FaceKey,
    pub wire_edges: Vec<(EdgeKey, Vec<usize>)>,
}

pub(crate) struct FaceLoopData {
    pub face_key: FaceKey,
    pub wire_edges: Vec<(EdgeKey, Vec<usize>)>,
    pub loops: Option<FaceUvLoops>,
}

pub(crate) struct ChunkOutput {
    pub vertices: Vec<PVec3>,
    pub normals: Vec<PVec3>,
    pub indices: Vec<i32>,
    pub face_ranges: Vec<FaceMeshRange>,
    pub report: ShellMeshReport,
    pub wire_diag: Vec<String>,
}

pub(crate) struct BoundaryPoolResult {
    pub boundary_vertices: Vec<PVec3>,
    pub boundary_normals: Vec<PVec3>,
    pub boundary_pos_to_idx: BoundaryPosIndex,
    pub boundary_vertex_count: usize,
    pub edge_boundary_idx: HashMap<(FaceKey, EdgeKey, usize), usize>,
    pub face_infos: Vec<FaceWireInfo>,
    pub face_orient: HashMap<FaceKey, bool>,
    pub heal_skipped_faces: Vec<FaceKey>,
    pub global_vertices: Vec<PVec3>,
}
