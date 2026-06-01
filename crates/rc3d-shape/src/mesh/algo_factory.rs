//! Surface-type mesh dispatch (OCC BRepMesh_MeshAlgoFactory subset).

use super::face_dispatch::prefers_native_uv_trim;
use super::face_uv::{FaceUvLoops, UvSource};
use crate::topo::BRepFace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceMeshAlgo {
    /// Constrained Delaunay on trimmed UV domain (Plane, Sphere, Cylinder, ...).
    TrimmedCdt,
    /// Full native parametric domain (VERTEX_LOOP / empty wire).
    ClosedParametric,
    /// 3D boundary fan / fill when UV projection fails.
    SurfaceFill3d,
}

/// Choose mesh algorithm from surface geometry and loop validity (OCC factory mapping).
pub fn select_face_mesh_algo(face: &BRepFace, loops: &FaceUvLoops, wire_empty: bool) -> FaceMeshAlgo {
    if wire_empty {
        return FaceMeshAlgo::ClosedParametric;
    }
    if !loops.is_fillable() || loops.uv_source == UvSource::SurfaceFill {
        return FaceMeshAlgo::SurfaceFill3d;
    }
    if !loops.is_valid() && !prefers_native_uv_trim(&face.surface) {
        return FaceMeshAlgo::SurfaceFill3d;
    }
    // All remaining cases default to TrimmedCdt — the UV domain is valid
    // enough for constrained Delaunay tessellation.
    FaceMeshAlgo::TrimmedCdt
}
