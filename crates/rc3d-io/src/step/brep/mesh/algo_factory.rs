//! Surface-type mesh dispatch (OCC BRepMesh_MeshAlgoFactory subset).

use super::face_uv::{FaceUvLoops, UvSource};
use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::topo::BRepFace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceMeshAlgo {
    /// Constrained Delaunay on trimmed UV domain (Plane, Sphere, Cylinder, ...).
    TrimmedCdt,
    /// Full native parametric domain (VERTEX_LOOP / empty wire).
    ClosedParametric,
    /// 3D boundary fan / fill when UV projection fails.
    SurfaceFill3d,
}

/// Surfaces that must use native UV CDT even when STEP loops fail area/validity checks.
fn prefers_native_uv_trim(surface: &SurfaceGeom) -> bool {
    matches!(
        surface,
        SurfaceGeom::Revolution { .. }
            | SurfaceGeom::BSpline(_)
            | SurfaceGeom::Offset { .. }
            | SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Cone { .. }
    )
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
    match &face.surface {
        SurfaceGeom::Plane { .. }
        | SurfaceGeom::Sphere { .. }
        | SurfaceGeom::Cylinder { .. }
        | SurfaceGeom::Cone { .. }
        | SurfaceGeom::Torus { .. } => FaceMeshAlgo::TrimmedCdt,
        _ => FaceMeshAlgo::TrimmedCdt,
    }
}
