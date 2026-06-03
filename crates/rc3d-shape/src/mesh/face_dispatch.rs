//! Per-face mesh algorithm selection (OCC BRepMesh_MeshAlgoFactory mapping).

use super::algo_factory::{select_face_mesh_algo, FaceMeshAlgo};
use super::face_uv::{FaceUvLoops, UvSource};
use crate::geom::SurfaceGeom;
use crate::topo::BRepFace;

/// Planned mesh strategy for one face (single decision point for fallbacks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceMeshPlan {
    TrimmedCdt,
    ClosedParametric,
    SurfaceFill3d { reason: SurfaceFillReason },
    SkippedHeal,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceFillReason {
    InvalidUvLoops,
    AlgoFactory,
    CdtFailed,
}

/// Choose mesh plan from surface, loops, and heal skip state.
pub fn plan_face_mesh(
    face: &BRepFace,
    loops: &FaceUvLoops,
    wire_empty: bool,
    heal_skipped: bool,
) -> FaceMeshPlan {
    if heal_skipped {
        return FaceMeshPlan::SkippedHeal;
    }
    if wire_empty {
        return FaceMeshPlan::ClosedParametric;
    }
    let algo = select_face_mesh_algo(face, loops, wire_empty);
    match algo {
        FaceMeshAlgo::ClosedParametric => FaceMeshPlan::ClosedParametric,
        FaceMeshAlgo::SurfaceFill3d => FaceMeshPlan::SurfaceFill3d {
            reason: if !loops.is_fillable() || loops.uv_source == UvSource::SurfaceFill {
                SurfaceFillReason::InvalidUvLoops
            } else {
                SurfaceFillReason::AlgoFactory
            },
        },
        FaceMeshAlgo::TrimmedCdt => {
            if loops.is_valid() || prefers_native_uv_trim(&face.surface) {
                FaceMeshPlan::TrimmedCdt
            } else {
                FaceMeshPlan::SurfaceFill3d {
                    reason: SurfaceFillReason::InvalidUvLoops,
                }
            }
        }
    }
}

/// Map a face plan to the mesh algorithm enum used by the face mesher.
pub fn algo_from_plan(plan: FaceMeshPlan) -> Option<FaceMeshAlgo> {
    match plan {
        FaceMeshPlan::TrimmedCdt => Some(FaceMeshAlgo::TrimmedCdt),
        FaceMeshPlan::ClosedParametric => Some(FaceMeshAlgo::ClosedParametric),
        FaceMeshPlan::SurfaceFill3d { .. } => Some(FaceMeshAlgo::SurfaceFill3d),
        FaceMeshPlan::SkippedHeal | FaceMeshPlan::Failed => None,
    }
}

/// Surfaces that must use native UV CDT even when STEP loops fail area/validity checks.
pub fn prefers_native_uv_trim(surface: &SurfaceGeom) -> bool {
    matches!(
        surface,
        SurfaceGeom::Revolution { .. }
            | SurfaceGeom::BSpline(_)
            | SurfaceGeom::Offset { .. }
            | SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Cone { .. }
            | SurfaceGeom::Torus { .. }
            | SurfaceGeom::Extrusion { .. }
    )
}
