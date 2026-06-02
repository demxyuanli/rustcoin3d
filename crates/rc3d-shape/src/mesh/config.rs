use rc3d_core::math::Vec3;
use crate::geom::SurfaceGeom;
use super::edge_disc::EdgeDiscConfig;
use super::face_fill::FaceFillConfig;
use super::optimize::OptimizeConfig;
use super::refiner::RefineConfig;

/// UV grid resolution fallback when deflection config is unavailable.
pub const MESH_CLOSED_SURFACE_SEGS: u32 = 48;

pub fn min_adequate_trim_tris(surface: &SurfaceGeom, _wire_edge_count: usize) -> usize {
    match surface {
        SurfaceGeom::Revolution { .. }
        | SurfaceGeom::BSpline(_)
        | SurfaceGeom::Offset { .. } => 64,
        _ => 8,
    }
}

#[derive(Debug, Clone)]
pub struct BRepMeshConfig {
    pub edge: EdgeDiscConfig,
    pub face: FaceFillConfig,
    pub refine: RefineConfig,
    pub optimize: OptimizeConfig,
    /// When > 0, deflection = shell_bbox_diagonal * relative_deflection (OCC Relative).
    pub relative_deflection: f32,
    pub same_parameter_tol: f32,
    /// Fast tessellation: skip chord retry, grid quality trials, and per-face chord remeasure.
    pub fast_export: bool,
    /// Tolerance for vertex welding after void shell merge. Prevents visible seams.
    pub weld_tolerance: f32,
}

impl Default for BRepMeshConfig {
    fn default() -> Self {
        Self {
            edge: EdgeDiscConfig::default(),
            face: FaceFillConfig::default(),
            refine: RefineConfig::default(),
            optimize: OptimizeConfig::default(),
            relative_deflection: 0.0,
            same_parameter_tol: 1e-4,
            fast_export: false,
            weld_tolerance: 1e-6,
        }
    }
}

impl BRepMeshConfig {
    /// Fast tessellation for export/regression: skips post-refine, shell-wide optimize,
    /// chord retries, and expensive grid quality comparisons.
    pub fn preview() -> Self {
        let mut cfg = Self::default();
        cfg.refine.enable_post_refine = false;
        cfg.optimize.max_iterations = 0;
        cfg.fast_export = true;
        cfg.face.max_adapt_iterations = 4;
        cfg.face.parameter_division_max_depth = 3;
        cfg.face.skip_interior_edge_split = false;
        cfg
    }
}

/// Count triangles with non-zero area in a face index range (before global cull).
pub fn count_valid_tris_in_range(
    indices: &[i32],
    vertices: &[Vec3],
    first_tri: usize,
    tri_count: usize,
) -> usize {
    let mut valid = 0usize;
    for ti in first_tri..first_tri.saturating_add(tri_count) {
        let base = ti * 4;
        if base + 3 >= indices.len() {
            break;
        }
        if indices[base + 3] != -1 {
            continue;
        }
        let (i0, i1, i2) = (
            indices[base] as usize,
            indices[base + 1] as usize,
            indices[base + 2] as usize,
        );
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let area = (vertices[i0] - vertices[i1])
            .cross(vertices[i0] - vertices[i2])
            .length();
        if area > 1e-12 {
            valid += 1;
        }
    }
    valid
}
