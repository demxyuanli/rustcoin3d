use rc3d_core::math::Vec3;
use crate::geom::SurfaceGeom;
use super::edge_disc::EdgeDiscConfig;
use super::face_fill::FaceFillConfig;
use super::optimize::OptimizeConfig;
use super::refiner::RefineConfig;

/// UV grid resolution fallback when deflection config is unavailable.
pub const MESH_CLOSED_SURFACE_SEGS: u32 = 48;

// ── Tiered tessellation policy ───────────────────────────────────────

/// Tessellation quality tier controlling the entire mesh pipeline.
///
/// Maps to OCC's meshing parameters + fallback strategy selection.
/// `StepImportMode::Strict` → `Precision`; `Preview` → `Preview`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TessellationTier {
    /// Fast preview: bounded fallbacks, no self-intersect output, faces may be skipped.
    Preview,
    /// Default engineering visualization: OCC-quality CDT with retries.
    Standard,
    /// Strict mesh contract: no heuristic fill, explicit failure on quality violation.
    Precision,
}

/// Deflection strategy for edge and interior sampling.
#[derive(Debug, Clone)]
pub enum DeflectionPolicy {
    /// Absolute chord height in world units.
    Absolute(f32),
    /// Relative to shell bounding-box diagonal (OCC `isRelative` mode).
    Relative(f32),
}

impl Default for DeflectionPolicy {
    fn default() -> Self {
        DeflectionPolicy::Absolute(0.01)
    }
}

/// Which mesh strategies are allowed as fallbacks.
///
/// `plane_center_fan` and `surface_fill_3d` are globally disabled
/// (they do not respect boundary constraints).
#[derive(Debug, Clone)]
pub struct FallbackAllowlist {
    pub trimmed_cdt: bool,
    pub closed_parametric: bool,
    pub ruled_strip: bool,
    pub parametric_grid: bool,
    pub plane_center_fan: bool,
    pub surface_fill_3d: bool,
}

impl Default for FallbackAllowlist {
    fn default() -> Self {
        Self {
            trimmed_cdt: true,
            closed_parametric: true,
            ruled_strip: true,
            parametric_grid: true,
            plane_center_fan: false,  // globally disabled — no boundary respect
            surface_fill_3d: false,   // globally disabled — drops holes
        }
    }
}

/// Per-face quality checks applied after meshing, before shell merge.
#[derive(Debug, Clone)]
pub struct MeshQualityGate {
    /// Max fraction of degenerate (zero-area) triangles before action.
    pub max_degenerate_rate: f32,
    /// Max triangle aspect ratio (longest/shortest edge).
    pub max_aspect_ratio: f32,
    /// Chord error multiplier over deflection; exceeding triggers retry or skip.
    pub chord_error_factor: f32,
}

impl Default for MeshQualityGate {
    fn default() -> Self {
        Self {
            max_degenerate_rate: 0.10,
            max_aspect_ratio: 20.0,
            chord_error_factor: 2.0,
        }
    }
}

/// Parallel tessellation policy.
#[derive(Debug, Clone)]
pub struct ParallelPolicy {
    pub enabled: bool,
    pub chunk_size: usize,
}

impl Default for ParallelPolicy {
    fn default() -> Self {
        Self { enabled: false, chunk_size: 64 }
    }
}

/// Complete tessellation policy — the single parameter source for S6.
///
/// Derived from `TessellationTier` via `for_tier()`; feeds into
/// `BRepMeshConfig`, `HealPolicy`, and the face strategy chain.
#[derive(Debug, Clone)]
pub struct TessellationPolicy {
    pub tier: TessellationTier,
    pub deflection: DeflectionPolicy,
    pub fallback: FallbackAllowlist,
    pub quality_gate: MeshQualityGate,
    pub parallel: ParallelPolicy,
}

impl TessellationPolicy {
    /// Create a policy preset for the given tier.
    pub fn for_tier(tier: TessellationTier) -> Self {
        let base = Self::default();
        match tier {
            TessellationTier::Preview => Self {
                tier,
                deflection: DeflectionPolicy::Absolute(0.02),
                quality_gate: MeshQualityGate {
                    max_degenerate_rate: 0.15,
                    ..base.quality_gate
                },
                ..base
            },
            TessellationTier::Standard => base,
            TessellationTier::Precision => Self {
                tier,
                deflection: DeflectionPolicy::Relative(0.001),
                fallback: FallbackAllowlist {
                    ruled_strip: false,
                    parametric_grid: false,
                    ..base.fallback
                },
                quality_gate: MeshQualityGate {
                    max_degenerate_rate: 0.01,
                    max_aspect_ratio: 10.0,
                    chord_error_factor: 1.5,
                },
                ..base
            },
        }
    }

    /// Derive a `BRepMeshConfig` from this policy.
    ///
    /// Applies `DeflectionPolicy::Relative(r)` as `relative_deflection`;
    /// `Absolute(d)` is applied via the calling code's deflection parameter.
    /// Fallback allowlist and quality gate are enforced at the strategy-chain level.
    pub fn to_mesh_config(&self) -> BRepMeshConfig {
        let mut cfg = match self.tier {
            TessellationTier::Preview => BRepMeshConfig::preview(),
            TessellationTier::Standard => BRepMeshConfig::default(),
            TessellationTier::Precision => BRepMeshConfig::quality(),
        };
        // Apply deflection policy — Relative overrides the config.
        if let DeflectionPolicy::Relative(r) = self.deflection {
            cfg.relative_deflection = r;
        }
        // Propagate tier-specific fallback policy
        cfg.fallback = self.fallback.clone();
        cfg
    }
}

impl Default for TessellationPolicy {
    fn default() -> Self {
        Self {
            tier: TessellationTier::Standard,
            deflection: DeflectionPolicy::default(),
            fallback: FallbackAllowlist::default(),
            quality_gate: MeshQualityGate::default(),
            parallel: ParallelPolicy::default(),
        }
    }
}

pub fn min_adequate_trim_tris(surface: &SurfaceGeom, _wire_edge_count: usize) -> usize {
    match surface {
        SurfaceGeom::Offset { basis, .. }
            if matches!(basis.as_ref(), SurfaceGeom::Plane { .. }) =>
        {
            8
        }
        SurfaceGeom::Revolution { .. } | SurfaceGeom::BSpline(_) | SurfaceGeom::Offset { .. } => 64,
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
    /// Which fallback strategies are allowed when primary algorithm fails.
    /// Default disables plane_center_fan and surface_fill_3d (他們不尊重边界约束).
    pub fallback: FallbackAllowlist,
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
            fallback: FallbackAllowlist::default(),
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
        cfg.face.max_adapt_iterations = 1;
        cfg.face.parameter_division_max_depth = 3;
        cfg.face.skip_interior_edge_split = true;
        cfg
    }

    /// Production-quality tessellation: relative deflection, chord retries, Steiner refinement.
    pub fn quality() -> Self {
        let mut cfg = Self::default();
        cfg.relative_deflection = 0.001;
        cfg.refine.enable_post_refine = true;
        cfg.optimize.max_iterations = 3;
        cfg.fast_export = false;
        cfg.face.max_adapt_iterations = 4;
        cfg.face.parameter_division_max_depth = 6;
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
