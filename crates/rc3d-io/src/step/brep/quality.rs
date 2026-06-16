//! Quality metrics for STEP import pipeline (OCC quality baseline).

use rc3d_core::math::Real;
use std::collections::HashSet;

use rc3d_shape::BRepStore;
use rc3d_shape::topo::ShellKey;

/// Quality metrics for a single shell.
#[derive(Debug, Clone, Default)]
pub struct ShellQualityMetrics {
    pub face_count: usize,
    pub edge_count: usize,
    pub vertex_count: usize,
    /// Euler characteristic (2 for closed genus-0).
    pub euler_characteristic: Option<i32>,
    /// Number of faces that skipped meshing.
    pub skipped_faces: usize,
    /// Maximum chord error (normalized by bbox diagonal).
    pub max_chord_error: Real,
    /// Average chord error.
    pub avg_chord_error: Real,
    /// Number of degenerate faces (zero-area).
    pub degenerate_faces: usize,
    /// Whether shell passed Euler-Poincaré check.
    pub euler_valid: bool,
}

/// Compute quality metrics for a shell.
pub fn compute_shell_quality(
    shell_key: ShellKey,
    reg: &BRepStore,
) -> ShellQualityMetrics {
    let mut metrics = ShellQualityMetrics::default();

    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return metrics,
    };

    metrics.face_count = shell.faces.len();

    // Count unique edges and vertices
    let mut edge_set: HashSet<rc3d_shape::topo::EdgeKey> = HashSet::new();
    let mut vertex_set: HashSet<rc3d_shape::topo::VertexKey> = HashSet::new();
    for &(fk, _) in &shell.faces {
        if let Some(face) = reg.faces.get(fk) {
            if let Some(wire) = reg.wires.get(face.outer_wire) {
                for &(ek, _) in &wire.edges {
                    edge_set.insert(ek);
                    if let Some(edge) = reg.edges.get(ek) {
                        vertex_set.insert(edge.v_low);
                        vertex_set.insert(edge.v_high);
                    }
                }
            }
        }
    }
    metrics.edge_count = edge_set.len();
    metrics.vertex_count = vertex_set.len();

    // Euler-Poincaré: χ = V - E + F
    let v = metrics.vertex_count as i32;
    let e = metrics.edge_count as i32;
    let f = metrics.face_count as i32;
    let chi = v - e + f;
    metrics.euler_characteristic = Some(chi);
    metrics.euler_valid = chi == 2 || chi == 0; // genus 0 closed or torus-like

    metrics
}

/// Pipeline-level quality summary.
#[derive(Debug, Default)]
pub struct PipelineQualityReport {
    pub parse_success: bool,
    pub shells_total: usize,
    pub shells_valid: usize,
    pub faces_total: usize,
    pub faces_meshed: usize,
    pub heal_passes_run: usize,
    pub heal_errors: usize,
    pub avg_quality_score: Real,
}

/// Compute a 0.0–1.0 quality score from shell metrics.
/// 1.0 = perfect (Euler valid, no degenerate faces, low chord error).
pub fn quality_score(metrics: &ShellQualityMetrics) -> Real {
    let mut score = 1.0_f64;

    // Euler validity
    if !metrics.euler_valid {
        score -= 0.3;
    }

    // Degenerate face penalty
    if metrics.face_count > 0 {
        let degen_ratio = metrics.degenerate_faces as Real / metrics.face_count as Real;
        score -= degen_ratio * 0.3;
    }

    // Chord error penalty (normalized, assume < 0.01 is good)
    if metrics.max_chord_error > 0.01 {
        score -= ((metrics.max_chord_error - 0.01) * 10.0).min(0.4);
    }

    score.max(0.0)
}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_shape::BRepStore;
    use rc3d_shape::topo::*;

    #[test]
    fn test_quality_empty_registry() {
        let reg = BRepStore::new();
        let metrics = compute_shell_quality(ShellKey::default(), &reg);
        assert_eq!(metrics.face_count, 0);
        assert!(!metrics.euler_valid);
    }

    #[test]
    fn test_quality_single_face() {
        let mut reg = BRepStore::new();
        // Create a simple triangular face
        let v0 = reg.vertices.insert(BRepVertex { position: rc3d_core::math::PVec3::ZERO, tolerance: 1e-6 });
        let v1 = reg.vertices.insert(BRepVertex { position: rc3d_core::math::PVec3::X, tolerance: 1e-6 });
        let v2 = reg.vertices.insert(BRepVertex { position: rc3d_core::math::PVec3::Y, tolerance: 1e-6 });

        let e0 = reg.edges.insert(BRepEdge {
            v_low: v0, v_high: v1,
            curve: crate::step::brep::geom::CurveGeom::Line {
                origin: rc3d_core::math::PVec3::ZERO,
                direction: rc3d_core::math::PVec3::X,
            },
            tolerance: 1e-6,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: Default::default(),
        });
        let e1 = reg.edges.insert(BRepEdge {
            v_low: v1, v_high: v2,
            curve: crate::step::brep::geom::CurveGeom::Line {
                origin: rc3d_core::math::PVec3::X,
                direction: rc3d_core::math::PVec3::Y - rc3d_core::math::PVec3::X,
            },
            tolerance: 1e-6,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: Default::default(),
        });
        let e2 = reg.edges.insert(BRepEdge {
            v_low: v2, v_high: v0,
            curve: crate::step::brep::geom::CurveGeom::Line {
                origin: rc3d_core::math::PVec3::Y,
                direction: -rc3d_core::math::PVec3::Y,
            },
            tolerance: 1e-6,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: Default::default(),
        });

        let wire = reg.wires.insert(BRepWire {
            edges: vec![
                (e0, Orientation::Forward),
                (e1, Orientation::Forward),
                (e2, Orientation::Forward),
            ],
        });

        let face = reg.faces.insert(BRepFace {
            surface: crate::step::brep::geom::SurfaceGeom::Plane {
                origin: rc3d_core::math::PVec3::ZERO,
                normal: rc3d_core::math::PVec3::Z,
                u_dir: rc3d_core::math::PVec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            seam_edges: vec![],
            same_sense: true,
            tolerance: 1e-6,
            color: None,
            degenerated_edges: vec![],
        });

        let shell = reg.shells.insert(BRepShell {
            faces: vec![(face, Orientation::Forward)],
            closed: false,
            step_id: None,
        });

        let metrics = compute_shell_quality(shell, &reg);
        assert_eq!(metrics.face_count, 1);
        assert_eq!(metrics.edge_count, 3);
        assert_eq!(metrics.vertex_count, 3);
        // χ = 3 - 3 + 1 = 1 (open shell, not genus-0 closed)
        assert_eq!(metrics.euler_characteristic, Some(1));
    }

    #[test]
    fn test_quality_score_perfect() {
        let m = ShellQualityMetrics {
            face_count: 10,
            edge_count: 15,
            vertex_count: 7,
            euler_characteristic: Some(2),
            euler_valid: true,
            degenerate_faces: 0,
            max_chord_error: 0.001,
            ..Default::default()
        };
        let s = quality_score(&m);
        assert!(s > 0.9, "expected high quality score, got {s}");
    }

    #[test]
    fn test_quality_score_bad_euler() {
        let m = ShellQualityMetrics {
            euler_valid: false,
            ..Default::default()
        };
        let s = quality_score(&m);
        assert!(s < 0.8, "expected penalized score, got {s}");
    }
}
