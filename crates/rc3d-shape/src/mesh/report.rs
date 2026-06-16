//! Shell-level mesh diagnostics (OCC BRepMesh reporting subset).

use rc3d_core::math::{Real, PVec3};

use super::face_uv::UvSource;
use super::t4_quality::DeflectionMetrics;
use super::BRepMeshConfig;
use crate::store::BRepStore;
use crate::topo::{FaceKey, ShellKey};

#[derive(Debug, Clone)]
pub struct FaceMeshStats {
    pub face_key: FaceKey,
    pub tri_count: usize,
    pub first_tri: usize,
    pub uv_source: UvSource,
    pub max_chord_error: Real,
    pub grid_fallback: bool,
    pub cdt_constraint_failures: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ShellMeshReport {
    pub face_count: usize,
    pub meshed_faces: usize,
    pub grid_fallback_count: usize,
    pub cdt_constraint_failure_count: usize,
    pub total_tris: usize,
    pub shell_diag: Real,
    /// Max 3D gap between boundary samples on duplicate EdgeKeys (same vertex pair).
    pub max_equiv_edge_weld_gap: Real,
    pub faces: Vec<FaceMeshStats>,
}

impl ShellMeshReport {
    pub fn grid_fallback_rate(&self) -> Real {
        if self.face_count == 0 {
            0.0
        } else {
            self.grid_fallback_count as Real / self.face_count as Real
        }
    }

    pub fn log_summary(&self, shell_key: ShellKey) {
        log::debug!(
            "[BRep mesh report] shell {:?}: {} faces, {} meshed, {} tris, \
             grid_fallback {}/{} ({:.1}%), diag={:.4}",
            shell_key,
            self.face_count,
            self.meshed_faces,
            self.total_tris,
            self.grid_fallback_count,
            self.face_count,
            self.grid_fallback_rate() * 100.0,
            self.shell_diag,
        );
    }
}

/// Bounding-box diagonal of all vertices referenced by a shell's faces/edges.
pub fn shell_bbox_diagonal(shell_key: ShellKey, reg: &BRepStore) -> Real {
    shell_vertex_bbox(shell_key, reg)
        .map(|(min, max)| (max - min).length())
        .unwrap_or(0.0)
}

/// Axis-aligned bbox (min, max) of BRep vertices on a shell.
pub fn shell_vertex_bbox(shell_key: ShellKey, reg: &BRepStore) -> Option<(PVec3, PVec3)> {
    let shell = reg.shells.get(shell_key)?;
    let mut min = PVec3::splat(f64::MAX);
    let mut max = PVec3::splat(f64::MIN);
    let mut any = false;

    for &(face_key, _) in &shell.faces {
        let face = reg.faces.get(face_key)?;
        let wires = std::iter::once(face.outer_wire)
            .chain(face.inner_wires.iter().copied());
        for wire_key in wires {
            let wire = reg.wires.get(wire_key)?;
            for &(ek, _) in &wire.edges {
                if face.seam_edges.contains(&ek) {
                    continue;
                }
                let edge = reg.edges.get(ek)?;
                for vk in [edge.v_low, edge.v_high] {
                    if let Some(v) = reg.vertices.get(vk) {
                        min = min.min(v.position);
                        max = max.max(v.position);
                        any = true;
                    }
                }
            }
        }
    }

    if any { Some((min, max)) } else { None }
}

/// Scale mesh deflection by shell bbox diagonal (OCC IMeshTools_Parameters::Relative).
pub fn apply_relative_deflection(config: &mut BRepMeshConfig, shell_diag: Real) {
    if config.relative_deflection <= 0.0 || shell_diag <= 0.0 {
        return;
    }
    let rel = shell_diag * config.relative_deflection;
    config.edge.deflection = rel;
    config.face.deflection_interior = rel;
    config.edge.relative_deflection = true;
}

/// T4 deflection from per-face `max_chord_error` (matches `adapt_tris_to_deflection` guarantee).
pub fn deflection_from_report(report: &ShellMeshReport) -> DeflectionMetrics {
    let chords: Vec<Real> = report
        .faces
        .iter()
        .filter(|f| f.tri_count > 0 && !f.grid_fallback)
        .map(|f| f.max_chord_error)
        .collect();
    if chords.is_empty() {
        return DeflectionMetrics::default();
    }
    let mut sorted = chords;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    DeflectionMetrics {
        sample_count: n,
        max: *sorted.last().unwrap_or(&0.0),
        p95: sorted[((n as Real * 0.95) as usize).min(n.saturating_sub(1))],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepShell, BRepSolid, BRepWire, Orientation};

    fn tiny_plane_shell() -> (BRepStore, ShellKey) {
        let mut reg = BRepStore::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: PVec3::ZERO,
                normal: PVec3::Z,
                u_dir: PVec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let edges_data = [
            (PVec3::ZERO, PVec3::new(1.0, 0.0, 0.0)),
            (PVec3::new(1.0, 0.0, 0.0), PVec3::new(1.0, 1.0, 0.0)),
            (PVec3::new(1.0, 1.0, 0.0), PVec3::new(0.0, 1.0, 0.0)),
            (PVec3::new(0.0, 1.0, 0.0), PVec3::ZERO),
        ];
        let mut wire_edges = Vec::new();
        for (a, b) in edges_data {
            let v0 = reg.find_or_add_vertex(a, 1e-4);
            let v1 = reg.find_or_add_vertex(b, 1e-4);
            let curve_3d = CurveGeom::Line { origin: a, direction: b - a };
            let pcurve = Curve2d::Line { origin: (a.x, a.y), direction: (b.x - a.x, b.y - a.y) };
            let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve, true);
            wire_edges.push((ek, Orientation::Forward));
        }
        let outer_wire = reg.wires.insert(BRepWire { edges: wire_edges });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer_wire;
        }
        let shell_key = reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let _ = reg.solids.insert(BRepSolid {
            outer_shell: shell_key,
            void_shells: vec![],
        });
        (reg, shell_key)
    }

    #[test]
    fn relative_deflection_scales_with_shell_diag() {
        let (reg, shell_key) = tiny_plane_shell();
        let diag = shell_bbox_diagonal(shell_key, &reg);
        assert!(diag > 0.0 && diag < 2.0, "expected ~sqrt(2), got {diag}");

        let mut config = BRepMeshConfig::default();
        config.relative_deflection = 0.01;
        apply_relative_deflection(&mut config, diag);
        let expected = diag * 0.01;
        assert!(
            (config.edge.deflection - expected).abs() < 1e-6,
            "edge deflection should equal shell_diag * factor"
        );
        assert!(config.edge.relative_deflection);
    }

    #[test]
    fn relative_deflection_shell_and_edge_mode() {
        let (reg_small, sk_small) = tiny_plane_shell();
        let diag_small = shell_bbox_diagonal(sk_small, &reg_small);
        let mut cfg_small = BRepMeshConfig::default();
        cfg_small.relative_deflection = 0.005;
        apply_relative_deflection(&mut cfg_small, diag_small);

        let _reg_large = BRepStore::new();
        let mut cfg_large = BRepMeshConfig::default();
        cfg_large.relative_deflection = 0.005;
        apply_relative_deflection(&mut cfg_large, diag_small * 10.0);

        assert!(
            cfg_large.edge.deflection > cfg_small.edge.deflection,
            "larger shell should yield larger absolute deflection"
        );
    }
}
