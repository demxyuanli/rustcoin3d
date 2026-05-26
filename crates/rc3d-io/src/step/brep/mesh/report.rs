//! Shell-level mesh diagnostics (OCC BRepMesh reporting subset).

use rc3d_core::math::Vec3;

use super::face_uv::UvSource;
use super::t4_quality::DeflectionMetrics;
use super::BRepMeshConfig;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{FaceKey, ShellKey};

#[derive(Debug, Clone)]
pub struct FaceMeshStats {
    pub face_key: FaceKey,
    pub tri_count: usize,
    pub uv_source: UvSource,
    pub max_chord_error: f32,
    pub grid_fallback: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ShellMeshReport {
    pub face_count: usize,
    pub meshed_faces: usize,
    pub grid_fallback_count: usize,
    pub total_tris: usize,
    pub shell_diag: f32,
    pub faces: Vec<FaceMeshStats>,
}

impl ShellMeshReport {
    pub fn grid_fallback_rate(&self) -> f32 {
        if self.face_count == 0 {
            0.0
        } else {
            self.grid_fallback_count as f32 / self.face_count as f32
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
pub fn shell_bbox_diagonal(shell_key: ShellKey, reg: &BRepRegistry) -> f32 {
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return 0.0,
    };

    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut any = false;

    for &(face_key, _) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wires = std::iter::once(face.outer_wire)
            .chain(face.inner_wires.iter().copied());
        for wire_key in wires {
            let wire = match reg.wires.get(wire_key) {
                Some(w) => w,
                None => continue,
            };
            for &(ek, _) in &wire.edges {
                let edge = match reg.edges.get(ek) {
                    Some(e) => e,
                    None => continue,
                };
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

    if any {
        (max - min).length()
    } else {
        0.0
    }
}

/// Scale mesh deflection by shell bbox diagonal (OCC IMeshTools_Parameters::Relative).
pub fn apply_relative_deflection(config: &mut BRepMeshConfig, shell_diag: f32) {
    if config.relative_deflection <= 0.0 || shell_diag <= 0.0 {
        return;
    }
    let rel = shell_diag * config.relative_deflection;
    config.edge.deflection = config.edge.deflection.min(rel);
    config.face.deflection_interior = config.face.deflection_interior.min(rel);
}

/// T4 deflection from per-face `max_chord_error` (matches `adapt_tris_to_deflection` guarantee).
pub fn deflection_from_report(report: &ShellMeshReport) -> DeflectionMetrics {
    let chords: Vec<f32> = report
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
        p95: sorted[((n as f32 * 0.95) as usize).min(n.saturating_sub(1))],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::topo::{BRepFace, BRepShell, BRepSolid, BRepWire, Orientation};

    fn tiny_plane_shell() -> (BRepRegistry, ShellKey) {
        let mut reg = BRepRegistry::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
        });
        let edges_data = [
            (Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)),
            (Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
            (Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO),
        ];
        let mut wire_edges = Vec::new();
        for (a, b) in edges_data {
            let v0 = reg.find_or_add_vertex(a, 1e-4);
            let v1 = reg.find_or_add_vertex(b, 1e-4);
            let curve_3d = CurveGeom::Line { origin: a, direction: b - a };
            let pcurve = CurveGeom::Line {
                origin: Vec3::new(a.x, a.y, 0.0),
                direction: Vec3::new(b.x - a.x, b.y - a.y, 0.0),
            };
            let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve);
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
    fn relative_deflection_scales_down_for_small_shell() {
        let (reg, shell_key) = tiny_plane_shell();
        let diag = shell_bbox_diagonal(shell_key, &reg);
        assert!(diag > 0.0 && diag < 2.0, "expected ~sqrt(2), got {diag}");

        let mut config = BRepMeshConfig::default();
        config.relative_deflection = 0.01;
        let base_edge = config.edge.deflection;
        apply_relative_deflection(&mut config, diag);
        assert!(
            config.edge.deflection <= base_edge,
            "small shell relative deflection should not increase deflection: {} vs {}",
            config.edge.deflection,
            base_edge,
        );
    }
}
