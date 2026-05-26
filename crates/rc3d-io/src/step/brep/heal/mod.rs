pub mod reorder;
pub mod gap;
pub mod orient;
pub mod seam;
pub mod check;
pub mod connected;

use super::topo::{ShellKey, FaceKey, Orientation};
use super::registry::BRepRegistry;
use reorder::reorder_wire_edges;
use gap::close_wire_gaps;
use orient::fix_shell_orientation;
use seam::fix_missing_seams;
use connected::fix_connected_wire;
pub use check::{check_shell, CheckReport};

#[derive(Debug, Default)]
pub struct HealReport {
    pub reordered_wires: usize,
    pub closed_gaps: usize,
    pub flipped_faces: usize,
    pub added_seams: usize,
    pub merged_vertices: usize,
    pub skip_face_keys: Vec<FaceKey>,
    pub check_errors: usize,
    pub check_warnings: usize,
}

impl HealReport {
    pub fn merge(&mut self, other: HealReport) {
        self.reordered_wires += other.reordered_wires;
        self.closed_gaps += other.closed_gaps;
        self.flipped_faces += other.flipped_faces;
        self.added_seams += other.added_seams;
        self.merged_vertices += other.merged_vertices;
        self.skip_face_keys.extend(other.skip_face_keys);
        self.check_errors += other.check_errors;
        self.check_warnings += other.check_warnings;
    }
}

#[derive(Debug, Clone)]
pub struct HealConfig {
    pub gap_tolerance: f32,
    pub fix_orientation: bool,
    pub fix_reorder: bool,
    pub fix_missing_seams: bool,
    pub fix_connected: bool,
    pub fix_vertex_tolerance: bool,
    pub fix_small_area: bool,
}

impl Default for HealConfig {
    fn default() -> Self {
        Self {
            gap_tolerance: 1e-4,
            fix_orientation: true,
            fix_reorder: true,
            fix_missing_seams: true,
            fix_connected: true,
            fix_vertex_tolerance: true,
            fix_small_area: true,
        }
    }
}

/// Run all healing passes on a shell.
pub fn heal_shell(
    shell_key: ShellKey,
    reg: &mut BRepRegistry,
    config: &HealConfig,
) -> HealReport {
    let mut report = HealReport::default();

    let face_keys: Vec<(FaceKey, Orientation)> = {
        match reg.shells.get(shell_key) {
            Some(s) => s.faces.clone(),
            None => return report,
        }
    };

    if config.fix_connected {
        for (face_key, _) in &face_keys {
            let face = match reg.faces.get(*face_key) {
                Some(f) => f,
                None => continue,
            };
            let cr = fix_connected_wire(face.outer_wire, reg, config.gap_tolerance);
            report.merged_vertices += cr.merged_vertices;
            if cr.merged_vertices > 0 {
                log::debug!(
                    "[BRep heal] FixConnected face {:?}: merged {} verts, {} already connected",
                    face_key, cr.merged_vertices, cr.already_connected
                );
            }
        }
    }

    for (face_key, _) in &face_keys {
        let face = match reg.faces.get(*face_key) {
            Some(f) => f,
            None => continue,
        };

        if config.fix_reorder {
            let wire = match reg.wires.get(face.outer_wire) {
                Some(w) => w,
                None => continue,
            };
            if let Some(reordered) = reorder_wire_edges(&wire.edges, reg) {
                if let Some(w) = reg.wires.get_mut(face.outer_wire) {
                    w.edges = reordered;
                    report.reordered_wires += 1;
                }
            }
        }

        if config.gap_tolerance > 0.0 {
            let closed = close_wire_gaps(face.outer_wire, reg, config.gap_tolerance);
            report.closed_gaps += closed;
        }

        if config.fix_missing_seams {
            report.added_seams += fix_missing_seams(reg, *face_key);
        }
    }

    if config.fix_vertex_tolerance {
        let fixed = fix_vertex_tolerance(reg);
        if fixed > 0 {
            log::debug!("[BRep heal] fixed vertex tolerance on {} edges", fixed);
        }
    }

    if config.fix_small_area {
        report.skip_face_keys = fix_small_area(shell_key, reg);
    }

    if config.fix_orientation {
        report.flipped_faces = fix_shell_orientation(shell_key, reg);
    }

    let check_report = check_shell(shell_key, reg);
    report.check_errors = check_report.errors.len();
    report.check_warnings = check_report.warnings.len();
    for &fk in &check_report.failed_faces {
        if !report.skip_face_keys.contains(&fk) {
            report.skip_face_keys.push(fk);
            log::warn!("[BRep heal] face {:?} failed topology check, skipping mesh", fk);
        }
    }
    for e in &check_report.errors {
        log::warn!("[BRep check] {}", e);
    }
    for w in &check_report.warnings {
        log::debug!("[BRep check] {}", w);
    }

    report
}

fn fix_vertex_tolerance(reg: &mut BRepRegistry) -> usize {
    let mut fixed = 0usize;
    for (_, edge) in reg.edges.iter_mut() {
        let v_lo = reg.vertices.get(edge.v_low).map(|v| v.position);
        let v_hi = reg.vertices.get(edge.v_high).map(|v| v.position);
        if let (Some(p_lo), Some(p_hi)) = (v_lo, v_hi) {
            let curve_lo = edge.curve.d0(0.0);
            let curve_hi = edge.curve.d0(1.0);
            let max_gap = (curve_lo - p_lo).length().max((curve_hi - p_hi).length());
            if max_gap > edge.tolerance {
                edge.tolerance = max_gap * 1.01;
                fixed += 1;
            }
        }
    }
    fixed
}

fn fix_small_area(shell_key: ShellKey, reg: &BRepRegistry) -> Vec<FaceKey> {
    let shell = match reg.shells.get(shell_key) { Some(s) => s, None => return vec![] };
    let mut skip = Vec::new();
    for &(face_key, _) in &shell.faces {
        let face = match reg.faces.get(face_key) { Some(f) => f, None => continue };
        let wire = match reg.wires.get(face.outer_wire) { Some(w) => w, None => { skip.push(face_key); continue; } };
        if wire.edges.is_empty() && face.seam_edges.is_empty() {
            log::warn!("[BRep heal] face {:?} has zero area, marking for skip", face_key);
            skip.push(face_key);
        }
    }
    skip
}
