pub mod reorder;
pub mod gap;
pub mod orient;
pub mod seam;
pub mod check;

use super::topo::{ShellKey, FaceKey, Orientation};
use super::registry::BRepRegistry;
use reorder::reorder_wire_edges;
use gap::close_wire_gaps;
use orient::fix_shell_orientation;
use seam::fix_missing_seams;
pub use check::{check_shell, CheckReport};

#[derive(Debug, Default)]
pub struct HealReport {
    pub reordered_wires: usize,
    pub closed_gaps: usize,
    pub flipped_faces: usize,
    pub added_seams: usize,
}

impl HealReport {
    pub fn merge(&mut self, other: HealReport) {
        self.reordered_wires += other.reordered_wires;
        self.closed_gaps += other.closed_gaps;
        self.flipped_faces += other.flipped_faces;
        self.added_seams += other.added_seams;
    }
}

#[derive(Debug, Clone)]
pub struct HealConfig {
    pub gap_tolerance: f32,
    pub fix_orientation: bool,
    pub fix_reorder: bool,
    pub fix_missing_seams: bool,
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

    if config.fix_orientation {
        report.flipped_faces = fix_shell_orientation(shell_key, reg);
    }

    let check_report = check_shell(shell_key, reg);
    for e in &check_report.errors {
        log::warn!("[BRep check] {}", e);
    }
    for w in &check_report.warnings {
        log::debug!("[BRep check] {}", w);
    }

    report
}
