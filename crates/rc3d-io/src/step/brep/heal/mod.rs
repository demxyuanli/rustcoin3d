pub mod reorder;
pub mod gap;
pub mod orient;

use super::topo::{ShellKey, FaceKey, Orientation};
use super::registry::BRepRegistry;
use reorder::reorder_wire_edges;
use gap::close_wire_gaps;
use orient::fix_shell_orientation;

#[derive(Debug, Default)]
pub struct HealReport {
    pub reordered_wires: usize,
    pub closed_gaps: usize,
    pub flipped_faces: usize,
}

impl HealReport {
    pub fn merge(&mut self, other: HealReport) {
        self.reordered_wires += other.reordered_wires;
        self.closed_gaps += other.closed_gaps;
        self.flipped_faces += other.flipped_faces;
    }
}

#[derive(Debug, Clone)]
pub struct HealConfig {
    pub gap_tolerance: f32,
    pub fix_orientation: bool,
    pub fix_reorder: bool,
}

impl Default for HealConfig {
    fn default() -> Self {
        Self { gap_tolerance: 1e-4, fix_orientation: true, fix_reorder: true }
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
    }

    if config.fix_orientation {
        report.flipped_faces = fix_shell_orientation(shell_key, reg);
    }

    report
}
