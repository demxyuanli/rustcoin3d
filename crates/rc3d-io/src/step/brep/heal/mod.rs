pub mod reorder;
pub mod gap;
pub mod orient;

use super::topo::ShellKey;
use super::registry::BRepRegistry;

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

pub fn heal_shell(_shell: &mut super::topo::BRepShell, _reg: &mut BRepRegistry, _config: &HealConfig) -> HealReport {
    HealReport::default() // TODO: implement after T3.x
}
