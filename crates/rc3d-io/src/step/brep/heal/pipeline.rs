//! Iterative auto-heal pipeline (OCC ShapeHealing equivalent).
//! Runs heal passes, checks results, re-applies fixes until converged.

use super::super::topo::ShellKey;
use super::super::registry::BRepRegistry;
use super::{HealConfig, HealReport, heal_shell, check_shell};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealLevel {
    /// FixConnected + FixSmall + GapClose3d + FixOrientation only
    Basic,
    /// Basic + FixGaps2d + FixShifted + FixEdgeCurves + FixMissingSeams
    Standard,
    /// Standard + FixSelfIntersection + FixDegenerated + FixIntersectingWires + FixPeriodic
    Advanced,
}

impl Default for HealLevel {
    fn default() -> Self { HealLevel::Standard }
}

/// Run iterative auto-heal on a shell.
/// Each iteration: heal → check → if no improvement → converged.
/// Returns the accumulated HealReport for backward compatibility.
/// Pipeline diagnostics are logged at debug level.
pub fn auto_heal_shell(
    shell_key: ShellKey,
    reg: &mut BRepRegistry,
    level: HealLevel,
    max_iterations: usize,
) -> HealReport {
    let mut config = HealConfig::default();
    apply_level(&mut config, level);

    // Baseline check
    let check_before = check_shell(shell_key, reg);
    let check_before_count = check_before.errors.len() + check_before.warnings.len();

    let mut total_report = HealReport::default();
    let mut prev_errors = check_before.errors.len();
    let mut prev_warnings = check_before.warnings.len();

    log::debug!(
        "[BRep pipeline] level={:?}, baseline: {} errors, {} warnings",
        level, prev_errors, prev_warnings
    );

    for iter in 0..max_iterations {
        let hr = heal_shell(shell_key, reg, &config);
        total_report.merge(hr);

        let check_after = check_shell(shell_key, reg);
        let curr_errors = check_after.errors.len();
        let curr_warnings = check_after.warnings.len();

        log::debug!(
            "[BRep pipeline] iter {}: errors {}→{}, warnings {}→{} ({} fixes)",
            iter + 1,
            prev_errors, curr_errors,
            prev_warnings, curr_warnings,
            total_report.total_fix_count(),
        );

        total_report.check_errors = curr_errors;
        total_report.check_warnings = curr_warnings;

        if curr_errors == prev_errors && curr_warnings == prev_warnings {
            log::debug!(
                "[BRep pipeline] converged after {} iteration(s), check delta: {}→{}",
                iter + 1,
                check_before_count,
                curr_errors + curr_warnings,
            );
            break;
        }

        prev_errors = curr_errors;
        prev_warnings = curr_warnings;
    }

    total_report
}

impl HealReport {
    /// Sum of all categories to give a quick total-fixes count for pipeline diagnostics.
    fn total_fix_count(&self) -> usize {
        self.reordered_wires
            + self.closed_gaps
            + self.closed_uv_gaps
            + self.flipped_faces
            + self.added_seams
            + self.merged_vertices
            + self.removed_small_edges
            + self.shifted_pcurves
            + self.adjusted_edge_curves
            + self.lacking_tolerance_fixes
            + self.degenerate_edges_created
            + self.periodic_degen_created
            + self.self_intersections_fixed
            + self.inner_wires_fixed
            + self.vertex_positions_fixed
    }
}

fn apply_level(config: &mut HealConfig, level: HealLevel) {
    match level {
        HealLevel::Basic => {
            config.fix_connected = true;
            config.fix_small_edges = true;
            config.fix_reorder = true;
            config.fix_orientation = true;
            config.fix_missing_seams = false;
            config.fix_shifted = false;
            config.fix_edge_curves = false;
            config.fix_lacking = false;
            config.fix_degenerated = false;
            config.fix_self_intersection = false;
            config.fix_intersecting_wires = false;
            config.fix_periodic_degenerated = false;
            config.uv_gap_tolerance = 0.0;
        }
        HealLevel::Standard => {
            // Standard = Basic + extended gap/edge checks, but no self-intersect/degen fixes.
            config.fix_connected = true;
            config.fix_small_edges = true;
            config.fix_reorder = true;
            config.fix_orientation = true;
            config.fix_missing_seams = true;
            config.fix_shifted = true;
            config.fix_edge_curves = true;
            config.fix_lacking = true;
            config.fix_degenerated = false;
            config.fix_self_intersection = false;
            config.fix_intersecting_wires = false;
            config.fix_periodic_degenerated = false;
            // uv_gap_tolerance stays at default (1e-5)
        }
        HealLevel::Advanced => {
            // Advanced = Standard + self-intersection, degenerated, intersecting wires, periodic.
            config.fix_connected = true;
            config.fix_small_edges = true;
            config.fix_reorder = true;
            config.fix_orientation = true;
            config.fix_missing_seams = true;
            config.fix_shifted = true;
            config.fix_edge_curves = true;
            config.fix_lacking = true;
            config.fix_degenerated = true;
            config.fix_self_intersection = true;
            config.fix_intersecting_wires = true;
            config.fix_periodic_degenerated = true;
        }
    }
}
