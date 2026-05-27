//! Iterative auto-heal pipeline (OCC ShapeHealing equivalent).
//! Runs heal passes, checks results, re-applies fixes until converged.

use super::super::topo::ShellKey;
use super::super::registry::BRepRegistry;
use super::{CheckReport, HealConfig, HealReport, check_shell, heal_shell};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

/// Run iterative auto-heal with adaptive fix selection.
/// First iteration runs foundational fixes (connected, small, reorder, orientation).
/// Subsequent iterations selectively enable fixes based on check results.
pub fn auto_heal_shell(
    shell_key: ShellKey,
    reg: &mut BRepRegistry,
    level: HealLevel,
    max_iterations: usize,
) -> HealReport {
    let mut total_report = HealReport::default();

    // Baseline check
    let check_before = check_shell(shell_key, reg);
    let mut prev_errors = check_before.errors.len();
    let mut prev_warnings = check_before.warnings.len();

    log::debug!(
        "[BRep pipeline] level={:?}, baseline: {} errors, {} warnings",
        level, prev_errors, prev_warnings
    );

    for iter in 0..max_iterations {
        let config = select_fixes(level, iter, &check_before);
        let hr = heal_shell(shell_key, reg, &config);
        let curr_errors = hr.check_errors;
        let curr_warnings = hr.check_warnings;
        total_report.merge(hr);
        total_report.check_errors = curr_errors;
        total_report.check_warnings = curr_warnings;

        log::debug!(
            "[BRep pipeline] iter {}: errors {}→{}, warnings {}→{}",
            iter + 1, prev_errors, curr_errors, prev_warnings, curr_warnings,
        );

        if curr_errors == prev_errors && curr_warnings == prev_warnings {
            log::debug!("[BRep pipeline] converged after {} iteration(s)", iter + 1);
            break;
        }
        prev_errors = curr_errors;
        prev_warnings = curr_warnings;
    }

    total_report
}

/// Select which fixes to apply based on heal level and iteration.
/// First iteration: foundational fixes only (connected, small, reorder, orientation).
/// Standard+: enable UV fixes if UV gaps detected.
/// Advanced: enable topology fixes if self-intersection or degeneracies detected.
fn select_fixes(level: HealLevel, _iteration: usize, check: &CheckReport) -> HealConfig {
    let mut config = HealConfig::default();
    // Start with everything disabled
    config.fix_connected = false;
    config.fix_small_edges = false;
    config.fix_reorder = false;
    config.fix_orientation = false;
    config.fix_missing_seams = false;
    config.fix_shifted = false;
    config.fix_edge_curves = false;
    config.fix_lacking = false;
    config.fix_degenerated = false;
    config.fix_self_intersection = false;
    config.fix_intersecting_wires = false;
    config.fix_periodic_degenerated = false;
    config.fix_vertex_tolerance = false;
    config.fix_small_area = false;
    config.fix_vertex_position = false;
    config.uv_gap_tolerance = 0.0;

    // Iteration 0: foundational fixes always run first
    config.fix_connected = true;
    config.fix_small_edges = true;
    config.fix_reorder = true;
    config.fix_orientation = true;
    config.fix_edge_curves = true;

    if level >= HealLevel::Standard {
        config.fix_missing_seams = true;
        config.uv_gap_tolerance = 1e-5;
        // Only enable shifted/lacking if check detected issues
        if has_uv_issues(check) {
            config.fix_shifted = true;
        }
        config.fix_lacking = true;
        config.fix_vertex_tolerance = true;
        config.fix_small_area = true;
    }

    if level >= HealLevel::Advanced {
        if has_self_intersections(check) {
            config.fix_self_intersection = true;
        }
        if has_singularities(check) {
            config.fix_degenerated = true;
            config.fix_periodic_degenerated = true;
        }
        if has_multiple_wires(check) {
            config.fix_intersecting_wires = true;
        }
    }

    config
}

fn has_uv_issues(check: &CheckReport) -> bool {
    check.warnings.iter().any(|w| w.contains("UV") || w.contains("PCurve") || w.contains("parameter"))
}

fn has_self_intersections(check: &CheckReport) -> bool {
    check.warnings.iter().any(|w| w.contains("self-intersection"))
}

fn has_singularities(check: &CheckReport) -> bool {
    check.warnings.iter().any(|w| w.contains("degeneracy") || w.contains("singularity"))
}

fn has_multiple_wires(check: &CheckReport) -> bool {
    check.warnings.iter().any(|w| w.contains("inner wire") || w.contains("intersecting"))
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
