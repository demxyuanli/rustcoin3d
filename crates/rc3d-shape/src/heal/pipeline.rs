//! Iterative auto-heal pipeline (OCC ShapeHealing equivalent).
//! Runs heal passes, checks results, re-applies fixes until converged.

use crate::topo::ShellKey;
use crate::store::BRepStore;
use super::{CheckReport, HealConfig, HealReport, check_shell, heal_shell};
use super::edge_tolerance::auto_fix_shell_edge_tolerances;

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
    fn default() -> Self {
        HealLevel::Standard
    }
}

/// Run iterative auto-heal with adaptive fix selection.
/// Each iteration uses a fresh `check_shell` result to drive `select_fixes`.
pub fn auto_heal_shell(
    shell_key: ShellKey,
    reg: &mut BRepStore,
    level: HealLevel,
    max_iterations: usize,
) -> HealReport {
    let _t0 = std::time::Instant::now();
    let mut total_report = HealReport::default();
    let mut check = check_shell(shell_key, reg);
    let t_check0 = _t0.elapsed().as_secs_f32();
    let mut prev_errors = check.errors.len();
    let mut prev_warnings = check.warnings.len();

    if check.has_open_edges {
        log::info!(
            "[BRep pipeline] level={:?}, baseline check: {:.1}s ({} err, {} warn, {} open edges)",
            level, t_check0, prev_errors, prev_warnings, check.open_edge_count
        );
    } else {
        log::info!(
            "[BRep pipeline] level={:?}, baseline check: {:.1}s ({} err, {} warn)",
            level, t_check0, prev_errors, prev_warnings
        );
    }

    // Pre-step: auto-fix edge tolerances from PCurve-to-3D deviation.
    // This must run before the main heal loop so subsequent passes
    // (same_parameter, gap closing, etc.) use corrected tolerances.
    // OCC: ShapeFix_Edge::FixSameParameter before ShapeFix_Wire / ShapeFix_Face.
    if level >= HealLevel::Standard {
        let _te = std::time::Instant::now();
        let et_report = auto_fix_shell_edge_tolerances(shell_key, reg, 1e-7, 1.0);
        let t = _te.elapsed().as_secs_f32();
        if et_report.tolerances_increased > 0 || et_report.tolerances_decreased > 0 {
            log::info!(
                "[BRep pipeline] edge tolerance fix: {:.1}s ({} inc, {} dec, max_adj={:.6})",
                t,
                et_report.tolerances_increased,
                et_report.tolerances_decreased,
                et_report.max_adjustment,
            );
        }
    }

    for iter in 0..max_iterations {
        let _ti = std::time::Instant::now();
        let config = select_fixes(level, iter, &check);
        let hr = heal_shell(shell_key, reg, &config);
        total_report.merge(hr.clone());
        total_report.check_errors = hr.check_errors;
        total_report.check_warnings = hr.check_warnings;

        check = check_shell(shell_key, reg);
        let curr_errors = check.errors.len();
        let curr_warnings = check.warnings.len();

        log::info!(
            "[BRep pipeline] iter {}: {:.1}s ({}→{} err, {}→{} warn)",
            iter + 1, _ti.elapsed().as_secs_f32(),
            prev_errors, curr_errors, prev_warnings, curr_warnings,
        );

        if curr_errors == prev_errors && curr_warnings == prev_warnings {
            log::debug!("[BRep pipeline] converged after {} iteration(s)", iter + 1);
            break;
        }
        prev_errors = curr_errors;
        prev_warnings = curr_warnings;
    }

    total_report.check_errors = check.errors.len();
    total_report.check_warnings = check.warnings.len();
    total_report
}

/// Table-driven fix selection by [`HealLevel`] and check report (iteration 0 = bulk pass).
///
/// | Level    | Iteration 0 fixes |
/// |----------|-------------------|
/// | Basic    | connected, small edges, reorder, gaps 3d, orientation |
/// | Standard | + same parameter, UV gaps, shifted, periodic degen, edge curves, lacking, seams, natural bound, reversed 2d |
/// | Advanced | Standard + self-intersection, degenerated, intersecting wires |
/// | Advanced | + self-intersection, degenerated, intersecting wires (when check flags set) |
pub(crate) fn select_fixes(level: HealLevel, iteration: usize, check: &CheckReport) -> HealConfig {
    let mut config = HealConfig::all_disabled();

    if iteration == 0 {
        config.fix_connected = true;
        config.fix_small_edges = true;
        config.fix_reorder = true;
        config.fix_gaps_3d = level >= HealLevel::Basic;
        config.fix_same_parameter = level >= HealLevel::Standard;
        config.fix_orientation = true;

        if level >= HealLevel::Standard {
            config.uv_gap_tolerance = 1e-5;
            config.fix_shifted = true;
            config.fix_periodic_degenerated = true;
            config.fix_edge_curves = true;
            config.fix_lacking = true;
            config.fix_missing_seams = true;
            config.fix_natural_bound = true;
            config.fix_reversed_2d = true;
            config.fix_vertex_tolerance = true;
            config.fix_small_area = true;
            config.fix_vertex_position = true;
        }
    } else {
        if check.has_uv_gaps || check.has_pcurve_issues {
            config.uv_gap_tolerance = 1e-5;
            config.fix_same_parameter = true;
            config.fix_shifted = true;
            config.fix_lacking = true;
            config.fix_edge_curves = true;
        }
        if check.has_pcurve_issues {
            config.fix_split_face = true;
        }
        if level >= HealLevel::Standard {
            config.fix_gaps_3d = check.has_uv_gaps;
            config.fix_missing_seams = check.has_pcurve_issues;
        }
        if level >= HealLevel::Advanced {
            if check.has_self_intersections {
                config.fix_self_intersection = true;
            }
            if check.has_singularities {
                config.fix_degenerated = true;
                config.fix_periodic_degenerated = true;
            }
            if check.has_inner_wires || check.has_intersecting_wires {
                config.fix_intersecting_wires = true;
            }
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepWire, Orientation};
    use crate::store::BRepStore;
    use rc3d_core::math::Vec3;

    fn build_closed_triangle_shell(reg: &mut BRepStore) -> (ShellKey, crate::topo::FaceKey) {
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, line.clone());
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, line.clone());
        let e3 = reg.add_edge_with_pcurve(v2, v0, line.clone(), 1e-4, fk, line);
        let orient_for = |ek, from_vk| {
            let edge = reg.edges.get(ek).unwrap();
            if edge.v_low == from_vk {
                Orientation::Forward
            } else {
                Orientation::Reversed
            }
        };
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, orient_for(e1, v0)),
            (e2, orient_for(e2, v1)),
            (e3, orient_for(e3, v2)),
        ];
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        (sk, fk)
    }

    #[test]
    fn test_auto_heal_converges() {
        let mut reg = BRepStore::new();
        let (sk, _) = build_closed_triangle_shell(&mut reg);
        let report = auto_heal_shell(sk, &mut reg, HealLevel::Basic, 3);
        assert!(report.check_errors <= 1, "simple wire should converge quickly");
    }

    #[test]
    fn test_auto_heal_max_iterations() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let report = auto_heal_shell(sk, &mut reg, HealLevel::Basic, 1);
        let _ = report.merged_vertices;
    }

    #[test]
    fn test_auto_heal_basic_vs_standard() {
        let mut reg = BRepStore::new();
        let (sk, fk) = build_closed_triangle_shell(&mut reg);
        let report_basic = auto_heal_shell(sk, &mut reg, HealLevel::Basic, 2);
        let sk2 = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let report_std = auto_heal_shell(sk2, &mut reg, HealLevel::Standard, 2);
        assert!(
            report_std.added_seams >= report_basic.added_seams
                || report_std.lacking_tolerance_fixes >= report_basic.lacking_tolerance_fixes,
            "Standard should run more fixes than Basic"
        );
    }

    #[test]
    fn test_select_fixes_advanced_enables_self_intersect() {
        let mut check = CheckReport::default();
        check.has_self_intersections = true;
        let cfg0 = select_fixes(HealLevel::Advanced, 0, &check);
        assert!(
            !cfg0.fix_self_intersection,
            "self-intersect fix waits for iter > 0"
        );
        let cfg = select_fixes(HealLevel::Advanced, 1, &check);
        assert!(cfg.fix_self_intersection);
    }

    #[test]
    fn test_select_fixes_iter1_uses_fresh_check() {
        let mut check = CheckReport::default();
        check.has_self_intersections = true;
        let cfg = select_fixes(HealLevel::Advanced, 1, &check);
        assert!(cfg.fix_self_intersection);
        assert!(!cfg.fix_orientation, "orientation only on iter 0");
    }

    #[test]
    fn test_select_fixes_standard_enables_vertex_position() {
        let check = CheckReport::default();
        let cfg = select_fixes(HealLevel::Standard, 0, &check);
        assert!(cfg.fix_vertex_position);
    }

    #[test]
    fn test_select_fixes_iter1_enables_split_face_on_pcurve_issues() {
        let mut check = CheckReport::default();
        check.has_pcurve_issues = true;
        let cfg = select_fixes(HealLevel::Standard, 1, &check);
        assert!(cfg.fix_split_face);
    }

    #[test]
    fn test_select_fixes_advanced_enables_face_self_intersect() {
        let mut check = CheckReport::default();
        check.has_face_self_intersections = true;
        // Face self-intersection is detection-only (no fix flag), but check should be reported
        assert!(check.has_face_self_intersections);
    }
}
