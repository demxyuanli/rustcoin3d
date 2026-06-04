pub(crate) mod wire_ops;
pub(crate) mod wire_join;
pub(crate) mod pcurve_fix;
pub(crate) mod same_param_fix;
pub(crate) mod face_fix;
pub(crate) mod shell_fix;
pub(crate) mod seam;
pub mod check;
pub(crate) mod lacking;
pub(crate) mod degenerated;
pub(crate) mod self_intersect;
pub(crate) mod intersecting_wires;
pub mod continuity;
pub mod pipeline;
pub mod curve_trim;
pub(crate) mod geom2d;
pub mod topo_diag;

use std::collections::HashSet;

use crate::topo::{FaceKey, ShellKey, WireKey};
use crate::store::BRepStore;
use wire_ops::{reorder_wire_edges, remove_small_edges};
use wire_join::{close_wire_gaps, close_wire_gaps_2d, fix_connected_wire};
use same_param_fix::fix_same_parameter_wire;
use pcurve_fix::{fix_shifted_pcurves, fix_edge_curves_wire};
use face_fix::{fix_add_natural_bound, fix_reversed_2d};
use shell_fix::{fix_shell_orientation, fix_split_face, fix_vertex_positions};
use seam::fix_missing_seams;
use lacking::fix_lacking_edges;
use degenerated::{fix_degenerated_edges, fix_periodic_degenerated};
use self_intersect::fix_self_intersecting_wire;
use intersecting_wires::fix_intersecting_wires;
pub use check::{check_shell, check_uv_self_intersection, CheckReport};
pub use topo_diag::{
    check_shell_topo_diag, log_shell_topo_diag, topo_diag_enabled, TopoDiagReport,
};
pub use continuity::check_shell_continuity;
pub use pipeline::{auto_heal_shell, HealLevel};

/// Why a face was excluded from meshing after heal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceSkipReason {
    HealPipeline,
    CheckError,
    SelfIntersection,
}

#[derive(Debug, Clone)]
pub struct HealReport {
    pub reordered_wires: usize,
    pub closed_gaps: usize,
    pub closed_uv_gaps: usize,
    pub flipped_faces: usize,
    pub added_seams: usize,
    pub merged_vertices: usize,
    pub removed_small_edges: usize,
    pub shifted_pcurves: usize,
    pub same_param_fixed: usize,
    pub adjusted_edge_curves: usize,
    pub skip_face_keys: Vec<FaceKey>,
    pub face_skip_reasons: Vec<(FaceKey, FaceSkipReason)>,
    pub lacking_tolerance_fixes: usize,
    pub degenerate_edges_created: usize,
    pub periodic_degen_created: usize,
    pub self_intersections_fixed: usize,
    pub inner_wires_fixed: usize,
    pub vertex_positions_fixed: usize,
    pub split_faces_created: usize,
    pub natural_bounds_added: usize,
    pub reversed_2d_fixed: usize,
    pub check_errors: usize,
    pub check_warnings: usize,
    skip_faces_seen: HashSet<FaceKey>,
}

impl Default for HealReport {
    fn default() -> Self {
        Self {
            reordered_wires: 0,
            closed_gaps: 0,
            closed_uv_gaps: 0,
            flipped_faces: 0,
            added_seams: 0,
            merged_vertices: 0,
            removed_small_edges: 0,
            shifted_pcurves: 0,
            same_param_fixed: 0,
            adjusted_edge_curves: 0,
            skip_face_keys: Vec::new(),
            face_skip_reasons: Vec::new(),
            lacking_tolerance_fixes: 0,
            degenerate_edges_created: 0,
            periodic_degen_created: 0,
            self_intersections_fixed: 0,
            inner_wires_fixed: 0,
            vertex_positions_fixed: 0,
            split_faces_created: 0,
            natural_bounds_added: 0,
            reversed_2d_fixed: 0,
            check_errors: 0,
            check_warnings: 0,
            skip_faces_seen: HashSet::new(),
        }
    }
}

impl HealReport {
    pub fn merge(&mut self, other: HealReport) {
        self.reordered_wires += other.reordered_wires;
        self.closed_gaps += other.closed_gaps;
        self.closed_uv_gaps += other.closed_uv_gaps;
        self.flipped_faces += other.flipped_faces;
        self.added_seams += other.added_seams;
        self.merged_vertices += other.merged_vertices;
        self.removed_small_edges += other.removed_small_edges;
        self.shifted_pcurves += other.shifted_pcurves;
        self.adjusted_edge_curves += other.adjusted_edge_curves;
        self.lacking_tolerance_fixes += other.lacking_tolerance_fixes;
        self.degenerate_edges_created += other.degenerate_edges_created;
        self.periodic_degen_created += other.periodic_degen_created;
        self.self_intersections_fixed += other.self_intersections_fixed;
        self.inner_wires_fixed += other.inner_wires_fixed;
        self.vertex_positions_fixed += other.vertex_positions_fixed;
        self.split_faces_created += other.split_faces_created;
        self.natural_bounds_added += other.natural_bounds_added;
        self.reversed_2d_fixed += other.reversed_2d_fixed;
        for (fk, reason) in other.face_skip_reasons {
            push_skip_face(self, fk, reason);
        }
        self.check_errors += other.check_errors;
        self.check_warnings += other.check_warnings;
    }
}

fn push_skip_face(report: &mut HealReport, face_key: FaceKey, reason: FaceSkipReason) {
    if report.skip_faces_seen.insert(face_key) {
        report.skip_face_keys.push(face_key);
        report.face_skip_reasons.push((face_key, reason));
    }
}

#[derive(Debug, Clone)]
pub struct HealConfig {
    pub gap_tolerance: f32,
    pub fix_orientation: bool,
    pub fix_reorder: bool,
    pub fix_missing_seams: bool,
    pub fix_connected: bool,
    pub fix_gaps_3d: bool,
    pub fix_vertex_tolerance: bool,
    pub fix_small_area: bool,
    pub fix_small_edges: bool,
    pub fix_same_parameter: bool,
    pub fix_shifted: bool,
    pub fix_edge_curves: bool,
    pub fix_lacking: bool,
    pub fix_degenerated: bool,
    pub fix_periodic_degenerated: bool,
    pub fix_self_intersection: bool,
    pub fix_intersecting_wires: bool,
    pub fix_vertex_position: bool,
    pub fix_split_face: bool,
    pub fix_natural_bound: bool,
    pub fix_reversed_2d: bool,
    pub small_edge_min_length: f32,
    pub uv_gap_tolerance: f32,
}

impl HealConfig {
    /// All fix flags disabled; gap_tolerance preserved from caller when needed.
    pub fn all_disabled() -> Self {
        Self {
            gap_tolerance: 1e-4,
            fix_orientation: false,
            fix_reorder: false,
            fix_missing_seams: false,
            fix_connected: false,
            fix_gaps_3d: false,
            fix_vertex_tolerance: false,
            fix_small_area: false,
            fix_small_edges: false,
            fix_same_parameter: false,
            fix_shifted: false,
            fix_edge_curves: false,
            fix_lacking: false,
            fix_degenerated: false,
            fix_periodic_degenerated: false,
            fix_self_intersection: false,
            fix_intersecting_wires: false,
            fix_vertex_position: false,
            fix_split_face: false,
            fix_natural_bound: false,
            fix_reversed_2d: false,
            small_edge_min_length: 1e-6,
            uv_gap_tolerance: 0.0,
        }
    }

    /// Minimal wire repair after self-intersection split.
    pub fn wire_reconnect_only(gap_tolerance: f32) -> Self {
        let mut cfg = Self::all_disabled();
        cfg.gap_tolerance = gap_tolerance;
        cfg.fix_connected = true;
        cfg.fix_gaps_3d = true;
        cfg.fix_reorder = true;
        cfg
    }
}

impl Default for HealConfig {
    fn default() -> Self {
        Self {
            gap_tolerance: 1e-4,
            fix_orientation: true,
            fix_reorder: true,
            fix_missing_seams: true,
            fix_connected: true,
            fix_gaps_3d: true,
            fix_vertex_tolerance: true,
            fix_small_area: true,
            fix_small_edges: true,
            fix_same_parameter: true,
            fix_shifted: true,
            fix_edge_curves: true,
            fix_lacking: true,
            fix_degenerated: true,
            fix_periodic_degenerated: true,
            fix_self_intersection: true,
            fix_intersecting_wires: true,
            fix_vertex_position: true,
            fix_split_face: true,
            fix_natural_bound: true,
            fix_reversed_2d: true,
            small_edge_min_length: 1e-6,
            uv_gap_tolerance: 1e-5,
        }
    }
}

fn wire_is_closed(wire_key: WireKey, reg: &BRepStore) -> bool {
    let Some(wire) = reg.wires.get(wire_key) else {
        return false;
    };
    if wire.edges.len() < 2 {
        return false;
    }
    let mut first_v = None;
    let mut last_v = None;
    for &(ek, orient) in &wire.edges {
        let Some(edge) = reg.edges.get(ek) else {
            return false;
        };
        let (v_start, v_end) = if orient == crate::topo::Orientation::Forward {
            (edge.v_low, edge.v_high)
        } else {
            (edge.v_high, edge.v_low)
        };
        if first_v.is_none() {
            first_v = Some(v_start);
        }
        if let Some(lv) = last_v {
            if lv != v_start {
                return false;
            }
        }
        last_v = Some(v_end);
    }
    matches!((first_v, last_v), (Some(f), Some(l)) if f == l)
}

/// Wire-level heal passes (outer and inner wires).
fn heal_wire_passes(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepStore,
    config: &HealConfig,
    seam_edges: &[crate::topo::EdgeKey],
    report: &mut HealReport,
) -> bool {
    if config.fix_connected {
        let cr = fix_connected_wire(wire_key, reg, config.gap_tolerance);
        report.merged_vertices += cr.merged_vertices;
    }

    if config.fix_small_edges {
        let old_len = reg.wires.get(wire_key).map(|w| w.edges.len()).unwrap_or(0);
        if let Some(updated) =
            remove_small_edges(wire_key, reg, seam_edges, config.small_edge_min_length)
        {
            if updated.len() != old_len {
                report.removed_small_edges += 1;
            }
        } else {
            return false;
        }
    }

    if config.fix_reorder {
        let wire = match reg.wires.get(wire_key) {
            Some(w) => w,
            None => return false,
        };
        if let Some(reordered) = reorder_wire_edges(&wire.edges, reg) {
            if let Some(w) = reg.wires.get_mut(wire_key) {
                w.edges = reordered;
                report.reordered_wires += 1;
            }
        }
    }

    if config.fix_gaps_3d {
        let closed = wire_is_closed(wire_key, reg);
        let gaps = close_wire_gaps(wire_key, reg, config.gap_tolerance, closed);
        if gaps > 0 {
            report.closed_gaps += gaps;
        }
    }

    if config.uv_gap_tolerance > 0.0 {
        let uv_closed = close_wire_gaps_2d(
            wire_key,
            face_key,
            reg,
            config.gap_tolerance,
            config.uv_gap_tolerance,
        );
        if uv_closed > 0 {
            report.closed_uv_gaps += uv_closed;
        }
    }

    if config.fix_same_parameter {
        let fixed = fix_same_parameter_wire(wire_key, face_key, reg, config.gap_tolerance);
        report.same_param_fixed += fixed;
    }

    if config.fix_shifted {
        let sr = fix_shifted_pcurves(wire_key, face_key, reg);
        report.shifted_pcurves += sr.shifts_applied;
    }

    if config.fix_edge_curves {
        let adjusted = fix_edge_curves_wire(wire_key, face_key, reg, config.gap_tolerance);
        report.adjusted_edge_curves += adjusted;
    }

    true
}

/// Face-level heal passes after all wires are processed.
fn heal_face_passes(
    face_key: FaceKey,
    reg: &mut BRepStore,
    config: &HealConfig,
    report: &mut HealReport,
) -> bool {
    let (outer_wire, inner_wires) = {
        let Some(face) = reg.faces.get(face_key) else {
            return false;
        };
        (face.outer_wire, face.inner_wires.clone())
    };

    if config.fix_lacking {
        if !heal_lacking_on_wire(outer_wire, face_key, reg, config, report) {
            return false;
        }
        for &iw in &inner_wires {
            if !heal_lacking_on_wire(iw, face_key, reg, config, report) {
                return false;
            }
        }
    }

    if config.fix_periodic_degenerated {
        let pr = fix_periodic_degenerated(face_key, reg);
        report.periodic_degen_created += pr.pole_edges_created;
    }

    if config.fix_self_intersection {
        if !heal_self_intersect_on_wire(outer_wire, face_key, reg, config, report) {
            return false;
        }
        for &iw in &inner_wires {
            if !heal_self_intersect_on_wire(iw, face_key, reg, config, report) {
                return false;
            }
        }
    }

    if config.fix_intersecting_wires {
        let iwr = fix_intersecting_wires(face_key, reg);
        let total_fixes = iwr.inner_wires_trimmed + iwr.inner_wires_removed + iwr.inner_wires_merged;
        report.inner_wires_fixed += total_fixes;
    }

    if config.fix_natural_bound {
        if fix_add_natural_bound(reg, face_key) {
            report.natural_bounds_added += 1;
        }
    }

    if config.fix_reversed_2d {
        if fix_reversed_2d(reg, face_key) {
            report.reversed_2d_fixed += 1;
        }
    }

    if config.fix_missing_seams {
        report.added_seams += fix_missing_seams(reg, face_key);
    }

    if config.fix_degenerated {
        let dr = fix_degenerated_edges(face_key, reg);
        report.degenerate_edges_created += dr.degenerate_edges_created;
    }

    true
}

fn heal_lacking_on_wire(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepStore,
    config: &HealConfig,
    report: &mut HealReport,
) -> bool {
    if reg
        .wires
        .get(wire_key)
        .map(|w| w.edges.is_empty())
        .unwrap_or(true)
    {
        return true;
    }
    let lr = fix_lacking_edges(
        wire_key,
        face_key,
        reg,
        config.gap_tolerance,
        config.uv_gap_tolerance,
    );
    report.lacking_tolerance_fixes += lr.tolerance_fixes;
    !reg.wires.get(wire_key).map(|w| w.edges.is_empty()).unwrap_or(true)
}

fn heal_self_intersect_on_wire(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepStore,
    config: &HealConfig,
    report: &mut HealReport,
) -> bool {
    if reg
        .wires
        .get(wire_key)
        .map(|w| w.edges.is_empty())
        .unwrap_or(true)
    {
        return true;
    }
    let sir = fix_self_intersecting_wire(wire_key, face_key, reg);
    if sir.intersections_found > 0 {
        report.self_intersections_fixed += sir.intersections_found;
    }
    if sir.edges_split > 0 || sir.wires_rebuilt {
        let wire_cfg = HealConfig::wire_reconnect_only(config.gap_tolerance);
        heal_wire_passes(wire_key, face_key, reg, &wire_cfg, &[], report);
    }
    if sir.intersections_found > 0 {
        let si_warnings = check_uv_self_intersection(face_key, reg);
        if !si_warnings.is_empty() {
            log::warn!(
                "[BRep heal] FixSelfIntersection face {:?}: still self-intersecting after fix",
                face_key
            );
            push_skip_face(report, face_key, FaceSkipReason::CheckError);
            return false;
        }
    }
    !reg.wires.get(wire_key).map(|w| w.edges.is_empty()).unwrap_or(true)
}

/// Run all healing passes on a shell.
pub fn heal_shell(
    shell_key: ShellKey,
    reg: &mut BRepStore,
    config: &HealConfig,
) -> HealReport {
    let mut report = HealReport::default();

    let Some(shell) = reg.shells.get(shell_key) else {
        return report;
    };
    let face_keys = shell.faces.clone();

    for (face_key, _) in &face_keys {
        let (outer_wire, inner_wires, protected_edges) = {
            let face = match reg.faces.get(*face_key) {
                Some(f) => f,
                None => continue,
            };
            let mut protected = face.seam_edges.clone();
            for &ek in &face.degenerated_edges {
                if !protected.contains(&ek) {
                    protected.push(ek);
                }
            }
            (face.outer_wire, face.inner_wires.clone(), protected)
        };

        let mut wire_keys = vec![outer_wire];
        wire_keys.extend(inner_wires);

        for wk in wire_keys {
            if !heal_wire_passes(wk, *face_key, reg, config, &protected_edges, &mut report) {
                push_skip_face(&mut report, *face_key, FaceSkipReason::HealPipeline);
                log::warn!("[BRep heal] face {:?}: wire {:?} failed wire heal", face_key, wk);
                break;
            }
        }

        if report.skip_faces_seen.contains(face_key) {
            continue;
        }

        if !heal_face_passes(*face_key, reg, config, &mut report) {
            push_skip_face(&mut report, *face_key, FaceSkipReason::SelfIntersection);
        }
    }

    if config.fix_vertex_tolerance {
        let fixed = fix_vertex_tolerance(reg);
        if fixed > 0 {
            log::debug!("[BRep heal] fixed vertex tolerance on {} edges", fixed);
        }
    }

    if config.fix_vertex_position {
        let vp_fixed = fix_vertex_positions(shell_key, reg, config.gap_tolerance);
        report.vertex_positions_fixed += vp_fixed;
    }

    if config.fix_small_area {
        let small = fix_small_area(shell_key, reg);
        for fk in small {
            push_skip_face(&mut report, fk, FaceSkipReason::HealPipeline);
        }
    }

    if config.fix_orientation {
        report.flipped_faces = fix_shell_orientation(shell_key, reg);
    }

    if config.fix_split_face {
        let sr = fix_split_face(shell_key, reg);
        report.split_faces_created += sr.faces_created;
    }

    let check_report = check_shell(shell_key, reg);
    report.check_errors = check_report.errors.len();
    report.check_warnings = check_report.warnings.len();
    for e in &check_report.errors {
        log::debug!("[BRep check] {}", e);
    }
    if !check_report.errors.is_empty() {
        log::info!(
            "[BRep check] {} topology error(s) after heal pass",
            check_report.errors.len()
        );
    }
    for w in &check_report.warnings {
        log::debug!("[BRep check] {}", w);
    }

    report
}

fn fix_vertex_tolerance(reg: &mut BRepStore) -> usize {
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

fn fix_small_area(shell_key: ShellKey, reg: &BRepStore) -> Vec<FaceKey> {
    let Some(shell) = reg.shells.get(shell_key) else {
        return vec![];
    };
    let mut skip = Vec::new();
    for &(face_key, _) in &shell.faces {
        let Some(face) = reg.faces.get(face_key) else {
            continue;
        };
        let Some(wire) = reg.wires.get(face.outer_wire) else {
            skip.push(face_key);
            continue;
        };
        if wire.edges.is_empty()
            && face.seam_edges.is_empty()
            && face.degenerated_edges.is_empty()
        {
            log::warn!("[BRep heal] face {:?} has zero area, marking for skip", face_key);
            skip.push(face_key);
        }
    }
    skip
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepWire, Orientation};
    use rc3d_core::math::Vec3;

    #[test]
    fn test_close_3d_gap_in_heal_shell() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 1.0, 0.0), 1e-4);
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
        let line = |a, b| CurveGeom::Line {
            origin: a,
            direction: b - a,
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line(Vec3::ZERO, Vec3::X), 1e-4, fk, line(Vec3::ZERO, Vec3::X));
        let gap_v = reg.vertices.insert(crate::topo::BRepVertex {
            position: Vec3::new(1.0, 0.00005, 0.0),
            tolerance: 1e-6,
        });
        let e2 = reg.add_edge_with_pcurve(gap_v, v2, line(Vec3::new(1.0, 0.00005, 0.0), Vec3::new(1.0, 1.0, 0.0)), 1e-4, fk, line(Vec3::new(1.0, 0.1, 0.0), Vec3::new(1.0, 1.0, 0.0)));
        let e2_orient = if gap_v < v2 {
            Orientation::Forward
        } else {
            Orientation::Reversed
        };
        let e3 = reg.add_edge_with_pcurve(v2, v3, line(Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)), 1e-4, fk, line(Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)));
        let e5 = reg.add_edge_with_pcurve(v3, v0, line(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO), 1e-4, fk, line(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO));
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, e2_orient),
            (e3, Orientation::Forward),
            (e5, Orientation::Forward),
        ];
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let mut cfg = HealConfig::default();
        cfg.gap_tolerance = 1e-3;
        cfg.fix_connected = false;
        cfg.fix_reorder = false;
        cfg.fix_small_edges = false;
        cfg.fix_shifted = false;
        cfg.fix_edge_curves = false;
        cfg.fix_vertex_tolerance = false;
        cfg.fix_small_area = false;
        cfg.fix_vertex_position = false;
        cfg.uv_gap_tolerance = 0.0;
        cfg.fix_gaps_3d = true;
        cfg.fix_orientation = false;
        cfg.fix_missing_seams = false;
        cfg.fix_self_intersection = false;
        cfg.fix_degenerated = false;
        cfg.fix_intersecting_wires = false;
        cfg.fix_periodic_degenerated = false;
        cfg.fix_lacking = false;
        cfg.fix_split_face = false;
        let report = heal_shell(sk, &mut reg, &cfg);
        assert!(report.closed_gaps > 0 || report.merged_vertices > 0);
    }
}
