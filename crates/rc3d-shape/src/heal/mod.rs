use rc3d_core::math::Real;
pub(crate) mod edge_tolerance;
pub(crate) mod wire_ops;
pub(crate) mod wire_join;
pub(crate) mod pcurve_fix;
pub(crate) mod same_param_fix;
pub(crate) mod face_fix;
pub(crate) mod shell_fix;
pub mod seam;
pub mod check;
pub(crate) mod lacking;
pub(crate) mod degenerated;
pub(crate) mod self_intersect;
pub(crate) mod face_self_intersect;
pub(crate) mod intersecting_wires;
pub mod continuity;
pub mod pipeline;
pub mod same_param_reparam;
pub mod curve_trim;
pub(crate) mod geom2d;
pub mod topo_diag;
pub(crate) mod free_bounds;
pub(crate) mod compose_shell;
pub(crate) mod face_fold;
pub(crate) mod edge_connect;
pub mod unify_same_domain;
pub mod canonical;
pub mod solid_fix;
use std::collections::HashSet;

use crate::topo::{FaceKey, ShellKey, SolidKey, WireKey};
use crate::store::BRepStore;
use wire_ops::{reorder_wire_edges, remove_small_edges};
use wire_join::{close_wire_gaps, close_wire_gaps_2d, fix_connected_wire};
use same_param_fix::fix_same_parameter_wire;
use pcurve_fix::{fix_shifted_pcurves, fix_edge_curves_wire};
use face_fix::{fix_add_natural_bound, fix_reversed_2d, fix_small_faces};
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
pub use pipeline::{auto_heal_shell, HealLevel, HealPassId, HealPolicy};

/// Why a face was excluded from meshing after heal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceSkipReason {
    HealPipeline,
    CheckError,
    SelfIntersection,
}

#[derive(Debug, Clone)]
#[derive(Default)]
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
    pub free_bounds_closed: usize,
    pub free_bounds_open_found: usize,
    pub shells_composed: usize,
    pub face_folds_repaired: usize,
    pub face_self_intersections_fixed: usize,
    pub check_errors: usize,
    pub check_warnings: usize,
    pub unify_merges: usize,
    pub small_faces_merged: usize,
    pub removed_small_solids: usize,
    pub removed_empty_shells: usize,
    skip_faces_seen: HashSet<FaceKey>,
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
        self.face_self_intersections_fixed += other.face_self_intersections_fixed;
        for (fk, reason) in other.face_skip_reasons {
            push_skip_face(self, fk, reason);
        }
        self.check_errors += other.check_errors;
        self.check_warnings += other.check_warnings;
        self.unify_merges += other.unify_merges;
        self.small_faces_merged += other.small_faces_merged;
        self.removed_small_solids += other.removed_small_solids;
        self.removed_empty_shells += other.removed_empty_shells;
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
    pub gap_tolerance: Real,
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
    pub fix_free_bounds: bool,
    pub fix_compose_shell: bool,
    pub fix_face_fold: bool,
    pub fix_face_self_intersect: bool,
    pub fix_edge_connect: bool,
    pub fix_unify_same_domain: bool,
    pub fix_small_faces: bool,
    pub fix_small_solids: bool,
    pub small_edge_min_length: Real,
    pub small_face_min_area: Real,
    pub small_solid_min_volume: Real,
    pub uv_gap_tolerance: Real,
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
            fix_free_bounds: false,
            fix_compose_shell: false,
            fix_face_fold: false,
            fix_face_self_intersect: false,
            fix_edge_connect: false,
            fix_unify_same_domain: false,
            fix_small_faces: false,
            fix_small_solids: false,
            small_edge_min_length: 1e-6,
            small_face_min_area: 0.01,
            small_solid_min_volume: 0.001,
            uv_gap_tolerance: 0.0,
        }
    }

    /// Minimal wire repair after self-intersection split.
    pub fn wire_reconnect_only(gap_tolerance: Real) -> Self {
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
            fix_free_bounds: true,
            fix_compose_shell: false,
            fix_face_fold: true,
            fix_face_self_intersect: false,
            fix_edge_connect: false,
            fix_unify_same_domain: false,
            fix_small_faces: false,
            fix_small_solids: false,
            small_edge_min_length: 1e-6,
            small_face_min_area: 0.01,
            small_solid_min_volume: 0.001,
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
        let _t = std::time::Instant::now();
        let cr = fix_connected_wire(wire_key, reg, config.gap_tolerance);
        report.merged_vertices += cr.merged_vertices;
        let t = _t.elapsed().as_secs_f32();
        if t > 0.1 { log::debug!("[heal timer]    fix_connected: {:.1}s", t); }
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
        let _t = std::time::Instant::now();
        let closed = wire_is_closed(wire_key, reg);
        let gaps = close_wire_gaps(wire_key, reg, config.gap_tolerance, closed);
        let t = _t.elapsed().as_secs_f32();
        if t > 0.1 { log::debug!("[heal timer]    fix_gaps_3d: {:.1}s", t); }
        if gaps > 0 { report.closed_gaps += gaps; }
    }

    if config.uv_gap_tolerance > 0.0 {
        let uv_closed = close_wire_gaps_2d(
            wire_key, face_key, reg, config.gap_tolerance, config.uv_gap_tolerance,
        );
        if uv_closed > 0 { report.closed_uv_gaps += uv_closed; }
    }

    if config.fix_same_parameter {
        let _t = std::time::Instant::now();
        let fixed = fix_same_parameter_wire(wire_key, face_key, reg, config.gap_tolerance);
        let t = _t.elapsed().as_secs_f32();
        if t > 0.1 { log::debug!("[heal timer]    fix_same_param: {:.1}s ({} edges fixed)", t, fixed); }
        report.same_param_fixed += fixed;
    }

    if config.fix_shifted {
        let sr = fix_shifted_pcurves(wire_key, face_key, reg);
        report.shifted_pcurves += sr.shifts_applied;
    }

    if config.fix_edge_curves {
        let _t = std::time::Instant::now();
        let adjusted = fix_edge_curves_wire(wire_key, face_key, reg, config.gap_tolerance);
        let t = _t.elapsed().as_secs_f32();
        if t > 0.1 { log::debug!("[heal timer]    fix_edge_curves: {:.1}s", t); }
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

    if config.fix_natural_bound
        && fix_add_natural_bound(reg, face_key) {
            report.natural_bounds_added += 1;
        }

    if config.fix_reversed_2d
        && fix_reversed_2d(reg, face_key) {
            report.reversed_2d_fixed += 1;
        }

    if config.fix_missing_seams {
        report.added_seams += fix_missing_seams(reg, face_key);
    }

    if config.fix_degenerated {
        let dr = fix_degenerated_edges(face_key, reg);
        report.degenerate_edges_created += dr.degenerate_edges_created;
    }

    if config.fix_face_fold {
        let fr = face_fold::fix_face_folds(face_key, reg);
        report.face_folds_repaired += fr.folds_repaired;
    }

    if config.fix_face_self_intersect {
        let sir = face_self_intersect::fix_face_self_intersections(face_key, reg, config.gap_tolerance);
        report.face_self_intersections_fixed += sir.faces_fixed;
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
    let _t_total = std::time::Instant::now();
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

    if config.fix_small_faces {
        let merged = fix_small_faces(shell_key, reg, config.small_face_min_area);
        report.small_faces_merged = merged;
        if merged > 0 {
            log::debug!(
                "[BRep heal] FixSmallFaces: merged/removed {} small faces on shell {:?}",
                merged, shell_key
            );
        }
    }

    if config.fix_orientation {
        report.flipped_faces = fix_shell_orientation(shell_key, reg);
    }

    if config.fix_split_face {
        let sr = fix_split_face(shell_key, reg);
        report.split_faces_created += sr.faces_created;
    }

    // Free bounds: detect and close open edges
    if config.fix_free_bounds {
        let fb = free_bounds::close_free_bounds(shell_key, reg, config.gap_tolerance);
        report.free_bounds_closed += fb.edges_closed;
        report.free_bounds_open_found += fb.open_edges_found;
    }

    // EdgeConnect: merge geometrically coincident vertices across adjacent faces.
    // OCC alignment: ShapeFix_EdgeConnect — merges shared edge endpoints.
    if config.fix_edge_connect {
        let merges = edge_connect::connect_shell_edges(reg, shell_key, config.gap_tolerance);
        if merges > 0 {
            log::debug!("[BRep heal] EdgeConnect: merged {} vertices on shell {:?}", merges, shell_key);
        }
    }

    // Compose shell: group disconnected faces into separate shells.
    // OCC alignment: ShapeFix_ComposeShell — stitches faces into shells.
    if config.fix_compose_shell {
        let faces: Vec<FaceKey> = face_keys.iter()
            .filter(|(fk, _)| !report.skip_faces_seen.contains(fk))
            .map(|(fk, _)| *fk)
            .collect();
        if !faces.is_empty() {
            let groups = compose_shell::compose_shells(&faces, reg);
            if groups.len() > 1 {
                // Multiple disconnected face groups — create separate shells.
                let cr = compose_shell::compose_shells_into_store(&faces, reg);
                report.shells_composed = cr.shells_composed;
            } else {
                report.shells_composed = groups.len();
            }
        }
    }

    // UnifySameDomain: merge adjacent faces sharing the same surface geometry.
    // OCC alignment: ShapeUpgrade_UnifySameDomain::Perform()
    if config.fix_unify_same_domain {
        let ur = unify_same_domain::unify_same_domain(shell_key, reg, config.gap_tolerance);
        report.unify_merges = ur.merges;
        if ur.merges > 0 {
            log::debug!(
                "[BRep heal] UnifySameDomain: merged {} face clusters on shell {:?}",
                ur.merges, shell_key
            );
        }
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
    log::debug!("[heal timer] check_shell: {:.1}s  total heal_shell: {:.1}s",
        _t_total.elapsed().as_secs_f32(), _t_total.elapsed().as_secs_f32());

    report
}


/// Run solid-level healing passes (FixSmallSolids, remove empty shells).
pub fn heal_solid(
    solid_key: SolidKey,
    reg: &mut BRepStore,
    config: &HealConfig,
) -> HealReport {
    let mut report = HealReport::default();

    if config.fix_small_solids {
        let removed = solid_fix::fix_small_solids(&[solid_key], reg, config.small_solid_min_volume);
        report.removed_small_solids = removed.len();
    }

    let empty_count = solid_fix::remove_empty_shells(reg);
    report.removed_empty_shells = empty_count;
    if empty_count > 0 {
        log::info!("[heal] FixSmallSolid: removed {} empty shells", empty_count);
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
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepWire, Orientation};
    use rc3d_core::math::PVec3;

    #[test]
    fn test_close_3d_gap_in_heal_shell() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(PVec3::new(0.0, 1.0, 0.0), 1e-4);
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
        let pc = |a: rc3d_core::math::PVec3, b: rc3d_core::math::PVec3| Curve2d::Line {
            origin: (a.x, a.y),
            direction: (b.x - a.x, b.y - a.y),
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line(PVec3::ZERO, PVec3::X), 1e-4, fk, pc(PVec3::ZERO, PVec3::X), true);
        let gap_v = reg.vertices.insert(crate::topo::BRepVertex {
            position: PVec3::new(1.0, 0.00005, 0.0),
            tolerance: 1e-6,
        });
        let e2 = reg.add_edge_with_pcurve(gap_v, v2, line(PVec3::new(1.0, 0.00005, 0.0), PVec3::new(1.0, 1.0, 0.0)), 1e-4, fk, pc(PVec3::new(1.0, 0.1, 0.0), PVec3::new(1.0, 1.0, 0.0)), true);
        let e2_orient = if gap_v < v2 {
            Orientation::Forward
        } else {
            Orientation::Reversed
        };
        let e3 = reg.add_edge_with_pcurve(v2, v3, line(PVec3::new(1.0, 1.0, 0.0), PVec3::new(0.0, 1.0, 0.0)), 1e-4, fk, pc(PVec3::new(1.0, 1.0, 0.0), PVec3::new(0.0, 1.0, 0.0)), true);
        let e5 = reg.add_edge_with_pcurve(v3, v0, line(PVec3::new(0.0, 1.0, 0.0), PVec3::ZERO), 1e-4, fk, pc(PVec3::new(0.0, 1.0, 0.0), PVec3::ZERO), true);
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
