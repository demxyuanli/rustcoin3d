pub mod reorder;
pub mod gap;
pub mod orient;
pub mod seam;
pub mod check;
pub mod connected;
pub mod small;
pub mod shifted;
pub mod edge_curve;
pub mod lacking;
pub mod degenerated;
pub mod periodic;
pub mod self_intersect;
pub mod intersecting_wires;

use super::topo::{ShellKey, FaceKey, Orientation};
use super::registry::BRepRegistry;
use reorder::reorder_wire_edges;
use gap::{close_wire_gaps, close_wire_gaps_2d};
use orient::fix_shell_orientation;
use seam::fix_missing_seams;
use connected::fix_connected_wire;
use small::remove_small_edges;
use shifted::fix_shifted_pcurves;
use edge_curve::fix_edge_curves;
use lacking::fix_lacking_edges;
use degenerated::fix_degenerated_edges;
use periodic::fix_periodic_degenerated;
use self_intersect::fix_self_intersecting_wire;
use intersecting_wires::fix_intersecting_wires;
pub use check::{check_shell, CheckReport};

#[derive(Debug, Default)]
pub struct HealReport {
    pub reordered_wires: usize,
    pub closed_gaps: usize,
    pub closed_uv_gaps: usize,
    pub flipped_faces: usize,
    pub added_seams: usize,
    pub merged_vertices: usize,
    pub removed_small_edges: usize,
    pub shifted_pcurves: usize,
    pub adjusted_edge_curves: usize,
    pub skip_face_keys: Vec<FaceKey>,
    pub lacking_tolerance_fixes: usize,
    pub degenerate_edges_created: usize,
    pub periodic_degen_created: usize,
    pub self_intersections_fixed: usize,
    pub inner_wires_fixed: usize,
    pub check_errors: usize,
    pub check_warnings: usize,
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
    pub fix_small_edges: bool,
    pub fix_shifted: bool,
    pub fix_edge_curves: bool,
    pub fix_lacking: bool,
    pub fix_degenerated: bool,
    pub fix_periodic_degenerated: bool,
    pub fix_self_intersection: bool,
    pub fix_intersecting_wires: bool,
    pub small_edge_min_length: f32,
    pub uv_gap_tolerance: f32,
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
            fix_small_edges: true,
            fix_shifted: true,
            fix_edge_curves: true,
            fix_lacking: true,
            fix_degenerated: true,
            fix_periodic_degenerated: true,
            fix_self_intersection: true,
            fix_intersecting_wires: true,
            small_edge_min_length: 1e-6,
            uv_gap_tolerance: 1e-5,
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

    if config.fix_small_edges {
        for (face_key, _) in &face_keys {
            let face = match reg.faces.get(*face_key) {
                Some(f) => f,
                None => continue,
            };
            let seam_edges = face.seam_edges.clone();
            let outer_wire = face.outer_wire;
            let old_len = reg.wires.get(outer_wire).map(|w| w.edges.len()).unwrap_or(0);
            if let Some(updated) = remove_small_edges(outer_wire, reg, &seam_edges, config.small_edge_min_length) {
                if updated.len() != old_len {
                    report.removed_small_edges += 1;
                    log::debug!("[BRep heal] FixSmall face {:?}: removed small edges, {} remain", face_key, updated.len());
                }
            } else {
                report.skip_face_keys.push(*face_key);
                log::warn!("[BRep heal] FixSmall face {:?}: wire emptied, marking for skip", face_key);
            }
        }
    }

    for (face_key, _) in &face_keys {
        let outer_wire = match reg.faces.get(*face_key) {
            Some(f) => f.outer_wire,
            None => continue,
        };

        if config.fix_reorder {
            let wire = match reg.wires.get(outer_wire) {
                Some(w) => w,
                None => continue,
            };
            if let Some(reordered) = reorder_wire_edges(&wire.edges, reg) {
                if let Some(w) = reg.wires.get_mut(outer_wire) {
                    w.edges = reordered;
                    report.reordered_wires += 1;
                }
            }
        }

        if config.gap_tolerance > 0.0 {
            let closed = close_wire_gaps(outer_wire, reg, config.gap_tolerance);
            report.closed_gaps += closed;
        }

        if config.uv_gap_tolerance > 0.0 {
            let uv_closed = close_wire_gaps_2d(
                outer_wire,
                *face_key,
                reg,
                config.gap_tolerance,
                config.uv_gap_tolerance,
            );
            if uv_closed > 0 {
                report.closed_uv_gaps += uv_closed;
                log::debug!("[BRep heal] FixGaps2d face {:?}: closed {} UV gap(s)", face_key, uv_closed);
            }
        }

        if config.fix_shifted {
            let sr = fix_shifted_pcurves(outer_wire, *face_key, reg);
            report.shifted_pcurves += sr.shifts_applied;
            if sr.shifts_applied > 0 {
                log::debug!("[BRep heal] FixShifted face {:?}: {} pcurve shift(s) applied", face_key, sr.shifts_applied);
            }
        }

        if config.fix_periodic_degenerated {
            let pr = fix_periodic_degenerated(*face_key, reg);
            report.periodic_degen_created += pr.pole_edges_created;
            if pr.degeneracies_reconstructed > 0 {
                log::debug!("[BRep heal] FixPeriodicDegenerated face {:?}: {} pole degeneracies", face_key, pr.degeneracies_reconstructed);
            }
        }

        if config.fix_missing_seams {
            report.added_seams += fix_missing_seams(reg, *face_key);
        }

        if config.fix_intersecting_wires {
            let iwr = fix_intersecting_wires(*face_key, reg);
            let total_fixes = iwr.inner_wires_trimmed + iwr.inner_wires_removed + iwr.inner_wires_merged;
            report.inner_wires_fixed += total_fixes;
            if total_fixes > 0 {
                log::debug!("[BRep heal] FixIntersectingWires face {:?}: {} inner wire fix(es)", face_key, total_fixes);
            }
        }

        if config.fix_lacking {
            let lr = fix_lacking_edges(outer_wire, *face_key, reg, config.gap_tolerance, config.uv_gap_tolerance);
            report.lacking_tolerance_fixes += lr.tolerance_fixes;
            if lr.tolerance_fixes > 0 {
                log::debug!("[BRep heal] FixLacking face {:?}: {} tolerance fix(es)", face_key, lr.tolerance_fixes);
            }
        }

        if config.fix_self_intersection {
            let sir = fix_self_intersecting_wire(outer_wire, *face_key, reg);
            if sir.intersections_found > 0 {
                report.self_intersections_fixed += sir.intersections_found;
                log::debug!("[BRep heal] FixSelfIntersection face {:?}: {} intersections, {} edges split, rebuilt={}",
                    face_key, sir.intersections_found, sir.edges_split, sir.wires_rebuilt);
            }
            let wire_empty = reg.wires.get(outer_wire).map(|w| w.edges.is_empty()).unwrap_or(false);
            if wire_empty {
                report.skip_face_keys.push(*face_key);
                log::warn!("[BRep heal] FixSelfIntersection face {:?}: unfixable, marking for skip", face_key);
                continue;
            }
        }

        if config.fix_degenerated {
            let dr = fix_degenerated_edges(*face_key, reg);
            report.degenerate_edges_created += dr.degenerate_edges_created;
            if dr.degeneracies_found > 0 {
                log::debug!("[BRep heal] FixDegenerated face {:?}: {} singularities, {} edges created",
                    face_key, dr.degeneracies_found, dr.degenerate_edges_created);
            }
        }
    }

    if config.fix_edge_curves {
        let adjusted = fix_edge_curves(shell_key, reg, config.gap_tolerance);
        report.adjusted_edge_curves += adjusted;
        if adjusted > 0 {
            log::debug!("[BRep heal] FixEdgeCurves: adjusted {} edge(s)", adjusted);
        }
    }

    if config.fix_vertex_tolerance {
        let fixed = fix_vertex_tolerance(reg);
        if fixed > 0 {
            log::debug!("[BRep heal] fixed vertex tolerance on {} edges", fixed);
        }
    }

    if config.fix_small_area {
        let small = fix_small_area(shell_key, reg);
        for fk in small {
            if !report.skip_face_keys.contains(&fk) {
                report.skip_face_keys.push(fk);
            }
        }
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
