//! Topology sewing diagnostics (loop closure, pcurve vs 3D edge, wire orientation).
//! Enable at import: `RC3D_BREP_TOPO_DIAG=1` (stderr) or path for a log file.

use std::io::Write;

use rc3d_core::math::Vec3;

use crate::geom::{CurveGeom, SurfaceGeom, eval_pcurve_on_surface, signed_area_2d};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation, ShellKey, WireKey};
use super::geom2d::collect_wire_uv_polygon;

const PCURVE_DRIFT_SAMPLES: usize = 9;

#[derive(Debug, Clone, Default)]
pub struct TopoDiagReport {
    pub wire_junction_gaps: Vec<WireJunctionGap>,
    pub pcurve_drifts: Vec<PcurveDrift>,
    pub orientation_flags: Vec<WireOrientationFlag>,
    pub max_wire_junction_gap: f32,
    pub max_pcurve_drift: f32,
}

#[derive(Debug, Clone)]
pub struct WireJunctionGap {
    pub face_key: FaceKey,
    pub wire_key: WireKey,
    pub edge_prev: EdgeKey,
    pub edge_next: EdgeKey,
    pub gap_3d: f32,
    pub tol_used: f32,
}

#[derive(Debug, Clone)]
pub struct PcurveDrift {
    pub face_key: FaceKey,
    pub edge_key: EdgeKey,
    pub max_drift: f32,
    pub tol_used: f32,
}

#[derive(Debug, Clone)]
pub struct WireOrientationFlag {
    pub face_key: FaceKey,
    pub wire_key: WireKey,
    pub signed_uv_area: f32,
    pub same_sense: bool,
    pub is_inner: bool,
}

pub fn topo_diag_enabled() -> bool {
    match std::env::var("RC3D_BREP_TOPO_DIAG") {
        Ok(s) if !s.is_empty() && s != "0" => true,
        _ => false,
    }
}

pub fn check_shell_topo_diag(shell_key: ShellKey, reg: &BRepStore) -> TopoDiagReport {
    let mut report = TopoDiagReport::default();
    let Some(shell) = reg.shells.get(shell_key) else {
        return report;
    };

    for &(face_key, _) in &shell.faces {
        let Some(face) = reg.faces.get(face_key) else {
            continue;
        };
        let tol = face.tolerance.max(1e-4);

        let mut wires: Vec<(WireKey, bool)> = vec![(face.outer_wire, false)];
        for &iw in &face.inner_wires {
            wires.push((iw, true));
        }

        for (wire_key, is_inner) in wires {
            report
                .wire_junction_gaps
                .extend(measure_wire_junction_gaps(face_key, wire_key, tol, reg));
            if let Some(flag) = check_wire_uv_orientation(face_key, wire_key, is_inner, reg) {
                report.orientation_flags.push(flag);
            }
        }

        for ek in crate::topo_iter::iter_edges_of_face(face_key, reg) {
            if let Some(drift) = measure_pcurve_drift(face_key, ek, tol, reg) {
                report.pcurve_drifts.push(drift);
            }
        }
    }

    report.max_wire_junction_gap = report
        .wire_junction_gaps
        .iter()
        .map(|g| g.gap_3d)
        .fold(0.0f32, f32::max);
    report.max_pcurve_drift = report
        .pcurve_drifts
        .iter()
        .map(|d| d.max_drift)
        .fold(0.0f32, f32::max);
    report
}

fn oriented_curve_endpoints(
    ek: EdgeKey,
    orient: Orientation,
    reg: &BRepStore,
) -> Option<(Vec3, Vec3)> {
    let edge = reg.edges.get(ek)?;
    let p_lo = reg.vertices.get(edge.v_low)?.position;
    let p_hi = reg.vertices.get(edge.v_high)?.position;
    let (mut p0, mut p1) = (edge.curve.d0(0.0), edge.curve.d0(1.0));
    if (p0 - p_lo).length() > (p0 - p_hi).length() {
        std::mem::swap(&mut p0, &mut p1);
    }
    if orient == Orientation::Forward {
        Some((p0, p1))
    } else {
        Some((p1, p0))
    }
}

fn measure_wire_junction_gaps(
    face_key: FaceKey,
    wire_key: WireKey,
    tol: f32,
    reg: &BRepStore,
) -> Vec<WireJunctionGap> {
    let wire = match reg.wires.get(wire_key) {
        Some(w) => w,
        None => return Vec::new(),
    };
    if wire.edges.len() < 2 {
        return Vec::new();
    }

    let n = wire.edges.len();
    let mut out = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ek_prev, o_prev) = wire.edges[i];
        let (ek_next, o_next) = wire.edges[j];
        let Some((_, p_end)) = oriented_curve_endpoints(ek_prev, o_prev, reg) else {
            continue;
        };
        let Some((p_start, _)) = oriented_curve_endpoints(ek_next, o_next, reg) else {
            continue;
        };
        let gap = (p_end - p_start).length();
        if gap > tol * 0.1 {
            out.push(WireJunctionGap {
                face_key,
                wire_key,
                edge_prev: ek_prev,
                edge_next: ek_next,
                gap_3d: gap,
                tol_used: tol,
            });
        }
    }
    out
}

fn measure_pcurve_drift(
    face_key: FaceKey,
    ek: EdgeKey,
    tol: f32,
    reg: &BRepStore,
) -> Option<PcurveDrift> {
    let edge = reg.edges.get(ek)?;
    let face = reg.faces.get(face_key)?;
    let pcurve = edge.pcurves.get(&face_key)?;
    let surface = &face.surface;
    let match_tol = pcurve_match_tol(&edge.curve, tol);

    let mut max_drift = 0.0f32;
    for i in 0..=PCURVE_DRIFT_SAMPLES {
        let t = i as f32 / PCURVE_DRIFT_SAMPLES as f32;
        max_drift = max_drift.max(pcurve_on_surface_gap(surface, pcurve, t, match_tol));
    }
    let _ = edge;

    if max_drift <= tol * 0.1 {
        return None;
    }
    Some(PcurveDrift {
        face_key,
        edge_key: ek,
        max_drift,
        tol_used: match_tol,
    })
}

/// Gap between PCURVE-induced point and face surface (periodic UV branches included).
fn pcurve_on_surface_gap(surface: &SurfaceGeom, pcurve: &CurveGeom, t: f32, match_tol: f32) -> f32 {
    let pt = eval_pcurve_on_surface(pcurve, surface, t);
    let uv = pcurve.d0(t);
    if pcurve_uv_matches_surface(surface, pt, uv, match_tol) {
        return 0.0;
    }
    let mut best = (pt - surface.d0_native(uv.x, uv.y)).length();
    if matches!(surface, SurfaceGeom::Revolution { .. }) {
        const TAU: f32 = std::f32::consts::TAU;
        if uv.x <= 1.0 + 1e-4 {
            best = best.min((pt - surface.d0_native(uv.x * TAU, uv.y)).length());
        }
        if uv.x >= TAU * 0.25 {
            let un = uv.x / TAU;
            if (un - uv.x).abs() > 1e-6 {
                best = best.min((pt - surface.d0_native(un, uv.y)).length());
            }
        }
    }
    if let Some(pu) = surface.native_u_period() {
        for shift in [-1.0f32, 1.0] {
            best = best.min((pt - surface.d0_native(uv.x + shift * pu, uv.y)).length());
        }
    }
    if let Some(pv) = surface.native_v_period() {
        for shift in [-1.0f32, 1.0] {
            best = best.min((pt - surface.d0_native(uv.x, uv.y + shift * pv)).length());
        }
    }
    best
}

fn pcurve_match_tol(curve: &crate::geom::CurveGeom, tol: f32) -> f32 {
    let edge_len = (curve.d0(0.0) - curve.d0(1.0)).length();
    (tol.max(1e-4) * 10.0).max(edge_len * 0.05).max(1e-3)
}

fn pcurve_uv_matches_surface(
    surface: &SurfaceGeom,
    p3: Vec3,
    uv: Vec3,
    match_tol: f32,
) -> bool {
    let mut candidates = vec![(uv.x, uv.y)];
    if matches!(surface, SurfaceGeom::Revolution { .. }) {
        const TAU: f32 = std::f32::consts::TAU;
        if uv.x <= 1.0 + 1e-4 {
            candidates.push((uv.x * TAU, uv.y));
        }
        if uv.x >= TAU * 0.25 {
            let un = uv.x / TAU;
            if (un - uv.x).abs() > 1e-6 {
                candidates.push((un, uv.y));
            }
        }
    }
    for (u, v) in candidates {
        if (p3 - surface.d0_native(u, v)).length() <= match_tol {
            return true;
        }
        if let Some(period_u) = surface.native_u_period() {
            for shift in [-1.0f32, 1.0] {
                if (p3 - surface.d0_native(u + shift * period_u, v)).length() <= match_tol {
                    return true;
                }
            }
        }
        if let Some(period_v) = surface.native_v_period() {
            for shift in [-1.0f32, 1.0] {
                if (p3 - surface.d0_native(u, v + shift * period_v)).length() <= match_tol {
                    return true;
                }
            }
        }
    }
    false
}

fn check_wire_uv_orientation(
    face_key: FaceKey,
    wire_key: WireKey,
    is_inner: bool,
    reg: &BRepStore,
) -> Option<WireOrientationFlag> {
    let face = reg.faces.get(face_key)?;
    let wire = reg.wires.get(wire_key)?;
    let uv_points = collect_wire_uv_polygon(wire, face_key, reg);
    if uv_points.len() < 3 {
        return None;
    }
    let area = signed_area_2d(&uv_points) as f32;
    let outer_positive = face.same_sense;
    let bad = if is_inner {
        (area > 0.0) == outer_positive
    } else {
        (area > 0.0) != outer_positive
    };
    if !bad {
        return None;
    }
    Some(WireOrientationFlag {
        face_key,
        wire_key,
        signed_uv_area: area,
        same_sense: face.same_sense,
        is_inner,
    })
}

pub fn format_topo_diag_lines(shell_key: ShellKey, report: &TopoDiagReport) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!(
        "[BRep topo diag] shell {:?}: max_wire_gap={:.6} max_pcurve_drift={:.6} junctions={} drifts={} orient={}",
        shell_key,
        report.max_wire_junction_gap,
        report.max_pcurve_drift,
        report.wire_junction_gaps.len(),
        report.pcurve_drifts.len(),
        report.orientation_flags.len(),
    ));

    let mut gaps: Vec<_> = report.wire_junction_gaps.iter().collect();
    gaps.sort_by(|a, b| b.gap_3d.partial_cmp(&a.gap_3d).unwrap_or(std::cmp::Ordering::Equal));
    for g in gaps.into_iter().take(12) {
        lines.push(format!(
            "[BRep topo diag]   wire gap face {:?} wire {:?} {:?}->{:?} gap={:.6} tol={:.6}",
            g.face_key, g.wire_key, g.edge_prev, g.edge_next, g.gap_3d, g.tol_used
        ));
    }

    let mut drifts: Vec<_> = report.pcurve_drifts.iter().collect();
    drifts.sort_by(|a, b| b.max_drift.partial_cmp(&a.max_drift).unwrap_or(std::cmp::Ordering::Equal));
    for d in drifts.into_iter().take(12) {
        lines.push(format!(
            "[BRep topo diag]   pcurve drift face {:?} edge {:?} drift={:.6} tol={:.6}",
            d.face_key, d.edge_key, d.max_drift, d.tol_used
        ));
    }

    for o in &report.orientation_flags {
        lines.push(format!(
            "[BRep topo diag]   orient face {:?} wire {:?} inner={} signed_area={:.6} same_sense={}",
            o.face_key, o.wire_key, o.is_inner, o.signed_uv_area, o.same_sense
        ));
    }
    lines
}

pub fn log_shell_topo_diag(shell_key: ShellKey, reg: &BRepStore) {
    if !topo_diag_enabled() {
        return;
    }
    let report = check_shell_topo_diag(shell_key, reg);
    let lines = format_topo_diag_lines(shell_key, &report);
    match std::env::var("RC3D_BREP_TOPO_DIAG") {
        Ok(path) if path.contains('\\') || path.contains('/') || path.ends_with(".txt") => {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                for line in &lines {
                    let _ = writeln!(f, "{line}");
                }
            }
        }
        _ => {
            for line in lines {
                eprintln!("{line}");
            }
        }
    }
}
