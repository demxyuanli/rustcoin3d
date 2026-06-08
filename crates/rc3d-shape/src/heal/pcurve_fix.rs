//! PCurve repair: period-shift detection + edge curve adjustment.

use crate::geom::{Curve2d, CurveGeom};
use crate::geom::SurfaceGeom;
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation, WireKey};
use rc3d_core::math::Vec3;

// ── Period-shift detection ─────────────────────────────────────────

#[derive(Debug, Default)]
pub(crate) struct ShiftedReport {
    pub shifts_applied: usize,
}

/// Detect and fix PCurves shifted by a surface period.
pub(crate) fn fix_shifted_pcurves(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepStore,
) -> ShiftedReport {
    let mut report = ShiftedReport::default();

    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return report,
    };
    let surface = face.surface.clone();

    let (period_u, period_v) = surface_periods(&surface);
    if period_u == 0.0 && period_v == 0.0 {
        return report;
    }

    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return report; };
        wire.edges.clone()
    };

    let mut uv_sum = (0.0f64, 0.0f64);
    let mut count = 0u32;
    for &(ek, _) in &edges {
        if let Some(edge) = reg.edges.get(ek) {
            if let Some(pc) = edge.pcurves.get(&face_key) {
                let uv_start = pc.d0(0.0);
                let uv_end = pc.d0(1.0);
                uv_sum.0 += (uv_start.0 + uv_end.0) as f64 * 0.5;
                uv_sum.1 += (uv_start.1 + uv_end.1) as f64 * 0.5;
                count += 1;
            }
        }
    }

    if count == 0 {
        return report;
    }
    let uv_center = (uv_sum.0 / count as f64, uv_sum.1 / count as f64);

    for &(ek, _) in &edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        let pc = match edge.pcurves.get(&face_key) {
            Some(p) => p,
            None => continue,
        };

        let mid_uv = pc.d0(0.5);
        let (mu, mv) = (mid_uv.0 as f64, mid_uv.1 as f64);

        let mut shift_u: f64 = 0.0;
        let mut shift_v: f64 = 0.0;

        if period_u > 0.0 {
            let dist_u = (mu - uv_center.0).abs();
            if dist_u > period_u as f64 * 0.4 {
                let shifted_up = mu - period_u as f64;
                let shifted_down = mu + period_u as f64;
                if (shifted_up - uv_center.0).abs() < dist_u * 0.5 {
                    shift_u = -period_u as f64;
                } else if (shifted_down - uv_center.0).abs() < dist_u * 0.5 {
                    shift_u = period_u as f64;
                }
            }
        }

        if period_v > 0.0 {
            let dist_v = (mv - uv_center.1).abs();
            if dist_v > period_v as f64 * 0.4 {
                let shifted_up = mv - period_v as f64;
                let shifted_down = mv + period_v as f64;
                if (shifted_up - uv_center.1).abs() < dist_v * 0.5 {
                    shift_v = -period_v as f64;
                } else if (shifted_down - uv_center.1).abs() < dist_v * 0.5 {
                    shift_v = period_v as f64;
                }
            }
        }

        if shift_u != 0.0 || shift_v != 0.0 {
            if let Some(pc) = reg.pcurve_mut(ek, face_key) {
                *pc = shift_pcurve(pc, shift_u as f32, shift_v as f32);
                report.shifts_applied += 1;
            }
        }
    }

    report
}

fn surface_periods(surface: &SurfaceGeom) -> (f64, f64) {
    use std::f32::consts::TAU;
    match surface {
        SurfaceGeom::Cylinder { .. } => (TAU as f64, 0.0),
        SurfaceGeom::Torus { .. } => (TAU as f64, TAU as f64),
        SurfaceGeom::Sphere { .. } => (TAU as f64, 0.0),
        SurfaceGeom::Cone { .. } => (TAU as f64, 0.0),
        SurfaceGeom::Revolution { .. } => (TAU as f64, 0.0),
        _ => (0.0, 0.0),
    }
}

fn shift_pcurve(pc: &Curve2d, du: f32, dv: f32) -> Curve2d {
    match pc {
        Curve2d::Line { origin, direction } => Curve2d::Line {
            origin: (origin.0 + du, origin.1 + dv),
            direction: *direction,
        },
        Curve2d::Circle { center, radius } => Curve2d::Circle {
            center: (center.0 + du, center.1 + dv),
            radius: *radius,
        },
        Curve2d::Ellipse { center, semi_major, semi_minor } => Curve2d::Ellipse {
            center: (center.0 + du, center.1 + dv),
            semi_major: *semi_major,
            semi_minor: *semi_minor,
        },
        other => other.clone(),
    }
}

// ── Edge curve adjustment ──────────────────────────────────────────

/// Adjust edge 3D curves on one wire to match vertex positions.
pub(crate) fn fix_edge_curves_wire(
    wire_key: WireKey,
    _face_key: FaceKey,
    reg: &mut BRepStore,
    tolerance: f32,
) -> usize {
    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else {
            return 0;
        };
        wire.edges.clone()
    };

    let mut adjusted = 0usize;
    for &(ek, orient) in &edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };

        let p_lo = reg.vertices.get(edge.v_low).map(|v| v.position);
        let p_hi = reg.vertices.get(edge.v_high).map(|v| v.position);
        let (Some(p_lo), Some(p_hi)) = (p_lo, p_hi) else {
            continue;
        };

        let (v_start, v_end) = if orient == Orientation::Reversed {
            (p_hi, p_lo)
        } else {
            (p_lo, p_hi)
        };

        let curve = &edge.curve;
        let dev_start = (curve.d0(0.0) - v_start).length();
        let dev_end = (curve.d0(1.0) - v_end).length();

        if dev_start < tolerance && dev_end < tolerance {
            continue;
        }

        let new_curve = adjust_curve(curve, v_start, v_end, tolerance);
        if let Some(edge_mut) = reg.edges.get_mut(ek) {
            edge_mut.curve = new_curve;
            adjusted += 1;
        }
    }
    adjusted
}

/// Shell-level entry (all outer wires).
#[allow(dead_code)]
pub(crate) fn fix_edge_curves(
    shell_key: crate::topo::ShellKey,
    reg: &mut BRepStore,
    tolerance: f32,
) -> usize {
    let face_keys: Vec<_> = {
        let Some(shell) = reg.shells.get(shell_key) else {
            return 0;
        };
        shell.faces.iter().map(|&(fk, _)| fk).collect()
    };

    let mut adjusted = 0usize;
    for fk in face_keys {
        let wires = {
            let Some(face) = reg.faces.get(fk) else {
                continue;
            };
            let mut ws = vec![face.outer_wire];
            ws.extend(face.inner_wires.iter().copied());
            ws
        };
        for wk in wires {
            adjusted += fix_edge_curves_wire(wk, fk, reg, tolerance);
        }
    }
    adjusted
}

fn circle_angle(center: Vec3, axis: Vec3, radius: f32, point: Vec3) -> Option<f32> {
    let a = axis.normalize();
    let ref_dir = if a.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let y_dir = a.cross(ref_dir).normalize();
    let x_dir = y_dir.cross(a);
    let to_p = point - center;
    let in_plane = to_p - a * to_p.dot(a);
    if in_plane.length() < radius * 1e-6 {
        return None;
    }
    let u = in_plane.dot(x_dir) / radius;
    let v = in_plane.dot(y_dir) / radius;
    Some(v.atan2(u))
}

fn normalize_arc_params(t0: f32, mut t1: f32) -> (f32, f32) {
    while t1 < t0 {
        t1 += std::f32::consts::TAU;
    }
    if t1 - t0 > std::f32::consts::TAU {
        t1 = t0 + std::f32::consts::TAU;
    }
    (t0, t1)
}

fn adjust_curve(curve: &CurveGeom, v_start: Vec3, v_end: Vec3, _tolerance: f32) -> CurveGeom {
    match curve {
        CurveGeom::Line { .. } => CurveGeom::Line {
            origin: v_start,
            direction: v_end - v_start,
        },
        CurveGeom::Circle { center, axis, radius, .. } => {
            let Some(a0) = circle_angle(*center, *axis, *radius, v_start) else {
                return curve.clone();
            };
            let Some(a1) = circle_angle(*center, *axis, *radius, v_end) else {
                return curve.clone();
            };
            let (t_min, t_max) = normalize_arc_params(a0, a1);
            CurveGeom::Trimmed {
                basis: Box::new(CurveGeom::circle(*center, *axis, *radius)),
                t_min,
                t_max,
            }
        }
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor, .. } => {
            let Some(a0) = circle_angle(*center, *axis, *semi_major, v_start) else {
                return curve.clone();
            };
            let Some(a1) = circle_angle(*center, *axis, *semi_major, v_end) else {
                return curve.clone();
            };
            let (t_min, t_max) = normalize_arc_params(a0, a1);
            CurveGeom::Trimmed {
                basis: Box::new(CurveGeom::ellipse(*center, *axis, *semi_major, *semi_minor)),
                t_min,
                t_max,
            }
        }
        CurveGeom::Trimmed { basis, .. } => adjust_curve(basis, v_start, v_end, _tolerance),
        CurveGeom::BSpline { degree, control_points, knots, weights } => {
            if control_points.len() < 2 {
                return curve.clone();
            }
            let cp_start = control_points[0];
            let cp_end = control_points[control_points.len() - 1];
            let delta_start = v_start - cp_start;
            let delta_end = v_end - cp_end;
            let mut new_cp = control_points.clone();
            let n = new_cp.len();
            for (i, cp) in new_cp.iter_mut().enumerate() {
                let t = i as f32 / (n - 1) as f32;
                let delta = delta_start * (1.0 - t) + delta_end * t;
                *cp = *cp + delta;
            }
            CurveGeom::BSpline {
                degree: *degree,
                control_points: new_cp,
                knots: knots.clone(),
                weights: weights.clone(),
            }
        }
        CurveGeom::Polyline { points } => {
            if points.len() < 2 {
                return curve.clone();
            }
            let mut new_pts = points.clone();
            new_pts[0] = v_start;
            let last = new_pts.len() - 1;
            new_pts[last] = v_end;
            CurveGeom::Polyline { points: new_pts }
        }
        other => other.clone(),
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepEdge, BRepFace, BRepShell, BRepWire, Orientation, WireKey};
    use rc3d_core::math::Vec3;
    use std::collections::HashMap;

    // ── shifted tests ──

    #[test]
    fn test_cylinder_periods() {
        let s = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 1.0);
        let (pu, pv) = surface_periods(&s);
        assert!(pu > 0.0);
        assert_eq!(pv, 0.0);
    }

    #[test]
    fn test_torus_periods() {
        let s = SurfaceGeom::torus(Vec3::ZERO, Vec3::Z, 3.0, 1.0);
        let (pu, pv) = surface_periods(&s);
        assert!(pu > 0.0);
        assert!(pv > 0.0);
    }

    #[test]
    fn test_plane_no_period() {
        let s = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let (pu, pv) = surface_periods(&s);
        assert_eq!(pu, 0.0);
        assert_eq!(pv, 0.0);
    }

    #[test]
    fn test_shifted_cylinder_pcurve() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 1.0);
        let v0 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 1.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let curve_3d = CurveGeom::Line {
            origin: Vec3::new(1.0, 0.0, 0.0),
            direction: Vec3::new(0.0, 0.0, 1.0),
        };
        let pc_normal = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let pc_shifted = Curve2d::Line {
            origin: (std::f32::consts::TAU, 0.0),
            direction: (1.0, 0.0),
        };
        let e1 = reg.edges.insert(BRepEdge {
            v_low: v0, v_high: v1, curve: curve_3d.clone(), tolerance: 1e-4,
            pcurves: HashMap::from([(fk, pc_normal.clone())]),
        });
        let e2 = reg.edges.insert(BRepEdge {
            v_low: v0, v_high: v1, curve: curve_3d.clone(), tolerance: 1e-4,
            pcurves: HashMap::from([(fk, pc_normal.clone())]),
        });
        let e3 = reg.edges.insert(BRepEdge {
            v_low: v0, v_high: v1, curve: curve_3d.clone(), tolerance: 1e-4,
            pcurves: HashMap::from([(fk, pc_normal)]),
        });
        let e4 = reg.edges.insert(BRepEdge {
            v_low: v0, v_high: v1, curve: curve_3d, tolerance: 1e-4,
            pcurves: HashMap::from([(fk, pc_shifted)]),
        });
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
            (e3, Orientation::Forward),
            (e4, Orientation::Forward),
        ];
        let report = fix_shifted_pcurves(wk, fk, &mut reg);
        assert!(report.shifts_applied > 0, "shifted cylinder PCurve should be detected and corrected");
    }

    #[test]
    fn test_no_false_positive() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 1.0);
        let v0 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 1.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });
        let line = CurveGeom::Line {
            origin: Vec3::new(1.0, 0.0, 0.0),
            direction: Vec3::new(0.0, 0.0, 1.0),
        };
        let pc = Curve2d::Line {
            origin: (0.1, 0.0),
            direction: (0.5, 0.0),
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone());
        let e2 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc);
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
        ];
        let report = fix_shifted_pcurves(wk, fk, &mut reg);
        assert_eq!(report.shifts_applied, 0, "valid unshifted PCurves should not be modified");
    }

    // ── edge_curve tests ──

    fn make_shell_with_line_edge(reg: &mut BRepStore, endpoint_offset: f32) -> crate::topo::ShellKey {
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let curve = CurveGeom::Line {
            origin: Vec3::new(0.0, endpoint_offset, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        };
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: WireKey::default(),
            inner_wires: vec![],
            same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });
        let ek = reg.edges.insert(BRepEdge {
            curve,
            tolerance: 1e-4,
            v_low: v0.min(v1),
            v_high: v0.max(v1),
            pcurves: HashMap::from([(fk, Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) })]),
        });
        let wk = reg.wires.insert(BRepWire {
            edges: vec![(ek, Orientation::Forward)],
        });
        if let Some(face) = reg.faces.get_mut(fk) {
            face.outer_wire = wk;
        }
        reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        })
    }

    #[test]
    fn test_fix_line_endpoints() {
        let mut reg = BRepStore::new();
        let sk = make_shell_with_line_edge(&mut reg, 0.01);
        let adjusted = fix_edge_curves(sk, &mut reg, 1e-3);
        assert!(adjusted > 0);
    }

    #[test]
    fn test_no_adjust_when_aligned() {
        let mut reg = BRepStore::new();
        let sk = make_shell_with_line_edge(&mut reg, 0.0);
        let adjusted = fix_edge_curves(sk, &mut reg, 1e-3);
        assert_eq!(adjusted, 0);
    }

    #[test]
    fn test_fix_circle_arc() {
        let mut reg = BRepStore::new();
        let center = Vec3::ZERO;
        let v0 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::Y, 1e-4);
        let circle = CurveGeom::circle(center, Vec3::Z, 1.0);
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });
        let ek = reg.edges.insert(BRepEdge {
            curve: circle.clone(),
            tolerance: 1e-4,
            v_low: v0.min(v1),
            v_high: v0.max(v1),
            pcurves: HashMap::from([(fk, Curve2d::Circle { center: (0.0, 0.0), radius: 1.0 })]),
        });
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let adjusted = fix_edge_curves(sk, &mut reg, 1e-3);
        assert!(adjusted > 0);
        let edge = reg.edges.get(ek).unwrap();
        let tol = 1e-3;
        assert!((edge.curve.d0(0.0) - Vec3::X).length() < tol, "curve start should match vertex");
        assert!((edge.curve.d0(1.0) - Vec3::Y).length() < tol, "curve end should match vertex");
    }

    #[test]
    fn test_fix_bspline_translate() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.5, 0.0), 1e-4);
        let bspline = CurveGeom::BSpline {
            degree: 2,
            control_points: vec![
                Vec3::new(0.0, 0.1, 0.0),
                Vec3::new(0.5, 0.3, 0.0),
                Vec3::new(1.0, 0.6, 0.0),
            ],
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            weights: None,
        };
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });
        let ek = reg.edges.insert(BRepEdge {
            curve: bspline,
            tolerance: 1e-4,
            v_low: v0.min(v1),
            v_high: v0.max(v1),
            pcurves: HashMap::from([(fk, Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) })]),
        });
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let adjusted = fix_edge_curves(sk, &mut reg, 1e-3);
        assert!(adjusted > 0);
    }
}
