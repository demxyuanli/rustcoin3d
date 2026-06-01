//! Edge 3D curve to vertex alignment (OCC ShapeFix_Wire::FixEdgeCurves).

use crate::geom::CurveGeom;
use crate::store::BRepRegistry;
use crate::topo::{Orientation, WireKey, FaceKey};
use rc3d_core::math::Vec3;

/// Adjust edge 3D curves on one wire to match vertex positions.
pub fn fix_edge_curves_wire(
    wire_key: WireKey,
    _face_key: FaceKey,
    reg: &mut BRepRegistry,
    tolerance: f32,
) -> usize {
    let edges: Vec<(crate::topo::EdgeKey, Orientation)> = {
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
pub fn fix_edge_curves(
    shell_key: crate::topo::ShellKey,
    reg: &mut BRepRegistry,
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
        CurveGeom::Ellipse {
            center,
            axis,
            semi_major,
            semi_minor,
            ..
        } => {
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
        CurveGeom::BSpline {
            degree,
            control_points,
            knots,
            weights,
        } => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::SurfaceGeom;
    use crate::topo::{BRepEdge, BRepFace, BRepShell, BRepWire, WireKey};
    use std::collections::HashMap;

    fn make_shell_with_line_edge(reg: &mut BRepRegistry, endpoint_offset: f32) -> crate::topo::ShellKey {
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let curve = CurveGeom::Line {
            origin: Vec3::new(0.0, endpoint_offset, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        };
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: WireKey::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let ek = reg.edges.insert(BRepEdge {
            curve,
            tolerance: 1e-4,
            v_low: v0.min(v1),
            v_high: v0.max(v1),
            pcurves: {
                let mut m = HashMap::new();
                m.insert(
                    fk,
                    CurveGeom::Line {
                        origin: Vec3::ZERO,
                        direction: Vec3::X,
                    },
                );
                m
            },
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
        let mut reg = BRepRegistry::new();
        let sk = make_shell_with_line_edge(&mut reg, 0.01);
        let adjusted = fix_edge_curves(sk, &mut reg, 1e-3);
        assert!(adjusted > 0);
    }

    #[test]
    fn test_no_adjust_when_aligned() {
        let mut reg = BRepRegistry::new();
        let sk = make_shell_with_line_edge(&mut reg, 0.0);
        let adjusted = fix_edge_curves(sk, &mut reg, 1e-3);
        assert_eq!(adjusted, 0);
    }

    #[test]
    fn test_fix_circle_arc() {
        let mut reg = BRepRegistry::new();
        let center = Vec3::ZERO;
        let v0 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::Y, 1e-4);
        let circle = CurveGeom::circle(center, Vec3::Z, 1.0);
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
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
        let ek = reg.edges.insert(BRepEdge {
            curve: circle.clone(),
            tolerance: 1e-4,
            v_low: v0.min(v1),
            v_high: v0.max(v1),
            pcurves: {
                let mut m = HashMap::new();
                m.insert(fk, circle);
                m
            },
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
        assert!(
            (edge.curve.d0(0.0) - Vec3::X).length() < tol,
            "curve start should match vertex"
        );
        assert!(
            (edge.curve.d0(1.0) - Vec3::Y).length() < tol,
            "curve end should match vertex"
        );
    }

    #[test]
    fn test_fix_bspline_translate() {
        let mut reg = BRepRegistry::new();
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
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
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
        let ek = reg.edges.insert(BRepEdge {
            curve: bspline,
            tolerance: 1e-4,
            v_low: v0.min(v1),
            v_high: v0.max(v1),
            pcurves: {
                let mut m = HashMap::new();
                m.insert(
                    fk,
                    CurveGeom::Line {
                        origin: Vec3::ZERO,
                        direction: Vec3::X,
                    },
                );
                m
            },
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
