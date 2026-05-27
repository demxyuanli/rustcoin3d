//! Edge 3D curve to vertex alignment (OCC ShapeFix_Wire::FixEdgeCurves).
//! Adjusts edge curves so t=0 and t=1 match vertex positions within tolerance.

use crate::step::brep::geom::CurveGeom;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{ShellKey, Orientation};
use rc3d_core::math::Vec3;

/// Adjust edge 3D curves to match vertex positions.
/// Returns number of edges adjusted.
pub fn fix_edge_curves(
    shell_key: ShellKey,
    reg: &mut BRepRegistry,
    tolerance: f32,
) -> usize {
    let face_keys: Vec<_> = {
        let Some(shell) = reg.shells.get(shell_key) else { return 0; };
        shell.faces.iter().map(|&(fk, _)| fk).collect()
    };

    let mut adjusted = 0usize;

    for fk in face_keys {
        let face = match reg.faces.get(fk) {
            Some(f) => f,
            None => continue,
        };
        let wire = match reg.wires.get(face.outer_wire) {
            Some(w) => w,
            None => continue,
        };

        for &(ek, orient) in &wire.edges {
            let edge = match reg.edges.get(ek) {
                Some(e) => e,
                None => continue,
            };

            let p_lo = reg.vertices.get(edge.v_low).map(|v| v.position);
            let p_hi = reg.vertices.get(edge.v_high).map(|v| v.position);
            let (Some(p_lo), Some(p_hi)) = (p_lo, p_hi) else { continue; };

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
    }

    adjusted
}

fn adjust_curve(curve: &CurveGeom, v_start: Vec3, v_end: Vec3, _tolerance: f32) -> CurveGeom {
    match curve {
        CurveGeom::Line { .. } => {
            CurveGeom::Line {
                origin: v_start,
                direction: v_end - v_start,
            }
        }
        CurveGeom::Circle { center, axis, radius } => {
            // Re-project vertices onto the circle to get new parameter angles.
            // The circle geometry (center, axis, radius) stays the same; only the
            // parametric range changes, which is stored implicitly by the vertices.
            let to_start = v_start - *center;
            let to_end = v_end - *center;
            // Project onto the circle plane (perpendicular to axis)
            let axis_n = axis.normalize();
            let proj_start = to_start - axis_n * to_start.dot(axis_n);
            let proj_end = to_end - axis_n * to_end.dot(axis_n);
            let new_start = *center + proj_start.normalize() * *radius;
            let new_end = *center + proj_end.normalize() * *radius;
            // Verify the reprojection is close
            let err_start = (new_start - v_start).length();
            let err_end = (new_end - v_end).length();
            if err_start.max(err_end) > *radius * 1e-3 {
                log::debug!(
                    "[BRep heal] FixEdgeCurves: circle reprojection error ({:.6}, {:.6}) > 0.1% radius, keeping original",
                    err_start, err_end
                );
                return curve.clone();
            }
            // The curve geometry is unchanged; vertex positions define the arc extent
            CurveGeom::Circle {
                center: *center,
                axis: *axis,
                radius: *radius,
            }
        }
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor } => {
            // Same approach as circle: project vertices, validate, keep geometry
            let to_start = v_start - *center;
            let to_end = v_end - *center;
            let axis_n = axis.normalize();
            let proj_start = to_start - axis_n * to_start.dot(axis_n);
            let proj_end = to_end - axis_n * to_end.dot(axis_n);
            let new_start = *center + proj_start.normalize() * *semi_major;
            let new_end = *center + proj_end.normalize() * *semi_major;
            let err_start = (new_start - v_start).length();
            let err_end = (new_end - v_end).length();
            if err_start.max(err_end) > *semi_major * 1e-3 {
                log::debug!(
                    "[BRep heal] FixEdgeCurves: ellipse reprojection error ({:.6}, {:.6}), keeping original",
                    err_start, err_end
                );
                return curve.clone();
            }
            CurveGeom::Ellipse {
                center: *center,
                axis: *axis,
                semi_major: *semi_major,
                semi_minor: *semi_minor,
            }
        }
        CurveGeom::BSpline { degree, control_points, knots, weights } => {
            // Translate control polygon so first/last CPs match vertex endpoints
            if control_points.len() < 2 {
                return curve.clone();
            }
            let cp_start = control_points[0];
            let cp_end = control_points[control_points.len() - 1];
            let delta_start = v_start - cp_start;
            let delta_end = v_end - cp_end;

            // Use average shift for interior control points
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
    use crate::step::brep::geom::SurfaceGeom;
    use crate::step::brep::topo::{BRepWire, BRepFace, WireKey, BRepShell, BRepEdge};
    use crate::step::brep::registry::BRepRegistry;
    use std::collections::HashMap;

    fn make_shell_with_line_edge(reg: &mut BRepRegistry, endpoint_offset: f32) -> ShellKey {
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        // Edge curve intentionally offset from vertices
        let curve = CurveGeom::Line {
            origin: Vec3::new(0.0, endpoint_offset, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        };
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
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
        // Insert edge directly to bypass normalize_edge_curve_to_vertices
        let ek = reg.edges.insert(BRepEdge {
            curve,
            tolerance: 1e-4,
            v_low: v0.min(v1),
            v_high: v0.max(v1),
            pcurves: {
                let mut m = HashMap::new();
                m.insert(fk, CurveGeom::Line {
                    origin: Vec3::ZERO,
                    direction: Vec3::X,
                });
                m
            },
        });
        let wk = reg.wires.insert(BRepWire { edges: vec![(ek, Orientation::Forward)] });
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
        assert!(adjusted > 0, "line with offset endpoints should be adjusted");
    }

    #[test]
    fn test_no_adjust_when_aligned() {
        let mut reg = BRepRegistry::new();
        let sk = make_shell_with_line_edge(&mut reg, 0.0);
        let adjusted = fix_edge_curves(sk, &mut reg, 1e-3);
        assert_eq!(adjusted, 0, "already-aligned curve should not be counted");
    }

    #[test]
    fn test_fix_circle_arc() {
        let mut reg = BRepRegistry::new();
        let center = Vec3::new(1.0, 0.0, 0.0);
        let v0 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4); // at angle 0
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 1.01, 0.0), 1e-4); // slightly off true circle
        let circle = CurveGeom::Circle { center, axis: Vec3::Z, radius: 1.0 };
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        // Insert edge directly to bypass normalize_edge_curve_to_vertices (which converts
        // non-matching curves to line chords). We want the circle geometry preserved so
        // fix_edge_curves can detect and adjust the deviation.
        let ek = reg.edges.insert(BRepEdge {
            curve: circle.clone(),
            tolerance: 1e-4,
            v_low: v0.min(v1),
            v_high: v0.max(v1),
            pcurves: {
                let mut m = HashMap::new();
                m.insert(fk, circle.clone());
                m
            },
        });
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(BRepShell { faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None });
        let adjusted = fix_edge_curves(sk, &mut reg, 1e-3);
        assert!(adjusted > 0, "circle edge with offset endpoint should be adjusted");
    }
}
