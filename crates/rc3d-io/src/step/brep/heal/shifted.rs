//! PCurve period-shift detection and correction (OCC ShapeFix_Wire::FixShifted).
//! Detects PCurves offset by a full parameter period on periodic surfaces.

use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{EdgeKey, FaceKey, Orientation, WireKey};

#[derive(Debug, Default)]
pub struct ShiftedReport {
    pub shifts_applied: usize,
}

/// Detect and fix PCurves shifted by a surface period.
pub fn fix_shifted_pcurves(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepRegistry,
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

    // Compute UV centroid of all PCurve endpoints
    let mut uv_sum = (0.0f64, 0.0f64);
    let mut count = 0u32;
    for &(ek, _) in &edges {
        if let Some(edge) = reg.edges.get(ek) {
            if let Some(pc) = edge.pcurves.get(&face_key) {
                let uv_start = pc.d0(0.0);
                let uv_end = pc.d0(1.0);
                uv_sum.0 += (uv_start.x + uv_end.x) as f64 * 0.5;
                uv_sum.1 += (uv_start.y + uv_end.y) as f64 * 0.5;
                count += 1;
            }
        }
    }

    if count == 0 {
        return report;
    }
    let uv_center = (uv_sum.0 / count as f64, uv_sum.1 / count as f64);

    // Check each edge's PCurve midpoint vs the cluster center
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
        let (mu, mv) = (mid_uv.x as f64, mid_uv.y as f64);

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

/// Return (U_period, V_period) for a surface. Returns (0,0) for non-periodic.
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

/// Translate all UV coordinates of a PCurve by (du, dv).
fn shift_pcurve(
    pc: &crate::step::brep::geom::CurveGeom,
    du: f32,
    dv: f32,
) -> crate::step::brep::geom::CurveGeom {
    use crate::step::brep::geom::CurveGeom;
    let shift = rc3d_core::math::Vec3::new(du, dv, 0.0);
    match pc {
        CurveGeom::Line { origin, direction } => CurveGeom::Line {
            origin: *origin + shift,
            direction: *direction,
        },
        CurveGeom::Circle { center, axis, radius } => CurveGeom::Circle {
            center: *center + shift,
            axis: *axis,
            radius: *radius,
        },
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor } => CurveGeom::Ellipse {
            center: *center + shift,
            axis: *axis,
            semi_major: *semi_major,
            semi_minor: *semi_minor,
        },
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::CurveGeom;
    use crate::step::brep::topo::{BRepWire, Orientation};
    use rc3d_core::math::Vec3;

    #[test]
    fn test_cylinder_periods() {
        let s = SurfaceGeom::Cylinder { origin: Vec3::ZERO, axis: Vec3::Z, radius: 1.0 };
        let (pu, pv) = surface_periods(&s);
        assert!(pu > 0.0);
        assert_eq!(pv, 0.0);
    }

    #[test]
    fn test_torus_periods() {
        let s = SurfaceGeom::Torus { center: Vec3::ZERO, axis: Vec3::Z, major_r: 3.0, minor_r: 1.0 };
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
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Cylinder {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 1.0,
        };
        let v0 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 1.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
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
        let pc_normal = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        };
        let pc_shifted = CurveGeom::Line {
            origin: Vec3::new(std::f32::consts::TAU, 0.0, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        };
        // Direct insert to get distinct EdgeKeys (add_edge_with_pcurve would dedup on vertex pair)
        let e1 = reg.edges.insert(crate::step::brep::topo::BRepEdge {
            v_low: v0, v_high: v1, curve: curve_3d.clone(), tolerance: 1e-4,
            pcurves: std::collections::HashMap::from([(fk, pc_normal.clone())]),
        });
        let e2 = reg.edges.insert(crate::step::brep::topo::BRepEdge {
            v_low: v0, v_high: v1, curve: curve_3d.clone(), tolerance: 1e-4,
            pcurves: std::collections::HashMap::from([(fk, pc_normal.clone())]),
        });
        let e3 = reg.edges.insert(crate::step::brep::topo::BRepEdge {
            v_low: v0, v_high: v1, curve: curve_3d.clone(), tolerance: 1e-4,
            pcurves: std::collections::HashMap::from([(fk, pc_normal)]),
        });
        let e4 = reg.edges.insert(crate::step::brep::topo::BRepEdge {
            v_low: v0, v_high: v1, curve: curve_3d, tolerance: 1e-4,
            pcurves: std::collections::HashMap::from([(fk, pc_shifted)]),
        });
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
            (e3, Orientation::Forward),
            (e4, Orientation::Forward),
        ];
        let report = fix_shifted_pcurves(wk, fk, &mut reg);
        // One PCurve should be shifted back to match the cluster
        assert!(
            report.shifts_applied > 0,
            "shifted cylinder PCurve should be detected and corrected"
        );
    }

    #[test]
    fn test_no_false_positive() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Cylinder {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 1.0,
        };
        let v0 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 1.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
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
            origin: Vec3::new(1.0, 0.0, 0.0),
            direction: Vec3::new(0.0, 0.0, 1.0),
        };
        // Both PCurves are close to U=0 — not shifted
        let pc = CurveGeom::Line {
            origin: Vec3::new(0.1, 0.0, 0.0),
            direction: Vec3::new(0.5, 0.0, 0.0),
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone());
        let e2 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc);
        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
        ];
        let report = fix_shifted_pcurves(wk, fk, &mut reg);
        assert_eq!(
            report.shifts_applied, 0,
            "valid unshifted PCurves should not be modified"
        );
    }
}
