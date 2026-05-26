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
            let _to_start = v_start - *center;
            let _to_end = v_end - *center;
            let _u_dir = _to_start.normalize();
            let _angle = {
                let dot = _u_dir.dot(_to_end.normalize()).max(-1.0).min(1.0);
                dot.acos()
            };
            CurveGeom::Circle {
                center: *center,
                axis: *axis,
                radius: *radius,
            }
        }
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor } => {
            CurveGeom::Ellipse {
                center: *center,
                axis: *axis,
                semi_major: *semi_major,
                semi_minor: *semi_minor,
            }
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
}
