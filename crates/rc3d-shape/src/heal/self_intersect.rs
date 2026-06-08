//! UV boundary self-intersection repair (OCC ShapeFix_Wire::FixSelfIntersection).

use super::curve_trim::split_edge_at_params;
use super::geom2d::segment_intersection_strict;
use super::wire_ops::reorder_wire_edges;
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation, WireKey};

/// Represents an intersection point between two PCurve segments.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct IntersectionPoint {
    pub ta: f32,       // parameter along segment_a (0..1)
    pub tb: f32,       // parameter along segment_b (0..1)
    pub uv: (f64, f64), // UV coordinates of intersection
    pub edge_a: usize,  // index of first intersecting edge
    pub edge_b: usize,  // index of second intersecting edge
}

#[derive(Debug, Default)]
pub struct SelfIntersectReport {
    pub intersections_found: usize,
    pub edges_split: usize,
    pub wires_rebuilt: bool,
}

/// Detect and fix self-intersections in a face's outer wire UV boundary.
/// Returns report. If unfixable (>50 intersections), marks face failed via
/// returning an empty wire (the caller should push to skip_face_keys).
pub fn fix_self_intersecting_wire(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepStore,
) -> SelfIntersectReport {
    let mut report = SelfIntersectReport::default();

    let edges: Vec<(EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(wire_key) else { return report; };
        wire.edges.clone()
    };

    if edges.len() < 3 {
        return report;
    }

    // Collect UV segments
    let n = edges.len();
    let mut segments: Vec<(usize, (f32, f32), (f32, f32))> = Vec::with_capacity(n);
    for (i, &(ek, orient)) in edges.iter().enumerate() {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => return report,
        };
        let pc = match edge.pcurves.get(&face_key) {
            Some(p) => p,
            None => return report,
        };
        let (t0, t1) = if orient == Orientation::Reversed {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };
        let uv0 = pc.d0(t0);
        let uv1 = pc.d0(t1);
        segments.push((i, (uv0.0, uv0.1), (uv1.0, uv1.1)));
    }

    // Find non-adjacent intersections
    let mut intersections: Vec<IntersectionPoint> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            // Skip adjacent and wrap-around-adjacent
            if j == i + 1 {
                continue;
            }
            if i == 0 && j == n - 1 {
                continue;
            }

            let (_, a0, a1) = segments[i];
            let (_, b0, b1) = segments[j];
            if let Some((t, u)) = segment_intersection_2d(a0, a1, b0, b1) {
                // Discard endpoint intersections
                if t < 1e-6 || t > 1.0 - 1e-6 || u < 1e-6 || u > 1.0 - 1e-6 {
                    continue;
                }
                let uv_x = a0.0 + t * (a1.0 - a0.0);
                let uv_y = a0.1 + t * (a1.1 - a0.1);
                intersections.push(IntersectionPoint {
                    ta: t,
                    tb: u,
                    uv: (uv_x as f64, uv_y as f64),
                    edge_a: i,
                    edge_b: j,
                });
                report.intersections_found += 1;
            }
        }
    }

    if intersections.is_empty() {
        return report;
    }

    if intersections.len() > 50 {
        log::warn!(
            "[BRep heal] FixSelfIntersection face {:?}: {} intersections > 50, marking failed",
            face_key,
            intersections.len()
        );
        if let Some(wire) = reg.wires.get_mut(wire_key) {
            wire.edges.clear(); // signal failure
        }
        return report;
    }

    // Collect split parameters per edge
    let mut edge_splits: Vec<Vec<f32>> = vec![Vec::new(); n];
    for ip in &intersections {
        edge_splits[ip.edge_a].push(ip.ta);
        edge_splits[ip.edge_b].push(ip.tb);
    }

    // Split edges at intersection parameters using trimmed sub-curves
    let mut new_edges: Vec<(EdgeKey, Orientation)> = Vec::new();
    for (i, _) in edges.iter().enumerate() {
        let splits = &edge_splits[i];
        let (ek, orient) = edges[i];
        if splits.is_empty() {
            new_edges.push(edges[i]);
            continue;
        }
        let mut split_params = splits.clone();
        split_params.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        split_params.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        let parts = split_edge_at_params(ek, face_key, orient, &split_params, reg);
        report.edges_split += parts.len().saturating_sub(1);
        new_edges.extend(parts);
    }

    if let Some(ordered) = reorder_wire_edges(&new_edges, reg) {
        if ordered.len() >= 3 {
            if let Some(wire) = reg.wires.get_mut(wire_key) {
                wire.edges = ordered;
                report.wires_rebuilt = true;
            }
        } else {
            log::warn!(
                "[BRep heal] FixSelfIntersection face {:?}: wire too short after split",
                face_key
            );
            if let Some(wire) = reg.wires.get_mut(wire_key) {
                wire.edges.clear();
            }
        }
    } else if new_edges.len() >= 3 {
        if let Some(wire) = reg.wires.get_mut(wire_key) {
            wire.edges = new_edges;
            report.wires_rebuilt = true;
        }
    } else {
        log::warn!(
            "[BRep heal] FixSelfIntersection face {:?}: could not reorder split wire",
            face_key
        );
        if let Some(wire) = reg.wires.get_mut(wire_key) {
            wire.edges.clear();
        }
    }

    report
}

/// Compute intersection of two 2D segments. Returns (t_a, t_b) where intersection = a0 + t*(a1-a0).
fn segment_intersection_2d(
    a0: (f32, f32),
    a1: (f32, f32),
    b0: (f32, f32),
    b1: (f32, f32),
) -> Option<(f32, f32)> {
    segment_intersection_strict(a0, a1, b0, b1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
    use crate::topo::BRepWire;
    use rc3d_core::math::Vec3;

    fn make_self_intersecting_wire(reg: &mut BRepStore) -> (WireKey, FaceKey) {
        // Create wire first so the face can reference it
        let wk = reg.wires.insert(BRepWire { edges: vec![] });

        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            degenerated_edges: vec![],
            color: None,
        });

        // Figure-8 shape: edges (0,0)→(2,2)→(0,2)→(2,0)→(0,0) with a crossing
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(2.0, 2.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(0.0, 2.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);

        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        // PCurves in UV space (2D), encoded as 3D lines with z=0
        let pc1 = Curve2d::Line { origin: (0.0, 0.0), direction: (2.0, 2.0) };
        let pc2 = Curve2d::Line { origin: (2.0, 2.0), direction: (-2.0, 0.0) };
        let pc3 = Curve2d::Line { origin: (0.0, 2.0), direction: (2.0, -2.0) };
        let pc4 = Curve2d::Line { origin: (2.0, 0.0), direction: (-2.0, 0.0) };

        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc1);
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc2);
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc3);
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc4);

        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
            (e3, Orientation::Forward),
            (e4, Orientation::Forward),
        ];

        (wk, fk)
    }

    #[test]
    fn test_fix_self_intersect_simple_cross() {
        let mut reg = BRepStore::new();
        let (wk, fk) = make_self_intersecting_wire(&mut reg);
        let report = fix_self_intersecting_wire(wk, fk, &mut reg);
        assert!(
            report.intersections_found > 0,
            "should detect intersection"
        );
        assert!(report.edges_split > 0, "should split edges");
        assert!(report.wires_rebuilt, "should rebuild wire");
    }

    #[test]
    fn test_no_false_positive_adjacent() {
        let mut reg = BRepStore::new();

        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            degenerated_edges: vec![],
            color: None,
        });

        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);

        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone());
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc.clone());
        let e3 = reg.add_edge_with_pcurve(v2, v0, line.clone(), 1e-4, fk, pc);

        reg.wires.get_mut(wk).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
            (e3, Orientation::Forward),
        ];

        let report = fix_self_intersecting_wire(wk, fk, &mut reg);
        assert_eq!(
            report.intersections_found, 0,
            "simple triangle should have no self-intersections"
        );
    }

    #[test]
    fn test_fix_self_intersect_too_many() {
        let mut reg = BRepStore::new();
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        // Create edges that all cross through the origin (0,0) at interior t≈0.5.
        // Each edge goes from one point on the unit circle to the opposite point,
        // so every non-adjacent pair intersects at (0,0) — far more than the 50 cap.
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let mut edges = Vec::new();
        for i in 0..60 {
            let angle = (i as f32 / 60.0) * std::f32::consts::TAU;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let a = reg.find_or_add_vertex(Vec3::new(cos_a, sin_a, 0.0), 1e-4);
            let b = reg.find_or_add_vertex(Vec3::new(-cos_a, -sin_a, 0.0), 1e-4);
            let pc = Curve2d::Line { origin: (cos_a, sin_a), direction: (-2.0 * cos_a, -2.0 * sin_a) };
            let ek = reg.add_edge_with_pcurve(a, b, line.clone(), 1e-4, fk, pc);
            edges.push((ek, Orientation::Forward));
        }
        reg.wires.get_mut(wk).unwrap().edges = edges;
        let _report = fix_self_intersecting_wire(wk, fk, &mut reg);
        // Should detect many intersections and clear the wire (>50 cap)
        let wire_empty = reg.wires.get(wk).unwrap().edges.is_empty();
        assert!(wire_empty, ">50 intersections should clear the wire");
    }
}
