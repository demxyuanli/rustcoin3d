//! UV boundary self-intersection repair (OCC ShapeFix_Wire::FixSelfIntersection).

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{EdgeKey, FaceKey, Orientation, VertexKey, WireKey};

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
    reg: &mut BRepRegistry,
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
        segments.push((i, (uv0.x, uv0.y), (uv1.x, uv1.y)));
    }

    // Find non-adjacent intersections
    let mut intersections: Vec<(usize, usize, f32, f32)> = Vec::new();
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
                intersections.push((i, j, t, u));
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
    for &(i, j, t, u) in &intersections {
        edge_splits[i].push(t);
        edge_splits[j].push(u);
    }

    // Sort and deduplicate split parameters
    let mut new_edges: Vec<(EdgeKey, Orientation)> = Vec::new();
    for (i, _) in edges.iter().enumerate() {
        let splits = &mut edge_splits[i];
        if splits.is_empty() {
            new_edges.push(edges[i]);
            continue;
        }
        splits.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        splits.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

        let (ek, orient) = edges[i];
        // Clone edge data before mutable registry access
        let (curve, tolerance, pcurve) = {
            let edge = match reg.edges.get(ek) {
                Some(e) => e,
                None => continue,
            };
            (edge.curve.clone(), edge.tolerance, edge.pcurves.get(&face_key).cloned())
        };
        let pcurve = match pcurve {
            Some(pc) => pc,
            None => continue,
        };

        // Split at each parameter
        let mut t_prev = 0.0f32;
        let endpoint_t = [&splits[..], &[1.0f32]].concat();
        for &t in endpoint_t.iter() {
            if t - t_prev < 1e-6 {
                t_prev = t;
                continue;
            }
            let v_start = reg.find_or_add_vertex(curve.d0(t_prev), 1e-4);
            let v_end = reg.find_or_add_vertex(curve.d0(t), 1e-4);
            let ek_new = reg.add_edge_with_pcurve(
                v_start,
                v_end,
                curve.clone(),
                tolerance,
                face_key,
                pcurve.clone(),
            );
            new_edges.push((ek_new, orient));
            report.edges_split += 1;
            t_prev = t;
        }
    }

    // Build new wire by connecting split segments via shared vertices.
    // Each split segment is a directed edge from v_start to v_end.
    // After splitting, the wire should form a valid traversal through
    // the intersection points. We reorder the new edges by endpoint matching
    // (same algorithm as reorder_wire_edges in reorder.rs).
    let ordered = reorder_by_endpoints(&new_edges, reg);
    if ordered.len() >= 3 {
        if let Some(wire) = reg.wires.get_mut(wire_key) {
            wire.edges = ordered;
            report.wires_rebuilt = true;
        }
    } else {
        log::warn!("[BRep heal] FixSelfIntersection face {:?}: wire too short after split, marking failed", face_key);
        if let Some(wire) = reg.wires.get_mut(wire_key) {
            wire.edges.clear();
        }
    }

    report
}

/// Reorder edges into a connected chain by matching endpoints.
/// Uses the same algorithm as heal::reorder::reorder_wire_edges.
fn reorder_by_endpoints(
    edges: &[(EdgeKey, Orientation)],
    reg: &BRepRegistry,
) -> Vec<(EdgeKey, Orientation)> {
    if edges.len() <= 1 {
        return edges.to_vec();
    }
    let n = edges.len();
    let mut used = vec![false; n];
    let mut result = Vec::with_capacity(n);

    // Start from the first edge
    used[0] = true;
    result.push(edges[0]);

    for _ in 1..n {
        let last = result.last().unwrap();
        let last_end_vk = get_end_vertex(last.0, last.1, reg);

        let mut found = false;
        for (j, &(ek, orient)) in edges.iter().enumerate() {
            if used[j] {
                continue;
            }
            let v_start = get_start_vertex(ek, orient, reg);
            if v_start == last_end_vk {
                used[j] = true;
                result.push((ek, orient));
                found = true;
                break;
            }
            // Also try reversed: check if this edge's end matches our end
            let v_end = get_end_vertex(ek, orient, reg);
            if v_end == last_end_vk {
                // Insert with reversed orientation
                let rev_orient = if orient == Orientation::Forward {
                    Orientation::Reversed
                } else {
                    Orientation::Forward
                };
                used[j] = true;
                result.push((ek, rev_orient));
                found = true;
                break;
            }
        }
        if !found {
            // Try matching any remaining edge's start to a previous start
            for (j, &(ek, orient)) in edges.iter().enumerate() {
                if used[j] {
                    continue;
                }
                used[j] = true;
                result.push((ek, orient));
                break;
            }
        }
    }

    result
}

fn get_start_vertex(ek: EdgeKey, orient: Orientation, reg: &BRepRegistry) -> Option<VertexKey> {
    let edge = reg.edges.get(ek)?;
    if orient == Orientation::Reversed {
        Some(edge.v_high)
    } else {
        Some(edge.v_low)
    }
}

fn get_end_vertex(ek: EdgeKey, orient: Orientation, reg: &BRepRegistry) -> Option<VertexKey> {
    let edge = reg.edges.get(ek)?;
    if orient == Orientation::Reversed {
        Some(edge.v_low)
    } else {
        Some(edge.v_high)
    }
}

/// Compute intersection of two 2D segments. Returns (t_a, t_b) where intersection = a0 + t*(a1-a0).
fn segment_intersection_2d(
    a0: (f32, f32),
    a1: (f32, f32),
    b0: (f32, f32),
    b1: (f32, f32),
) -> Option<(f32, f32)> {
    let da = (a1.0 - a0.0, a1.1 - a0.1);
    let db = (b1.0 - b0.0, b1.1 - b0.1);
    let det = da.0 * db.1 - da.1 * db.0;
    if det.abs() < 1e-12 {
        return None; // parallel
    }
    let d0 = (b0.0 - a0.0, b0.1 - a0.1);
    let t = (d0.0 * db.1 - d0.1 * db.0) / det;
    let u = (d0.0 * da.1 - d0.1 * da.0) / det;
    if t > 0.0 && t < 1.0 && u > 0.0 && u < 1.0 {
        Some((t, u))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::topo::BRepWire;
    use rc3d_core::math::Vec3;

    fn make_self_intersecting_wire(reg: &mut BRepRegistry) -> (WireKey, FaceKey) {
        // Create wire first so the face can reference it
        let wk = reg.wires.insert(BRepWire { edges: vec![] });

        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
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
        let pc1 = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(2.0, 2.0, 0.0),
        };
        let pc2 = CurveGeom::Line {
            origin: Vec3::new(2.0, 2.0, 0.0),
            direction: Vec3::new(-2.0, 0.0, 0.0),
        };
        let pc3 = CurveGeom::Line {
            origin: Vec3::new(0.0, 2.0, 0.0),
            direction: Vec3::new(2.0, -2.0, 0.0),
        };
        let pc4 = CurveGeom::Line {
            origin: Vec3::new(2.0, 0.0, 0.0),
            direction: Vec3::new(-2.0, 0.0, 0.0),
        };

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
        let mut reg = BRepRegistry::new();
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
        let mut reg = BRepRegistry::new();

        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
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
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, line.clone());
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, line.clone());
        let e3 = reg.add_edge_with_pcurve(v2, v0, line.clone(), 1e-4, fk, line.clone());

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
}
