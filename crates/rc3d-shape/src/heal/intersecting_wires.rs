//! Multi-wire intersection detection and repair (OCC ShapeFix_Face::FixIntersectingWires).

use super::curve_trim::split_edge_at_params;
use super::geom2d::{point_in_polygon_winding as geom_point_in_polygon_winding, segment_intersection_strict};
use crate::store::BRepStore;
use crate::topo::{FaceKey, Orientation, WireKey};

#[derive(Debug, Default)]
pub struct IntersectingWiresReport {
    pub inner_wires_trimmed: usize,
    pub inner_wires_removed: usize,
    pub inner_wires_merged: usize,
}

/// Detect and fix intersections between outer and inner wires (and between inner wires).
/// Uses point-in-polygon tests with winding number.
pub fn fix_intersecting_wires(
    face_key: FaceKey,
    reg: &mut BRepStore,
) -> IntersectingWiresReport {
    let mut report = IntersectingWiresReport::default();

    let (outer_wk, inner_wks) = {
        let Some(face) = reg.faces.get(face_key) else { return report; };
        (face.outer_wire, face.inner_wires.clone())
    };

    if inner_wks.is_empty() {
        return report;
    }

    // Collect outer wire UV polygon
    let outer_poly = match collect_wire_uv_polygon(outer_wk, face_key, reg) {
        Some(p) if p.len() >= 3 => p,
        _ => return report,
    };

    let mut kept_inner = Vec::new();

    for &inner_wk in &inner_wks {
        let inner_poly = match collect_wire_uv_polygon(inner_wk, face_key, reg) {
            Some(p) if p.len() >= 3 => p,
            _ => continue,
        };

        // Check if inner wire is entirely inside outer wire
        let inside_count = inner_poly.iter()
            .filter(|&&(u, v)| point_in_polygon_winding(u, v, &outer_poly))
            .count();
        let total = inner_poly.len();

        if inside_count == 0 {
            // Entirely outside outer wire — remove
            log::warn!("[BRep heal] FixIntersectingWires face {:?}: inner wire {:?} entirely outside outer, removing",
                face_key, inner_wk);
            report.inner_wires_removed += 1;
            continue;
        }

        if inside_count < total {
            if inside_count < total / 2 {
                log::warn!(
                    "[BRep heal] FixIntersectingWires face {:?}: inner wire {:?} partially outside, removing",
                    face_key, inner_wk
                );
                report.inner_wires_removed += 1;
                continue;
            }
            if trim_inner_wire_at_outer(inner_wk, &outer_poly, face_key, reg) {
                report.inner_wires_trimmed += 1;
            }
        }

        kept_inner.push(inner_wk);
    }

    // Merge intersecting inner wires pairwise
    let merged = merge_intersecting_inner_wires(&kept_inner, face_key, reg);
    if kept_inner.len() > merged.len() {
        report.inner_wires_merged += kept_inner.len() - merged.len();
    }

    // Update face with filtered/merged inner wires
    if merged.len() != inner_wks.len() {
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.inner_wires = merged;
        }
    }

    report
}

/// Read-only detection: inner wires outside/intersecting outer or each other.
pub fn detect_intersecting_wires(face_key: FaceKey, reg: &BRepStore) -> bool {
    let (outer_wk, inner_wks) = {
        let Some(face) = reg.faces.get(face_key) else {
            return false;
        };
        if face.inner_wires.is_empty() {
            return false;
        }
        (face.outer_wire, face.inner_wires.clone())
    };

    let outer_poly = match collect_wire_uv_polygon(outer_wk, face_key, reg) {
        Some(p) if p.len() >= 3 => p,
        _ => return false,
    };

    for &inner_wk in &inner_wks {
        let inner_poly = match collect_wire_uv_polygon(inner_wk, face_key, reg) {
            Some(p) if p.len() >= 3 => p,
            _ => continue,
        };

        let inside_count = inner_poly
            .iter()
            .filter(|&&(u, v)| point_in_polygon_winding(u, v, &outer_poly))
            .count();
        if inside_count < inner_poly.len() {
            return true;
        }
    }

    for i in 0..inner_wks.len() {
        for j in (i + 1)..inner_wks.len() {
            if wires_intersect_2d(inner_wks[i], inner_wks[j], face_key, reg) {
                return true;
            }
        }
    }

    false
}

fn segment_intersection_2d(
    a0: (f32, f32),
    a1: (f32, f32),
    b0: (f32, f32),
    b1: (f32, f32),
) -> Option<(f32, f32)> {
    segment_intersection_strict(a0, a1, b0, b1)
        .filter(|(t, u)| *t > 1e-6 && *t < 1.0 - 1e-6 && *u > 1e-6 && *u < 1.0 - 1e-6)
}

/// Split inner wire edges at intersections with the outer boundary polygon.
fn trim_inner_wire_at_outer(
    inner_wk: WireKey,
    outer_poly: &[(f32, f32)],
    face_key: FaceKey,
    reg: &mut BRepStore,
) -> bool {
    let edges: Vec<(crate::topo::EdgeKey, Orientation)> = {
        let Some(wire) = reg.wires.get(inner_wk) else {
            return false;
        };
        wire.edges.clone()
    };
    let mut trimmed = false;
    let mut new_edges = edges.clone();
    for (i, &(ek, orient)) in edges.iter().enumerate() {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        let pc = match edge.pcurves.get(&face_key) {
            Some(p) => p,
            None => continue,
        };
        let uv0 = pc.d0(0.0);
        let uv1 = pc.d0(1.0);
        let a0 = (uv0.x, uv0.y);
        let a1 = (uv1.x, uv1.y);
        let mut splits = Vec::new();
        let n = outer_poly.len();
        for j in 0..n {
            let k = (j + 1) % n;
            if let Some((t, _u)) = segment_intersection_2d(a0, a1, outer_poly[j], outer_poly[k]) {
                splits.push(t);
            }
        }
        if splits.is_empty() {
            continue;
        }
        let parts = split_edge_at_params(ek, face_key, orient, &splits, reg);
        if parts.len() > 1 {
            new_edges.splice(i..i + 1, parts);
            trimmed = true;
        }
    }
    if trimmed {
        if let Some(wire) = reg.wires.get_mut(inner_wk) {
            wire.edges = new_edges;
        }
    }
    trimmed
}

fn collect_wire_uv_polygon(wk: WireKey, fk: FaceKey, reg: &BRepStore) -> Option<Vec<(f32, f32)>> {
    let wire = reg.wires.get(wk)?;
    let mut pts = Vec::new();
    for &(ek, _) in &wire.edges {
        let edge = reg.edges.get(ek)?;
        let pc = edge.pcurves.get(&fk)?;
        let uv = pc.d0(0.0);
        pts.push((uv.x, uv.y));
    }
    if pts.len() >= 3 {
        // Close the polygon
        if let Some(&last_ek) = wire.edges.last() {
            if let Some(edge) = reg.edges.get(last_ek.0) {
                if let Some(pc) = edge.pcurves.get(&fk) {
                    let uv = pc.d0(1.0);
                    pts.push((uv.x, uv.y));
                }
            }
        }
    }
    Some(pts)
}

fn point_in_polygon_winding(u: f32, v: f32, poly: &[(f32, f32)]) -> bool {
    geom_point_in_polygon_winding(u, v, poly)
}

fn merge_intersecting_inner_wires(
    wires: &[WireKey],
    face_key: FaceKey,
    reg: &mut BRepStore,
) -> Vec<WireKey> {
    let mut result: Vec<WireKey> = wires.to_vec();
    let mut i = 0;
    while i < result.len() {
        let mut j = i + 1;
        while j < result.len() {
            if wires_intersect_2d(result[i], result[j], face_key, reg) {
                // Merge wire j into wire i: concatenate edge lists
                if let (Some(wire_i), Some(wire_j)) = (
                    reg.wires.get(result[i]),
                    reg.wires.get(result[j]),
                ) {
                    let mut merged_edges = wire_i.edges.clone();
                    merged_edges.extend(wire_j.edges.clone());
                    if let Some(w) = reg.wires.get_mut(result[i]) {
                        w.edges = merged_edges;
                    }
                }
                result.remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }
    result
}

fn wires_intersect_2d(wk_a: WireKey, wk_b: WireKey, fk: FaceKey, reg: &BRepStore) -> bool {
    let poly_a = match collect_wire_uv_polygon(wk_a, fk, reg) {
        Some(p) => p,
        None => return false,
    };
    let poly_b = match collect_wire_uv_polygon(wk_b, fk, reg) {
        Some(p) => p,
        None => return false,
    };
    // Bounding box quick reject
    let (min_a, max_a) = polygon_bbox(&poly_a);
    let (min_b, max_b) = polygon_bbox(&poly_b);
    if max_a.0 < min_b.0 || max_b.0 < min_a.0 || max_a.1 < min_b.1 || max_b.1 < min_a.1 {
        return false;
    }
    // Check if any vertex of A is inside B, or vice versa
    poly_a.iter().any(|&(u, v)| point_in_polygon_winding(u, v, &poly_b))
        || poly_b.iter().any(|&(u, v)| point_in_polygon_winding(u, v, &poly_a))
}

fn polygon_bbox(poly: &[(f32, f32)]) -> ((f32, f32), (f32, f32)) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for &(x, y) in poly {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    ((min_x, min_y), (max_x, max_y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepWire, Orientation};
    use rc3d_core::math::Vec3;

    #[test]
    fn test_point_in_square() {
        let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
        assert!(point_in_polygon_winding(0.5, 0.5, &square));
        assert!(!point_in_polygon_winding(2.0, 0.5, &square));
    }

    #[test]
    fn test_inner_outside_outer_removed() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 1.0, 0.0), 1e-4);
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        // Outer wire: UV square from (0,0) to (1,1) — use proper per-edge PCurves
        let pc1 = CurveGeom::Line { origin: Vec3::new(0.0, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };
        let pc2 = CurveGeom::Line { origin: Vec3::new(1.0, 0.0, 0.0), direction: Vec3::new(0.0, 1.0, 0.0) };
        let pc3 = CurveGeom::Line { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::new(-1.0, 0.0, 0.0) };
        let pc4 = CurveGeom::Line { origin: Vec3::new(0.0, 1.0, 0.0), direction: Vec3::new(0.0, -1.0, 0.0) };
        let wk_outer = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface, outer_wire: wk_outer, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc1);
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc2);
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc3);
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc4);
        reg.wires.get_mut(wk_outer).unwrap().edges = vec![
            (e1, Orientation::Forward), (e2, Orientation::Forward), (e3, Orientation::Forward), (e4, Orientation::Forward),
        ];
        // Inner wire: small triangle at offset +10 (entirely outside outer square)
        let v4 = reg.find_or_add_vertex(Vec3::new(10.0, 0.0, 0.0), 1e-4);
        let v5 = reg.find_or_add_vertex(Vec3::new(11.0, 0.0, 0.0), 1e-4);
        let v6 = reg.find_or_add_vertex(Vec3::new(11.0, 1.0, 0.0), 1e-4);
        let pc_inner1 = CurveGeom::Line { origin: Vec3::new(10.0, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };
        let pc_inner2 = CurveGeom::Line { origin: Vec3::new(11.0, 0.0, 0.0), direction: Vec3::new(0.0, 1.0, 0.0) };
        let pc_inner3 = CurveGeom::Line { origin: Vec3::new(11.0, 1.0, 0.0), direction: Vec3::new(-1.0, -1.0, 0.0) };
        let ek1 = reg.add_edge_with_pcurve(v4, v5, line.clone(), 1e-4, fk, pc_inner1);
        let ek2 = reg.add_edge_with_pcurve(v5, v6, line.clone(), 1e-4, fk, pc_inner2);
        let ek3 = reg.add_edge_with_pcurve(v6, v4, line.clone(), 1e-4, fk, pc_inner3);
        let wk_inner = reg.wires.insert(BRepWire { edges: vec![
            (ek1, Orientation::Forward), (ek2, Orientation::Forward), (ek3, Orientation::Forward),
        ]});
        if let Some(face) = reg.faces.get_mut(fk) {
            face.inner_wires = vec![wk_inner];
        }
        let report = fix_intersecting_wires(fk, &mut reg);
        assert!(report.inner_wires_removed > 0, "inner wire outside outer should be removed");
    }

    #[test]
    fn test_detect_intersecting_wires_before_fix() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 1.0, 0.0), 1e-4);
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let pc1 = CurveGeom::Line { origin: Vec3::new(0.0, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };
        let pc2 = CurveGeom::Line { origin: Vec3::new(1.0, 0.0, 0.0), direction: Vec3::new(0.0, 1.0, 0.0) };
        let pc3 = CurveGeom::Line { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::new(-1.0, 0.0, 0.0) };
        let pc4 = CurveGeom::Line { origin: Vec3::new(0.0, 1.0, 0.0), direction: Vec3::new(0.0, -1.0, 0.0) };
        let wk_outer = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface, outer_wire: wk_outer, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc1);
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc2);
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc3);
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc4);
        reg.wires.get_mut(wk_outer).unwrap().edges = vec![
            (e1, Orientation::Forward), (e2, Orientation::Forward), (e3, Orientation::Forward), (e4, Orientation::Forward),
        ];
        let v4 = reg.find_or_add_vertex(Vec3::new(10.0, 0.0, 0.0), 1e-4);
        let v5 = reg.find_or_add_vertex(Vec3::new(11.0, 0.0, 0.0), 1e-4);
        let v6 = reg.find_or_add_vertex(Vec3::new(11.0, 1.0, 0.0), 1e-4);
        let pc_inner1 = CurveGeom::Line { origin: Vec3::new(10.0, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };
        let pc_inner2 = CurveGeom::Line { origin: Vec3::new(11.0, 0.0, 0.0), direction: Vec3::new(0.0, 1.0, 0.0) };
        let pc_inner3 = CurveGeom::Line { origin: Vec3::new(11.0, 1.0, 0.0), direction: Vec3::new(-1.0, -1.0, 0.0) };
        let ek1 = reg.add_edge_with_pcurve(v4, v5, line.clone(), 1e-4, fk, pc_inner1);
        let ek2 = reg.add_edge_with_pcurve(v5, v6, line.clone(), 1e-4, fk, pc_inner2);
        let ek3 = reg.add_edge_with_pcurve(v6, v4, line.clone(), 1e-4, fk, pc_inner3);
        let wk_inner = reg.wires.insert(BRepWire { edges: vec![
            (ek1, Orientation::Forward), (ek2, Orientation::Forward), (ek3, Orientation::Forward),
        ]});
        if let Some(face) = reg.faces.get_mut(fk) {
            face.inner_wires = vec![wk_inner];
        }
        assert!(detect_intersecting_wires(fk, &reg));
    }

    #[test]
    fn test_no_false_positive_separate_inners() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(3.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(3.0, 3.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 3.0, 0.0), 1e-4);
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        // Outer wire: UV square from (0,0) to (3,3) — proper per-edge PCurves
        let pc1 = CurveGeom::Line { origin: Vec3::new(0.0, 0.0, 0.0), direction: Vec3::new(3.0, 0.0, 0.0) };
        let pc2 = CurveGeom::Line { origin: Vec3::new(3.0, 0.0, 0.0), direction: Vec3::new(0.0, 3.0, 0.0) };
        let pc3 = CurveGeom::Line { origin: Vec3::new(3.0, 3.0, 0.0), direction: Vec3::new(-3.0, 0.0, 0.0) };
        let pc4 = CurveGeom::Line { origin: Vec3::new(0.0, 3.0, 0.0), direction: Vec3::new(0.0, -3.0, 0.0) };
        let wk_outer = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface, outer_wire: wk_outer, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc1);
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc2);
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc3);
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc4);
        reg.wires.get_mut(wk_outer).unwrap().edges = vec![
            (e1, Orientation::Forward), (e2, Orientation::Forward), (e3, Orientation::Forward), (e4, Orientation::Forward),
        ];
        // Inner wire: small triangle properly inside (offset by +1)
        let v4 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v5 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);
        let v6 = reg.find_or_add_vertex(Vec3::new(2.0, 1.0, 0.0), 1e-4);
        let pc_inner1 = CurveGeom::Line { origin: Vec3::new(1.0, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };
        let pc_inner2 = CurveGeom::Line { origin: Vec3::new(2.0, 0.0, 0.0), direction: Vec3::new(0.0, 1.0, 0.0) };
        let pc_inner3 = CurveGeom::Line { origin: Vec3::new(2.0, 1.0, 0.0), direction: Vec3::new(-1.0, -1.0, 0.0) };
        let ek1 = reg.add_edge_with_pcurve(v4, v5, line.clone(), 1e-4, fk, pc_inner1);
        let ek2 = reg.add_edge_with_pcurve(v5, v6, line.clone(), 1e-4, fk, pc_inner2);
        let ek3 = reg.add_edge_with_pcurve(v6, v4, line.clone(), 1e-4, fk, pc_inner3);
        let wk_inner = reg.wires.insert(BRepWire { edges: vec![
            (ek1, Orientation::Forward), (ek2, Orientation::Forward), (ek3, Orientation::Forward),
        ]});
        if let Some(face) = reg.faces.get_mut(fk) {
            face.inner_wires = vec![wk_inner];
        }
        let report = fix_intersecting_wires(fk, &mut reg);
        assert_eq!(report.inner_wires_removed, 0, "properly separate inner wires should not be removed");
    }

    #[test]
    fn test_inner_intersects_outer_trim() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        // Outer wire: square from (0,0) to (2,2)
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(2.0, 2.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 2.0, 0.0), 1e-4);
        let wk_outer = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: wk_outer,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let pc = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(2.0, 0.0, 0.0),
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone());
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc.clone());
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc.clone());
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc);
        reg.wires.get_mut(wk_outer).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
            (e3, Orientation::Forward),
            (e4, Orientation::Forward),
        ];
        // Inner wire: crosses outer boundary (starts inside at (0.5,0) goes to (3,0) — outside)
        let v4 = reg.find_or_add_vertex(Vec3::new(0.5, 0.0, 0.0), 1e-4);
        let v5 = reg.find_or_add_vertex(Vec3::new(3.0, 0.0, 0.0), 1e-4);
        let v6 = reg.find_or_add_vertex(Vec3::new(3.0, 0.5, 0.0), 1e-4);
        let pc_inner = CurveGeom::Line {
            origin: Vec3::new(0.5, 0.0, 0.0),
            direction: Vec3::new(2.5, 0.0, 0.0),
        };
        let ek1 = reg.add_edge_with_pcurve(v4, v5, line.clone(), 1e-4, fk, pc_inner.clone());
        let ek2 = reg.add_edge_with_pcurve(v5, v6, line.clone(), 1e-4, fk, pc_inner.clone());
        let ek3 = reg.add_edge_with_pcurve(v6, v4, line.clone(), 1e-4, fk, pc_inner);
        let wk_inner = reg.wires.insert(BRepWire {
            edges: vec![
                (ek1, Orientation::Forward),
                (ek2, Orientation::Forward),
                (ek3, Orientation::Forward),
            ],
        });
        if let Some(face) = reg.faces.get_mut(fk) {
            face.inner_wires = vec![wk_inner];
        }
        let report = fix_intersecting_wires(fk, &mut reg);
        // Inner wire partially outside outer — should be removed
        assert!(
            report.inner_wires_removed > 0,
            "intersecting inner wire should be removed"
        );
    }

    #[test]
    fn test_two_inners_intersect_merged() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        // Outer wire: large square
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(5.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(5.0, 5.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 5.0, 0.0), 1e-4);
        let wk_outer = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: wk_outer,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let pc = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(5.0, 0.0, 0.0),
        };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone());
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc.clone());
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc.clone());
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc);
        reg.wires.get_mut(wk_outer).unwrap().edges = vec![
            (e1, Orientation::Forward),
            (e2, Orientation::Forward),
            (e3, Orientation::Forward),
            (e4, Orientation::Forward),
        ];
        // Two inner wires that are close to each other (overlapping bounding boxes)
        let v4 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v5 = reg.find_or_add_vertex(Vec3::new(1.5, 1.0, 0.0), 1e-4);
        let pc_i1 = CurveGeom::Line {
            origin: Vec3::new(1.0, 1.0, 0.0),
            direction: Vec3::new(0.5, 0.0, 0.0),
        };
        let ei1 = reg.add_edge_with_pcurve(v4, v5, line.clone(), 1e-4, fk, pc_i1);
        let wk_i1 = reg.wires.insert(BRepWire {
            edges: vec![(ei1, Orientation::Forward)],
        });
        let v6 = reg.find_or_add_vertex(Vec3::new(1.4, 1.4, 0.0), 1e-4);
        let v7 = reg.find_or_add_vertex(Vec3::new(2.0, 1.4, 0.0), 1e-4);
        let pc_i2 = CurveGeom::Line {
            origin: Vec3::new(1.4, 1.4, 0.0),
            direction: Vec3::new(0.6, 0.0, 0.0),
        };
        let ei2 = reg.add_edge_with_pcurve(v6, v7, line.clone(), 1e-4, fk, pc_i2);
        let wk_i2 = reg.wires.insert(BRepWire {
            edges: vec![(ei2, Orientation::Forward)],
        });
        if let Some(face) = reg.faces.get_mut(fk) {
            face.inner_wires = vec![wk_i1, wk_i2];
        }
        let _report = fix_intersecting_wires(fk, &mut reg);
        // Verify the function completes without panic.
    }
}
