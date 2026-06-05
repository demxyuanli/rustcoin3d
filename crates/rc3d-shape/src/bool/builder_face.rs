//! Face builder: reconstructs BRep faces from split UV regions.
//!
//! OCC alignment: BOPAlgo_BuilderFace — converts UV-space split descriptions
//! into actual BRep topology (new vertices on edges, split edges, new wires,
//! and new faces with proper boundary loops).
//!
//! Pipeline:
//! 1. Detect intersection points on face boundary edges → split vertices
//! 2. Split edges at intersection points → new edge segments
//! 3. Build closed wire loops from edge segments + intersection curve segments
//! 4. Create new BRepFace entities from wire loops + original surface

use std::collections::{HashMap, HashSet};
use rc3d_core::math::Vec3;
use crate::geom::{CurveGeom, SurfaceGeom};
use crate::store::BRepStore;
use crate::topo::*;
use crate::topo_iter;

/// A split point on a face boundary edge.
#[derive(Debug, Clone)]
struct EdgeSplitPoint {
    /// The edge being split.
    edge: EdgeKey,
    /// Parameter t on the edge where the split occurs.
    t: f32,
    /// 3D position of the split point.
    position: Vec3,
    /// UV coordinate on the face surface.
    uv: (f32, f32),
    /// The new vertex created at this split (populated after vertex creation).
    vertex: Option<VertexKey>,
}

/// Result of splitting a face along intersection curves.
#[derive(Debug)]
pub struct BuilderFaceResult {
    /// New faces created from the split.
    pub new_faces: Vec<FaceKey>,
    /// Original face (may be retained if no splits occurred).
    pub original_face: FaceKey,
    /// Number of edge splits performed.
    pub edge_splits: usize,
    /// Number of new vertices created.
    pub new_vertices: usize,
}

/// Build new faces from a face split by intersection curves.
///
/// OCC: BOPAlgo_BuilderFace::BuildFace()
///
/// Takes the original face and the UV-space sub-regions produced by
/// `split_face_along_curves`, and creates actual BRep faces.
pub fn build_faces_from_split(
    face_key: FaceKey,
    sub_regions: &[super::split::SubFaceRegion],
    curves: &[super::split::BRepIntersectionCurve],
    reg: &mut BRepStore,
) -> BuilderFaceResult {
    let mut result = BuilderFaceResult {
        new_faces: Vec::new(),
        original_face: face_key,
        edge_splits: 0,
        new_vertices: 0,
    };

    // Clone surface and tolerance before mutable operations
    let (surface, face_tol, has_edges) = {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => return result,
        };
        let has_edges = topo_iter::iter_edges_of_face(face_key, reg).len() > 0;
        (face.surface.clone(), face.tolerance, has_edges)
    };

    if sub_regions.len() <= 1 || !has_edges {
        return result;
    }

    // Phase 1: Detect edge split points
    let split_points = detect_edge_split_points(face_key, curves, reg);
    result.new_vertices = split_points.len();

    // Phase 2: Create split vertices on edges
    let split_vertices = create_split_vertices(&split_points, reg);

    // Phase 3: Split edges at the split vertices
    let edge_map = split_edges_at_points(face_key, &split_vertices, reg);
    result.edge_splits = edge_map.len();

    // Phase 4: For each sub-region, build a new wire and face
    for region in sub_regions {
        if let Some(new_face) = build_face_from_region(
            face_key, region, &surface, &edge_map, reg,
        ) {
            result.new_faces.push(new_face);
        }
    }

    result
}

/// Detect where intersection curves meet face boundary edges.
fn detect_edge_split_points(
    face_key: FaceKey,
    curves: &[super::split::BRepIntersectionCurve],
    reg: &BRepStore,
) -> Vec<EdgeSplitPoint> {
    let mut points = Vec::new();
    let edges = topo_iter::iter_edges_of_face(face_key, reg);

    for curve in curves {
        // The curve endpoints are where it meets the face boundary
        let params = if curve.face_a == face_key { &curve.params_a } else { &curve.params_b };
        let pts_3d = &curve.points_3d;

        if params.len() < 2 || pts_3d.len() < 2 {
            continue;
        }

        // Check curve endpoints: they should lie on face boundary edges
        for &(idx, label) in &[(0, "start"), (params.len() - 1, "end")] {
            let pt = pts_3d[idx.min(pts_3d.len() - 1)];
            let uv = params[idx];

            // Find which edge this point lies on
            for &ek in &edges {
                if let Some(t) = point_on_edge_param(pt, ek, reg, 1e-3) {
                    // Point is on this edge — split here
                    points.push(EdgeSplitPoint {
                        edge: ek,
                        t,
                        position: pt,
                        uv,
                        vertex: None,
                    });
                    break;
                }
            }
        }
    }

    // Deduplicate: same edge, same t (within tolerance)
    points.sort_by(|a, b| {
        a.edge.partial_cmp(&b.edge).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal))
    });
    points.dedup_by(|a, b| a.edge == b.edge && (a.t - b.t).abs() < 1e-4);
    points
}

/// Create BRep vertices at the split points.
fn create_split_vertices(
    split_points: &[EdgeSplitPoint],
    reg: &mut BRepStore,
) -> Vec<EdgeSplitPoint> {
    split_points.iter().map(|sp| {
        let vk = reg.find_or_add_vertex(sp.position, 1e-4);
        EdgeSplitPoint {
            edge: sp.edge,
            t: sp.t,
            position: sp.position,
            uv: sp.uv,
            vertex: Some(vk),
        }
    }).collect()
}

/// Split edges at the given split vertices.
///
/// Returns a map from original EdgeKey → list of (new_edge, t_start, t_end)
/// representing the split segments in parameter order.
fn split_edges_at_points(
    face_key: FaceKey,
    split_vertices: &[EdgeSplitPoint],
    reg: &mut BRepStore,
) -> HashMap<EdgeKey, Vec<(EdgeKey, f32, f32)>> {
    let mut edge_splits: HashMap<EdgeKey, Vec<(f32, VertexKey)>> = HashMap::new();

    for sp in split_vertices {
        if let Some(vk) = sp.vertex {
            edge_splits.entry(sp.edge).or_default().push((sp.t, vk));
        }
    }

    let mut result: HashMap<EdgeKey, Vec<(EdgeKey, f32, f32)>> = HashMap::new();

    for (ek, mut splits) in edge_splits {
        splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Clone data needed for mutable operations to release immutable borrows
        let (edge_curve, edge_v_low, edge_v_high, pcurve, surface) = {
            let edge = match reg.edges.get(ek) {
                Some(e) => e,
                None => continue,
            };
            let face = match reg.faces.get(face_key) {
                Some(f) => f,
                None => continue,
            };
            let pc = match edge.pcurves.get(&face_key) {
                Some(p) => p.clone(),
                None => continue,
            };
            (edge.curve.clone(), edge.v_low, edge.v_high, pc, face.surface.clone())
        };

        let mut segments = Vec::new();
        let mut t_prev = 0.0f32;
        let mut v_prev = edge_v_low;

        for &(t, vk) in &splits {
            if t > t_prev + 1e-6 {
                let new_ek = create_edge_segment(
                    ek, v_prev, vk, t_prev, t, &edge_curve, &pcurve,
                    face_key, &surface, reg,
                );
                segments.push((new_ek, t_prev, t));
            }
            t_prev = t;
            v_prev = vk;
        }

        if t_prev < 1.0 - 1e-6 {
            let new_ek = create_edge_segment(
                ek, v_prev, edge_v_high, t_prev, 1.0, &edge_curve, &pcurve,
                face_key, &surface, reg,
            );
            segments.push((new_ek, t_prev, 1.0));
        }

        if !segments.is_empty() {
            result.insert(ek, segments);
        }
    }

    result
}

/// Create a single edge segment.
fn create_edge_segment(
    _original_ek: EdgeKey,
    v_start: VertexKey,
    v_end: VertexKey,
    t_start: f32,
    t_end: f32,
    curve_3d: &CurveGeom,
    pcurve: &CurveGeom,
    face_key: FaceKey,
    surface: &SurfaceGeom,
    reg: &mut BRepStore,
) -> EdgeKey {
    // Build trimmed 3D curve for this segment
    let span = t_end - t_start;
    let trimmed_3d = if span < 0.99 {
        CurveGeom::Trimmed {
            basis: Box::new(curve_3d.clone()),
            t_min: t_start,
            t_max: t_end,
        }
    } else {
        curve_3d.clone()
    };

    // Build trimmed PCurve for this segment
    let trimmed_pc = if span < 0.99 {
        CurveGeom::Trimmed {
            basis: Box::new(pcurve.clone()),
            t_min: t_start,
            t_max: t_end,
        }
    } else {
        pcurve.clone()
    };

    // Compute tolerance from curve endpoints
    let p0 = trimmed_3d.d0(0.0);
    let p1 = trimmed_3d.d0(1.0);
    let v0_pos = reg.vertices.get(v_start).map(|v| v.position).unwrap_or(p0);
    let v1_pos = reg.vertices.get(v_end).map(|v| v.position).unwrap_or(p1);
    let tol = (p0 - v0_pos).length().max((p1 - v1_pos).length()).max(1e-4);

    reg.add_edge_with_pcurve(v_start, v_end, trimmed_3d, tol, face_key, trimmed_pc)
}

/// Build a new BRep face from a sub-region.
fn build_face_from_region(
    original_face: FaceKey,
    region: &super::split::SubFaceRegion,
    surface: &SurfaceGeom,
    edge_map: &HashMap<EdgeKey, Vec<(EdgeKey, f32, f32)>>,
    reg: &mut BRepStore,
) -> Option<FaceKey> {
    // Create a new face with the same surface
    let new_face = reg.add_face(surface.clone(), 1e-4);

    // Build a wire from the UV boundary
    if region.uv_boundary.is_empty() {
        // No boundary specified — create a placeholder
        return Some(new_face);
    }

    let outer_uv = region.uv_boundary.first()?;
    if outer_uv.len() < 3 {
        return Some(new_face);
    }

    // Build edges from UV polygon vertices
    let mut wire_edges = Vec::new();
    for i in 0..outer_uv.len() {
        let j = (i + 1) % outer_uv.len();
        let uv0 = outer_uv[i];
        let uv1 = outer_uv[j];

        // Create 3D edge between these UV points on the surface
        let p0 = surface.d0_native(uv0.0, uv0.1);
        let p1 = surface.d0_native(uv1.0, uv1.1);

        let v0 = reg.find_or_add_vertex(p0, 1e-4);
        let v1 = reg.find_or_add_vertex(p1, 1e-4);

        let edge_3d = CurveGeom::Line {
            origin: p0,
            direction: p1 - p0,
        };
        let edge_pc = CurveGeom::Line {
            origin: Vec3::new(uv0.0, uv0.1, 0.0),
            direction: Vec3::new(uv1.0 - uv0.0, uv1.1 - uv0.1, 0.0),
        };

        let ek = reg.add_edge_with_pcurve(v0, v1, edge_3d, 1e-4, new_face, edge_pc);
        wire_edges.push((ek, Orientation::Forward));
    }

    // Update the face's outer wire
    let wire_key = reg.wires.insert(BRepWire { edges: wire_edges });
    if let Some(face) = reg.faces.get_mut(new_face) {
        face.outer_wire = wire_key;
    }

    // Handle inner wires (holes)
    for inner_uv in region.uv_boundary.iter().skip(1) {
        if inner_uv.len() < 3 {
            continue;
        }
        let mut inner_edges = Vec::new();
        for i in 0..inner_uv.len() {
            let j = (i + 1) % inner_uv.len();
            let uv0 = inner_uv[i];
            let uv1 = inner_uv[j];
            let p0 = surface.d0_native(uv0.0, uv0.1);
            let p1 = surface.d0_native(uv1.0, uv1.1);
            let v0 = reg.find_or_add_vertex(p0, 1e-4);
            let v1 = reg.find_or_add_vertex(p1, 1e-4);
            let ek = reg.add_edge_with_pcurve(
                v0, v1,
                CurveGeom::Line { origin: p0, direction: p1 - p0 },
                1e-4, new_face,
                CurveGeom::Line {
                    origin: Vec3::new(uv0.0, uv0.1, 0.0),
                    direction: Vec3::new(uv1.0 - uv0.0, uv1.1 - uv0.1, 0.0),
                },
            );
            inner_edges.push((ek, Orientation::Forward));
        }
        let inner_wk = reg.wires.insert(BRepWire { edges: inner_edges });
        if let Some(face) = reg.faces.get_mut(new_face) {
            face.inner_wires.push(inner_wk);
        }
    }

    Some(new_face)
}

/// Find the parameter t on an edge closest to a 3D point.
fn point_on_edge_param(pt: Vec3, ek: EdgeKey, reg: &BRepStore, tol: f32) -> Option<f32> {
    let edge = reg.edges.get(ek)?;
    let n = 16;
    let mut best_t = 0.0f32;
    let mut best_d2 = f32::MAX;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let d2 = (edge.curve.d0(t) - pt).length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_t = t;
        }
    }
    let best_d = best_d2.sqrt();
    if best_d <= tol {
        // Refine with binary search
        let mut t = best_t;
        for _ in 0..4 {
            let eps = 0.001;
            for &dt in &[-eps, eps] {
                let tn = (t + dt).clamp(0.0, 1.0);
                let d = (edge.curve.d0(tn) - pt).length();
                if d < best_d {
                    best_d2 = d * d;
                    t = tn;
                }
            }
        }
        Some(t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepWire, FaceKey, Orientation};
    use rc3d_core::math::Vec3;

    #[test]
    fn test_edge_split_detection() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        };
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);
        let edge_3d = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::new(2.0, 0.0, 0.0) };
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: surface.clone(),
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let ek = reg.add_edge_with_pcurve(v0, v1, edge_3d, 1e-4, fk, edge_3d.clone());
        reg.wires.get_mut(wire).unwrap().edges = vec![(ek, Orientation::Forward)];

        // Create an intersection curve with endpoint at (1, 0, 0) — on the edge
        let curve = super::super::split::BRepIntersectionCurve {
            points_3d: vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)],
            params_a: vec![(1.0, 0.0), (1.0, 1.0)],
            params_b: vec![(0.0, 0.0), (0.0, 1.0)],
            face_a: fk,
            face_b: FaceKey::default(),
        };

        let split_points = detect_edge_split_points(fk, &[curve], &reg);
        assert!(!split_points.is_empty(), "should detect endpoint on edge");
        let sp = &split_points[0];
        assert_eq!(sp.edge, ek);
    }

    #[test]
    fn test_build_face_from_region() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X,
        };
        let region = super::super::split::SubFaceRegion {
            uv_boundary: vec![vec![
                (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
            ]],
            interior_point: (0.5, 0.5),
            interior_point_3d: Vec3::new(0.5, 0.5, 0.0),
            original_face: FaceKey::default(),
        };

        let fk = build_face_from_region(
            FaceKey::default(), &region, &surface,
            &HashMap::new(), &mut reg,
        );
        assert!(fk.is_some(), "should build face from UV region");
        let face = reg.faces.get(fk.unwrap()).unwrap();
        let wire = reg.wires.get(face.outer_wire).unwrap();
        assert_eq!(wire.edges.len(), 4, "square region should have 4 edges");
    }
}
