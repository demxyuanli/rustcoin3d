//! PaveFiller: populates BOPDS from face-face intersection results.
//!
//! OCC alignment: BOPAlgo_PaveFiller — the first phase of boolean operations.
//! Takes raw face-face intersection data and fills the BOPDS with:
//! - Pave blocks on edges (parameter intervals between intersection points)
//! - Common blocks (groups of pave blocks sharing the same edge segment)
//! - Split vertices (new vertices at intersection points)
//!
//! Pipeline:
//! 1. Collect all face-face intersection curves
//! 2. For each intersection point on an edge, create a split vertex
//! 3. Sort split vertices along each edge by parameter t
//! 4. Create pave blocks between consecutive split vertices
//! 5. Group overlapping pave blocks into common blocks

use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, ShellKey, VertexKey};
use crate::topo_iter;
use super::bopds::{BopDS, CommonBlock, FaceFaceInterf, InterfPoint, PaveBlock};
use super::face_intersector;
use rc3d_core::math::Vec3;

/// Result of the pave filling phase.
#[derive(Debug, Default)]
pub struct PaveFillerReport {
    pub face_pairs_tested: usize,
    pub intersections_found: usize,
    pub total_curves: usize,
    pub split_vertices_created: usize,
    pub pave_blocks_created: usize,
    pub common_blocks_created: usize,
}

/// Run the pave filler on two sets of shells.
///
/// OCC: BOPAlgo_PaveFiller::Perform()
pub fn fill_paves(
    shells_a: &[ShellKey],
    shells_b: &[ShellKey],
    reg: &BRepStore,
    tolerance: f32,
) -> (BopDS, PaveFillerReport) {
    let mut ds = BopDS::new(tolerance);
    let mut report = PaveFillerReport::default();

    // Collect all faces from both shell sets
    let faces_a: Vec<(FaceKey, ShellKey)> = collect_shell_faces(shells_a, reg);
    let faces_b: Vec<(FaceKey, ShellKey)> = collect_shell_faces(shells_b, reg);

    // Build AABB acceleration, tracking faces without bboxes separately
    let mut bboxes_a: Vec<(FaceKey, super::aabb::AABB)> = Vec::new();
    let mut no_bbox_a: Vec<FaceKey> = Vec::new();
    for &(fk, _) in &faces_a {
        if let Some(b) = super::aabb::face_vertex_bbox(fk, reg) {
            bboxes_a.push((fk, b));
        } else {
            no_bbox_a.push(fk);
        }
    }
    let mut bboxes_b: Vec<(FaceKey, super::aabb::AABB)> = Vec::new();
    let mut no_bbox_b: Vec<FaceKey> = Vec::new();
    for &(fk, _) in &faces_b {
        if let Some(b) = super::aabb::face_vertex_bbox(fk, reg) {
            bboxes_b.push((fk, b));
        } else {
            no_bbox_b.push(fk);
        }
    }

    // Process AABB-accelerated face pairs
    let mut test_pair = |fka: FaceKey, fkb: FaceKey| {
        report.face_pairs_tested += 1;
        let face_a = match reg.faces.get(fka) { Some(f) => f, None => return };
        let face_b = match reg.faces.get(fkb) { Some(f) => f, None => return };
        if let Some(interf) = compute_face_pair_interf(fka, fkb, face_a, face_b, reg, tolerance) {
            report.intersections_found += 1;
            report.total_curves += interf.curves_3d.len();
            ds.add_face_face_interf(interf);
        }
    };

    for &(fka, ref bbox_a) in &bboxes_a {
        for &(fkb, ref bbox_b) in &bboxes_b {
            if !bbox_a.overlaps(bbox_b) { continue; }
            test_pair(fka, fkb);
        }
    }

    // Faces without AABB: test against all faces from the other set
    for &fka in &no_bbox_a {
        for &(fkb, _) in &bboxes_b { test_pair(fka, fkb); }
        for &fkb in &no_bbox_b { test_pair(fka, fkb); }
    }
    for &fkb in &no_bbox_b {
        for &(fka, _) in &bboxes_a { test_pair(fka, fkb); }
    }

    // Phase 1b: Edge-face interference — edges of one shell pierce faces of the other.
    // OCC: BOPAlgo_PaveFiller processes edge-face interferences alongside face-face.
    // Each edge-face hit becomes a single InterfPoint in a minimal FaceFaceInterf.
    for &sk_a in shells_a {
        for fk_a in topo_iter::iter_faces_of_shell(sk_a, reg) {
            for ek in topo_iter::iter_edges_of_face(fk_a, reg) {
                let edge = match reg.edges.get(ek) { Some(e) => e, None => continue };
                for &(fkb, ref bbox_b) in &bboxes_b {
                    if !edge_bbox_touches(edge, bbox_b, tolerance, reg) { continue; }
                    let face_b = match reg.faces.get(fkb) { Some(f) => f, None => continue };
                    let hits = super::intersect_edge::intersect_edge_face(
                        &edge.curve, &face_b.surface, tolerance,
                    );
                    for hit in &hits {
                        ds.face_face_interfs.push(FaceFaceInterf {
                            face_a: fk_a, face_b: fkb,
                            curves_3d: vec![],
                            pcurves_a: vec![],
                            pcurves_b: vec![],
                            points: vec![InterfPoint {
                                point_3d: hit.point,
                                uv_a: hit.uv_face,
                                uv_b: hit.uv_face, // same face hit from edge perspective
                            }],
                        });
                        report.intersections_found += 1;
                    }
                }
            }
        }
    }

    // Build pave blocks from the intersection data
    build_pave_blocks_from_interfs(&mut ds, reg, &mut report);

    // Build common blocks by grouping overlapping pave blocks
    build_common_blocks(&mut ds, &mut report);

    (ds, report)
}

/// Quick AABB check: does the edge's bounding box intersect the face bbox?
fn edge_bbox_touches(
    edge: &crate::topo::BRepEdge, face_bbox: &super::aabb::AABB, tol: f32, reg: &BRepStore,
) -> bool {
    use super::aabb::AABB;
    let p0 = edge.curve.d0(0.0);
    let p1 = edge.curve.d0(1.0);
    let mut edge_bb = AABB::empty();
    edge_bb.expand(p0);
    edge_bb.expand(p1);
    edge_bb.expand(Vec3::new(p0.x + tol, p0.y + tol, p0.z + tol));
    edge_bb.expand(Vec3::new(p1.x + tol, p1.y + tol, p1.z + tol));
    edge_bb.overlaps(face_bbox)
}

/// Compute intersection for a single face pair.
fn compute_face_pair_interf(
    fka: FaceKey, fkb: FaceKey,
    face_a: &crate::topo::BRepFace,
    face_b: &crate::topo::BRepFace,
    _reg: &BRepStore,
    tolerance: f32,
) -> Option<FaceFaceInterf> {
    // Try the general face intersector (marching + Newton for all surface types)
    face_intersector::intersect_faces(
        fka, fkb,
        &face_a.surface, &face_b.surface,
        tolerance,
    )
}

/// Build pave blocks by identifying intersection points that lie on edges.
fn build_pave_blocks_from_interfs(
    ds: &mut BopDS,
    reg: &BRepStore,
    report: &mut PaveFillerReport,
) {
    // For each intersection point, find which edges of each face it lies on
    for interf in &ds.face_face_interfs {
        for pt in &interf.points {
            // Check edges of face A
            let edges_a = face_boundary_edges(interf.face_a, reg);
            for ek in &edges_a {
                if point_on_edge(pt.point_3d, *ek, reg, ds.tolerance * 10.0) {
                    let t = point_param_on_edge(pt.point_3d, *ek, reg);
                    ds.pave_blocks.entry(*ek).or_default().push(PaveBlock {
                        edge: *ek,
                        t_range: (t, t),
                        vertices: (VertexKey::default(), VertexKey::default()),
                        face_refs: vec![interf.face_a, interf.face_b],
                        points_3d: (pt.point_3d, pt.point_3d),
                    });
                    report.pave_blocks_created += 1;
                }
            }

            // Check edges of face B
            let edges_b = face_boundary_edges(interf.face_b, reg);
            for ek in &edges_b {
                if point_on_edge(pt.point_3d, *ek, reg, ds.tolerance * 10.0) {
                    let t = point_param_on_edge(pt.point_3d, *ek, reg);
                    ds.pave_blocks.entry(*ek).or_default().push(PaveBlock {
                        edge: *ek,
                        t_range: (t, t),
                        vertices: (VertexKey::default(), VertexKey::default()),
                        face_refs: vec![interf.face_a, interf.face_b],
                        points_3d: (pt.point_3d, pt.point_3d),
                    });
                    report.pave_blocks_created += 1;
                }
            }
        }
    }

    // Sort pave blocks on each edge by parameter t
    for blocks in ds.pave_blocks.values_mut() {
        blocks.sort_by(|a, b| a.t_range.0.partial_cmp(&b.t_range.0).unwrap_or(std::cmp::Ordering::Equal));
        blocks.dedup_by(|a, b| (a.t_range.0 - b.t_range.0).abs() < 1e-6);
    }
}

/// Build common blocks by grouping pave blocks on the same edge.
fn build_common_blocks(ds: &mut BopDS, report: &mut PaveFillerReport) {
    for blocks in ds.pave_blocks.values() {
        if blocks.len() >= 2 {
            // All pave blocks on the same edge with overlapping parameter
            // ranges form a common block (they share the same geometric section)
            let cb = CommonBlock {
                pave_blocks: blocks.clone(),
                faces: blocks.iter().flat_map(|b| b.face_refs.iter().copied()).collect(),
            };
            ds.common_blocks.push(cb);
            report.common_blocks_created += 1;
        }
    }
}

/// Collect all (face_key, shell_key) pairs from a set of shells.
fn collect_shell_faces(shells: &[ShellKey], reg: &BRepStore) -> Vec<(FaceKey, ShellKey)> {
    let mut result = Vec::new();
    for &sk in shells {
        if let Some(shell) = reg.shells.get(sk) {
            for &(fk, _) in &shell.faces {
                result.push((fk, sk));
            }
        }
    }
    result
}

/// Collect the boundary edge keys of a face (outer wire + inner wires).
fn face_boundary_edges(fk: FaceKey, reg: &BRepStore) -> Vec<EdgeKey> {
    topo_iter::iter_edges_of_face(fk, reg)
}

/// Check if a 3D point lies on an edge within tolerance.
fn point_on_edge(pt: Vec3, ek: EdgeKey, reg: &BRepStore, tol: f32) -> bool {
    let edge = match reg.edges.get(ek) { Some(e) => e, None => return false };
    let t = find_param_on_edge(pt, edge);
    let curve_pt = edge.curve.d0(t);
    (pt - curve_pt).length() <= tol
}

/// Find the parameter t on an edge closest to a 3D point.
fn point_param_on_edge(pt: Vec3, ek: EdgeKey, reg: &BRepStore) -> f32 {
    let edge = match reg.edges.get(ek) { Some(e) => e, None => return 0.0 };
    find_param_on_edge(pt, edge)
}

fn find_param_on_edge(pt: Vec3, edge: &crate::topo::BRepEdge) -> f32 {
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
    // Refine with binary search
    for _ in 0..4 {
        let eps = 0.01 / (1 << 4) as f32;
        for &dt in &[-eps, eps] {
            let t = (best_t + dt).clamp(0.0, 1.0);
            let d2 = (edge.curve.d0(t) - pt).length_squared();
            if d2 < best_d2 {
                best_d2 = d2;
                best_t = t;
            }
        }
    }
    best_t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::SurfaceGeom;
    use crate::store::BRepStore;
    use crate::topo::*;
    use rc3d_core::math::Vec3;
    use std::collections::HashMap;

    fn make_plane_shell(reg: &mut BRepStore, origin: Vec3, normal: Vec3) -> ShellKey {
        let surface = SurfaceGeom::Plane { origin, normal, u_dir: Vec3::X };
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        })
    }

    #[test]
    fn test_fill_paves_finds_plane_intersection() {
        let mut reg = BRepStore::new();
        let sa = make_plane_shell(&mut reg, Vec3::ZERO, Vec3::Z);
        let sb = make_plane_shell(&mut reg, Vec3::ZERO, Vec3::Y);

        let (ds, report) = fill_paves(&[sa], &[sb], &reg, 1e-4);
        assert!(report.face_pairs_tested > 0, "should test face pairs");
        // Two intersecting planes should produce intersection
        assert!(!ds.face_face_interfs.is_empty() || report.intersections_found > 0,
            "intersecting planes should produce result, got {} interfs",
            ds.face_face_interfs.len());
    }

    #[test]
    fn test_fill_paves_parallel_planes_no_intersection() {
        let mut reg = BRepStore::new();
        let sa = make_plane_shell(&mut reg, Vec3::ZERO, Vec3::Z);
        let sb = make_plane_shell(&mut reg, Vec3::new(100.0, 0.0, 0.0), Vec3::Z);

        let (_ds, report) = fill_paves(&[sa], &[sb], &reg, 1e-4);
        // Parallel planes at different positions — no overlap
        assert_eq!(report.intersections_found, 0,
            "parallel separated planes should not intersect");
    }

    #[test]
    fn test_find_param_on_edge_line() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let curve = crate::geom::CurveGeom::Line {
            origin: Vec3::ZERO, direction: Vec3::X,
        };
        let ek = reg.edges.insert(BRepEdge {
            curve, tolerance: 1e-4,
            v_low: v0, v_high: v1,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: HashMap::new(),
        });
        let edge = reg.edges.get(ek).unwrap();
        let t = find_param_on_edge(Vec3::new(0.5, 0.0, 0.0), edge);
        assert!((t - 0.5).abs() < 0.1, "midpoint should map to t≈0.5, got {}", t);
    }
}
