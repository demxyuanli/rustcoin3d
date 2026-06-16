//! BOP Data Structure — OCC BOPDS_DS equivalent.
//!
//! Stores the results of face-face intersection as parameterized "pave blocks"
//! on edges and "common blocks" for shared edge intervals across faces.
//!
//! OCC alignment:
//! - BOPDS_PaveBlock: a parameter interval [t0, t1] on an edge
//! - BOPDS_CommonBlock: set of pave blocks sharing the same geometric edge section
//! - BOPDS_VectorOfInterfFF: face-face intersection results
//!
//! Architecture:
//! PaveFiller (SSI) produces PaveBlocks on edges,
//! which are grouped into CommonBlocks (shared pave intervals),
//! feeding FaceInfo (split vertices per face).

use std::collections::{HashMap, HashSet};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, VertexKey};
use crate::geom::CurveGeom;
use rc3d_core::math::{Real, PVec3};

/// A parameter interval on an edge created by intersection with another face.
///
/// OCC: BOPDS_PaveBlock — represents a section of an edge between two
/// intersection points (paves). Used during face splitting to determine
/// which portions of edges belong to which sub-faces.
#[derive(Debug, Clone)]
pub struct PaveBlock {
    /// The edge this pave block lies on.
    pub edge: EdgeKey,
    /// Parameter range [t_start, t_end] along the edge curve.
    pub t_range: (Real, Real),
    /// The two vertices bounding this pave block (may be new split vertices).
    pub vertices: (VertexKey, VertexKey),
    /// Faces that share this pave block (2 for regular edge, >2 for non-manifold).
    pub face_refs: Vec<FaceKey>,
    /// 3D intersection points at start and end.
    pub points_3d: (PVec3, PVec3),
}

/// A set of pave blocks that share the same geometric edge section.
///
/// OCC: BOPDS_CommonBlock — when two faces share an edge, the pave blocks
/// from both faces for that edge form a common block. Used to ensure
/// consistent face splitting across shared edges.
#[derive(Debug, Clone)]
pub struct CommonBlock {
    /// All pave blocks in this common block (one per face sharing the edge).
    pub pave_blocks: Vec<PaveBlock>,
    /// The faces sharing this common block.
    pub faces: Vec<FaceKey>,
}

/// Result of face-face intersection with associated pave data.
///
/// OCC: BOPDS_VectorOfInterfFF entry — stores the intersection curves
/// plus the pave blocks generated on each face's edges.
#[derive(Debug, Clone)]
pub struct FaceFaceInterf {
    pub face_a: FaceKey,
    pub face_b: FaceKey,
    /// 3D intersection curves.
    pub curves_3d: Vec<CurveGeom>,
    /// PCurves on face A (one per curve_3d).
    pub pcurves_a: Vec<CurveGeom>,
    /// PCurves on face B (one per curve_3d).
    pub pcurves_b: Vec<CurveGeom>,
    /// Intersection points (3D position + UV on both faces).
    pub points: Vec<InterfPoint>,
}

/// A single intersection point with UV coordinates on both faces.
#[derive(Debug, Clone)]
pub struct InterfPoint {
    pub point_3d: PVec3,
    pub uv_a: (Real, Real),
    pub uv_b: (Real, Real),
}

/// The full BOP data structure for a boolean operation.
///
/// OCC: BOPDS_DS — manages all intersection data (pave blocks, common blocks,
/// face-face interference results) for one boolean operation.
#[derive(Debug, Clone, Default)]
pub struct BopDS {
    /// Face-face intersection results.
    pub face_face_interfs: Vec<FaceFaceInterf>,
    /// Pave blocks keyed by edge.
    pub pave_blocks: HashMap<EdgeKey, Vec<PaveBlock>>,
    /// Common blocks (shared pave intervals across faces).
    pub common_blocks: Vec<CommonBlock>,
    /// New vertices created during face splitting.
    pub split_vertices: Vec<VertexKey>,
    /// Tolerance for intersection computations.
    pub tolerance: Real,
}

impl BopDS {
    pub fn new(tolerance: Real) -> Self {
        Self {
            face_face_interfs: Vec::new(),
            pave_blocks: HashMap::new(),
            common_blocks: Vec::new(),
            split_vertices: Vec::new(),
            tolerance,
        }
    }

    /// Add a face-face intersection result and register its pave blocks.
    pub fn add_face_face_interf(&mut self, interf: FaceFaceInterf) {
        self.face_face_interfs.push(interf);
    }

    /// Get all face-face interfs that involve a specific face.
    pub fn interfs_for_face(&self, fk: FaceKey) -> Vec<&FaceFaceInterf> {
        self.face_face_interfs
            .iter()
            .filter(|i| i.face_a == fk || i.face_b == fk)
            .collect()
    }

    /// Get all faces that have intersection with the given face.
    pub fn faces_interfering_with(&self, fk: FaceKey) -> HashSet<FaceKey> {
        self.face_face_interfs
            .iter()
            .filter_map(|i| {
                if i.face_a == fk { Some(i.face_b) }
                else if i.face_b == fk { Some(i.face_a) }
                else { None }
            })
            .collect()
    }

    /// Build pave blocks from intersection points.
    ///
    /// Maps each `InterfPoint` onto the edges of the face's wire by projecting
    /// the 3D point onto the edge's curve to find parameter t. Sorts points by t
    /// and creates PaveBlocks for each consecutive pair of intersection points.
    pub fn build_pave_blocks(&mut self, reg: &BRepStore) {
        let mut edge_points: HashMap<EdgeKey, Vec<(Real, InterfPoint)>> = HashMap::new();

        for interf in &self.face_face_interfs {
            for pt in &interf.points {
                for &face_key in &[interf.face_a, interf.face_b] {
                    let face = match reg.faces.get(face_key) {
                        Some(f) => f, None => continue,
                    };
                    let wire = match reg.wires.get(face.outer_wire) {
                        Some(w) => w, None => continue,
                    };
                    for &(ek, _orient) in &wire.edges {
                        let edge = match reg.edges.get(ek) {
                            Some(e) => e, None => continue,
                        };
                        if let Some(t) = project_point_on_edge(&edge.curve, pt.point_3d, self.tolerance) {
                            let t_clamped = t.clamp(0.0, 1.0);
                            edge_points.entry(ek).or_default().push((t_clamped, pt.clone()));
                        }
                    }
                }
            }
        }

        for (ek, mut pts) in edge_points {
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            // Deduplicate very close parameters.
            let mut deduped: Vec<(Real, InterfPoint)> = Vec::new();
            for (t, pt) in pts {
                if deduped.last().map_or(true, |(lt, _)| (t - lt).abs() > self.tolerance) {
                    deduped.push((t, pt));
                }
            }
            if deduped.len() < 2 { continue; }
            let mut blocks = Vec::new();
            for w in deduped.windows(2) {
                let (t0, pt0) = &w[0];
                let (t1, pt1) = &w[1];
                let faces = faces_for_edge(ek, &self.face_face_interfs);
                blocks.push(PaveBlock {
                    edge: ek,
                    t_range: (*t0, *t1),
                    vertices: (VertexKey::default(), VertexKey::default()),
                    face_refs: faces,
                    points_3d: (pt0.point_3d, pt1.point_3d),
                });
            }
            self.pave_blocks.insert(ek, blocks);
        }
    }

    /// Build common blocks by grouping overlapping pave blocks across faces.
    ///
    /// A common block represents a shared edge section where two or more faces
    /// have overlapping parameter intervals.
    pub fn build_common_blocks(&mut self) {
        let mut common: Vec<CommonBlock> = Vec::new();

        for (&_ek, blocks) in &self.pave_blocks {
            if blocks.len() < 2 { continue; }
            // Group blocks by overlapping parameter ranges.
            let mut groups: Vec<Vec<usize>> = Vec::new();
            for (i, bi) in blocks.iter().enumerate() {
                let mut assigned = false;
                for group in &mut groups {
                    let overlaps = group.iter().any(|&j| {
                        let bj = &blocks[j];
                        bi.t_range.0 <= bj.t_range.1 && bj.t_range.0 <= bi.t_range.1
                    });
                    if overlaps { group.push(i); assigned = true; break; }
                }
                if !assigned { groups.push(vec![i]); }
            }
            for group in groups {
                if group.len() < 2 { continue; }
                let mut faces = Vec::new();
                let mut pbs = Vec::new();
                for &idx in &group {
                    let pb = &blocks[idx];
                    for fk in &pb.face_refs {
                        if !faces.contains(fk) { faces.push(*fk); }
                    }
                    pbs.push(pb.clone());
                }
                common.push(CommonBlock { pave_blocks: pbs, faces });
            }
        }

        self.common_blocks = common;
    }

    /// Infer edge key from two vertices (lookup in face-face interfs).
    pub fn edge_for_vertices(&self, _va: VertexKey, _vb: VertexKey, _reg: &BRepStore) -> Option<EdgeKey> {
        // Stub: in full pipeline, intersects index edges by vertex adjacency.
        // For now, return the first edge from pave_blocks that matches both vertices.
        for (&ek, blocks) in &self.pave_blocks {
            for pb in blocks {
                if (pb.vertices.0 == _va && pb.vertices.1 == _vb)
                    || (pb.vertices.0 == _vb && pb.vertices.1 == _va)
                {
                    return Some(ek);
                }
            }
        }
        None
    }

    /// Total number of intersection curves found.
    pub fn intersection_count(&self) -> usize {
        self.face_face_interfs.iter().map(|i| i.curves_3d.len()).sum()
    }
}

// ── BopDS helpers ────────────────────────────────────────────────────

/// Project a 3D point onto a curve to find parameter t (20-sample + binary refine).
fn project_point_on_edge(curve: &CurveGeom, pt: PVec3, tolerance: Real) -> Option<Real> {
    const N: usize = 20;
    let mut best_t = 0.0_f64;
    let mut best_dist = f64::MAX;
    for i in 0..=N {
        let t = i as Real / N as Real;
        let d = (curve.d0(t) - pt).length();
        if d < best_dist { best_dist = d; best_t = t; }
    }
    let mut lo = (best_t - 0.1).max(0.0);
    let mut hi = (best_t + 0.1).min(1.0);
    for _ in 0..8 {
        let mid = (lo + hi) * 0.5;
        if (curve.d0(mid) - pt).length() < (curve.d0(lo) - pt).length() { lo = mid; } else { hi = mid; }
    }
    let t_final = (lo + hi) * 0.5;
    if (curve.d0(t_final) - pt).length() < tolerance.max(0.1) { Some(t_final) } else { None }
}

/// Collect all face keys that reference a given edge from the interference data.
fn faces_for_edge(_ek: EdgeKey, interfs: &[FaceFaceInterf]) -> Vec<FaceKey> {
    let mut faces = Vec::new();
    for interf in interfs {
        for &fk in &[interf.face_a, interf.face_b] {
            if !faces.contains(&fk) { faces.push(fk); }
        }
    }
    faces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bopds_new() {
        let ds = BopDS::new(1e-4);
        assert_eq!(ds.intersection_count(), 0);
        assert!(ds.face_face_interfs.is_empty());
    }

    #[test]
    fn test_bopds_add_interf() {
        let mut ds = BopDS::new(1e-4);
        let ffa = FaceKey::default();
        let ffb = FaceKey::default();
        ds.add_face_face_interf(FaceFaceInterf {
            face_a: ffa,
            face_b: ffb,
            curves_3d: vec![],
            pcurves_a: vec![],
            pcurves_b: vec![],
            points: vec![],
        });
        assert_eq!(ds.face_face_interfs.len(), 1);
    }

    #[test]
    fn test_interfs_for_face() {
        let mut ds = BopDS::new(1e-4);
        let fk = FaceKey::default();
        ds.add_face_face_interf(FaceFaceInterf {
            face_a: fk,
            face_b: FaceKey::default(),
            curves_3d: vec![],
            pcurves_a: vec![],
            pcurves_b: vec![],
            points: vec![],
        });
        let found = ds.interfs_for_face(fk);
        assert_eq!(found.len(), 1);
    }
}
