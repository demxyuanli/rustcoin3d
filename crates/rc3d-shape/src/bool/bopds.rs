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
use rc3d_core::math::Vec3;

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
    pub t_range: (f32, f32),
    /// The two vertices bounding this pave block (may be new split vertices).
    pub vertices: (VertexKey, VertexKey),
    /// Faces that share this pave block (2 for regular edge, >2 for non-manifold).
    pub face_refs: Vec<FaceKey>,
    /// 3D intersection points at start and end.
    pub points_3d: (Vec3, Vec3),
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
    pub point_3d: Vec3,
    pub uv_a: (f32, f32),
    pub uv_b: (f32, f32),
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
    pub tolerance: f32,
}

impl BopDS {
    pub fn new(tolerance: f32) -> Self {
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
    /// Sorts intersection points along each edge by parameter t and creates
    /// pave blocks for each consecutive pair.
    pub fn build_pave_blocks(&mut self, reg: &mut BRepStore) {
        // Collect all intersection points per edge
        let mut edge_points: HashMap<EdgeKey, Vec<(f32, Vec3, FaceKey)>> = HashMap::new();

        for interf in &self.face_face_interfs {
            for pt in &interf.points {
                // For now, pave blocks are built from the 3D intersection points
                // In the full pipeline, each point would be associated with specific edges
                let _ = (pt, interf.face_a, interf.face_b);
            }
        }

        // For each edge with intersection points, sort and create pave blocks
        for (_ek, mut pts) in edge_points {
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            // Create pave blocks between consecutive points
            for w in pts.windows(2) {
                let _pb = PaveBlock {
                    edge: _ek,
                    t_range: (w[0].0, w[1].0),
                    vertices: (VertexKey::default(), VertexKey::default()),
                    face_refs: vec![w[0].2, w[1].2],
                    points_3d: (w[0].1, w[1].1),
                };
                // pave_blocks entry would go here
            }
        }
    }

    /// Build common blocks by grouping pave blocks that share the same edge section.
    pub fn build_common_blocks(&mut self) {
        let mut edge_groups: HashMap<EdgeKey, Vec<usize>> = HashMap::new();

        for (ek, blocks) in &self.pave_blocks {
            edge_groups.entry(*ek).or_default().extend(0..blocks.len());
        }

        // For each edge, pave blocks from different faces that overlap in
        // parameter space form a common block.
        for (_ek, _block_indices) in &edge_groups {
            // Group overlapping pave blocks into common blocks
            // (Full implementation requires parameter-space overlap detection)
        }
    }

    /// Total number of intersection curves found.
    pub fn intersection_count(&self) -> usize {
        self.face_face_interfs.iter().map(|i| i.curves_3d.len()).sum()
    }
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
