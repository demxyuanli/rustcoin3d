//! Shape caching module for efficient GPU mesh reuse.
//!
//! Provides shape key hashing and caching for primitive shapes (Cube, Sphere, Cone, Cylinder)
//! and indexed face sets to avoid redundant tessellation.

use rc3d_core::Aabb;
use rc3d_mesh::MeshletData;
use std::sync::Arc;

use crate::vertex::Vertex;

/// Shape key for mesh caching - uniquely identifies a geometric shape.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShapeKey {
    Cube {
        w: u32,
        h: u32,
        d: u32,
    },
    Sphere {
        r: u32,
        slices: u32,
        stacks: u32,
    },
    Cone {
        r: u32,
        h: u32,
        segments: u32,
    },
    Cylinder {
        r: u32,
        h: u32,
        segments: u32,
    },
    Torus {
        major_r: u32,
        minor_r: u32,
        major_segments: u32,
        minor_segments: u32,
    },
    IndexedFaceSet {
        node: u64,
        coord_len: u32,
        coord_index_len: u32,
        points_sig: [u32; 6],
        index_sig: [i32; 2],
        tex_len: u32,
        tex_sig: [u32; 4],
        /// Explicit normals from `NormalNode` (len must match coord when used); 0 = use computed only.
        normal_len: u32,
        normal_sig: [u32; 6],
    },
}

/// Cached shape data tuple:
/// (vertices, indices, feature edge positions, full edge positions, AABB, meshlet data)
pub type CachedShapeData = (
    Arc<Vec<Vertex>>,
    Arc<Vec<u32>>,
    Arc<Vec<[f32; 3]>>,
    Arc<Vec<[f32; 3]>>,
    Aabb,
    Option<Arc<MeshletData>>,
);

/// Maximum number of edge positions to prevent memory exhaustion.
pub const MAX_EDGE_POSITIONS: usize = 50_000_000;

/// Triangle count threshold above which meshlet generation is skipped.
pub const MESHLET_TRIANGLE_THRESHOLD: usize = 500_000;

/// Runtime-configurable feature edge crease angle (degrees).
/// Default 12°. Set via `set_feature_crease_angle`.
#[allow(clippy::incompatible_msrv)]
static FEATURE_CREASE_ANGLE_BITS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(12.0f32.to_bits());

/// Get the current feature edge crease angle in degrees.
pub fn feature_crease_angle() -> f32 {
    f32::from_bits(FEATURE_CREASE_ANGLE_BITS.load(std::sync::atomic::Ordering::SeqCst))
}

/// Set the feature edge crease angle in degrees.
pub fn set_feature_crease_angle(deg: f32) {
    FEATURE_CREASE_ANGLE_BITS.store(deg.to_bits(), std::sync::atomic::Ordering::SeqCst);
}

/// Clamp edge positions to MAX_EDGE_POSITIONS to prevent memory issues.
/// Returns an empty Arc if the input exceeds the limit.
pub fn clamp_edge_positions(arc: Arc<Vec<[f32; 3]>>) -> Arc<Vec<[f32; 3]>> {
    if arc.len() > MAX_EDGE_POSITIONS {
        Arc::new(Vec::new())
    } else {
        arc
    }
}
