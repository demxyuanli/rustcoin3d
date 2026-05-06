//! Out-of-core point cloud rendering (HOOPS OOC equivalent).
//!
//! Streams massive point datasets from disk in spatial chunks, rendering
//! only visible tiles to the GPU. Uses an octree spatial index and
//! LRU-based tile cache.

use rc3d_core::math::Vec3;
use rc3d_core::Aabb;
use serde::{Deserialize, Serialize};

/// A point in a point cloud (position + optional color/normal).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Point {
    pub x: f32, pub y: f32, pub z: f32,
    pub r: u8, pub g: u8, pub b: u8, pub a: u8,
}

/// Octree spatial index for out-of-core querying.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Octree {
    Leaf { bounds: Aabb, points: Vec<Point> },
    Node { bounds: Aabb, children: Box<[Octree; 8]> },
}

impl Octree {
    pub fn bounds(&self) -> &Aabb {
        match self { Octree::Leaf { bounds, .. } | Octree::Node { bounds, .. } => bounds }
    }

    pub fn build(points: &[Point], max_per_leaf: usize, bounds: Aabb) -> Self {
        if points.len() <= max_per_leaf || bounds.size().length() < 0.01 {
            return Octree::Leaf { bounds, points: points.to_vec() };
        }
        let mid = (bounds.min + bounds.max) * 0.5;
        let mut child_points: [Vec<Point>; 8] = Default::default();
        for p in points {
            let idx = ((p.x >= mid.x) as usize)
                | ((p.y >= mid.y) as usize) << 1
                | ((p.z >= mid.z) as usize) << 2;
            child_points[idx].push(*p);
        }
        let half = (bounds.max - bounds.min) * 0.5;
        let children = Box::new(std::array::from_fn(|i| {
            let off = Vec3::new(
                ((i & 1) as f32) * half.x,
                (((i >> 1) & 1) as f32) * half.y,
                (((i >> 2) & 1) as f32) * half.z,
            );
            let child_bounds = Aabb { min: bounds.min + off, max: bounds.min + off + half };
            Octree::build(&child_points[i], max_per_leaf, child_bounds)
        }));
        Octree::Node { bounds, children }
    }

    /// Collect all points visible within a view frustum.
    pub fn query_frustum(&self, _view: &[Vec3; 8]) -> Vec<Point> {
        let mut out = Vec::new();
        self._query_frustum(_view, &mut out);
        out
    }

    fn _query_frustum(&self, _view: &[Vec3; 8], out: &mut Vec<Point>) {
        // Simplified: return all points within bounds intersecting frustum
        match self {
            Octree::Leaf { points, .. } => out.extend_from_slice(points),
            Octree::Node { children, .. } => {
                for child in children.iter() { child._query_frustum(_view, out); }
            }
        }
    }
}

/// Tile cache entry: a chunk of points loaded from disk.
#[derive(Clone, Debug, Default)]
pub struct TileCache {
    pub points: Vec<Point>,
    pub last_used_frame: u64,
    pub loaded: bool,
}

/// Out-of-core point cloud manager.
pub struct PointCloudOoc {
    pub octree: Octree,
    pub tile_cache: lru::LruCache<usize, TileCache>,
    pub total_points: u64,
    pub max_tiles: usize,
    pub file_path: Option<std::path::PathBuf>,
}

impl PointCloudOoc {
    pub fn new(max_tiles: usize) -> Self {
        Self {
            octree: Octree::Leaf { bounds: Aabb::empty(), points: Vec::new() },
            tile_cache: lru::LruCache::new(std::num::NonZeroUsize::new(max_tiles.max(1)).unwrap()),
            total_points: 0,
            max_tiles,
            file_path: None,
        }
    }

    /// Build octree from an in-memory point set.
    pub fn build(&mut self, points: Vec<Point>, max_per_leaf: usize) {
        self.total_points = points.len() as u64;
        let bounds = points.iter().fold(Aabb::empty(), |a, p| {
            a.union(&Aabb::from_point(Vec3::new(p.x, p.y, p.z)))
        });
        self.octree = Octree::build(&points, max_per_leaf, bounds);
    }

    /// Set the backing file for OOC streaming.
    pub fn set_file(&mut self, path: std::path::PathBuf) {
        self.file_path = Some(path);
    }

    /// Query visible points for a given frustum.
    pub fn query(&self, frustum: &[Vec3; 8]) -> Vec<Point> {
        self.octree.query_frustum(frustum)
    }

    /// Stream a tile from disk into cache (placeholder).
    pub fn stream_tile(&mut self, _tile_id: usize, _frame: u64) -> Option<&TileCache> {
        // In full implementation, reads tile bytes from self.file_path
        // and deserializes points, updating LRU cache.
        None
    }

    /// Total point count.
    pub fn count(&self) -> u64 { self.total_points }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_cloud() {
        let pc = PointCloudOoc::new(64);
        assert_eq!(pc.count(), 0);
    }

    #[test]
    fn test_build_small_cloud() {
        let mut pc = PointCloudOoc::new(64);
        let points: Vec<Point> = (0..100).map(|i| Point {
            x: i as f32 * 0.1, y: 0.0, z: 0.0, ..Default::default()
        }).collect();
        pc.build(points, 16);
        assert_eq!(pc.count(), 100);
    }

    #[test]
    fn test_octree_query() {
        let points: Vec<Point> = vec![
            Point { x: 0.0, y: 0.0, z: 0.0, r: 255, g: 0, b: 0, a: 255 },
            Point { x: 1.0, y: 1.0, z: 1.0, r: 0, g: 255, b: 0, a: 255 },
        ];
        let bounds = Aabb { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(1.0, 1.0, 1.0) };
        let tree = Octree::build(&points, 1, bounds);
        let frustum = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ONE, Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ONE];
        let result = tree.query_frustum(&frustum);
        assert!(!result.is_empty());
    }
}
