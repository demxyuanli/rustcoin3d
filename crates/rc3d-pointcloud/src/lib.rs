//! Out-of-core point cloud rendering (HOOPS OOC equivalent).
//!
//! Streams massive point datasets from disk in spatial chunks, rendering
//! only visible tiles to the GPU. Uses an octree spatial index and
//! LRU-based tile cache.

use std::io::{Read, Seek, Write};

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
    ///
    /// Uses 6 exact plane tests (not AABB approximation).
    pub fn query_frustum(&self, view: &[Vec3; 8]) -> Vec<Point> {
        let planes = frustum_planes_from_corners(view);
        let mut out = Vec::new();
        self._query_frustum(&planes, &mut out);
        out
    }

    fn _query_frustum(&self, planes: &[(Vec3, f32); 6], out: &mut Vec<Point>) {
        match self {
            Octree::Leaf { bounds, points } => {
                if !frustum_intersects_aabb(planes, bounds) {
                    return;
                }
                out.extend(points.iter().copied().filter(|p| {
                    point_inside_frustum(planes, Vec3::new(p.x, p.y, p.z))
                }));
            }
            Octree::Node { children, .. } => {
                for child in children.iter() {
                    if frustum_intersects_aabb(planes, child.bounds()) {
                        child._query_frustum(planes, out);
                    }
                }
            }
        }
    }

    /// Count of non-empty leaf nodes (tiles) in the octree.
    pub fn leaf_count(&self) -> usize {
        match self {
            Octree::Leaf { points, .. } => if points.is_empty() { 0 } else { 1 },
            Octree::Node { children, .. } => children.iter().map(|c| c.leaf_count()).sum(),
        }
    }

    fn leaf_points_by_index(&self, tile_id: usize) -> Option<Vec<Point>> {
        let mut current = 0usize;
        self.find_leaf_points(tile_id, &mut current)
    }

    fn find_leaf_points(&self, tile_id: usize, current: &mut usize) -> Option<Vec<Point>> {
        match self {
            Octree::Leaf { points, .. } => {
                if points.is_empty() {
                    return None;
                }
                let is_match = *current == tile_id;
                *current += 1;
                is_match.then(|| points.clone())
            }
            Octree::Node { children, .. } => {
                for child in children.iter() {
                    if let Some(points) = child.find_leaf_points(tile_id, current) {
                        return Some(points);
                    }
                }
                None
            }
        }
    }
}

/// Compute 6 inward-facing frustum planes from 8 corner points.
///
/// Corner ordering (near face, then far face):
/// `[nbl, nbr, ntl, ntr, fbl, fbr, ftl, ftr]`
///
/// `nbl` = near-bottom-left, `ftr` = far-top-right, etc.
fn frustum_planes_from_corners(corners: &[Vec3; 8]) -> [(Vec3, f32); 6] {
    let nbl = corners[0]; let nbr = corners[1];
    let ntl = corners[2]; let ntr = corners[3];
    let fbl = corners[4]; let fbr = corners[5];
    let ftl = corners[6]; let ftr = corners[7];
    // Choose a reference point inside the frustum (centroid of all corners).
    let inside = (nbl + nbr + ntl + ntr + fbl + fbr + ftl + ftr) / 8.0;
    [
        inward_plane(ntl, nbl, nbr, inside),  // near
        inward_plane(fbr, fbl, ftl, inside),  // far
        inward_plane(nbl, fbl, ftl, inside),  // left
        inward_plane(ftr, fbr, nbr, inside),  // right
        inward_plane(nbr, fbr, fbl, inside),  // bottom
        inward_plane(ntl, ntr, ftr, inside),  // top
    ]
}

/// Plane from 3 vertices: computes normal via cross product, then orients it
/// so `inside_point` satisfies `normal·inside_point + d >= 0`.
fn inward_plane(a: Vec3, b: Vec3, c: Vec3, inside: Vec3) -> (Vec3, f32) {
    let n = (b - a).cross(c - a);
    if n.length_squared() < 1e-12 {
        return (Vec3::Y, 0.0);
    }
    let nn = n.normalize();
    let d = -nn.dot(a);
    // Flip if reference point is outside
    if nn.dot(inside) + d < 0.0 {
        (-nn, -d)
    } else {
        (nn, d)
    }
}

/// True if point lies on the inside of all 6 frustum planes.
fn point_inside_frustum(planes: &[(Vec3, f32); 6], p: Vec3) -> bool {
    for (n, d) in planes {
        if n.dot(p) + d < 0.0 {
            return false;
        }
    }
    true
}

/// True if AABB is at least partially inside the frustum (n-vertex test).
fn frustum_intersects_aabb(planes: &[(Vec3, f32); 6], aabb: &Aabb) -> bool {
    for (n, d) in planes {
        let p = Vec3::new(
            if n.x >= 0.0 { aabb.max.x } else { aabb.min.x },
            if n.y >= 0.0 { aabb.max.y } else { aabb.min.y },
            if n.z >= 0.0 { aabb.max.z } else { aabb.min.z },
        );
        if n.dot(p) + d < 0.0 {
            return false;
        }
    }
    true
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
    /// Lazily-loaded tile (offset, point_count) pairs from disk file.
    stream_tile_offsets: Option<Vec<(u64, u32)>>,
    /// Total tile count from disk file (set by load_offsets).
    stream_tile_count: usize,
}

impl PointCloudOoc {
    pub fn new(max_tiles: usize) -> Self {
        Self {
            octree: Octree::Leaf { bounds: Aabb::empty(), points: Vec::new() },
            tile_cache: lru::LruCache::new(std::num::NonZeroUsize::new(max_tiles.max(1)).unwrap()),
            total_points: 0,
            max_tiles,
            file_path: None,
            stream_tile_offsets: None,
            stream_tile_count: 0,
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

    /// Number of leaf tiles.
    pub fn tile_count(&self) -> usize {
        if self.stream_tile_count > 0 {
            self.stream_tile_count
        } else {
            self.octree.leaf_count()
        }
    }

    /// Save octree tiles to a binary file.
    ///
    /// Format: `[tile_count: u32][offsets: u64*N][tile_data...]`
    /// Each tile: `[point_count: u32][points: Point*count]`
    pub fn save(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let tiles: Vec<Vec<Point>> = (0..self.tile_count())
            .filter_map(|i| self.octree.leaf_points_by_index(i))
            .collect();
        let n = tiles.len() as u32;
        let header_size = 4u64 + 8 * n as u64;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&n.to_le_bytes())?;
        let mut cursor = header_size;
        for t in &tiles {
            f.write_all(&cursor.to_le_bytes())?;
            cursor += 4 + t.len() as u64 * 16;
        }
        // Pad to header_size if needed
        let current = f.stream_position()?;
        if current < header_size {
            f.write_all(&vec![0u8; (header_size - current) as usize])?;
        }
        for t in &tiles {
            f.write_all(&(t.len() as u32).to_le_bytes())?;
            let byte_slice: &[u8] = unsafe {
                std::slice::from_raw_parts(t.as_ptr() as *const u8, t.len() * 16)
            };
            f.write_all(byte_slice)?;
        }
        Ok(())
    }

    /// Load tile metadata from the backing file.
    pub fn load_offsets(&mut self) -> Result<(), std::io::Error> {
        let path = self.file_path.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "no file path set")
        })?;
        let mut f = std::fs::File::open(path)?;
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;
        let mut offsets = vec![0u64; n];
        let mut buf8 = [0u8; 8];
        for i in 0..n {
            f.read_exact(&mut buf8)?;
            offsets[i] = u64::from_le_bytes(buf8);
        }
        let mut info = Vec::with_capacity(n);
        for i in 0..n {
            f.seek(std::io::SeekFrom::Start(offsets[i]))?;
            f.read_exact(&mut buf4)?;
            let count = u32::from_le_bytes(buf4);
            info.push((offsets[i], count));
        }
        self.stream_tile_offsets = Some(info);
        self.stream_tile_count = n;
        Ok(())
    }

    /// Query visible points for a given frustum.
    pub fn query(&self, frustum: &[Vec3; 8]) -> Vec<Point> {
        self.octree.query_frustum(frustum)
    }

    /// Stream a tile from disk or in-memory octree into the LRU cache.
    ///
    /// If a backing file is set, reads from disk. Falls back to octree lookup.
    pub fn stream_tile(&mut self, tile_id: usize, frame: u64) -> Option<&TileCache> {
        if let Some(tile) = self.tile_cache.get_mut(&tile_id) {
            tile.last_used_frame = frame;
            return self.tile_cache.get(&tile_id);
        }
        // Try disk
        if self.stream_tile_offsets.is_none() && self.file_path.is_some() {
            let _ = self.load_offsets();
        }
        if let Some(ref offsets) = self.stream_tile_offsets {
            if tile_id < offsets.len() {
                if let Ok(points) = read_tile_from_disk(
                    self.file_path.as_ref().unwrap(), offsets, tile_id,
                ) {
                    self.tile_cache.put(tile_id, TileCache {
                        points, last_used_frame: frame, loaded: true,
                    });
                    return self.tile_cache.get(&tile_id);
                }
            }
        }
        // Fallback to memory
        let points = self.octree.leaf_points_by_index(tile_id)?;
        self.tile_cache.put(tile_id, TileCache {
            points,
            last_used_frame: frame,
            loaded: true,
        });
        self.tile_cache.get(&tile_id)
    }

    /// Total point count.
    pub fn count(&self) -> u64 { self.total_points }
}

/// Read a tile's points from disk.
fn read_tile_from_disk(
    path: &std::path::Path,
    offsets: &[(u64, u32)],
    tile_id: usize,
) -> Result<Vec<Point>, std::io::Error> {
    let (offset, count) = offsets[tile_id];
    let mut f = std::fs::File::open(path)?;
    f.seek(std::io::SeekFrom::Start(offset + 4))?;
    let mut points = vec![Point::default(); count as usize];
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(points.as_mut_ptr() as *mut u8, count as usize * 16)
    };
    f.read_exact(byte_slice)?;
    Ok(points)
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

    #[test]
    fn query_frustum_filters_points_outside_frustum_bounds() {
        let mut pc = PointCloudOoc::new(64);
        pc.build(
            vec![
                Point { x: 0.0, y: 0.0, z: 0.0, r: 255, a: 255, ..Default::default() },
                Point { x: 10.0, y: 10.0, z: 10.0, g: 255, a: 255, ..Default::default() },
            ],
            1,
        );
        let frustum = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];

        let result = pc.query(&frustum);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].r, 255);
    }

    #[test]
    fn stream_tile_loads_octree_leaf_into_lru_cache() {
        let mut pc = PointCloudOoc::new(2);
        pc.build(
            vec![
                Point { x: 0.0, y: 0.0, z: 0.0, r: 255, a: 255, ..Default::default() },
                Point { x: 1.0, y: 1.0, z: 1.0, g: 255, a: 255, ..Default::default() },
            ],
            1,
        );

        let tile = pc.stream_tile(0, 42).expect("tile 0 should exist");

        assert!(tile.loaded);
        assert_eq!(tile.last_used_frame, 42);
        assert!(!tile.points.is_empty());
    }

    #[test]
    fn save_and_stream_tile_from_disk_roundtrip() {
        let mut pc = PointCloudOoc::new(64);
        pc.build(
            (0..200)
                .map(|i| Point {
                    x: i as f32 * 0.1, y: 0.0, z: 0.0,
                    r: (i % 255) as u8, a: 255, ..Default::default()
                })
                .collect(),
            50,
        );
        let tile_count = pc.tile_count();
        assert!(tile_count >= 2, "expected at least 2 tiles, got {tile_count}");

        let tmp = std::env::temp_dir().join("rc3d_pc_rt.bin");
        pc.save(&tmp).expect("save");

        let mut pc2 = PointCloudOoc::new(64);
        pc2.set_file(tmp.clone());
        for i in 0..tile_count {
            let tile = pc2.stream_tile(i, 1).expect("stream from disk");
            assert!(tile.loaded, "tile {i} not loaded");
            assert!(!tile.points.is_empty(), "tile {i} empty");
        }

        let mut total = 0usize;
        for i in 0..tile_count {
            if let Some(t) = pc2.tile_cache.get(&i) { total += t.points.len(); }
        }
        assert_eq!(total, 200);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn stream_tile_falls_back_to_memory_when_no_file() {
        let mut pc = PointCloudOoc::new(64);
        pc.build(
            vec![Point { x: 1.0, y: 0.0, z: 0.0, r: 100, a: 255, ..Default::default() }],
            1,
        );
        let tile = pc.stream_tile(0, 5).expect("memory fallback");
        assert!(tile.loaded);
        assert_eq!(tile.points.len(), 1);
        assert_eq!(tile.points[0].r, 100);
    }
}
