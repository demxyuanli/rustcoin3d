//! Tolerance-aware 3D spatial index for vertex deduplication.
//!
//! Uses a uniform grid with adaptive neighborhood radius so points within a
//! given tolerance are found in O(1) average time without single-bucket
//! hash collisions.
//!
//! Provided in f32 (rendering) and f64 (geometry kernel) variants.

use std::collections::HashMap;

macro_rules! impl_spatial_index {
    ($F:ty, $SpatialIndex:ident, $spatial_cell:ident) => {
        /// Grid cell for a 3D position.
        #[inline]
        pub fn $spatial_cell(p: [$F; 3], cell_size: $F) -> (i64, i64, i64) {
            let s = if cell_size < 1e-6 { 1e-6 as $F } else { cell_size };
            (
                (p[0] / s).floor() as i64,
                (p[1] / s).floor() as i64,
                (p[2] / s).floor() as i64,
            )
        }

        /// Uniform-grid spatial index mapping 3D positions to stored keys.
        #[derive(Debug, Clone)]
        pub struct $SpatialIndex<K: Copy> {
            cell_size: $F,
            grid: HashMap<(i64, i64, i64), Vec<K>>,
        }

        impl<K: Copy> Default for $SpatialIndex<K> {
            fn default() -> Self {
                Self {
                    cell_size: 1e-4 as $F,
                    grid: HashMap::new(),
                }
            }
        }

        impl<K: Copy> $SpatialIndex<K> {
            /// Create an index with the given cell size (typically the dedup tolerance).
            pub fn with_cell_size(cell_size: $F) -> Self {
                Self {
                    cell_size: if cell_size < 1e-6 { 1e-6 as $F } else { cell_size },
                    grid: HashMap::new(),
                }
            }

            pub fn cell_size(&self) -> $F {
                self.cell_size
            }

            /// Find a stored key whose position is within `tolerance` of `p`.
            pub fn find_near<FN>(&self, p: [$F; 3], tolerance: $F, mut position_of: FN) -> Option<K>
            where
                FN: FnMut(K) -> [$F; 3],
            {
                if tolerance <= 0.0 {
                    return None;
                }
                let tol_sq = tolerance * tolerance;
                let (cx, cy, cz) = $spatial_cell(p, self.cell_size);
                let radius = ((tolerance / self.cell_size).ceil() as i64).max(1);
                for dx in -radius..=radius {
                    for dy in -radius..=radius {
                        for dz in -radius..=radius {
                            if let Some(keys) = self.grid.get(&(cx + dx, cy + dy, cz + dz)) {
                                for &key in keys {
                                    let q = position_of(key);
                                    let ddx = p[0] - q[0];
                                    let ddy = p[1] - q[1];
                                    let ddz = p[2] - q[2];
                                    if ddx * ddx + ddy * ddy + ddz * ddz <= tol_sq {
                                        return Some(key);
                                    }
                                }
                            }
                        }
                    }
                }
                None
            }

            /// Register `key` at position `p`.
            pub fn insert(&mut self, key: K, p: [$F; 3]) {
                let cell = $spatial_cell(p, self.cell_size);
                self.grid.entry(cell).or_default().push(key);
            }
        }
    };
}

// ── f32 (rendering) ──
impl_spatial_index!(f32, SpatialIndex, spatial_cell);

// ── f64 (geometry kernel) ──
impl_spatial_index!(f64, SpatialIndexF64, spatial_cell_f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_near_within_tolerance() {
        let mut idx = SpatialIndex::with_cell_size(1e-4);
        idx.insert(0usize, [0.0, 0.0, 0.0]);
        idx.insert(1usize, [1.0, 0.0, 0.0]);
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let found = idx.find_near([5e-5, 0.0, 0.0], 1e-4, |i| positions[i]);
        assert_eq!(found, Some(0));
        let miss = idx.find_near([0.5, 0.0, 0.0], 1e-4, |i| positions[i]);
        assert_eq!(miss, None);
    }

    #[test]
    fn distinct_keys_same_cell_beyond_tolerance() {
        let mut idx = SpatialIndex::with_cell_size(1e-4);
        idx.insert(0usize, [0.0, 0.0, 0.0]);
        let positions = [[0.0, 0.0, 0.0], [2e-4, 0.0, 0.0]];
        let found = idx.find_near([2e-4, 0.0, 0.0], 1e-5, |i| positions[i]);
        assert_eq!(found, None);
    }

    #[test]
    fn spatial_index_f64() {
        let mut idx = SpatialIndexF64::with_cell_size(1e-4);
        idx.insert(0usize, [0.0, 0.0, 0.0]);
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let found = idx.find_near([5e-5, 0.0, 0.0], 1e-4, |i| positions[i]);
        assert_eq!(found, Some(0));
    }
}
