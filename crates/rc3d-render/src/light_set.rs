//! Shared light-set table for deduplicating light parameters across draw calls.
//!
//! Most draw calls share identical light configurations. Instead of storing
//! 5 arrays of `[[f32;4]; MAX_LIGHTS]` (1280 bytes) per draw call, each draw
//! call stores a `u32` index into this table. For 1M objects this saves ~1.2 GB.

use std::collections::HashMap;

use crate::vertex::MAX_LIGHTS;

/// Packed light parameter arrays matching `DrawCall` light fields.
pub type PackedLights = (
    [[f32; 4]; MAX_LIGHTS],  // light_dirs
    [[f32; 4]; MAX_LIGHTS],  // light_colors
    [[f32; 4]; MAX_LIGHTS],  // light_types
    [[f32; 4]; MAX_LIGHTS],  // light_positions
    [[f32; 4]; MAX_LIGHTS],  // spot_params
    u32,                      // light_count
);

/// Deduplicated light-set storage. Each unique light configuration is stored
/// once and referenced by a `LightSetId` (u32 index).
#[derive(Clone)]
pub struct LightSetTable {
    sets: Vec<PackedLights>,
    /// Maps light_key → index into `sets`. The light_key is computed by
    /// `hash_light_params` (defined in `render_action`).
    key_to_index: HashMap<u64, u32>,
}

impl LightSetTable {
    pub fn new() -> Self {
        Self {
            sets: Vec::with_capacity(64),
            key_to_index: HashMap::with_capacity(64),
        }
    }

    /// Intern a light configuration. Returns the index.
    /// If the configuration already exists, returns the existing index.
    pub fn intern(&mut self, key: u64, packed: PackedLights) -> u32 {
        if let Some(&idx) = self.key_to_index.get(&key) {
            idx
        } else {
            let idx = self.sets.len() as u32;
            self.sets.push(packed);
            self.key_to_index.insert(key, idx);
            idx
        }
    }

    /// Look up a light set by index. Returns a zeroed fallback if the
    /// table is empty (can happen when the camera-only fast-path reuses
    /// cached draw calls before the first full traversal completes).
    #[inline]
    pub fn get(&self, id: u32) -> &PackedLights {
        static EMPTY: PackedLights = (
            [[0.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            0,
        );
        self.sets.get(id as usize).unwrap_or(&EMPTY)
    }

    /// Number of unique light sets stored.
    pub fn len(&self) -> usize {
        self.sets.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }
}

impl Default for LightSetTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lights(
        dirs: [[f32; 4]; MAX_LIGHTS],
        colors: [[f32; 4]; MAX_LIGHTS],
        types: [[f32; 4]; MAX_LIGHTS],
        positions: [[f32; 4]; MAX_LIGHTS],
        spots: [[f32; 4]; MAX_LIGHTS],
        count: u32,
    ) -> PackedLights {
        (dirs, colors, types, positions, spots, count)
    }

    #[test]
    fn test_intern_deduplicates() {
        let mut tbl = LightSetTable::new();
        let lights = make_lights(
            [[1.0, 0.0, 0.0, 0.0]; MAX_LIGHTS],
            [[1.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            1,
        );
        // Compute a key (in practice comes from hash_light_params)
        let id1 = tbl.intern(42, lights.clone());
        let id2 = tbl.intern(42, lights);
        assert_eq!(id1, id2); // same key → same index
        assert_eq!(tbl.len(), 1);
    }

    #[test]
    fn test_different_lights_get_different_ids() {
        let mut tbl = LightSetTable::new();
        let lights1 = make_lights(
            [[1.0, 0.0, 0.0, 0.0]; MAX_LIGHTS],
            [[1.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            1,
        );
        let lights2 = make_lights(
            [[0.0, 1.0, 0.0, 0.0]; MAX_LIGHTS],
            [[1.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            [[0.0; 4]; MAX_LIGHTS],
            1,
        );
        let id1 = tbl.intern(1, lights1);
        let id2 = tbl.intern(2, lights2);
        assert_ne!(id1, id2);
        assert_eq!(tbl.len(), 2);
    }
}
