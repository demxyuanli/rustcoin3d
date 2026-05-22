//! Light packing utilities for GPU-driven lighting.
//!
//! Provides functions to pack light parameters into buffer formats and compute
//! light parameter hashes for draw-call grouping.

use crate::vertex::MAX_LIGHTS;

/// Packed light data for GPU upload:
/// (directions, colors, types, positions, spot params, light count)
pub type PackedLights = (
    [[f32; 4]; MAX_LIGHTS],
    [[f32; 4]; MAX_LIGHTS],
    [[f32; 4]; MAX_LIGHTS],
    [[f32; 4]; MAX_LIGHTS],
    [[f32; 4]; MAX_LIGHTS],
    u32,
);

/// Pre-compute a u64 hash of all light parameters for fast draw-call grouping.
pub fn hash_light_params(
    light_dirs: &[[f32; 4]; MAX_LIGHTS],
    light_colors: &[[f32; 4]; MAX_LIGHTS],
    light_types: &[[f32; 4]; MAX_LIGHTS],
    light_positions: &[[f32; 4]; MAX_LIGHTS],
    spot_params: &[[f32; 4]; MAX_LIGHTS],
    light_count: u32,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = twox_hash::XxHash64::with_seed(0);
    for arr in [light_dirs, light_colors, light_types, light_positions, spot_params] {
        for v in arr {
            for f in v {
                f.to_bits().hash(&mut h);
            }
        }
    }
    light_count.hash(&mut h);
    h.finish()
}
