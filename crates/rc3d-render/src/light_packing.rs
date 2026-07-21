//! Light packing utilities for GPU-driven lighting.
//!
//! Provides functions to pack light parameters into buffer formats and compute
//! light parameter hashes for draw-call grouping.

use crate::vertex::MAX_LIGHTS;
use rc3d_scene::{LightData, LightType};

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

/// Collect scene lights into packed GPU buffers.
///
/// Returns a [`PackedLights`] tuple ready for upload. If no lights are present,
/// a default directional light is inserted so shaders always have at least one light.
pub fn collect_lights(lights: &[LightData]) -> PackedLights {
    let mut dirs = [[0.0f32; 4]; MAX_LIGHTS];
    let mut colors = [[0.0f32; 4]; MAX_LIGHTS];
    let mut types = [[0.0f32; 4]; MAX_LIGHTS];
    let mut positions = [[0.0f32; 4]; MAX_LIGHTS];
    let mut spot_params = [[0.0f32; 4]; MAX_LIGHTS];
    let mut count = 0u32;
    let mut warned = false;
    let total_lights = lights.len();
    for light in lights {
        if (count as usize) < MAX_LIGHTS {
            let idx = count as usize;
            dirs[idx] = [light.direction.x, light.direction.y, light.direction.z, 0.0];
            let c = light.color * light.intensity;
            colors[idx] = [c.x, c.y, c.z, 1.0];
            positions[idx] = [light.location.x, light.location.y, light.location.z, 1.0];
            types[idx][0] = match light.light_type {
                LightType::Directional => 0.0,
                LightType::Point => 1.0,
                LightType::Spot => 2.0,
            };
            spot_params[idx] = [light.cut_off_angle.cos(), light.drop_off_rate, 0.0, 0.0];
            count += 1;
        } else {
            warned = true;
            break;
        }
    }
    if warned {
        log::warn!("MAX_LIGHTS ({}) exceeded; {} lights truncated", MAX_LIGHTS, total_lights.saturating_sub(MAX_LIGHTS));
    }
    if count == 0 {
        dirs[0] = [0.0, 0.0, -1.0, 0.0];
        colors[0] = [1.0, 1.0, 1.0, 1.0];
        types[0][0] = 0.0;
        count = 1;
    }
    (dirs, colors, types, positions, spot_params, count)
}
