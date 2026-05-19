// Velocity-buffer based motion blur.
// Samples scene color along the per-pixel motion vector with a reconstruction filter.

struct MotionBlurParams {
    max_samples: u32,     // max samples along velocity (e.g. 16)
    intensity: f32,       // overall blur intensity (e.g. 0.5)
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var t_color: texture_2d<f32>;     // current frame HDR
@group(0) @binding(1) var t_velocity: texture_2d<f32>; // Rgba16Float velocity
@group(0) @binding(2) var t_depth: texture_2d<f32>;    // linear depth
@group(0) @binding(3) var s_linear: sampler;
@group(0) @binding(4) var s_point: sampler;
@group(0) @binding(5) var<uniform> params: MotionBlurParams;
@group(0) @binding(6) var output_tex: texture_storage_2d<rgba16float, write>;

// Soft depth-aware reconstruction weights (favors foreground objects)
fn compute_weight(depth_sample: f32, depth_center: f32, velocity_mag: f32) -> f32 {
    let depth_diff = abs(depth_sample - depth_center);
    return 1.0 / (1.0 + depth_diff * 10.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_color);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let tex_size = vec2<f32>(dims);

    let velocity = textureSampleLevel(t_velocity, s_point, uv, 0.0).rg;
    let center_depth = textureSampleLevel(t_depth, s_point, uv, 0.0).r;

    // Velocity magnitude in pixels
    let vel_pixels = velocity * tex_size;
    let vel_mag = length(vel_pixels);

    if vel_mag < 0.05 || params.intensity <= 0.0 {
        // No motion — write center sample
        let color = textureSampleLevel(t_color, s_linear, uv, 0.0);
        textureStore(output_tex, vec2<i32>(gid.xy), color);
        return;
    }

    let num_samples = min(u32(vel_mag * params.intensity * 2.0) + 1u, params.max_samples);
    let step = velocity / f32(num_samples);

    var accum = vec4<f32>(0.0);
    var total_weight = 0.0;

    // Gather samples along the velocity vector
    for (var i = 0u; i < num_samples; i++) {
        let t = (f32(i) + 0.5) / f32(num_samples) - 0.5; // [-0.5, 0.5]
        let sample_uv = uv + step * t;

        // Clamp to screen bounds
        let clamped_uv = clamp(sample_uv, vec2<f32>(0.001), vec2<f32>(0.999));
        let sample_color = textureSampleLevel(t_color, s_linear, clamped_uv, 0.0);
        let sample_depth = textureSampleLevel(t_depth, s_point, clamped_uv, 0.0).r;

        let w = compute_weight(sample_depth, center_depth, vel_mag);
        accum += sample_color * w;
        total_weight += w;
    }

    let result = accum / max(total_weight, 1e-6);
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(result.rgb, 1.0));
}
