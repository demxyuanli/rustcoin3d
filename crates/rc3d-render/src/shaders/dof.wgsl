// Depth of Field — circle-of-confusion computation + composite.
// Uses separable blur passes on half-res buffer for performance.

struct DofParams {
    focus_distance: f32,  // world-space focus plane distance
    aperture: f32,        // lens aperture size (larger = more blur)
    max_coc: f32,         // max CoC radius in pixels
    near_blur_scale: f32, // scale near blur (usually 1.0)
    far_blur_scale: f32,  // scale far blur (usually 1.0)
    _pad0: f32,
};

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_2d<f32>;
@group(0) @binding(2) var s_linear: sampler;
@group(0) @binding(3) var s_point: sampler;
@group(0) @binding(4) var<uniform> params: DofParams;
@group(0) @binding(5) var output_tex: texture_storage_2d<rgba16float, write>;

// Compute circle-of-confusion radius from depth.
fn compute_coc(depth: f32) -> f32 {
    let focus = params.focus_distance;
    let coc = params.aperture * abs(depth - focus) / max(depth, 1e-5);
    return clamp(coc * 5.0, 0.0, params.max_coc);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_color);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let px = 1.0 / vec2<f32>(dims);

    let color = textureSampleLevel(t_color, s_linear, uv, 0.0);
    let depth = textureSampleLevel(t_depth, s_point, uv, 0.0).r;
    let coc = compute_coc(depth);

    if coc < 1.0 {
        // In focus — pass through
        textureStore(output_tex, vec2<i32>(gid.xy), color);
        return;
    }

    // Gather blur samples in a circle pattern
    let sample_count: u32 = 12u;
    let radius = coc * 0.02; // UV-space radius
    var accum = vec4<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < sample_count; i++) {
        let angle = f32(i) * 6.283185 / f32(sample_count);
        let r = radius * (f32(i) / f32(sample_count));
        let offset = vec2<f32>(cos(angle), sin(angle)) * r;

        let sample_uv = clamp(uv + offset, vec2<f32>(0.001), vec2<f32>(0.999));
        let sample_color = textureSampleLevel(t_color, s_linear, sample_uv, 0.0);
        let sample_depth = textureSampleLevel(t_depth, s_point, sample_uv, 0.0).r;

        // Reduce weight for samples at very different depths (avoid bleeding)
        let depth_diff = abs(depth - sample_depth);
        let depth_weight = 1.0 / (1.0 + depth_diff * 20.0);

        accum += sample_color * depth_weight;
        total_weight += depth_weight;
    }

    // Blend with center sample
    let blurred = accum / max(total_weight, 1e-6);
    let result = mix(color, blurred, 0.8);

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(result.rgb, 1.0));
}
