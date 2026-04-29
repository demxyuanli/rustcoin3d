// TAA resolve: blends current frame color with reprojected history.

struct TaaParams {
    blend_factor: f32,     // typically 0.05-0.1
    clip_factor: f32,      // color clamp range (1.0 = tight, 2.0 = loose)
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var t_current: texture_2d<f32>;      // Current frame HDR color
@group(0) @binding(1) var t_history: texture_2d<f32>;      // Previous frame resolved color
@group(0) @binding(2) var t_velocity: texture_2d<f32>;    // Screen-space velocity (Rg16Float)
@group(0) @binding(3) var t_depth: texture_2d<f32>;       // Current frame depth
@group(0) @binding(4) var s_linear: sampler;
@group(0) @binding(5) var s_point: sampler;
@group(0) @binding(6) var<uniform> params: TaaParams;

@group(0) @binding(7) var output_tex: texture_storage_2d<rgba16float, write>;

struct YCoCgBounds {
    min_c: vec3<f32>,
    max_c: vec3<f32>,
}

fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    let y = 0.25 * c.r + 0.5 * c.g + 0.25 * c.b;
    let co = 0.5 * c.r - 0.5 * c.b;
    let cg = -0.25 * c.r + 0.5 * c.g - 0.25 * c.b;
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let r = c.r + c.g - c.b;
    let g = c.r + c.b;
    let b = c.r - c.g - c.b;
    return vec3<f32>(r, g, b);
}

fn compute_color_bounds(uv: vec2<f32>, tex_size: vec2<f32>) -> YCoCgBounds {
    let px = 1.0 / tex_size;
    var min_c = vec3<f32>(1e10);
    var max_c = vec3<f32>(-1e10);
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let suv = uv + vec2<f32>(f32(x), f32(y)) * px;
            let sample_c = textureSampleLevel(t_current, s_point, suv, 0.0).rgb;
            let ycocg = rgb_to_ycocg(sample_c);
            min_c = min(min_c, ycocg);
            max_c = max(max_c, ycocg);
        }
    }
    return YCoCgBounds(min_c, max_c);
}

@compute @workgroup_size(8, 8)
fn taa_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_current);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    let current_color = textureSampleLevel(t_current, s_linear, uv, 0.0).rgb;
    let velocity = textureSampleLevel(t_velocity, s_point, uv, 0.0).rg;

    // Reproject UV using velocity
    let history_uv = uv - velocity;
    let history_color = textureSampleLevel(t_history, s_linear, history_uv, 0.0).rgb;

    // Neighborhood color clamping in YCoCg
    let bounds = compute_color_bounds(uv, vec2<f32>(dims));
    let min_ycocg = bounds.min_c;
    let max_ycocg = bounds.max_c;
    let history_ycocg = rgb_to_ycocg(history_color);

    // Expand bounds slightly for tolerance
    let expand = (max_ycocg - min_ycocg) * (1.0 - params.clip_factor);
    let clip_min = min_ycocg - expand;
    let clip_max = max_ycocg + expand;

    let clamped_ycocg = clamp(history_ycocg, clip_min, clip_max);
    let clamped_history = ycocg_to_rgb(clamped_ycocg);

    // Blend
    let blend = params.blend_factor;
    // Reduce blend weight at disocclusions (where history UV is out of bounds)
    let edge_factor = select(blend, 1.0,
        history_uv.x < 0.0 || history_uv.x > 1.0 ||
        history_uv.y < 0.0 || history_uv.y > 1.0);

    let resolved = mix(clamped_history, current_color, edge_factor);

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(resolved, 1.0));
}
