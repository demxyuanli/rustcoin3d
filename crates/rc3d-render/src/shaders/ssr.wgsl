// Screen-space reflections via HiZ-accelerated ray marching.

struct SsrParams {
    inv_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    max_steps: u32,          // max ray march steps (e.g. 64)
    max_distance: f32,       // max world-space ray distance
    thickness: f32,          // depth tolerance
    stride: f32,             // step size in UV space
    roughness_cutoff: f32,   // max roughness for SSR (e.g. 0.8)
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var t_color: texture_2d<f32>;      // HDR scene color
@group(0) @binding(1) var t_depth: texture_2d<f32>;      // linear depth
@group(0) @binding(2) var t_hzb: texture_2d<f32>;        // HiZ max pyramid (reverse-Z)
@group(0) @binding(3) var s_point: sampler;
@group(0) @binding(4) var s_linear: sampler;
@group(0) @binding(5) var<uniform> params: SsrParams;
@group(0) @binding(6) var output_tex: texture_storage_2d<rgba16float, write>;

// Reconstruct view-space position from depth and screen UV
fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view = params.inv_proj * clip;
    return view.xyz / view.w;
}

// Ray march in screen space with HiZ acceleration
fn trace_ssr(ray_origin_vs: vec3<f32>, ray_dir_vs: vec3<f32>, hzb_dims: vec2<f32>) -> vec2<f32> {
    // Project ray start and end
    let start_clip = params.inv_proj * vec4<f32>(ray_origin_vs, 1.0);
    let start_uv = (start_clip.xy / start_clip.w) * 0.5 + 0.5;

    let end_vs = ray_origin_vs + ray_dir_vs * params.max_distance;
    let end_clip = params.inv_proj * vec4<f32>(end_vs, 1.0);
    let end_uv = (end_clip.xy / end_clip.w) * 0.5 + 0.5;

    var ray_uv = start_uv;
    var ray_vs = ray_origin_vs;
    let step_uv = (end_uv - start_uv) / f32(params.max_steps);
    let step_vs = ray_dir_vs * params.max_distance / f32(params.max_steps);

    for (var i = 0u; i < params.max_steps; i++) {
        ray_uv += step_uv;
        ray_vs += step_vs;

        // Bounds check
        if any(ray_uv < vec2<f32>(0.0)) || any(ray_uv > vec2<f32>(1.0)) {
            return vec2<f32>(-1.0);
        }

        // Sample depth at current step
        let scene_depth = textureSampleLevel(t_depth, s_point, ray_uv, 0.0).r;
        let scene_vs = view_pos_from_depth(ray_uv, scene_depth);

        // Check for intersection: ray_z goes behind scene_z
        let ray_z = -ray_vs.z; // camera looks -Z in view space
        let scene_z = -scene_vs.z;

        if ray_z > scene_z && (ray_z - scene_z) < params.thickness {
            return ray_uv;
        }
    }

    return vec2<f32>(-1.0); // no hit
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_color);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    let depth = textureSampleLevel(t_depth, s_point, uv, 0.0).r;
    let view_pos = view_pos_from_depth(uv, depth);

    // Simplified: use view-space reflection about origin
    // In production, this needs the per-pixel normal from a G-buffer
    let N = normalize(-view_pos); // approximate normal as viewing direction (toward origin)
    let V = normalize(-view_pos);
    let R = reflect(-V, N);

    // Trace reflection
    let hit_uv = trace_ssr(view_pos, R, vec2<f32>(dims));

    if hit_uv.x < 0.0 {
        // No reflection — write black
        textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // Sample reflected color
    let reflected_color = textureSampleLevel(t_color, s_linear, hit_uv, 0.0).rgb;

    // Edge fade: fade near screen borders
    let edge_dist = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
    let edge_fade = smoothstep(0.0, 0.1, edge_dist);

    // Distance fade
    let dist = length(view_pos);
    let dist_fade = 1.0 - smoothstep(0.0, params.max_distance, dist);

    let fade = edge_fade * dist_fade * 0.5; // 50% max contribution

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(reflected_color * fade, 1.0));
}
