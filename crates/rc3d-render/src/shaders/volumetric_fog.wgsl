// Volumetric fog: screen-space ray marching through frustum-aligned depth slices.
// Uses exponential height fog + directional light scattering.

struct FogParams {
    inv_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,      // primary directional light direction
    light_color: vec4<f32>,    // directional light color * intensity
    fog_color: vec4<f32>,      // base fog / ambient color
    fog_density: f32,          // global density multiplier
    height_falloff: f32,       // exponential height falloff (higher = denser at low altitude)
    global_density: f32,       // base density at height 0
    max_distance: f32,         // max ray march distance
    num_steps: u32,            // ray march steps
    scattering: f32,           // in-scattering strength
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var t_depth: texture_2d<f32>;       // linear depth
@group(0) @binding(1) var s_point: sampler;
@group(0) @binding(2) var<uniform> params: FogParams;
@group(0) @binding(3) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var t_scene: texture_2d<f32>;       // scene HDR color

// Reconstruct world position from depth and UV
fn world_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view = params.inv_proj * clip;
    return view.xyz / view.w;
}

// Exponential height fog density at a world position
fn fog_density_at(world_pos: vec3<f32>) -> f32 {
    let h = world_pos.y;
    let height_factor = exp(-h * params.height_falloff);
    return params.global_density * params.fog_density * height_factor;
}

// Phase function: Henyey-Greenstein for fog scattering
fn phase_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159 * max(denom, 1e-5) * sqrt(denom));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_depth);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let depth = textureSampleLevel(t_depth, s_point, uv, 0.0).r;
    let world_pos = world_pos_from_depth(uv, depth);
    let camera_pos = params.camera_pos.xyz;

    // Ray from camera to world position
    var ray_dir = world_pos - camera_pos;
    let ray_len = length(ray_dir);
    ray_dir = ray_dir / max(ray_len, 1e-6);

    let march_dist = min(ray_len, params.max_distance);
    let step_size = march_dist / f32(params.num_steps);

    var transmittance = 1.0;
    var scattered_light = vec3<f32>(0.0);

    // Jitter start position for temporal stability
    let jitter = fract(sin(dot(uv, vec2<f32>(12.9898, 78.233))) * 43758.5453) * step_size;
    var current_pos = camera_pos + ray_dir * jitter;

    for (var i = 0u; i < params.num_steps; i++) {
        let density = fog_density_at(current_pos);

        if density > 0.0001 {
            let step_transmittance = exp(-density * step_size);

            // In-scattering from directional light
            let light_dir = normalize(params.light_dir.xyz);
            let cos_theta = dot(ray_dir, light_dir);
            let phase = phase_hg(cos_theta, 0.3); // forward scattering

            let scattering = params.light_color.rgb * density * phase * params.scattering;
            scattered_light += scattering * transmittance * step_size;

            transmittance *= step_transmittance;
        }

        current_pos += ray_dir * step_size;

        // Early exit if nearly opaque
        if transmittance < 0.01 {
            break;
        }
    }

    // Combine fog color with scattered light, then composite over the scene:
    // out = scene * transmittance + fog. Writing the composited image keeps
    // this pass a valid ping-pong stage in the post-processing chain.
    let fog = params.fog_color.rgb * (1.0 - transmittance) + scattered_light;
    let scene = textureLoad(t_scene, vec2<i32>(gid.xy), 0).rgb;
    let result = scene * transmittance + fog;

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(result, 1.0));
}
