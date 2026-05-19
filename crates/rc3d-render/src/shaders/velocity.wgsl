// Compute screen-space velocity (motion vectors) from depth + camera matrices.
// Uses current and previous frame view-projection to compute per-pixel UV displacement.

struct VelocityParams {
    inv_vp_curr: mat4x4<f32>,
    vp_prev: mat4x4<f32>,
    _pad0: vec2<f32>,
    _pad1: vec2<f32>,
};

@group(0) @binding(0) var t_depth: texture_2d<f32>;
@group(0) @binding(1) var s_point: sampler;
@group(0) @binding(2) var<uniform> params: VelocityParams;
@group(0) @binding(3) var output_vel: texture_storage_2d<rg16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_depth);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let depth = textureSampleLevel(t_depth, s_point, uv, 0.0).r;

    // Reconstruct world position from depth
    let clip = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let world = params.inv_vp_curr * clip;
    let world_pos = world.xyz / world.w;

    // Reproject to previous frame
    let prev_clip = params.vp_prev * vec4<f32>(world_pos, 1.0);
    let prev_uv = (prev_clip.xy / max(prev_clip.w, 1e-6)) * 0.5 + 0.5;

    // Motion vector in UV space
    let velocity = uv - prev_uv;
    textureStore(output_vel, vec2<i32>(gid.xy), vec4<f32>(velocity, 0.0, 0.0));
}
