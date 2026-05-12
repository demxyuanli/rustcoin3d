// Cubemap skybox — view-space direction from inverse projection matrix.
// The cubemap "sticks" to the camera so the +Z face is always forward.
// FOV controls how much of the side/top/bottom faces are visible.

struct BgParams {
    mode: u32,
    image_fit: u32,
    _pad0: vec2<u32>,
    top_color: vec4<f32>,
    bot_color: vec4<f32>,
    image_size: vec2<f32>,
    screen_size: vec2<f32>,
    // inverse projection matrix (column-major, 4×4)
    inv_proj_0: vec4<f32>,
    inv_proj_1: vec4<f32>,
    inv_proj_2: vec4<f32>,
    inv_proj_3: vec4<f32>,
}

@group(0) @binding(0) var<uniform> bg: BgParams;
@group(0) @binding(1) var t_cube: texture_cube<f32>;
@group(0) @binding(2) var s_cube: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    var o: VsOut;
    o.clip_pos = vec4<f32>(positions[vi], 1.0, 1.0);
    o.uv = vec2<f32>((positions[vi].x + 1.0) * 0.5, (positions[vi].y + 1.0) * 0.5);
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let ndc = vec4<f32>(i.uv.x * 2.0 - 1.0, i.uv.y * 2.0 - 1.0, 1.0, 1.0);
    let inv_proj = mat4x4<f32>(
        bg.inv_proj_0, bg.inv_proj_1, bg.inv_proj_2, bg.inv_proj_3,
    );
    let view_h = inv_proj * ndc;
    let dir = normalize(view_h.xyz / view_h.w);
    return textureSample(t_cube, s_cube, dir);
}
