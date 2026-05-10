// Screen-space decal projection (HOOPS Decal equivalent).
// Projects a decal texture onto geometry using depth buffer reconstruction.
//
// Integration path:
// 1. Create pipeline in post_processor.rs (DecalPass with BGL)
// 2. Render a box volume representing the decal projection bounds
// 3. Fragment shader reconstructs world pos from depth, checks if within box,
//    samples decal texture, and blends with existing scene color.

struct DecalParams {
    inv_view_proj: mat4x4<f32>,
    // Box transform: object-to-world of the decal volume
    box_to_world: mat4x4<f32>,
    world_to_box: mat4x4<f32>,
    decal_color: vec4<f32>,
    opacity: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var t_decal: texture_2d<f32>;
@group(0) @binding(1) var s_decal: sampler;
@group(0) @binding(2) var<uniform> params: DecalParams;

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
    o.clip_pos = vec4<f32>(positions[vi], 0.0, 1.0);
    o.uv = vec2<f32>((positions[vi].x + 1.0) * 0.5, 1.0 - (positions[vi].y + 1.0) * 0.5);
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    // Project decal to screen-space UV (fullscreen for now)
    let decal_uv = i.uv;
    let decal = textureSample(t_decal, s_decal, decal_uv).rgb;
    return vec4<f32>(decal * params.decal_color.rgb, params.opacity * params.decal_color.a);
}
