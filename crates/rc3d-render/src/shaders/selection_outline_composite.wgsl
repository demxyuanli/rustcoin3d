struct OverlayParams {
    edge_strength: f32,
    _pad: vec3<f32>,
};

@group(0) @binding(0) var mask_tex: texture_2d<f32>;
@group(0) @binding(1) var edge_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> params: OverlayParams;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var o: VsOut;
    o.clip_pos = vec4<f32>(positions[vi], 0.0, 1.0);
    o.uv = uvs[vi];
    return o;
}

@fragment
fn fs_overlay(i: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(mask_tex));
    let p = vec2<i32>(
        clamp(i32(i.clip_pos.x), 0, dims.x - 1),
        clamp(i32(i.clip_pos.y), 0, dims.y - 1),
    );
    let mask_r = textureLoad(mask_tex, p, 0).r;
    let edge = textureSample(edge_tex, samp, i.uv);
    // mask.r is 1 outside the selection (clear white) and 0 inside.
    let rgb = edge.rgb * mask_r * params.edge_strength;
    return vec4<f32>(rgb, 1.0);
}
