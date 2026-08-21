struct EdgeParams {
    texel_size: vec2<f32>,
    _pad0: vec2<f32>,
    visible_color: vec3<f32>,
    _pad1: f32,
    hidden_color: vec3<f32>,
    _pad2: f32,
};

@group(0) @binding(0) var mask_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: EdgeParams;

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
fn fs_edge(i: VsOut) -> @location(0) vec4<f32> {
    let t = params.texel_size;
    let c1 = textureSample(mask_tex, samp, i.uv + vec2<f32>(t.x, 0.0));
    let c2 = textureSample(mask_tex, samp, i.uv - vec2<f32>(t.x, 0.0));
    let c3 = textureSample(mask_tex, samp, i.uv + vec2<f32>(0.0, t.y));
    let c4 = textureSample(mask_tex, samp, i.uv - vec2<f32>(0.0, t.y));
    let diff1 = (c1.r - c2.r) * 0.5;
    let diff2 = (c3.r - c4.r) * 0.5;
    let d = length(vec2<f32>(diff1, diff2));
    let a1 = min(c1.g, c2.g);
    let a2 = min(c3.g, c4.g);
    let visibility_factor = min(a1, a2);
    let edge_rgb = select(params.hidden_color, params.visible_color, (1.0 - visibility_factor) > 0.001);
    return vec4<f32>(edge_rgb * d, d);
}
