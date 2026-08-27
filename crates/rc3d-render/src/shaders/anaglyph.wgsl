// Red/cyan anaglyph composite of left (R) and right (GB) eye tiles.

@group(0) @binding(0) var left_tex: texture_2d<f32>;
@group(0) @binding(1) var right_tex: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_anaglyph(@builtin(vertex_index) vi: u32) -> VsOut {
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
    o.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    o.uv = uvs[vi];
    return o;
}

@fragment
fn fs_anaglyph(in: VsOut) -> @location(0) vec4<f32> {
    let l = textureSampleLevel(left_tex, src_sampler, in.uv, 0.0);
    let r = textureSampleLevel(right_tex, src_sampler, in.uv, 0.0);
    return vec4<f32>(l.r, r.g, r.b, 1.0);
}
