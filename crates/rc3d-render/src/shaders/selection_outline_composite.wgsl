@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var edge_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> strength: CompositeStrength;

struct CompositeStrength {
    params: vec4<f32>,
    texel_size: vec2<f32>,
    _pad: vec2<f32>,
}

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
fn fs_composite(i: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_tex, samp, i.uv);
    let t = strength.texel_size;
    var best_a = 0.0;
    var best_rgb = vec3<f32>(0.0, 0.0, 0.0);
    for (var j = 0u; j < 25u; j = j + 1u) {
        let dx = i32(j % 5u) - 2;
        let dy = i32(j / 5u) - 2;
        let o = vec2<f32>(f32(dx), f32(dy)) * t;
        let e = textureSample(edge_tex, samp, i.uv + o);
        if e.a > best_a {
            best_a = e.a;
            best_rgb = e.rgb;
        }
    }
    let s = best_a * strength.params.x;
    return vec4<f32>(scene.rgb + best_rgb * s, scene.a);
}
