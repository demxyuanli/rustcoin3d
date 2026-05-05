// Fullscreen FXAA for already-LDR scene (non-HDR path). Luminance-edge blend, same idea as post_tonemap tail.

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

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

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_fxaa(i: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(scene_tex));
    let r = 1.0 / max(dims, vec2<f32>(1.0));

    let c_m = textureSampleLevel(scene_tex, samp, i.uv, 0.0).rgb;
    let c_l = textureSampleLevel(scene_tex, samp, i.uv + vec2<f32>(-r.x, 0.0), 0.0).rgb;
    let c_r = textureSampleLevel(scene_tex, samp, i.uv + vec2<f32>(r.x, 0.0), 0.0).rgb;
    let c_d = textureSampleLevel(scene_tex, samp, i.uv + vec2<f32>(0.0, -r.y), 0.0).rgb;
    let c_u = textureSampleLevel(scene_tex, samp, i.uv + vec2<f32>(0.0, r.y), 0.0).rgb;

    let l_m = luma(c_m);
    let l_l = luma(c_l);
    let l_r = luma(c_r);
    let l_d = luma(c_d);
    let l_u = luma(c_u);
    let edge = max(max(abs(l_m - l_l), abs(l_m - l_r)), max(abs(l_m - l_d), abs(l_m - l_u)));
    let blend = clamp(edge * 8.0, 0.0, 1.0);
    let avg = (c_l + c_r + c_d + c_u) * 0.25;
    let filtered = mix(c_m, avg, blend);
    return vec4<f32>(filtered, 1.0);
}
