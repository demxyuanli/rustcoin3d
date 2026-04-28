@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
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

fn reinhard(c: vec3<f32>) -> vec3<f32> {
    return c / (c + vec3<f32>(1.0));
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_tex));
    let r = 1.0 / max(dims, vec2<f32>(1.0));
    let c_m = textureSampleLevel(hdr_tex, hdr_samp, i.uv, 0.0).rgb;
    let c_l = textureSampleLevel(hdr_tex, hdr_samp, i.uv + vec2<f32>(-r.x, 0.0), 0.0).rgb;
    let c_r = textureSampleLevel(hdr_tex, hdr_samp, i.uv + vec2<f32>(r.x, 0.0), 0.0).rgb;
    let c_d = textureSampleLevel(hdr_tex, hdr_samp, i.uv + vec2<f32>(0.0, -r.y), 0.0).rgb;
    let c_u = textureSampleLevel(hdr_tex, hdr_samp, i.uv + vec2<f32>(0.0, r.y), 0.0).rgb;
    let l_m = luma(reinhard(c_m));
    let l_l = luma(reinhard(c_l));
    let l_r = luma(reinhard(c_r));
    let l_d = luma(reinhard(c_d));
    let l_u = luma(reinhard(c_u));
    let edge = max(max(abs(l_m - l_l), abs(l_m - l_r)), max(abs(l_m - l_d), abs(l_m - l_u)));
    let blend = clamp(edge * 8.0, 0.0, 1.0);
    let avg = (c_l + c_r + c_d + c_u) * 0.25;
    let filtered = mix(c_m, avg, blend);
    let ldr = reinhard(filtered);
    return vec4<f32>(ldr, 1.0);
}
