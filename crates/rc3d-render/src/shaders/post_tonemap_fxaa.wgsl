// Post-processing: ACES tonemapping, Bloom compositing, SSAO application, FXAA.

@group(0) @binding(0) var t_hdr: texture_2d<f32>;
@group(0) @binding(1) var t_bloom: texture_2d<f32>;
@group(0) @binding(2) var t_ssao: texture_2d<f32>;
@group(0) @binding(3) var s_point: sampler;

struct PostParams {
    vignette: f32,
    chromatic: f32,
    bloom_str: f32,
    grain: f32,
}
@group(0) @binding(4) var<uniform> params: PostParams;

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

fn tonemap_aces(hdr: vec3<f32>) -> vec3<f32> {
    // ACES filmic (Narkowicz 2015 fit)
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(t_hdr));
    let r = 1.0 / max(dims, vec2<f32>(1.0));

    // Sample HDR scene (center + 4 neighbors for FXAA edge detection)
    let c_m = textureSampleLevel(t_hdr, s_point, i.uv, 0.0).rgb;
    let c_l = textureSampleLevel(t_hdr, s_point, i.uv + vec2<f32>(-r.x, 0.0), 0.0).rgb;
    let c_r = textureSampleLevel(t_hdr, s_point, i.uv + vec2<f32>( r.x, 0.0), 0.0).rgb;
    let c_d = textureSampleLevel(t_hdr, s_point, i.uv + vec2<f32>(0.0, -r.y), 0.0).rgb;
    let c_u = textureSampleLevel(t_hdr, s_point, i.uv + vec2<f32>(0.0,  r.y), 0.0).rgb;

    // FXAA edge detection on tonemapped luminance
    let l_m = luma(tonemap_aces(c_m));
    let l_l = luma(tonemap_aces(c_l));
    let l_r = luma(tonemap_aces(c_r));
    let l_d = luma(tonemap_aces(c_d));
    let l_u = luma(tonemap_aces(c_u));
    let edge = max(max(abs(l_m - l_l), abs(l_m - l_r)), max(abs(l_m - l_d), abs(l_m - l_u)));
    let blend = clamp(edge * 8.0, 0.0, 1.0);
    let avg = (c_l + c_r + c_d + c_u) * 0.25;
    var filtered = mix(c_m, avg, blend);

    // Chromatic aberration: offset R and B channels
    if params.chromatic > 0.0 {
        let center = i.uv - 0.5;
        let dist = length(center);
        let offset = center * dist * params.chromatic * 0.02;
        let r_sample = textureSampleLevel(t_hdr, s_point, i.uv + offset, 0.0).r;
        let b_sample = textureSampleLevel(t_hdr, s_point, i.uv - offset, 0.0).b;
        filtered = vec3<f32>(r_sample, filtered.g, b_sample);
    }

    // Bloom compositing
    let bloom_sample = textureSampleLevel(t_bloom, s_point, i.uv, 0.0).rgb;
    filtered = filtered + bloom_sample * params.bloom_str;

    // SSAO application
    let ao = textureSampleLevel(t_ssao, s_point, i.uv, 0.0).r;
    filtered = filtered * mix(0.85, 1.0, ao);

    // ACES tonemapping
    let ldr = tonemap_aces(filtered);

    // Vignette: radial darkening
    let vig = 1.0 - dot(i.uv - 0.5, i.uv - 0.5) * params.vignette * 2.0;
    let vig_ldr = ldr * clamp(vig, 0.0, 1.0);

    // Film grain
    let grain_noise = fract(sin(dot(i.clip_pos.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let grain_ldr = vig_ldr + (grain_noise - 0.5) * params.grain;

    return vec4<f32>(grain_ldr, 1.0);
}
