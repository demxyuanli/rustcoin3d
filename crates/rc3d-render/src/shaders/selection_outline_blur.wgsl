struct BlurParams {
    texel_size: vec2<f32>,
    direction: vec2<f32>,
    kernel_radius: f32,
    _pad0: f32,
    _pad1: vec2<f32>,
    _pad2: vec4<f32>,
};

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: BlurParams;

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

const MAX_RADIUS: i32 = 4;

fn gaussian_pdf(x: f32, sigma: f32) -> f32 {
    return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

@fragment
fn fs_blur(i: VsOut) -> @location(0) vec4<f32> {
    let sigma = max(params.kernel_radius * 0.5, 1.0e-4);
    var weight_sum = gaussian_pdf(0.0, sigma);
    var diffuse_sum = textureSample(color_tex, samp, i.uv) * weight_sum;
    let delta = params.direction * params.texel_size * params.kernel_radius / f32(MAX_RADIUS);
    var uv_offset = delta;
    for (var r = 1; r <= MAX_RADIUS; r = r + 1) {
        let x = params.kernel_radius * f32(r) / f32(MAX_RADIUS);
        let w = gaussian_pdf(x, sigma);
        let s1 = textureSample(color_tex, samp, i.uv + uv_offset);
        let s2 = textureSample(color_tex, samp, i.uv - uv_offset);
        diffuse_sum = diffuse_sum + (s1 + s2) * w;
        weight_sum = weight_sum + 2.0 * w;
        uv_offset = uv_offset + delta;
    }
    return diffuse_sum / weight_sum;
}
