struct EdgeParams {
    texel_size: vec2<f32>,
    visible_color: vec3<f32>,
    hidden_color: vec3<f32>,
    edge_strength: f32,
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

fn mask_fill_at(p: vec2<f32>) -> f32 {
    return 1.0 - textureSample(mask_tex, samp, p).r;
}

// Inside silhouette: R=0 -> fill=1. Outside clear: R=1 -> fill=0. Dilate(fill) rounds pixelized corners.
fn mask_fill_dilated(uv: vec2<f32>) -> f32 {
    let t = params.texel_size;
    var m = mask_fill_at(uv);
    m = max(m, mask_fill_at(uv + vec2<f32>(t.x, 0.0)));
    m = max(m, mask_fill_at(uv - vec2<f32>(t.x, 0.0)));
    m = max(m, mask_fill_at(uv + vec2<f32>(0.0, t.y)));
    m = max(m, mask_fill_at(uv - vec2<f32>(0.0, t.y)));
    m = max(m, mask_fill_at(uv + vec2<f32>(t.x, t.y)));
    m = max(m, mask_fill_at(uv - vec2<f32>(t.x, t.y)));
    m = max(m, mask_fill_at(uv + vec2<f32>(t.x, -t.y)));
    m = max(m, mask_fill_at(uv + vec2<f32>(-t.x, t.y)));
    return m;
}

fn sobel_mag_fill_dilated_se(uv: vec2<f32>) -> f32 {
    let t = params.texel_size;
    let c1 = mask_fill_dilated(uv + vec2<f32>(t.x, 0.0));
    let c2 = mask_fill_dilated(uv - vec2<f32>(t.x, 0.0));
    let c3 = mask_fill_dilated(uv + vec2<f32>(0.0, t.y));
    let c4 = mask_fill_dilated(uv - vec2<f32>(0.0, t.y));
    let diff1 = (c1 - c2) * 0.5;
    let diff2 = (c3 - c4) * 0.5;
    return length(vec2<f32>(diff1, diff2));
}

@fragment
fn fs_edge(i: VsOut) -> @location(0) vec4<f32> {
    let t = params.texel_size;
    let c1 = textureSample(mask_tex, samp, i.uv + vec2<f32>(t.x, 0.0));
    let c2 = textureSample(mask_tex, samp, i.uv - vec2<f32>(t.x, 0.0));
    let c3 = textureSample(mask_tex, samp, i.uv + vec2<f32>(0.0, t.y));
    let c4 = textureSample(mask_tex, samp, i.uv - vec2<f32>(0.0, t.y));
    let a1 = min(c1.g, c2.g);
    let a2 = min(c3.g, c4.g);
    let visibility_factor = min(a1, a2);
    let vis_test = 1.0 - visibility_factor;
    let edge_rgb = select(params.hidden_color, params.visible_color, vis_test > 0.001);

    let d0 = sobel_mag_fill_dilated_se(i.uv);
    let d1 = sobel_mag_fill_dilated_se(i.uv + vec2<f32>(t.x, 0.0));
    let d2 = sobel_mag_fill_dilated_se(i.uv - vec2<f32>(t.x, 0.0));
    let d3 = sobel_mag_fill_dilated_se(i.uv + vec2<f32>(0.0, t.y));
    let d4 = sobel_mag_fill_dilated_se(i.uv - vec2<f32>(0.0, t.y));
    let d5 = sobel_mag_fill_dilated_se(i.uv + vec2<f32>(t.x, t.y));
    let d6 = sobel_mag_fill_dilated_se(i.uv - vec2<f32>(t.x, t.y));
    let d7 = sobel_mag_fill_dilated_se(i.uv + vec2<f32>(t.x, -t.y));
    let d8 = sobel_mag_fill_dilated_se(i.uv + vec2<f32>(-t.x, t.y));
    let d9 = sobel_mag_fill_dilated_se(i.uv + vec2<f32>(2.0 * t.x, 0.0));
    let d10 = sobel_mag_fill_dilated_se(i.uv - vec2<f32>(2.0 * t.x, 0.0));
    let d11 = sobel_mag_fill_dilated_se(i.uv + vec2<f32>(0.0, 2.0 * t.y));
    let d12 = sobel_mag_fill_dilated_se(i.uv - vec2<f32>(0.0, 2.0 * t.y));
    var d = d0;
    d = max(d, d1);
    d = max(d, d2);
    d = max(d, d3);
    d = max(d, d4);
    d = max(d, d5);
    d = max(d, d6);
    d = max(d, d7);
    d = max(d, d8);
    d = max(d, d9);
    d = max(d, d10);
    d = max(d, d11);
    d = max(d, d12);

    let fw = fwidth(d);
    let cov = smoothstep(0.0, max(fw * 2.5, 1e-7), d);
    let strength = cov * params.edge_strength;
    return vec4<f32>(edge_rgb * strength, strength);
}
