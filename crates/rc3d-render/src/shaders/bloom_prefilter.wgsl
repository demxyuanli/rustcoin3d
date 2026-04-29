// Bloom prefilter: extract bright pixels from HDR scene, downsample 2x.
// Kawase-style 4-tap dual filter.
@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;
@group(0) @binding(2) var t_dst: texture_storage_2d<rgba16float, write>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = vec2<f32>(textureDimensions(t_dst));
    if gid.x >= u32(dst_dims.x) || gid.y >= u32(dst_dims.y) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / dst_dims;
    let src_texel = 1.0 / vec2<f32>(textureDimensions(t_src));

    // 4-tap rotated bilinear sampling (Kawase)
    let h = src_texel * 0.5;
    let c0 = textureSampleLevel(t_src, s_src, uv + vec2<f32>(-h.x, -h.y), 0.0).rgb;
    let c1 = textureSampleLevel(t_src, s_src, uv + vec2<f32>( h.x, -h.y), 0.0).rgb;
    let c2 = textureSampleLevel(t_src, s_src, uv + vec2<f32>(-h.x,  h.y), 0.0).rgb;
    let c3 = textureSampleLevel(t_src, s_src, uv + vec2<f32>( h.x,  h.y), 0.0).rgb;
    let avg = (c0 + c1 + c2 + c3) * 0.25;

    let brightness = luma(avg);
    let threshold = 1.0;
    let soft = brightness - threshold;
    let knee = 0.5;
    let weight = clamp(soft / (knee + soft), 0.0, 1.0);
    let bloom_color = avg * weight;

    textureStore(t_dst, vec2<i32>(gid.xy), vec4<f32>(bloom_color, 1.0));
}
