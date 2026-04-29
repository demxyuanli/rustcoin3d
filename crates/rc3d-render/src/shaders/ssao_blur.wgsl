// SSAO cross-bilateral blur: depth-aware horizontal + vertical blur.
// Uses 4-tap filter with depth-based weighting to preserve edges.

@group(0) @binding(0) var t_ao: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_2d<f32>;
@group(0) @binding(2) var s_point: sampler;

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

fn blur_h(uv: vec2<f32>, center_depth: f32, texel: vec2<f32>) -> f32 {
    let depth_sigma = 0.05;
    var sum = 0.0;
    var weight = 0.0;

    let offsets = array<f32, 4>(-1.5, -0.5, 0.5, 1.5);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let off_uv = uv + vec2<f32>(offsets[i] * texel.x, 0.0);
        let ao = textureSampleLevel(t_ao, s_point, off_uv, 0.0).r;
        let d = textureSampleLevel(t_depth, s_point, off_uv, 0.0).r;
        let w = exp(-abs(d - center_depth) / depth_sigma);
        sum += ao * w;
        weight += w;
    }
    return sum / max(weight, 0.0001);
}

fn blur_v(uv: vec2<f32>, center_depth: f32, texel: vec2<f32>) -> f32 {
    let depth_sigma = 0.05;
    var sum = 0.0;
    var weight = 0.0;

    let offsets = array<f32, 4>(-1.5, -0.5, 0.5, 1.5);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let off_uv = uv + vec2<f32>(0.0, offsets[i] * texel.y);
        let ao = textureSampleLevel(t_ao, s_point, off_uv, 0.0).r;
        let d = textureSampleLevel(t_depth, s_point, off_uv, 0.0).r;
        let w = exp(-abs(d - center_depth) / depth_sigma);
        sum += ao * w;
        weight += w;
    }
    return sum / max(weight, 0.0001);
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(t_ao));
    let depth = textureSampleLevel(t_depth, s_point, i.uv, 0.0).r;

    // Horizontal blur
    let h_result = blur_h(i.uv, depth, texel);
    // Vertical blur (use horizontal result as center for second pass)
    let v_result = blur_v(i.uv, depth, texel);

    let ao = (h_result + v_result) * 0.5;
    return vec4<f32>(ao, ao, ao, 1.0);
}
