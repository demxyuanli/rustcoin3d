// SMAA (Subpixel Morphological Anti-Aliasing) — Blend Pass
// Applies the edge pattern from the edge-detection pass as a 4-tap
// morphological filter, blending the center pixel with weighted neighbors.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var t_edges: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vi & 1u) << 2u);
    let y = f32((vi & 2u) << 1u);
    out.pos = vec4<f32>(x - 1.0, 1.0 - y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5, y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(t_input));
    let center = textureSample(t_input, s_input, in.uv);
    let edge = textureSampleLevel(t_edges, s_input, in.uv, 0.0).r;

    if (edge < 0.5) {
        return center;
    }

    // 4-neighbor blend with distance-weighted contribution
    let top = textureSample(t_input, s_input, in.uv + vec2<f32>(0.0, -texel.y));
    let bottom = textureSample(t_input, s_input, in.uv + vec2<f32>(0.0, texel.y));
    let left = textureSample(t_input, s_input, in.uv + vec2<f32>(-texel.x, 0.0));
    let right = textureSample(t_input, s_input, in.uv + vec2<f32>(texel.x, 0.0));

    // Weighted blur: blend only along detected edges
    let h_weight = 0.25;
    let v_weight = 0.25;
    let blend_h = mix(center, (left + right) * 0.5, h_weight);
    let blend_v = mix(center, (top + bottom) * 0.5, v_weight);

    return (blend_h + blend_v) * 0.5;
}
