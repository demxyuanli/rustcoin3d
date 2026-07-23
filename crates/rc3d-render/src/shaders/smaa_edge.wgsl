// SMAA (Subpixel Morphological Anti-Aliasing) — Edge Detection Pass
// Based on the SMAA technique by Jorge Jimenez et al.
// Detects color/luma edges in the input image and writes edge pattern to R8.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) offset0: vec4<f32>,
    @location(2) offset1: vec4<f32>,
    @location(3) offset2: vec4<f32>,
}

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vi & 1u) << 2u);
    let y = f32((vi & 2u) << 1u);
    out.pos = vec4<f32>(x - 1.0, 1.0 - y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5, y * 0.5);

    // SMAA pixel offsets for 4-neighbor pattern detection
    let texel = 1.0 / vec2<f32>(textureDimensions(t_input));
    out.offset0 = vec4<f32>(
        out.uv.x, out.uv.y - texel.y,  // top
        out.uv.x - texel.x, out.uv.y,  // left
    );
    out.offset1 = vec4<f32>(
        out.uv.x + texel.x, out.uv.y,  // right
        out.uv.y + texel.x, out.uv.y + texel.y, // bottom
    );
    out.offset2 = vec4<f32>(
        texel.x, texel.y, 0.0, 0.0,         // unused, padded
    );
    return out;
}

fn rgb2luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let center = rgb2luma(textureSample(t_input, s_input, in.uv).rgb);
    let top = rgb2luma(textureSample(t_input, s_input, in.offset0.xy).rgb);
    let left = rgb2luma(textureSample(t_input, s_input, in.offset0.zw).rgb);
    let right = rgb2luma(textureSample(t_input, s_input, in.offset1.xy).rgb);
    let bottom = rgb2luma(textureSample(t_input, s_input, in.offset1.zw).rgb);

    // SMAA color edge detection: mark pixel as edge if it differs enough
    // from any neighbor in cross pattern
    let d_top = abs(center - top);
    let d_left = abs(center - left);
    let d_right = abs(center - right);
    let d_bottom = abs(center - bottom);

    let edge_h = step(0.02, max(d_left, d_right));
    let edge_v = step(0.02, max(d_top, d_bottom));
    let edge = max(edge_h, edge_v);

    return vec4<f32>(edge, 0.0, 0.0, 1.0);
}
