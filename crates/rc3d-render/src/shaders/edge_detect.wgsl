// Screen-space edge detection via Sobel 3x3 depth gradient.
// Reads a depth texture (Depth32Float) and outputs edge intensity.

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var depth_sampler: sampler;

struct Uniforms {
    texel_size: vec2<f32>,
    threshold: f32,
    edge_color: vec3<f32>,
}

@group(0) @binding(2) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_sobel(@builtin(vertex_index) vi: u32) -> VsOut {
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
    o.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    o.uv = uvs[vi];
    return o;
}

fn sample_depth(uv: vec2<f32>, offset: vec2<f32>) -> f32 {
    return textureSampleLevel(depth_tex, depth_sampler, uv + offset * u.texel_size, 0.0);
}

@fragment
fn fs_sobel(in: VsOut) -> @location(0) vec4<f32> {
    let s00 = sample_depth(in.uv, vec2<f32>(-1.0, -1.0));
    let s10 = sample_depth(in.uv, vec2<f32>( 0.0, -1.0));
    let s20 = sample_depth(in.uv, vec2<f32>( 1.0, -1.0));
    let s01 = sample_depth(in.uv, vec2<f32>(-1.0,  0.0));
    let s21 = sample_depth(in.uv, vec2<f32>( 1.0,  0.0));
    let s02 = sample_depth(in.uv, vec2<f32>(-1.0,  1.0));
    let s12 = sample_depth(in.uv, vec2<f32>( 0.0,  1.0));
    let s22 = sample_depth(in.uv, vec2<f32>( 1.0,  1.0));

    let gx = -s00 - 2.0 * s01 - s02 + s20 + 2.0 * s21 + s22;
    let gy = -s00 - 2.0 * s10 - s20 + s02 + 2.0 * s12 + s22;
    let gradient = sqrt(gx * gx + gy * gy);
    let edge = smoothstep(u.threshold * 0.5, u.threshold * 1.5, gradient);
    return vec4<f32>(u.edge_color * edge, edge);
}
