// Composite an offscreen overlay tile onto the swapchain.
// Color + depth: pixels that never wrote depth (clear value) are discarded so
// the main film shows through (transparent nav cube, etc.).

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var s_color: sampler;
@group(0) @binding(2) var t_depth: texture_depth_2d;
@group(0) @binding(3) var s_depth: sampler;

struct BlitOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_blit(@builtin(vertex_index) vid: u32) -> BlitOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: BlitOut;
    out.clip = vec4<f32>(pos[vid], 0.0, 1.0);
    out.uv = uv[vid];
    return out;
}

@fragment
fn fs_blit(input: BlitOut) -> @location(0) vec4<f32> {
    let uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let d = textureSample(t_depth, s_depth, uv);
    // Forward-Z clear ~= 1, reverse-Z clear ~= 0. Either means "no geometry".
    if (d < 1.0e-4 || d > 0.9999) {
        discard;
    }
    return textureSample(t_color, s_color, uv);
}
