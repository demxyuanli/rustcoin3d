// WBOIT composite shader
// Outputs the weighted-blended transparent layer with premultiplied alpha.
// Hardware blending (src * ONE + dst * ONE_MINUS_SRC_ALPHA) composites
// the result onto the opaque scene, avoiding a texture read-back of the
// scene in the shader (and the resulting read-write hazard).

@group(0) @binding(0) var t_accum: texture_2d<f32>;
@group(0) @binding(1) var t_revealage: texture_2d<f32>;
@group(0) @binding(2) var s_accum: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    // Full-screen triangle
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
    var o: VertexOutput;
    o.clip_pos = vec4<f32>(positions[vid], 0.0, 1.0);
    o.uv = uvs[vid];
    return o;
}

const EPSILON: f32 = 1e-6;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let accum = textureSample(t_accum, s_accum, in.uv);
    let revealage = textureSample(t_revealage, s_accum, in.uv).r;

    // Reconstruct transparent layer:
    //   transparent_color = accum.rgb / max(accum.a, epsilon)
    //   transparent_alpha = 1.0 - revealage
    let trans_alpha = 1.0 - revealage;
    let trans_color = accum.rgb / max(accum.a, EPSILON);

    // Output premultiplied-alpha color.
    // Hardware blend (ONE, ONE_MINUS_SRC_ALPHA) computes:
    //   dst = src + dst * (1 - src.a)
    //       = trans_color * trans_alpha + scene * (1 - trans_alpha)
    return vec4<f32>(trans_color * trans_alpha, trans_alpha);
}
