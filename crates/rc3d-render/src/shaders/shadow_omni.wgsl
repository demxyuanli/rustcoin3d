// Omnidirectional shadow map rendering for point lights.
// Renders depth to a cube map (6 faces).

struct OmniShadowUniforms {
    view_proj: mat4x4<f32>,
    light_pos: vec3<f32>,
    far_plane: f32,
};

@group(0) @binding(0) var<uniform> u: OmniShadowUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos4 = vec4<f32>(in.position, 1.0);
    out.clip_position = u.view_proj * world_pos4;
    out.world_pos = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @builtin(frag_depth) f32 {
    let dist = length(in.world_pos - u.light_pos);
    return dist / u.far_plane;
}
