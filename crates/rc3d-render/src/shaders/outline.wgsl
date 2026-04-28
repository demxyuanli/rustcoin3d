struct OutlineUniforms {
    mvp: mat4x4<f32>,
    outline_width: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: OutlineUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    let offset_pos = in.position + normalize(in.normal) * u.outline_width;
    return u.mvp * vec4<f32>(offset_pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return u.color;
}
