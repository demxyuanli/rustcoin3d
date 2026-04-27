struct FlatUniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: FlatUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return u.mvp * vec4<f32>(in.position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return u.color;
}
