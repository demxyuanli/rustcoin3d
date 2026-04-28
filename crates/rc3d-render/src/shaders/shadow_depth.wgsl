struct ShadowUniforms {
    shadow_mvp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> su: ShadowUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return su.shadow_mvp * vec4<f32>(in.position, 1.0);
}
