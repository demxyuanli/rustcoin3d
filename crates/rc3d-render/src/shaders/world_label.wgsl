struct WorldLabelUniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: WorldLabelUniforms;
@group(0) @binding(1) var label_tex: texture_2d<f32>;
@group(0) @binding(2) var label_samp: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_world_label(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_world_label(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = textureSample(label_tex, label_samp, in.uv).r;
    return vec4<f32>(u.color.rgb, u.color.a * a);
}
