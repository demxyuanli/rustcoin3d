struct FlatUniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: FlatUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
}

struct LineVertexInput {
    @location(0) position: vec3<f32>,
}

struct MarkupVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct MarkupVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return u.mvp * vec4<f32>(in.position, 1.0);
}

@vertex
fn vs_line(in: LineVertexInput) -> @builtin(position) vec4<f32> {
    return u.mvp * vec4<f32>(in.position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return u.color;
}

// ── Colored markup lines: vertex color → fragment ──

@vertex
fn vs_markup(in: MarkupVertexInput) -> MarkupVertexOutput {
    var out: MarkupVertexOutput;
    out.position = u.mvp * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_markup(in: MarkupVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
