// Engine prelude for ShaderMaterial snippets.
// User code provides `fn material_fs(in: VertexOutput, u: PerDrawUniforms) -> vec4<f32>`.
// Full shaders that already contain `@fragment` skip this wrap.

struct PerDrawUniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_pos: vec4<f32>,
    custom: vec4<f32>,
    base_color: vec4<f32>,
    extra: vec4<f32>, // x = opacity, y = time
}

@group(0) @binding(0) var<uniform> u: PerDrawUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_dir: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = u.model * vec4<f32>(in.position, 1.0);
    out.clip_position = u.mvp * vec4<f32>(in.position, 1.0);
    out.world_pos = world.xyz;
    out.world_normal = normalize((u.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.texcoord;
    out.view_dir = normalize(u.camera_pos.xyz - world.xyz);
    return out;
}
