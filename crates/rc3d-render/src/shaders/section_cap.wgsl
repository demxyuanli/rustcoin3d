struct SectionCapUniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    color: vec4<f32>,
    plane: vec4<f32>,
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: SectionCapUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let world = (u.model * vec4<f32>(in.position, 1.0)).xyz;
    var o: VertexOutput;
    o.clip_pos = u.mvp * vec4<f32>(in.position, 1.0);
    o.world_pos = world;
    return o;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = dot(u.plane.xyz, in.world_pos) + u.plane.w;
    let w = max(fwidth(d) * 2.0, u.params.x);
    if (abs(d) > w) {
        discard;
    }
    return u.color;
}
