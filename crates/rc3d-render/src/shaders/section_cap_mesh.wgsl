// Section cap fill shader: renders back faces of mesh with clip planes
// to produce filled cross-section.
//
// FlatUniforms layout (must match Rust FlatUniforms in vertex.rs):
// - mvp at offset 0
// - color at offset 64
// - model at offset 80
// - clip_planes at offset 144
// - clip_count at offset 240

struct FlatUniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
    model: mat4x4<f32>,
    clip_planes: array<vec4<f32>, 6>,
    clip_count: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: FlatUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let world_pos = u.model * vec4<f32>(in.position, 1.0);
    var out: VertexOutput;
    out.pos = u.mvp * vec4<f32>(in.position, 1.0);
    out.world_pos = world_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Test clip planes
    let clip_count = i32(u.clip_count.x);
    for (var i = 0; i < clip_count; i = i + 1) {
        let plane = u.clip_planes[i];
        let dist = dot(plane.xyz, in.world_pos) + plane.w;
        if dist < 0.0 {
            discard;
        }
    }
    return u.color;
}