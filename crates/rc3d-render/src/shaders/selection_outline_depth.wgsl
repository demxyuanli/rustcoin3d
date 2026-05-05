struct FlatUniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: FlatUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) clip_z: f32,
    @location(1) clip_w: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VsOut {
    var o: VsOut;
    o.clip_pos = u.mvp * vec4<f32>(in.position, 1.0);
    o.clip_z = o.clip_pos.z;
    o.clip_w = o.clip_pos.w;
    return o;
}

@fragment
fn fs_main(@location(0) clip_z: f32, @location(1) clip_w: f32) -> @location(0) vec4<f32> {
    let nz = clip_z / clip_w;
    return vec4<f32>(nz, 0.0, 0.0, 1.0);
}
