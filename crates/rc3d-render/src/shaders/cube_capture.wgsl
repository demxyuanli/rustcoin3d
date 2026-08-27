struct Uniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_n: vec3<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    o.clip = u.mvp * vec4<f32>(v.position, 1.0);
    o.world_n = normalize((u.model * vec4<f32>(v.normal, 0.0)).xyz);
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(i.world_n);
    let ndl = max(dot(n, vec3<f32>(0.25, 0.85, 0.35)), 0.18);
    return vec4<f32>(u.color.rgb * ndl, 1.0);
}
