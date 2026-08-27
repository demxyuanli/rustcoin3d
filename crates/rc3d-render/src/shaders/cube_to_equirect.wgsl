@group(0) @binding(0) var cube_tex: texture_cube<f32>;
@group(0) @binding(1) var cube_samp: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var o: VsOut;
    o.clip_pos = vec4<f32>(positions[vi], 0.0, 1.0);
    o.uv = uvs[vi];
    return o;
}

const PI: f32 = 3.14159265359;

/// Inverse of `direction_to_uv` in pbr.wgsl.
@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let theta = (i.uv.x - 0.5) * 2.0 * PI;
    let phi = (i.uv.y - 0.5) * PI;
    let cp = cos(phi);
    let dir = vec3<f32>(cp * sin(theta), sin(phi), -cp * cos(theta));
    return textureSampleLevel(cube_tex, cube_samp, dir, 0.0);
}
