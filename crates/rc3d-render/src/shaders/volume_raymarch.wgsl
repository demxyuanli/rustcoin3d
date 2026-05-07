// Volume ray-march rendering (HOOPS Cellular Volumes equivalent).
// Ray-marches a 3D texture with density+color for volumetric visualization.

struct VolumeParams {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    dimensions: vec3<u32>,
    density_scale: f32,
    step_count: u32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var t_volume: texture_3d<f32>;
@group(0) @binding(1) var t_depth: texture_2d<f32>;
@group(0) @binding(2) var s_volume: sampler;
@group(0) @binding(3) var<uniform> params: VolumeParams;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    var o: VsOut;
    o.clip_pos = vec4<f32>(positions[vi], 0.0, 1.0);
    o.uv = vec2<f32>((positions[vi].x + 1.0) * 0.5, 1.0 - (positions[vi].y + 1.0) * 0.5);
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(t_depth, s_volume, i.uv).r;
    if depth >= 1.0 { discard; }

    let dims = vec2<f32>(textureDimensions(t_depth));
    let ndc = vec4<f32>(i.uv * 2.0 - 1.0, depth, 1.0);
    let world_pos = params.inv_view_proj * ndc;
    let world = world_pos.xyz / world_pos.w;

    let cam = params.camera_pos.xyz;
    let dir = normalize(world - cam);
    let dims_v = vec3<f32>(f32(params.dimensions.x), f32(params.dimensions.y), f32(params.dimensions.z));
    let texel_size = 1.0 / max(dims_v, vec3<f32>(1.0));

    var t = 0.0;
    var accumulated = vec4<f32>(0.0);
    let steps = f32(min(params.step_count, 256u));
    for (var s = 0u; s < params.step_count; s = s + 1u) {
        let sample_uvw = (world + dir * t + 0.5) / dims_v;
        if any(sample_uvw < 0.0) || any(sample_uvw > 1.0) { break; }
        let sample_texcoord = sample_uvw;
        let density = textureSampleLevel(t_volume, s_volume, sample_texcoord, 0.0).r * params.density_scale;
        accumulated.rgb = accumulated.rgb + (1.0 - accumulated.a) * density * vec3<f32>(density, density * 0.5, density * 0.25);
        accumulated.a = accumulated.a + (1.0 - accumulated.a) * density;
        if accumulated.a > 0.99 { break; }
        t = t + texel_size.x * 0.5;
    }
    return vec4<f32>(accumulated.rgb, accumulated.a);
}
