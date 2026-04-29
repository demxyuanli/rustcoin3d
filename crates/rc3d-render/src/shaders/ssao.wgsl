// SSAO: Hemisphere sampling with random rotation + range check.
// Inputs: depth texture, 4x4 noise texture.
// Output: single-channel AO factor [0, 1].

struct SsaoParams {
    proj: mat4x4<f32>,        // projection matrix for depth→view reconstruction
    inv_proj: mat4x4<f32>,    // inverse projection
    radius: f32,               // AO sample radius in view space
    bias: f32,                 // depth bias to avoid self-occlusion
    power: f32,                // AO power curve exponent
    _pad: vec2<f32>,
}

@group(0) @binding(0) var t_depth: texture_2d<f32>;
@group(0) @binding(1) var t_noise: texture_2d<f32>;
@group(0) @binding(2) var s_point: sampler;
@group(0) @binding(3) var<uniform> params: SsaoParams;

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

// Reconstruct view-space position from depth + UV using inverse projection
fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view = params.inv_proj * ndc;
    return view.xyz / view.w;
}

// Get view-space normal from depth using cross product of neighbors
fn view_normal(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(t_depth));
    let d_r = textureSampleLevel(t_depth, s_point, uv + vec2<f32>(texel.x, 0.0), 0.0).r;
    let d_l = textureSampleLevel(t_depth, s_point, uv + vec2<f32>(-texel.x, 0.0), 0.0).r;
    let d_u = textureSampleLevel(t_depth, s_point, uv + vec2<f32>(0.0, texel.y), 0.0).r;
    let d_d = textureSampleLevel(t_depth, s_point, uv + vec2<f32>(0.0, -texel.y), 0.0).r;
    let p = view_pos_from_depth(uv, depth);
    let p_r = view_pos_from_depth(uv + vec2<f32>(texel.x, 0.0), d_r);
    let p_u = view_pos_from_depth(uv + vec2<f32>(0.0, texel.y), d_u);
    return normalize(cross(p_r - p, p_u - p));
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSampleLevel(t_depth, s_point, i.uv, 0.0).r;
    if depth >= 1.0 || depth <= 0.0 {
        return vec4<f32>(1.0);
    }

    let P = view_pos_from_depth(i.uv, depth);
    let N = view_normal(i.uv, depth);

    // Sample hemisphere kernel (6 samples + random rotation)
    let noise_uv = i.uv * vec2<f32>(textureDimensions(t_noise)) * (1.0 / 4.0);
    let noise = textureSampleLevel(t_noise, s_point, noise_uv, 0.0).rgb * 2.0 - 1.0;

    let hemisphere = array<vec3<f32>, 8>(
        vec3<f32>( 0.538,  0.545, 0.642),
        vec3<f32>(-0.174,  0.841, 0.511),
        vec3<f32>( 0.707, -0.129, 0.695),
        vec3<f32>(-0.493, -0.493, 0.716),
        vec3<f32>( 0.106,  0.371, 0.922),
        vec3<f32>(-0.783,  0.278, 0.556),
        vec3<f32>( 0.331, -0.658, 0.676),
        vec3<f32>(-0.356, -0.729, 0.585),
    );

    let T = normalize(noise - N * dot(noise, N));
    let B = cross(N, T);

    var occlusion = 0.0;
    let sample_count = 8u;
    for (var j = 0u; j < sample_count; j = j + 1u) {
        let hs = hemisphere[j];
        var sample_dir = hs.x * T + hs.y * B + hs.z * N;
        sample_dir = normalize(sample_dir);

        var sample_pos = P + sample_dir * params.radius;

        // Project back to screen
        var sample_view = vec4<f32>(sample_pos, 1.0);
        var sample_clip = params.proj * sample_view;
        var sample_ndc = sample_clip.xyz / sample_clip.w;
        var sample_uv = sample_ndc.xy * 0.5 + 0.5;

        let sample_depth = textureSampleLevel(t_depth, s_point, sample_uv, 0.0).r;
        let sample_view_pos = view_pos_from_depth(sample_uv, sample_depth);

        let range_check = smoothstep(0.0, 1.0, params.radius / abs(P.z - sample_view_pos.z));
        let self_check = step(sample_view_pos.z, sample_pos.z + params.bias);
        occlusion += (1.0 - self_check) * range_check;
    }

    occlusion = 1.0 - (occlusion / f32(sample_count));
    occlusion = pow(occlusion, params.power);
    return vec4<f32>(occlusion, occlusion, occlusion, 1.0);
}
