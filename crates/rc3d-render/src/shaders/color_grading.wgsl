// 3D LUT-based color grading.
// Samples a 33x33x33 RGBA16Float 3D texture using the input color as UVW coordinates.

struct LutParams {
    lut_size: f32,   // e.g. 33.0
    intensity: f32,  // blend between original and graded (0.0-1.0)
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var t_lut: texture_3d<f32>;
@group(0) @binding(2) var s_lut: sampler;
@group(0) @binding(3) var<uniform> params: LutParams;
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba16float, write>;

// Sample 3D LUT with trilinear interpolation from a flat 3D texture.
// `c` is the input color in [0,1]
fn sample_lut(c: vec3<f32>) -> vec3<f32> {
    let size = params.lut_size;
    let max_coord = (size - 1.0) / size;
    let half_px = 0.5 / size;
    let uvw = c * max_coord + half_px;
    return textureSampleLevel(t_lut, s_lut, uvw, 0.0).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_input);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let input_color = textureSampleLevel(t_input, s_lut, uv, 0.0).rgb;

    let graded = sample_lut(input_color);
    let result = mix(input_color, graded, params.intensity);

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(result, 1.0));
}
