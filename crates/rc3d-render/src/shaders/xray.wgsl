// X-ray effect: depth-edge detection with glow overlay.
// Extracts edges from depth buffer and blends cyan wireframe over scene.

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_2d<f32>;
@group(0) @binding(2) var s_point: sampler;
@group(0) @binding(3) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_color);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let px = 1.0 / vec2<f32>(dims);

    // Sobel depth edge detection
    let d00 = textureSampleLevel(t_depth, s_point, uv + vec2<f32>(-px.x, -px.y), 0.0).r;
    let d10 = textureSampleLevel(t_depth, s_point, uv + vec2<f32>( 0.0, -px.y), 0.0).r;
    let d20 = textureSampleLevel(t_depth, s_point, uv + vec2<f32>( px.x, -px.y), 0.0).r;
    let d01 = textureSampleLevel(t_depth, s_point, uv + vec2<f32>(-px.x,  0.0), 0.0).r;
    let d21 = textureSampleLevel(t_depth, s_point, uv + vec2<f32>( px.x,  0.0), 0.0).r;
    let d02 = textureSampleLevel(t_depth, s_point, uv + vec2<f32>(-px.x,  px.y), 0.0).r;
    let d12 = textureSampleLevel(t_depth, s_point, uv + vec2<f32>( 0.0,  px.y), 0.0).r;
    let d22 = textureSampleLevel(t_depth, s_point, uv + vec2<f32>( px.x,  px.y), 0.0).r;

    let gx = -d00 + d20 - 2.0*d01 + 2.0*d21 - d02 + d22;
    let gy = -d00 - 2.0*d10 - d20 + d02 + 2.0*d12 + d22;
    let edge = length(vec2<f32>(gx, gy));

    let color = textureSampleLevel(t_color, s_point, uv, 0.0);
    // Blend: cyan glow on detected edges
    let xray_color = mix(color.rgb, vec3<f32>(0.2, 0.8, 1.0), smoothstep(0.01, 0.05, edge));
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(xray_color, 1.0));
}
