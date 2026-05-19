@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var s_linear: sampler;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let color = textureSampleLevel(src, s_linear, uv, 0.0);
    textureStore(dst, vec2<i32>(gid.xy), color);
}
