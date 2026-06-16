// Downsample the full-resolution depth buffer into a small grid used for
// CPU-side annotation occlusion testing. Each output texel maps to the
// corresponding position across the WHOLE screen (nearest sample), unlike a
// plain texture-to-buffer copy which would only capture the top-left region.

@group(0) @binding(0) var t_depth: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(out_tex);
    if gid.x >= out_dims.x || gid.y >= out_dims.y {
        return;
    }
    let in_dims = textureDimensions(t_depth);
    let src = vec2<u32>(
        min(gid.x * in_dims.x / out_dims.x, in_dims.x - 1u),
        min(gid.y * in_dims.y / out_dims.y, in_dims.y - 1u),
    );
    let d = textureLoad(t_depth, vec2<i32>(src), 0).r;
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 0.0));
}
