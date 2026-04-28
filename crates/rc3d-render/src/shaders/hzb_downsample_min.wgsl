@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(dst_tex);
    if gid.x >= dst_dims.x || gid.y >= dst_dims.y {
        return;
    }
    let sx = i32(gid.x) * 2;
    let sy = i32(gid.y) * 2;
    let src_dims = textureDimensions(src_tex);
    let v00 = textureLoad(src_tex, vec2<i32>(sx, sy), 0).x;
    let v10 = textureLoad(src_tex, vec2<i32>(min(sx + 1, i32(src_dims.x) - 1), sy), 0).x;
    let v01 = textureLoad(src_tex, vec2<i32>(sx, min(sy + 1, i32(src_dims.y) - 1)), 0).x;
    let v11 = textureLoad(
        src_tex,
        vec2<i32>(min(sx + 1, i32(src_dims.x) - 1), min(sy + 1, i32(src_dims.y) - 1)),
        0,
    ).x;
    let mn = min(min(v00, v10), min(v01, v11));
    textureStore(dst_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(mn, 0.0, 0.0, 0.0));
}
