// 2x2 box-filter downsampling: reads from src_mip-1 and writes to dst_mip.
// Each thread handles one output texel.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_tex);
    if gid.x >= dims.x / 2u || gid.y >= dims.y / 2u {
        return;
    }
    // 2x2 box filter
    let c00 = textureLoad(src_tex, vec2<i32>(i32(gid.x * 2u),     i32(gid.y * 2u)),      0);
    let c10 = textureLoad(src_tex, vec2<i32>(i32(gid.x * 2u + 1u), i32(gid.y * 2u)),      0);
    let c01 = textureLoad(src_tex, vec2<i32>(i32(gid.x * 2u),     i32(gid.y * 2u + 1u)), 0);
    let c11 = textureLoad(src_tex, vec2<i32>(i32(gid.x * 2u + 1u), i32(gid.y * 2u + 1u)), 0);
    let avg = (c00 + c10 + c01 + c11) * 0.25;
    textureStore(dst_tex, vec2<u32>(gid.x, gid.y), avg);
}
