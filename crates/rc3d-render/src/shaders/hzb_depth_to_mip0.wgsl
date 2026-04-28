@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var out_mip0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_mip0);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = textureLoad(depth_tex, coord, 0);
    textureStore(out_mip0, coord, vec4<f32>(d, 0.0, 0.0, 0.0));
}
