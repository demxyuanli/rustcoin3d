// Auto exposure: compute per-block log-luminance averages for CPU-side reduction.
// Dispatched over (width/8, height/8) workgroups, each writing one f32 to output buffer.

@group(0) @binding(0) var t_hdr: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> lum_buf: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) num_groups: vec3<u32>) {
    let dims = textureDimensions(t_hdr);
    if gid.x * 8u >= dims.x || gid.y * 8u >= dims.y {
        return;
    }

    // Compute average log-luminance for this 8x8 block
    var sum_log_lum = 0.0;
    var count = 0u;
    for (var dy = 0u; dy < 8u; dy++) {
        let py = gid.y * 8u + dy;
        if py >= dims.y { break; }
        for (var dx = 0u; dx < 8u; dx++) {
            let px = gid.x * 8u + dx;
            if px >= dims.x { break; }
            let color = textureLoad(t_hdr, vec2<i32>(i32(px), i32(py)), 0).rgb;
            let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
            sum_log_lum += log2(max(lum, 1e-5));
            count += 1u;
        }
    }

    let avg_log_lum = sum_log_lum / f32(count);
    let idx = gid.y * num_groups.x + gid.x;
    lum_buf[idx] = avg_log_lum;
}
