// Point cloud rendering with point sprites (HOOPS OOC PointCloud equivalent).
// Renders points as camera-facing quads with size attenuation.

struct PointCloudParams {
    mvp: mat4x4<f32>,
    view: mat4x4<f32>,
    point_size: f32,
    max_points: u32,
    _pad: vec2<f32>,
}

struct Point {
    position: vec4<f32>, // xyz = position, w = unused
    color: u32,           // packed RGBA8
}

@group(0) @binding(0) var<storage> t_points: array<Point>;
@group(0) @binding(1) var<uniform> params: PointCloudParams;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) point_size: f32,
}

fn unpack_color(c: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((c >> 24u) & 0xFFu) / 255.0,
        f32((c >> 16u) & 0xFFu) / 255.0,
        f32((c >> 8u) & 0xFFu) / 255.0,
        f32(c & 0xFFu) / 255.0,
    );
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let pt = t_points[vi];
    let pos = params.mvp * pt.position;
    var o: VsOut;
    o.clip_pos = pos;
    o.color = unpack_color(pt.color);
    o.point_size = params.point_size;
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    // Simple circular point with soft edge
    let d = length(i.color.xy * 2.0 - 1.0);
    if d > 1.0 { discard; }
    let alpha = 1.0 - smoothstep(0.8, 1.0, d);
    return vec4<f32>(i.color.rgb, i.color.a * alpha);
}
