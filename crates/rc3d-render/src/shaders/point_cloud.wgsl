// Point sprites: camera-facing quads with size attenuation (Three.js Points analogue).

struct PointCloudParams {
    mvp: mat4x4<f32>,
    view: mat4x4<f32>,
    point_size: f32,
    viewport_x: f32,
    viewport_y: f32,
    _pad: f32,
}

struct Point {
    position: vec4<f32>, // xyz = world, w = per-particle size scale
    color: vec4<f32>,
    velocity: vec4<f32>,
    extra: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> t_points: array<Point>;
@group(0) @binding(1) var<uniform> params: PointCloudParams;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let pid = vi / 6u;
    let corner = vi % 6u;
    var uv: vec2<f32>;
    switch corner {
        case 0u: { uv = vec2<f32>(-1.0, -1.0); }
        case 1u: { uv = vec2<f32>(1.0, -1.0); }
        case 2u: { uv = vec2<f32>(1.0, 1.0); }
        case 3u: { uv = vec2<f32>(-1.0, -1.0); }
        case 4u: { uv = vec2<f32>(1.0, 1.0); }
        default: { uv = vec2<f32>(-1.0, 1.0); }
    }
    let pt = t_points[pid];
    var clip = params.mvp * vec4<f32>(pt.position.xyz, 1.0);
    let size_px = max(params.point_size * max(pt.position.w, 0.1), 1.0);
    let w = max(clip.w, 1e-5);
    clip.x += uv.x * size_px * w / max(params.viewport_x, 1.0);
    clip.y += uv.y * size_px * w / max(params.viewport_y, 1.0);
    var o: VsOut;
    o.clip_pos = clip;
    o.color = pt.color;
    o.uv = uv;
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let d = length(i.uv);
    if d > 1.0 { discard; }
    let alpha = 1.0 - smoothstep(0.75, 1.0, d);
    return vec4<f32>(i.color.rgb, i.color.a * alpha);
}
