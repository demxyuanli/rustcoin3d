// Section cap fill shader: renders back faces of mesh with clip planes
// to produce filled cross-section, plus optional ANSI/ISO hatch.

struct FlatUniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
    model: mat4x4<f32>,
    clip_planes: array<vec4<f32>, 6>,
    clip_count: vec4<f32>,
    hatch_color: vec4<f32>,
    hatch_params: vec4<f32>,
    hatch_extra: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: FlatUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texcoord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let world_pos = u.model * vec4<f32>(in.position, 1.0);
    var out: VertexOutput;
    out.pos = u.mvp * vec4<f32>(in.position, 1.0);
    out.world_pos = world_pos.xyz;
    return out;
}

fn plane_tangent(n: vec3<f32>) -> vec3<f32> {
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    return normalize(cross(n, helper));
}

fn hatch_stripe(world_pos: vec3<f32>, dir: vec3<f32>, spacing: f32, width: f32) -> f32 {
    let s = dot(world_pos, dir) / max(spacing, 1e-5);
    let t = fract(s);
    let fw = max(fwidth(s), 1e-4);
    let d = min(t, 1.0 - t);
    return 1.0 - smoothstep(width, width + fw, d);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let clip_count = i32(u.clip_count.x);
    for (var i = 0; i < clip_count; i = i + 1) {
        let plane = u.clip_planes[i];
        let dist = dot(plane.xyz, in.world_pos) + plane.w;
        if dist < 0.0 {
            discard;
        }
    }

    var color = u.color;
    if u.hatch_params.w > 0.5 {
        let plane = u.clip_planes[0];
        let n = normalize(plane.xyz);
        let planar = in.world_pos - n * (dot(n, in.world_pos) + plane.w);
        let u_axis = plane_tangent(n);
        let v_axis = normalize(cross(n, u_axis));
        let ang = u.hatch_params.y;
        let ca = cos(ang);
        let sa = sin(ang);
        let dir0 = normalize(u_axis * ca + v_axis * sa);
        let spacing = u.hatch_params.x;
        let width = u.hatch_params.z * 0.5;
        var h = hatch_stripe(planar, dir0, spacing, width);
        if u.hatch_extra.x > 0.5 {
            let dir1 = normalize(u_axis * (-sa) + v_axis * ca);
            h = max(h, hatch_stripe(planar, dir1, spacing, width));
        }
        color = mix(color, u.hatch_color, h);
    }
    return color;
}
