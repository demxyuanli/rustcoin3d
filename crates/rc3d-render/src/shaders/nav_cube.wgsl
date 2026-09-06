struct NavUniforms {
    mvp: mat4x4<f32>,
    light_dir: vec4<f32>,
    hover: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: NavUniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) face_id: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) face_id: f32,
}

@vertex
fn vs_solid(input: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = u.mvp * vec4<f32>(input.position, 1.0);
    out.color = input.color;
    out.world_n = input.normal;
    out.face_id = input.face_id.x;
    return out;
}

@fragment
fn fs_solid(input: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(input.world_n);
    let l = normalize(u.light_dir.xyz);
    let wrap = 0.42 + 0.58 * max(dot(n, l), 0.0);
    var rgb = input.color.xyz * wrap;
    if (u.hover.x >= 0.0 && abs(input.face_id - u.hover.x) < 0.5) {
        rgb = rgb * 0.45 + vec3<f32>(0.38, 0.67, 0.94);
    }
    return vec4<f32>(rgb, 1.0);
}

struct LineIn {
    @location(0) position: vec3<f32>,
}

@vertex
fn vs_line(input: LineIn) -> @builtin(position) vec4<f32> {
    return u.mvp * vec4<f32>(input.position, 1.0);
}

@fragment
fn fs_line() -> @location(0) vec4<f32> {
    return vec4<f32>(0.12, 0.13, 0.16, 1.0);
}

struct BlitOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_blit(@builtin(vertex_index) vid: u32) -> BlitOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: BlitOut;
    out.clip = vec4<f32>(pos[vid], 0.0, 1.0);
    out.uv = uv[vid];
    return out;
}

@group(0) @binding(0) var t_cube: texture_2d<f32>;
@group(0) @binding(1) var s_cube: sampler;

@fragment
fn fs_blit(input: BlitOut) -> @location(0) vec4<f32> {
    let c = textureSample(t_cube, s_cube, input.uv);
    // Prefer alpha; also discard near-black so swapchain-format tiles that lose
    // alpha (opaque black clear) still composite as transparent. Keep threshold
    // below nav-cube label ink (~0.12) so face text is not punched out.
    let luma = max(c.r, max(c.g, c.b));
    if (c.a < 0.05 || luma < 0.02) {
        discard;
    }
    return c;
}
