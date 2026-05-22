struct FlatUniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
    // Model matrix for world-space operations (section cap clip planes)
    model: mat4x4<f32>,
    // Clip planes for section cap (if clip_count > 0)
    clip_planes: array<vec4<f32>, 6>,
    clip_count: vec4<f32>,
};

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

struct LineVertexInput {
    @location(0) position: vec3<f32>,
}

struct MarkupVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct MarkupVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// ── Main vertex: outputs world_pos for clip plane testing ──

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let world_pos = u.model * vec4<f32>(in.position, 1.0);
    var out: VertexOutput;
    out.pos = u.mvp * vec4<f32>(in.position, 1.0);
    out.world_pos = world_pos.xyz;
    return out;
}

// ── Main fragment: uses world_pos from VertexOutput ──

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Test clip planes (for section cap)
    let clip_count = i32(u.clip_count.x);
    for (var i = 0; i < clip_count; i = i + 1) {
        let plane = u.clip_planes[i];
        let dist = dot(plane.xyz, in.world_pos) + plane.w;
        if dist < 0.0 {
            discard;
        }
    }
    return u.color;
}

// ── Line vertex: no world_pos needed ──

@vertex
fn vs_line(in: LineVertexInput) -> @builtin(position) vec4<f32> {
    return u.mvp * vec4<f32>(in.position, 1.0);
}

// ── Line fragment: no world_pos needed ──

@fragment
fn fs_line_main() -> @location(0) vec4<f32> {
    return u.color;
}

// ── Colored markup lines: vertex color → fragment ──

@vertex
fn vs_markup(in: MarkupVertexInput) -> MarkupVertexOutput {
    var out: MarkupVertexOutput;
    out.position = u.mvp * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_markup(in: MarkupVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}