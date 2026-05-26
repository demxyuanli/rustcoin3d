// Anti-aliased line rendering via quad expansion + SDF.
//
// Each line segment is pre-expanded into 6 vertices (2 triangles) on the CPU.
// The vertex shader expands the quad to face the camera in screen space,
// and the fragment shader computes a smooth alpha gradient via SDF.

struct LineUniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
    line_params: vec4<f32>,  // (half_width_px, aa_px, viewport_w, viewport_h)
};

@group(0) @binding(0) var<uniform> u: LineUniforms;

struct LineVertexExpandedInput {
    @location(0) position: vec3<f32>,
    @location(1) partner: vec3<f32>,
    @location(2) side: f32,
}

struct LineExpandedOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) side: f32,
}

@vertex
fn vs_line_expanded(in: LineVertexExpandedInput) -> LineExpandedOutput {
    let m = u.mvp;

    // Project both endpoints to clip space
    let p0 = m * vec4<f32>(in.position, 1.0);
    let p1 = m * vec4<f32>(in.partner, 1.0);

    let p0_ndc = p0.xy / p0.w;
    let p1_ndc = p1.xy / p1.w;
    let p0_z = p0.z / p0.w;
    let p1_z = p1.z / p1.w;

    // Line direction in NDC
    let delta = p1_ndc - p0_ndc;
    let len_sq = max(dot(delta, delta), 1e-12);
    let dir = delta / sqrt(len_sq);
    let normal = vec2<f32>(-dir.y, dir.x);

    // Pixel-to-NDC scale
    let ndc_per_px_y = 2.0 / u.line_params.z;
    let ndc_per_px_x = 2.0 / u.line_params.w;
    let ndc_per_px = (ndc_per_px_x + ndc_per_px_y) * 0.5;

    let half_width = u.line_params.x;
    let aa_radius = u.line_params.y;
    let total = (half_width + aa_radius) * ndc_per_px;

    let offset_ndc = normal * in.side * total;
    let ndc = p0_ndc + offset_ndc;
    let w = p0.w;
    var out: LineExpandedOutput;
    out.position = vec4<f32>(ndc * w, p0_z * w, w);
    out.side = in.side;
    return out;
}

@fragment
fn fs_line_aa(in: LineExpandedOutput) -> @location(0) vec4<f32> {
    let half_width = u.line_params.x;
    let aa_radius = u.line_params.y;
    let total = half_width + aa_radius;

    let dist = abs(in.side) * total;
    let alpha = 1.0 - smoothstep(half_width, total, dist);
    return vec4<f32>(u.color.rgb, u.color.a * alpha);
}
