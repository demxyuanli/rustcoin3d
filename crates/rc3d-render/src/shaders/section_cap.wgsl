// Section cap fill — draws a filled cross-section at the clip plane.
// Uses a full-screen triangle + stencil test (compare=Equal, ref=1)
// to restrict fill to the interior of the clipped object silhouette.
//
// Fragment shader intersects the view ray with the clip plane to
// determine the correct depth for each pixel on the cap surface.

struct SectionCapUniforms {
    clip_to_world: mat4x4<f32>,
    color: vec4<f32>,
    plane: vec4<f32>,
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: SectionCapUniforms;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let uv = positions[vid] * 0.5 + 0.5;
    var o: VertexOutput;
    o.clip_pos = vec4<f32>(positions[vid], 0.5, 1.0);
    o.uv = uv;
    return o;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    // NDC x, y from screen UV
    let ndc_x = in.uv.x * 2.0 - 1.0;
    let ndc_y = in.uv.y * 2.0 - 1.0;

    // Solve for NDC z where the unprojected point lies on the clip plane.
    // world_h = clip_to_world * vec4(ndc_x, ndc_y, z, 1.0)
    // constraint: dot(plane.xyz, world_h.xyz/world_h.w) + plane.w = 0
    let ct = u.clip_to_world;
    let px = u.plane.x; let py = u.plane.y; let pz = u.plane.z; let pw = u.plane.w;

    let a13 = ct[0][2]; let a23 = ct[1][2]; let a33 = ct[2][2]; let a43 = ct[3][2];
    let a11 = ct[0][0]; let a12 = ct[0][1]; let a14 = ct[0][3];
    let a21 = ct[1][0]; let a22 = ct[1][1]; let a24 = ct[1][3];
    let a31 = ct[2][0]; let a32 = ct[2][1]; let a34 = ct[2][3];
    let a41 = ct[3][0]; let a42 = ct[3][1]; let a44 = ct[3][3];

    let num = -(px * (a11 * ndc_x + a12 * ndc_y + a14)
              + py * (a21 * ndc_x + a22 * ndc_y + a24)
              + pz * (a31 * ndc_x + a32 * ndc_y + a34)
              + pw * (a41 * ndc_x + a42 * ndc_y + a44));

    let denom = px * a13 + py * a23 + pz * a33 + pw * a43;

    // Ray parallel to plane → no intersection
    if abs(denom) < 1e-9 {
        discard;
    }

    let ndc_z = num / denom;

    // Reconstruct world position at intersection
    var world_h = ct * vec4<f32>(ndc_x, ndc_y, ndc_z, 1.0);
    if abs(world_h.w) < 1e-9 {
        discard;
    }
    let world_pos = world_h.xyz / world_h.w;

    // Verify we're near the clip plane (within tolerance band).
    // This is a safety check — the stencil should already restrict
    // rendering to the cross-section interior.
    let d = dot(u.plane.xyz, world_pos) + u.plane.w;
    let band = u.params.x;
    if abs(d) > band {
        discard;
    }

    // Compute NDC depth at the intersection point.
    // For wgpu, NDC z is in [0, 1]; clip z / w produces this range.
    // We compute clip-space position from ndc: clip = ndc * w (where w is from world_h)
    let clip_z = ndc_z * world_h.w;

    var out: FragmentOutput;
    out.color = u.color;
    out.depth = ndc_z * 0.5 + 0.5; // Convert [-1,1] → [0,1] wgpu NDC
    return out;
}
