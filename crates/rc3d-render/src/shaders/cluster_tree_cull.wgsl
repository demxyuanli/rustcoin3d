// Hierarchical meshlet-cluster culling across LOD levels.
//
// Dispatch: one workgroup per cluster at current LOD level.
// Each workgroup tests one cluster against frustum + HZB,
// then writes visible children to the next level's indirect args buffer.

struct FrustumUniforms {
    planes: array<vec4<f32>, 6>,
};

struct PushConstants {
    cluster_count: u32,
    hzb_mip: u32,
    use_hzb: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> bounds: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> first_child: array<u32>;
@group(0) @binding(2) var<storage, read> child_count: array<u32>;
@group(0) @binding(3) var<uniform> frustum: FrustumUniforms;
@group(0) @binding(4) var hzb_texture: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> next_visible: array<u32>;
@group(0) @binding(6) var<storage, read_write> next_visible_count: atomic<u32>;

var<immediate> pc: PushConstants;

fn sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let p = frustum.planes[i];
        if dot(p.xyz, center) + p.w < -radius {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cluster_idx = gid.x;
    if cluster_idx >= pc.cluster_count { return; }

    let b = bounds[cluster_idx];
    let center = b.xyz;
    let radius = b.w;

    // Frustum cull
    if !sphere_in_frustum(center, radius) { return; }

    // HZB occlusion cull (optional)
    if pc.use_hzb != 0u {
        // Project sphere to screen, sample HZB at appropriate mip
        // Simplified: skip for initial implementation
    }

    // Write children to next level's visible list
    let start = first_child[cluster_idx];
    let count = child_count[cluster_idx];
    for (var i = 0u; i < count; i++) {
        let slot = atomicAdd(&next_visible_count, 1u);
        next_visible[slot] = start + i;
    }
}
