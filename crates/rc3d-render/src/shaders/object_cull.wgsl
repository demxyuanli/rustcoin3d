// GPU compute: frustum-cull per-object transforms, write visible instance indices.
//
// Dispatch: workgroup_count = ceil(object_count / 256), 1, 1
// Push constants: object_count (u32), frustum_plane_count (u32), _pad (u32, u32)

struct GpuObjectTransform {
    model_matrix: array<array<f32, 4>, 4>,
    aabb_min: vec3<f32>,
    flags: u32,
    aabb_max: vec3<f32>,
    mesh_id: u32,
    material_id: u32,
    _pad: array<u32, 7>,
};

struct FrustumUniforms {
    planes: array<vec4<f32>, 6>,
};

struct PushConstants {
    object_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> transforms: array<GpuObjectTransform>;
@group(0) @binding(1) var<storage, read_write> indirect_args: array<DrawIndirectArgs>;
@group(0) @binding(2) var<uniform> frustum: FrustumUniforms;
@group(0) @binding(3) var<storage, read_write> instance_indices: array<u32>;

var<push_constant> pc: PushConstants;

// Test one AABB corner against a frustum plane.
// Returns true if the corner is on the positive side of the plane.
fn test_plane(p: vec4<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    let corner = vec3<f32>(
        select(aabb_max.x, aabb_min.x, p.x < 0.0),
        select(aabb_max.y, aabb_min.y, p.y < 0.0),
        select(aabb_max.z, aabb_min.z, p.z < 0.0),
    );
    return dot(p.xyz, corner) + p.w >= 0.0;
}

fn aabb_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i++) {
        if !test_plane(frustum.planes[i], aabb_min, aabb_max) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.object_count { return; }

    let obj = transforms[idx];
    if aabb_visible(obj.aabb_min, obj.aabb_max) {
        // Atomically append instance index to the draw's instance list
        let slot = atomicAdd(&indirect_args[obj.mesh_id].instance_count, 1u);
        instance_indices[slot] = idx;
    }
}
