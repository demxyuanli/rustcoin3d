// GPU skinning: applies bone transforms to skinned vertices.
// Reads bind-pose vertices + skin data, writes animated vertices.

struct SkinningUniforms {
    joint_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> u: SkinningUniforms;
@group(0) @binding(1) var<storage, read> src_vertices: array<f32>;    // interleaved pos(3)+norm(3)+uv(2)+tan(4)
@group(0) @binding(2) var<storage, read> skin_data: array<u32>;       // bone_indices[4] + bone_weights[4] per vertex
@group(0) @binding(3) var<storage, read> bone_mats: array<mat4x4<f32>>; // skinning matrices
@group(0) @binding(4) var<storage, read_write> dst_vertices: array<f32>;

// Layout: pos(3), norm(3), uv(2), tangent(4) = 12 floats per vertex
const VERTEX_STRIDE: u32 = 12u;

struct SkinVertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    tangent: vec4<f32>,
}

fn load_vertex(vertex_stride: u32, idx: u32) -> SkinVertex {
    let base = idx * vertex_stride;
    return SkinVertex(
        vec3<f32>(src_vertices[base], src_vertices[base + 1u], src_vertices[base + 2u]),
        vec3<f32>(src_vertices[base + 3u], src_vertices[base + 4u], src_vertices[base + 5u]),
        vec2<f32>(src_vertices[base + 6u], src_vertices[base + 7u]),
        vec4<f32>(src_vertices[base + 8u], src_vertices[base + 9u], src_vertices[base + 10u], src_vertices[base + 11u]),
    );
}

fn store_vertex(vertex_stride: u32, idx: u32, v: SkinVertex) {
    let base = idx * vertex_stride;
    dst_vertices[base] = v.position.x;
    dst_vertices[base + 1u] = v.position.y;
    dst_vertices[base + 2u] = v.position.z;
    dst_vertices[base + 3u] = v.normal.x;
    dst_vertices[base + 4u] = v.normal.y;
    dst_vertices[base + 5u] = v.normal.z;
    dst_vertices[base + 6u] = v.uv.x;
    dst_vertices[base + 7u] = v.uv.y;
    dst_vertices[base + 8u] = v.tangent.x;
    dst_vertices[base + 9u] = v.tangent.y;
    dst_vertices[base + 10u] = v.tangent.z;
    dst_vertices[base + 11u] = v.tangent.w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vertex_idx = gid.x;
    let vertex_count = arrayLength(&skin_data) / 8u; // 4 indices + 4 weights per vertex

    if vertex_idx >= vertex_count {
        return;
    }

    let v = load_vertex(VERTEX_STRIDE, vertex_idx);

    // Read skin data
    let skin_base = vertex_idx * 8u;
    let b0 = skin_data[skin_base];
    let b1 = skin_data[skin_base + 1u];
    let b2 = skin_data[skin_base + 2u];
    let b3 = skin_data[skin_base + 3u];
    let w0 = f32(skin_data[skin_base + 4u]) / 65535.0;
    let w1 = f32(skin_data[skin_base + 5u]) / 65535.0;
    let w2 = f32(skin_data[skin_base + 6u]) / 65535.0;
    let w3 = f32(skin_data[skin_base + 7u]) / 65535.0;

    // Apply skinning
    var animated_pos = vec4<f32>(0.0);
    var animated_norm = vec3<f32>(0.0);
    var sum_w: f32 = w0 + w1 + w2 + w3;

    if sum_w < 0.001 {
        // No skinning — use identity
        store_vertex(VERTEX_STRIDE, vertex_idx, v);
        return;
    }

    if b0 < u.joint_count {
        let m = bone_mats[b0];
        animated_pos += m * vec4<f32>(v.position, 1.0) * w0;
        animated_norm += (mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz) * v.normal) * w0;
    }
    if b1 < u.joint_count {
        let m = bone_mats[b1];
        animated_pos += m * vec4<f32>(v.position, 1.0) * w1;
        animated_norm += (mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz) * v.normal) * w1;
    }
    if b2 < u.joint_count {
        let m = bone_mats[b2];
        animated_pos += m * vec4<f32>(v.position, 1.0) * w2;
        animated_norm += (mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz) * v.normal) * w2;
    }
    if b3 < u.joint_count {
        let m = bone_mats[b3];
        animated_pos += m * vec4<f32>(v.position, 1.0) * w3;
        animated_norm += (mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz) * v.normal) * w3;
    }

    store_vertex(VERTEX_STRIDE, vertex_idx, SkinVertex(
        animated_pos.xyz / sum_w,
        normalize(animated_norm / sum_w),
        v.uv,
        v.tangent,
    ));
}
