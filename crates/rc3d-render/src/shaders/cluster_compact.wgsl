struct GpuMeshlet {
    index_offset: u32,
    index_count: u32,
    vertex_offset: u32,
    vertex_count: u32,
};

// Binding 0: visible list [0]=counter, [1..]=meshlet indices
@group(0) @binding(0) var<storage, read> visible: array<u32>;
// Binding 1: meshlets
@group(0) @binding(1) var<storage, read> meshlets: array<GpuMeshlet>;
// Binding 2: source indices
@group(0) @binding(2) var<storage, read> src_indices: array<u32>;
// Binding 3: destination compacted indices
@group(0) @binding(3) var<storage, read_write> dst_indices: array<u32>;
// Binding 4: indirect args
@group(0) @binding(4) var<storage, read_write> indirect_args: array<atomic<u32>>;

@compute @workgroup_size(64)
fn compact_meshlets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let visible_count = visible[0];
    let meshlet_idx = gid.x;

    if meshlet_idx >= visible_count {
        return;
    }

    let mi = visible[1u + meshlet_idx];
    let m = meshlets[mi];

    // Compute destination offset for this meshlet's indices
    // Each thread handles one meshlet, need to compute prefix sum of preceding meshlets
    var offset = 0u;
    for (var i = 0u; i < meshlet_idx; i = i + 1u) {
        let prev_mi = visible[1u + i];
        offset = offset + meshlets[prev_mi].index_count;
    }

    // Copy indices for this meshlet
    for (var j = 0u; j < m.index_count; j = j + 1u) {
        if (offset + j) < arrayLength(&dst_indices) {
            dst_indices[offset + j] = src_indices[m.index_offset + j];
        }
    }

    // Last thread writes indirect args
    if meshlet_idx == visible_count - 1u {
        let total_indices = offset + m.index_count;
        atomicStore(&indirect_args[0], total_indices); // index_count
        atomicStore(&indirect_args[1], 1u);            // instance_count
        atomicStore(&indirect_args[2], 0u);            // first_index
        atomicStore(&indirect_args[3], 0u);            // base_vertex
        atomicStore(&indirect_args[4], 0u);            // first_instance
    }
}
