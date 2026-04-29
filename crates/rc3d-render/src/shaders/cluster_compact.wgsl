struct GpuMeshlet {
    index_offset: u32,
    index_count: u32,
    vertex_offset: u32,
    vertex_count: u32,
};

// Binding 0: visible list [0]=visible_count, [1..]=meshlet indices
@group(0) @binding(0) var<storage, read> visible: array<u32>;
// Binding 1: meshlets
@group(0) @binding(1) var<storage, read> meshlets: array<GpuMeshlet>;
// Binding 2: source indices
@group(0) @binding(2) var<storage, read> src_indices: array<u32>;
// Binding 3: destination compacted indices
@group(0) @binding(3) var<storage, read_write> dst_indices: array<u32>;
// Binding 4: indirect args; [0]=atomic counter, [1]=index_count, [2]=instance_count, [3]=first_index, [4]=vertex_offset, [5]=first_instance
@group(0) @binding(4) var<storage, read_write> indirect_args: array<atomic<u32>>;

var<workgroup> wg_prefix: array<u32, 64>;
var<workgroup> wg_base: u32;

@compute @workgroup_size(64)
fn compact_meshlets(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    let visible_count = visible[0];
    let meshlet_idx = gid.x;

    if meshlet_idx >= visible_count {
        return;
    }

    let mi = visible[1u + meshlet_idx];
    let m = meshlets[mi];

    // Each thread contributes its meshlet's index_count
    let my_count = m.index_count;
    wg_prefix[local_idx] = my_count;

    // Hillis-Steele parallel prefix sum within workgroup (6 iterations for 64 threads)
    for (var step = 1u; step < 64u; step = step << 1u) {
        workgroupBarrier();
        let neighbor = select(0u, wg_prefix[local_idx - step], local_idx >= step);
        workgroupBarrier();
        if local_idx >= step {
            wg_prefix[local_idx] = wg_prefix[local_idx] + neighbor;
        }
    }

    // Exclusive prefix sum (offset before this meshlet's indices)
    let exclusive_offset = wg_prefix[local_idx] - my_count;

    // Last thread in workgroup claims global offset via atomic counter
    workgroupBarrier();
    if local_idx == 63u {
        wg_base = atomicAdd(&indirect_args[0], wg_prefix[63u]);
    }
    workgroupBarrier();

    let write_start = wg_base + exclusive_offset;

    // Copy indices for this meshlet
    for (var j = 0u; j < my_count; j = j + 1u) {
        if (write_start + j) < arrayLength(&dst_indices) {
            dst_indices[write_start + j] = src_indices[m.index_offset + j];
        }
    }
}
