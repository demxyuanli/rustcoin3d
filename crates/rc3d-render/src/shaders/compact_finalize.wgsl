// Single-thread dispatch that reads the atomic counter and writes the final indirect draw args.
// indirect_args layout: [0]=total_index_count (atomic counter result), [1]=index_count, [2]=instance_count, [3]=first_index, [4]=base_vertex, [5]=first_instance

@group(0) @binding(0) var<storage, read_write> indirect_args: array<atomic<u32>>;

@compute @workgroup_size(1)
fn finalize_args() {
    // Read the final total_index_count from the atomic counter (all workgroups have completed)
    let total_indices = atomicLoad(&indirect_args[0]);
    // Write draw indirect args starting at index 1
    atomicStore(&indirect_args[1], total_indices); // index_count
    atomicStore(&indirect_args[2], 1u);            // instance_count
    atomicStore(&indirect_args[3], 0u);            // first_index
    atomicStore(&indirect_args[4], 0u);            // base_vertex
    // first_instance is already 0 (written during clear)
}
