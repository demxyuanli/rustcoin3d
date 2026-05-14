## Why

The meshlet draw path was enabled for large models (500K+ triangles) but introduces GPU compute-based culling (frustum + HZB occlusion) and indirect drawing that corrupts rendering on integrated GPUs. The basic meshlet vertex/index buffer path works correctly (verified with both Flat and PBR pipelines drawing all triangles without culling). We need to separate the correct basic meshlet rendering from the problematic GPU cull+compact+indirect path so each can be used independently based on GPU capability.

## What Changes

- **Decouple meshlet vertex/index buffers from cull/compact/indirect draw**: Meshlet vertex and full index buffers are uploaded once and can be used for direct `draw_indexed` without any culling. This is the "basic" path that works everywhere.
- **Make GPU cull+compact+indirect optional**: The compute-based culling pipeline becomes a separate pass that produces compacted index buffers and indirect draw args, only activated on capable GPUs.
- **Remove InstanceData slot 0 hack**: Instead of pre-inserting meshlet InstanceData at slot 0 of the instance SSBO, meshlet draws use their own uniform buffer or a dedicated instance slot managed outside the standard draw batch.
- **Restore `draw_meshlets` fallback behavior**: When GPU culling is unavailable or disabled, meshlet-indexed geometry renders via the standard instanced draw path (as it did before the optimization changes).

## Capabilities

### New Capabilities
- `meshlet-basic-draw`: Render meshlet geometry using full vertex/index buffers with direct `draw_indexed` calls, bypassing all GPU cull/compute passes. Works on all GPUs including integrated.
- `meshlet-gpu-cull`: Optional compute-based frustum + HZB culling pass that produces compacted index buffers and indirect draw args for `draw_indexed_indirect`. Enabled only on GPUs that support the required features and pass capability checks.

### Modified Capabilities
- None (new capabilities only; existing behavior preserved).

## Impact

- `crates/rc3d-render/src/cluster.rs` — add `draw_clustered_basic()` method for direct draw without cull
- `crates/rc3d-render/src/render_passes/draw_opaque.rs` — route meshlet draws through basic path when GPU cull is unavailable
- `crates/rc3d-render/src/render_passes.rs` — gate cull compute passes on GPU capability
- `crates/rc3d-render/src/renderer_internals.rs` — add `meshlet_gpu_cull_enabled` field to `GpuCapability`
- No shader changes required
