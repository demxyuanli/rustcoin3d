## Context

Currently the meshlet draw path in `draw_opaque.rs` is gated by `can_meshlet_draw = draw_meshlets && hzb.is_some()`. When enabled, all meshlet-indexed draws go through the full GPU pipeline: cull compute pass → compact compute pass → finalize compute pass → `draw_indexed_indirect`. This pipeline produces rendering artifacts (broken faces, flickering) on integrated GPUs because:

1. The cull/compact/finalize passes write to `indirect_buffer` (STORAGE) and `compact_index_buffer` (STORAGE), which the render pass reads as INDIRECT and INDEX. The implicit compute→render barrier in wgpu may not properly flush caches on Intel Vulkan drivers.
2. The InstanceData for meshlet draws uses `first_instance=0` in the indirect buffer, requiring the meshlet instance to be at slot 0 of the instance SSBO, which conflicts with the standard draw batch write.

Diagnostic testing confirmed that meshlet vertex/index buffers work correctly with PBR shader when drawing ALL triangles via direct `draw_indexed` (bypassing cull+compact+indirect). The basic drawing path is correct; the GPU cull pipeline is the problem.

## Goals / Non-Goals

**Goals:**
- Provide a basic meshlet draw path that uses direct `draw_indexed` with full vertex/index buffers (no GPU culling), guaranteed to work on all hardware
- Make the GPU cull pipeline (cull+compact+indirect) optional, enabled only when GPU capability checks pass
- Eliminate the InstanceData slot 0 contention by managing meshlet instance data through the standard draw batch

**Non-Goals:**
- Shader changes (PBR shader stays as-is)
- Multi-draw-indirect changes (already verified correct via per-draw fallback test)
- HZB or other occlusion improvements

## Decisions

**Decision 1: Two-tier meshlet draw path**

Add `meshlet_gpu_cull_enabled: bool` to `GpuCapability`. When false, meshlet draws use `draw_clustered_full_diag` (full index buffer, direct draw). When true, the existing cull+compact+indirect path runs.

Rationale: The basic path works everywhere (verified with both Flat and PBR pipelines). The GPU cull path can be enabled later when the synchronization issues are resolved.

**Decision 2: InstanceData via pre-insert at all_instances[0]**

Keep the current approach where meshlet InstanceData is pre-inserted as the first element of `all_instances` before Phase 1. The Phase 2 batch write places it at slot 0 of the instance SSBO. Meshlet draws use `first_instance=0` (matching the indirect buffer).

Rationale: This avoids the staging buffer + encoder copy approach that adds complexity. The queue ordering works because Phase 2's `queue.write_buffer` is submitted to the queue before the encoder submit, so the meshlet InstanceData at slot 0 is visible when the render pass starts.

**Decision 3: Keep `draw_meshlets` flag, add separate `meshlet_gpu_cull_enabled`**

`draw_meshlets` controls whether meshlet-indexed draws use meshlet vertex/index buffers at all (vs routing to standard path). `meshlet_gpu_cull_enabled` controls whether the cull compute passes run and whether `draw_indexed_indirect` (compacted) is used vs `draw_indexed` (full).

Rationale: Two independent flags allow testing basic vs culled paths independently. On integrated GPUs, `draw_meshlets=true` and `meshlet_gpu_cull_enabled=false` gives correct rendering with meshlet buffers.

## Risks / Trade-offs

- **Basic path draws all triangles, no culling**: For a 2.7M triangle model, this means all triangles are submitted to the GPU even if off-screen. However, the GPU's fixed-function culling still clips off-screen triangles, so the performance impact is primarily vertex processing. → Mitigation: Only enable for Basic GPU tier; Standard/Enhanced tiers use the full cull pipeline.
- **Memory**: Both standard mesh buffers AND meshlet vertex/index buffers exist in GPU memory simultaneously. → Already the case; no regression.
- **First frame**: The meshlet InstanceData pre-insert happens during `draw_opaque_triangle_batches`. On the first frame, Phase 2 hasn't written yet, but queue.write_buffer executes before encoder submit, so the GPU sees correct data. → No issue.

## Open Questions

- Does the GPU cull path work correctly on discrete GPUs (Standard/Enhanced tier) where HZB and proper compute→render barriers are available? Needs testing.
