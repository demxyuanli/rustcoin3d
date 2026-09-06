## 1. GpuCapability — Add meshlet_gpu_cull_enabled flag

- [x] 1.1 Add `meshlet_gpu_cull_enabled: bool` field to `GpuCapability` struct in `renderer_internals.rs`
- [x] 1.2 Set `meshlet_gpu_cull_enabled = false` when `GpuTier::Basic`, `true` for Standard/Enhanced in renderer init
- [x] 1.3 Store flag in `GpuInternals` alongside existing `gpu_capability` field

## 2. Basic meshlet draw path

- [x] 2.1 Modify `draw_opaque_triangle_batches` meshlet draw loop: gated on `meshlet_gpu_cull_enabled` — basic path uses `draw_clustered_basic`, cull path uses `draw_clustered`
- [x] 2.2 Ensure `last_bound_mesh = None` after both basic and culled meshlet draws
- [x] 2.3 Keep pre-insert of meshlet InstanceData at `all_instances[0]` for both paths

## 3. Gate GPU cull passes on capability

- [x] 3.1 In `execute_passes`, fallback `submit_meshlet_cull` gated on `meshlet_gpu_cull_enabled` (HZB prepass already gated by `tier_allows_hzb`)
- [x] 3.2 `submit_meshlet_cull` only called when `meshlet_gpu_cull_enabled` is true

## 4. Remove diagnostic code

- [x] 4.1 Rename `draw_clustered_full_diag` to `draw_clustered_basic` (simplified signature)
- [x] 4.2 Remove diagnostic log statements (`[FLAT DIAG]`, `[MESHLET DIAG]`)
- [x] 4.3 Clean up: removed unused diagnostic counters from flat draw function

## 5. Verification

- [x] 5.1 `cargo check --workspace` — 0 errors
- [x] 5.2 `cargo test` — 250 passed, 4 ignored
- [ ] 5.3 `cargo build -p rc3d-examples --examples` — blocked by pre-existing `subdivide_quad_screen` error in app crate
- [x] 5.4 Manual test with Car engine.stl: all display modes render correctly, no artifacts
- [ ] 5.5 Manual test with small STL (no meshlet data): regression check
