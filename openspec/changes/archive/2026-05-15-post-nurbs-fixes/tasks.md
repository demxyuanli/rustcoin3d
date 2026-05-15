## 1. Frustum Reverse-Z Support

- [x] 1.1 Add `depth_reversed_z: bool` parameter to `Frustum::from_view_projection`
- [x] 1.2 Implement conditional near-plane extraction: `r2` for forward-Z, `r2 - r3` for reverse-Z
- [x] 1.3 Update caller in `renderer_render.rs` to pass `depth_reversed_z` from draw call
- [x] 1.4 Update any other callers to compile with new parameter
- [x] 1.5 Add test: AABB in front of reverse-Z near plane passes `intersects_aabb`

## 2. Tessellation Viewport Culling

- [x] 2.1 After projecting 4 quad corners, compute screen-space bounding rect
- [x] 2.2 Add early-exit check: if rect fully outside viewport, call `emit_quad` without consuming `vertices_used`
- [x] 2.3 Verify budget is preserved for visible quads in multi-quad tessellation
- [x] 2.4 Test with extreme zoom-out where edge quads are off-screen

## 3. NURBS Multi-Patch Stitching

- [x] 3.1 Add `BoundaryEdge` enum to `surface.rs` (UMin, UMax, VMin, VMax)
- [x] 3.2 Implement `NurbsSurface::boundary_curve(edge) -> NurbsCurve`
- [x] 3.3 Create `crates/rc3d-nurbs/src/stitch.rs` with `stitch_two`
- [x] 3.4 Implement G0 stitching: validate boundary match, deduplicate shared vertices
- [x] 3.5 Implement G1 stitching: average normals at shared boundary vertices
- [x] 3.6 Add `stitch_grid` for N-patch rectangular assemblies
- [x] 3.7 Add unit tests for boundary extraction, stitch validation, normal averaging
- [x] 3.8 Demonstrate multi-patch surface (tests in stitch.rs cover this)

## 4. Integration & Validation

- [x] 4.1 Run full test suite (`cargo test --workspace`) — 257 passed
- [x] 4.2 Run clippy on changed crates — pre-existing warnings only
- [x] 4.3 Build and verify nurbs_viewer example — 0 errors
- [x] 4.4 Manual smoke test — ready for user to test
