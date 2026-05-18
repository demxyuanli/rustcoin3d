=== Architecture Refactor: Final Report ===

Date: 2026-05-17
Branch: refactor/architecture-decouple

--- Test Results ---
Baseline tests: 257 passed, 4 ignored, 0 failed (35 suites)
Final tests:    283 passed, 5 ignored, 0 failed (47 suites)
Delta:          +26 passed, +1 ignored, +12 suites (+10.1% tests)
All green:      YES (0 failures)

--- Clippy ---
Errors:   0
New crate warnings: 0 (2 trivial warnings fixed in rc3d-editor)
Pre-existing warnings: remain in rc3d-core, rc3d-scene, rc3d-mesh, rc3d-render, rc3d-io, rc3d-nurbs, rc3d-engine

--- Documentation ---
cargo doc --no-deps --workspace: SUCCESS
New crate doc warnings: 0
Pre-existing doc warnings: rc3d-nurbs (3), rc3d-scene (4), rc3d-render (6), rc3d-fields (1), rc3d-mesh (1), rc3d-examples (1)

--- Example Build ---
cargo build --workspace --examples: 0 errors, 11 warnings
All examples link successfully.

--- New Crates ---
rc3d-engine-api: 1 test (unit), 1 test (doc)
rc3d-editor:     8 tests (unit), 1 test (doc)
rc3d-examples:   2 tests (unit), 37 examples

--- Architecture Metrics ---
rc3d-app lines: 76 (target: <150) PASS
Examples migrated to rc3d-examples: 37
Old examples in rc3d-app: 0 (all migrated)
All checks: PASS

--- Final Verification (2026-05-18) ---

All 46 examples compiled individually: PASS (0 errors each)

Example feature parity:
  Full interactive (HUD+keyboard+mouse): 10 examples (stl_diagnostic, import_viewer,
    import_viewer_async, render_features, editor, adaptive_stress_test,
    large_scene_stress, pbr_variant_viewer, profile_viewer, animation_control_panel)
  Simple render: 36 examples

Engine.render() fixes applied:
  1. renderer.set_materials() before traversal
  2. collector.material_library from World.materials
  3. set_hidden_nodes() support
  4. reserve_draw_calls() pre-allocation
  5. Default projection fallback (IDENTITY → perspective_rh)
  6. apply_world_camera() MVP computation
  7. Draw call caching for static frame fast path
  8. clear_all_dirty_flags() after traversal
  9. Legacy camera update_camera_recursive()
 10. renderer.update_hud() for HUD text overlay
 11. CameraController.dispatch_window_event() for mouse orbit/pan/zoom

All checks:
  cargo test --workspace: 283 passed, 0 failed
  cargo check --workspace: 0 errors
  cargo build --workspace --examples: 46/46 link
  clippy: 0 errors
  docs: built successfully

Refactor complete.
