## 1. Interaction detection consolidation

- [x] 1.1 Extract interaction detection into a shared helper `App::is_camera_interacting()` that checks both legacy `camera_controller` and `viewport_cameras`
- [x] 1.2 Use the helper to set `renderer.interaction_active` in the `RedrawRequested` handler (replace inline check at event_handler.rs:974-981)

## 2. Render scheduling switch

- [x] 2.1 In `AppState`, add `last_redraw_request: Instant` field for rate-limiting (revised: `pre_present_notify` is a no-op on Windows in winit 0.30.13)
- [x] 2.2 Add `request_redraw_rate_limited()` and `try_render_during_interaction()` methods on `App` — inline render during camera interaction bypasses winit's RedrawRequested mechanism entirely
- [x] 2.3 Use inline render in all camera event paths: CursorMoved handler, legacy camera handler, and dispatch_multi_viewport_camera

## 3. Viewport camera redraw fix

- [x] 3.1 In `dispatch_multi_viewport_camera`, add `request_redraw()` calls for `CursorMoved`, `MouseWheel`, and `MouseInput` handlers (currently missing; only legacy path calls them)
- [x] 3.2 Fix `cam_orbiting` check at event_handler.rs:107-109 to cover viewport cameras (now uses `is_camera_interacting()`)

## 4. Testing and verification

- [x] 4.1 Build with `rtk cargo check --workspace` to verify no compile errors
- [x] 4.2 Run `cargo build -p rc3d-examples --examples` to build all examples for linking verification
- [ ] 4.3 Manual test: run `rc3d-studio`, orbit/pan with middle/right mouse, verify smooth continuous rendering
- [ ] 4.4 Manual test: verify zoom still works smoothly
- [ ] 4.5 Manual test: verify static scene (no interaction) still renders correctly
