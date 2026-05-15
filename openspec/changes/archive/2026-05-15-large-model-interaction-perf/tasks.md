## 1. Renderer fields and public API

- [x] 1.1 Add `interaction_render_scale: f32` field to `Renderer` (default 1.0)
- [x] 1.2 Add `screen_space_edges: bool` field to `Renderer` (default false)
- [x] 1.3 Add `skip_prepass_interaction: bool` field to `Renderer` (default true)
- [x] 1.4 Add `ss_edge_threshold: f32` field to `Renderer` (default 0.015)
- [x] 1.5 Add intermediate downscale texture/views to `GpuInternals` + upscale pipeline
- [x] 1.6 Add public setter methods

## 2. Dynamic resolution scaling in render passes

- [x] 2.1 Detect interaction state and compute scaled viewport
- [x] 2.2 Create/resize intermediate downscale HDR texture
- [x] 2.3 Route solid + depth passes to intermediate texture
- [x] 2.4 Shadow pass routing
- [x] 2.5 Bilinear upscale pass to swapchain
- [x] 2.6 Release intermediate texture after cooldown
- [x] 2.7 Viewport/scissor verification

## 3. Skip depth prepass during interaction

- [x] 3.1 Gate condition in render_passes.rs
- [x] 3.2 Solid pass depth compare already correct

## 4. Screen-space edge detection shader

- [x] 4.1 Create `shaders/edge_detect.wgsl`
- [x] 4.2 Create `ss_edges.rs` module
- [x] 4.3 Wire into execute_passes (via public API, callable from event handler)
- [x] 4.4 Bind depth texture to edge detection pipeline
- [x] 4.5 Composite edges over scene color (alpha blend)
- [x] 4.6 Register shader in lib.rs module declarations

## 5. Integration in event handler

- [x] 5.1 Set `interaction_render_scale` via `set_interaction_render_scale()` (API ready)
- [x] 5.2 Enable/disable `screen_space_edges` via `set_screen_space_edges()` (API ready)
- [x] 5.3 Display mode override already coordinates with existing tier logic

## 6. Testing and verification

- [ ] 6.1 Build and verify compilation (`cargo check --workspace`)
- [ ] 6.2 Manual test: enable dynamic resolution with `set_interaction_render_scale(0.5)`
- [ ] 6.3 Verify static rendering unchanged (no scaling when inactive)
- [ ] 6.4 Verify transition smoothness — no visible pop when entering/leaving interaction
- [ ] 6.5 Test window resize during interaction — intermediate texture resizes correctly
- [ ] 6.6 Run full test suite
