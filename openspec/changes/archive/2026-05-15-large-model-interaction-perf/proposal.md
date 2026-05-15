## Why

When rendering large models (>500K triangles, e.g. 2.7M-triangle STL), the GPU solid pass alone takes ~29ms per frame. During mouse orbit/pan/zoom, this single-frame blocking causes mouse events to pile up (4-15 events per frame at standard polling rates), creating the appearance of "freeze then jump to final position." Small models (<200K triangles) do not exhibit this problem, confirming it's a pure GPU throughput bottleneck during interaction.

CAD/industrial viewers expect smooth interaction regardless of model complexity.

## What Changes

- **Dynamic resolution scaling during interaction**: Render internally at 50-70% of viewport resolution, then upscale to swapchain. Pixel shading load drops to 25-50%, directly proportional to GPU solid pass time.
- **Skip depth prepass during interaction**: The depth prepass (which reduces overdraw in solid pass) is skipped when the camera is moving, saving 3-5ms of GPU time per frame.
- **Screen-space edge detection shader**: A new compute/fragment shader that detects edges from the depth buffer (and optionally a normal buffer) as a single full-screen pass, decoupling edge rendering cost from triangle count. Replaces per-edge geometry draw calls during interaction.
- **Interaction-aware render path selection**: The renderer automatically switches between "quality" (static camera) and "performance" (interacting camera) configurations, toggling resolution, prepass, and edge rendering strategy.

## Capabilities

### New Capabilities
- `dynamic-resolution-scaling`: During camera interaction, the renderer internally renders at a reduced resolution and upscales the output, reducing GPU pixel load proportionally to the scale factor.
- `screen-space-edge-detection`: A post-processing shader that extracts edges from depth and normal buffers as a single full-screen pass, replacing per-edge geometry draw calls during interaction.
- `interaction-render-path`: The renderer toggles between a "quality" configuration (static camera) and a "performance" configuration (interacting camera), coordinating resolution, prepass, and edge rendering strategy.

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- **Renderer API**: New methods on `Renderer` for configuring dynamic resolution scale, and querying/forcing interaction render path
- **Render passes**: `execute_passes` and `render_draw_calls_core` modified to conditionally create intermediate render targets and skip passes during interaction
- **Shaders**: New WGSL shader `edge_detect.wgsl` (full-screen edge detection from depth + normals)
- **Event handler**: Existing `interaction_active` flag already set; leverage for render path switching
- **No breaking changes**: All new behavior is opt-in or gated behind interaction state; defaults unchanged
