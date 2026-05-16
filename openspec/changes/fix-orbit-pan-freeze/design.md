## Context

The app uses winit 0.30 with the `ApplicationHandler` trait. Rendering is triggered by `window.request_redraw()` which posts a synthetic `RedrawRequested` event. On Windows, winit drains all pending Windows messages before dispatching `RedrawRequested`. During mouse drag, the continuous stream of `WM_MOUSEMOVE` messages prevents the queue from draining, starving the render.

The app already has `continuous_redraw = true` (requests redraw at frame end) and explicitly calls `request_redraw()` in `CursorMoved` during camera orbit (line 141). Both are insufficient because the underlying winit dispatch model defers `RedrawRequested` until the Windows message queue is empty.

## Goals / Non-Goals

**Goals:**
- Camera orbit and pan produce smooth, continuous frame updates during mouse drag
- No regression to zoom behavior or static scene rendering
- Minimal changes — no architectural rewrite of the render loop

**Non-Goals:**
- Fixing general event loop latency (only interaction rendering)
- Changing the rendering pipeline itself (passes, shaders, culling)
- Supporting platforms other than Windows for this specific fix (the fix should be portable but the bug is Windows-specific)

## Decisions

### Decision 1: Use `pre_present_notify` for interaction rendering

**Chosen**: Convert the interaction render loop to use `Window::pre_present_notify()`.

`pre_present_notify` is a wgpu/Winit feature designed for continuous rendering. It fires a callback before each vsync, independently of the Windows message queue. This is the mechanism used by most modern wgpu examples for smooth rendering.

**Alternatives considered**:
- **`AboutToWait`**: Fires when the event loop is about to block. Calling `request_redraw()` here creates poll-like behavior. Simpler but doesn't directly solve the input-starvation problem — `AboutToWait` also fires after the message queue is drained.
- **Inline rendering in CursorMoved**: Would block input processing during render, adding latency to mouse response.
- **`ControlFlow::Poll`**: Forces continuous polling but spins CPU at 100% — wasteful for a 3D app.

### Decision 2: Keep existing `request_redraw()` path for static frames

When `interaction_active` is false (static camera, no user input), the existing `request_redraw()` + `continuous_redraw` mechanism is sufficient and more power-efficient. Only switch to `pre_present_notify` during active interaction.

### Decision 3: Per-frame interaction check

The interaction flag is already set correctly in the `RedrawRequested` handler (line 982: `renderer.interaction_active = cam_interacting`). We need to read this flag BEFORE the winit event dispatch to decide whether to use `pre_present_notify`. Since `pre_present_notify` is configured on the `Window`, we need to toggle it when interaction starts/ends.

## Risks / Trade-offs

- **Risk**: `pre_present_notify` may not be available on all wgpu backends → **Mitigation**: Feature-gate or fall back to current behavior when unavailable
- **Risk**: Toggling `pre_present_notify` mid-frame may cause frame pacing jitter → **Mitigation**: Use hysteresis (don't disable immediately on interaction end, wait 2-3 frames)
- **Risk**: Increased GPU/power usage during interaction → **Mitigation**: Only active during interaction, which is inherently short-lived
