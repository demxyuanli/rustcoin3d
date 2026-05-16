## Why

Orbit and pan camera interactions freeze the display on Windows — the viewport stops updating during mouse drag and only shows the final camera position after the mouse button is released. Zoom (scroll wheel) works normally. This is a fundamental usability blocker for interactive 3D scene exploration.

## What Changes

- Replace the `request_redraw()`-based render scheduling (which is starved by continuous `WM_MOUSEMOVE` on Windows) with `pre_present_notify`-driven rendering during camera interaction, ensuring frames are dispatched between input events regardless of event queue depth.
- Add winit `AboutToWait` handling to continuously request redraws during active interaction, keeping the event loop in poll mode.
- Fall back to the existing `continuous_redraw` / `request_redraw()` mechanism when not interacting, preserving power efficiency during static viewing.

## Capabilities

### New Capabilities

- `interaction-render-scheduling`: Continuous rendering during camera orbit/pan/zoom that is not starved by the Windows input event queue, using `pre_present_notify` or equivalent vsync-aligned scheduling.

### Modified Capabilities

<!-- No existing specs to modify. -->

## Impact

- `crates/rc3d-app/src/app/event_handler.rs` — render scheduling logic in `RedrawRequested` and camera event handlers
- `crates/rc3d-app/src/app/app_state.rs` — possible new fields for interaction render state
- `crates/rc3d-app/src/app/mod.rs` — `ApplicationHandler` trait implementation (new `AboutToWait` handler)
- No API or breaking changes
