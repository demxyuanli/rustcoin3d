# Graduated Recovery: Smooth Post-Interaction Quality Transition

**Date:** 2026-05-17
**Status:** approved
**Scope:** `rc3d-app` (app_state, event_handler, mod)

## Problem

After orbit/pan/zoom on large models (>1M tris), the first full-quality frame
can take 15-26ms (sometimes 1000ms+ for scene traversal). During interaction,
the renderer uses 0.5x Flat with no HDR. When the user releases the mouse,
the next `RedrawRequested` event triggers a full-quality Shaded + HDR render
at native resolution, causing a visible hitch.

## Design

### Graduated recovery state machine

```
interacting → recovery_step_0 → recovery_step_1 → recovery_step_2 → idle
 (0.5x Flat)   (0.67x Flat)      (1.0x Flat)      (1.0x Shaded)   (normal)
  no HDR         no HDR             no HDR            HDR on
```

Each step renders one frame at increasing quality, then requests the next
redraw. After the final step, the engine resumes normal rendering.

### RecoveryState struct

Added to `AppState`:

```rust
pub recovery: Option<RecoveryState>,

pub struct RecoveryState {
    pub step: u32,        // current step index (0-based)
    pub total_steps: u32, // fixed at 3
}
```

### Trigger

- **On button release** (both legacy `CameraController` and `ViewportCameraSet`
  paths in `event_handler.rs`): if `is_camera_interacting()` was true before
  release, set `recovery = Some(RecoveryState { step: 0, total_steps: 3 })`
  and call `window.request_redraw()`.
- **On `RedrawRequested`**: if `recovery.is_some()`, use the step to compute
  scale/mode/HDR overrides, render at the interpolated quality, increment
  step, and request next redraw. When `step >= total_steps`, clear recovery.

### Quality interpolation

| step | scale | mode      | HDR  | expected frame time |
|------|-------|-----------|------|---------------------|
| 0    | 0.67  | Flat      | off  | ~8-12ms             |
| 1    | 1.0   | Flat      | off  | ~10-15ms            |
| 2    | 1.0   | prev mode | prev | ~15-26ms            |

Step 2 restores the display mode and HDR setting that were active before
interaction began (saved at recovery init time).

### Edge cases

- **Re-interaction during recovery**: `is_camera_interacting()` becomes true →
  clear recovery, switch to inline interaction render path.
- **Resize during recovery**: no special handling; each frame picks up current
  surface size.
- **Scene load/import during recovery**: clear recovery, perform normal
  first-frame traversal.
- **Small models (<500K tris)**: recovery is skipped entirely; the full-quality
  frame is fast enough. Check `total_visible_triangles > 500_000` before
  initiating recovery.

## Files changed

| File | Change |
|------|--------|
| `crates/rc3d-app/src/app/app_state.rs` | Add `RecoveryState` struct, `recovery` field to `AppState` |
| `crates/rc3d-app/src/app/mod.rs` | Add `render_recovery_frame()` method: compute quality interpolation and render |
| `crates/rc3d-app/src/app/event_handler.rs` | Trigger recovery on button release; execute recovery in `RedrawRequested` |

## Non-goals

- Recovery for `Engine` API crate (not yet integrated with interaction path)
- Memory or load-time optimization (separate effort)
- Adaptive step count based on model size (fixed 3 steps is sufficient)
