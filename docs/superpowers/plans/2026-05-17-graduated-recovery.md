# Graduated Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Smooth 3-frame quality transition after camera interaction on large models to eliminate post-interaction frame hitches.

**Architecture:** A `RecoveryState` struct in `AppState` tracks a 3-step quality ramp (0.67x Flat → 1.0x Flat → full Shaded+HDR). Button release in `event_handler.rs` initiates recovery; `RedrawRequested` executes recovery frames by overriding render quality per step. Re-interaction or scene change clears recovery.

**Tech Stack:** Rust, wgpu, winit 0.30

---

### Task 1: Add RecoveryState struct and recovery field

**Files:**
- Modify: `crates/rc3d-app/src/app/app_state.rs`
- Modify: `crates/rc3d-app/src/app/tests.rs`

- [ ] **Step 1: Write unit test for RecoveryState defaults**

```rust
// crates/rc3d-app/src/app/tests.rs — add after existing tests

use super::app_state::RecoveryState;

#[test]
fn recovery_state_defaults() {
    let r = RecoveryState { step: 0, total_steps: 3 };
    assert_eq!(r.step, 0);
    assert_eq!(r.total_steps, 3);
}

#[test]
fn recovery_state_is_done_when_step_reaches_total() {
    let r = RecoveryState { step: 3, total_steps: 3 };
    assert!(r.step >= r.total_steps);
}

#[test]
fn app_state_recovery_is_none_by_default() {
    let graph = SceneGraph::new();
    let app = App::new(graph);
    assert!(app.state.recovery.is_none());
}
```

- [ ] **Step 2: Run test to verify it fails (RecoveryState not defined)**

```bash
cargo test -p rc3d-app -- app::tests
```
Expected: compilation error — `RecoveryState` not found, `recovery` field not found.

- [ ] **Step 3: Add RecoveryState struct and recovery field to AppState**

In `crates/rc3d-app/src/app/app_state.rs`:

Add RecoveryState struct after imports:

```rust
/// Graduated quality recovery after camera interaction ends.
/// Steps through increasing quality levels to avoid a single expensive frame.
#[derive(Clone, Debug)]
pub struct RecoveryState {
    pub step: u32,
    pub total_steps: u32,
}
```

Add field to `AppState` (after `interaction_render_count`):

```rust
    /// Graduated quality recovery after camera interaction on large scenes.
    /// Some during recovery: 3-step ramp from 0.67x Flat → 1.0x Flat → full quality.
    /// None when idle or interacting.
    pub recovery: Option<RecoveryState>,
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test -p rc3d-app -- app::tests
```
Expected: 8 tests pass (existing 5 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-app/src/app/app_state.rs crates/rc3d-app/src/app/tests.rs
git commit -m "feat: add RecoveryState struct and recovery field to AppState

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Write tests for recovery trigger and clear on re-interaction

**Files:**
- Create: `crates/rc3d-app/src/app/recovery_tests.rs` (or extend tests.rs)

- [ ] **Step 1: Write integration-level tests for recovery logic**

Since recovery trigger requires a winit event loop context, we test the state transitions directly.

```rust
// crates/rc3d-app/src/app/tests.rs — add after the RecoveryState tests

use super::app_state::RecoveryState;

#[test]
fn recovery_state_advances_correctly() {
    let mut r = RecoveryState { step: 0, total_steps: 3 };
    r.step += 1;
    assert_eq!(r.step, 1);
    assert!(r.step < r.total_steps);
    r.step += 1;
    assert_eq!(r.step, 2);
    r.step += 1;
    assert_eq!(r.step, 3);
    assert!(r.step >= r.total_steps);
}

#[test]
fn app_can_set_and_clear_recovery() {
    let graph = SceneGraph::new();
    let mut app = App::new(graph);
    app.state.recovery = Some(RecoveryState { step: 0, total_steps: 3 });
    assert!(app.state.recovery.is_some());
    app.state.recovery = None;
    assert!(app.state.recovery.is_none());
}
```

- [ ] **Step 2: Run test to verify they pass**

```bash
cargo test -p rc3d-app -- app::tests
```
Expected: 10 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/rc3d-app/src/app/tests.rs
git commit -m "test: add recovery state transition and lifecycle tests

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Implement recovery trigger on button release

**Files:**
- Modify: `crates/rc3d-app/src/app/event_handler.rs:1154-1169`

- [ ] **Step 1: Modify button release handler to init recovery for large scenes**

Replace the button-release block (lines 1158-1168) with:

```rust
        // Orbit/pan button release: queue a full redraw after every camera controller has been
        // updated. For large scenes, initiate graduated recovery to avoid a single expensive frame.
        if !app.editor.measurement_mode && !block_orbit {
            if let WindowEvent::MouseInput { state, button, .. } = &event {
                if *state == winit::event::ElementState::Released
                    && (matches!(button, MouseButton::Middle | MouseButton::Right)
                        || (left_orbit && matches!(button, MouseButton::Left)))
                {
                    // Compute total visible triangles to decide if recovery is needed
                    let total_tris: u64 = app.state.world.collector.draw_calls.iter()
                        .map(|dc| dc.indices.as_ref()
                            .map_or(dc.vertices.len() as u64 / 3, |i| i.len() as u64 / 3))
                        .sum();
                    if total_tris > 500_000 {
                        app.state.recovery = Some(RecoveryState { step: 0, total_steps: 3 });
                    }
                    if let Some(window) = &app.state.window {
                        window.request_redraw();
                    }
                }
            }
        }
```

Also add the import at the top of `event_handler.rs`:

```rust
use crate::app::app_state::RecoveryState;
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo check -p rc3d-app
```
Expected: compiles with no new errors.

- [ ] **Step 3: Commit**

```bash
git add crates/rc3d-app/src/app/event_handler.rs
git commit -m "feat: trigger graduated recovery on orbit/pan button release for large scenes

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Implement recovery rendering in RedrawRequested

**Files:**
- Modify: `crates/rc3d-app/src/app/event_handler.rs` (RedrawRequested handler, lines 535-1081)

This is the core change. Recovery frames render at interpolated quality and chain to the next step via `request_redraw`.

- [ ] **Step 1: Add recovery rendering logic before the inline_skip check**

Insert immediately after the `interaction_tris` block (after line 926) and before the `interaction_quality_override` block (line 941):

```rust
                // === Graduated recovery after interaction ===
                let recovery_step: Option<u32> = app.state.recovery.as_ref().map(|r| r.step);
                if let Some(ref mut rec) = app.state.recovery {
                    // Re-interaction during recovery: abort and switch to inline path
                    if cam_interacting {
                        app.state.recovery = None;
                    } else {
                        // Compute quality interpolation for this recovery step
                        let saved_scale = renderer.interaction_render_scale;
                        let saved_hdr_rec = renderer.hdr_post_processing;
                        let saved_display_rec = renderer.display_mode();
                        let total = rec.total_steps.max(1);
                        match rec.step {
                            0 => {
                                // Step 0: 0.67x Flat, no HDR
                                renderer.set_interaction_render_scale(0.67);
                                renderer.set_display_mode(DisplayMode::Flat);
                                renderer.hdr_post_processing = false;
                            }
                            1 => {
                                // Step 1: 1.0x Flat, no HDR
                                renderer.set_interaction_render_scale(1.0);
                                renderer.set_display_mode(DisplayMode::Flat);
                                renderer.hdr_post_processing = false;
                            }
                            _ => {
                                // Final step: restore full quality
                                renderer.set_interaction_render_scale(1.0);
                                renderer.set_display_mode(saved_display_rec);
                                renderer.hdr_post_processing = saved_hdr_rec;
                            }
                        }
                        // Advance step for next frame
                        rec.step += 1;
                        if rec.step >= total {
                            app.state.recovery = None;
                        }
                        // Prepare render state for this recovery frame
                        if let Some(ref bg) = app.state.bg_settings {
                            renderer.set_background(bg.clone());
                        }
                    }
                }
```

**Important:** This must be placed BEFORE the `interaction_quality_override` block (line 941) so recovery takes precedence over interaction overrides. And the `cam_interacting` check should still run — if the user starts interacting during recovery, recovery clears.

After the render completes and recovery is active, we need to chain to the next recovery frame. Find the block at lines 1070-1075:

```rust
                if app.state.continuous_redraw && !cam_interacting {
                    window.request_redraw();
                }
```

Extend it to also trigger when recovery is active:

```rust
                if (app.state.continuous_redraw && !cam_interacting)
                    || app.state.recovery.is_some()
                {
                    window.request_redraw();
                }
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo check -p rc3d-app
```
Expected: compiles with no new errors.

- [ ] **Step 3: Restore renderer state after recovery frame**

After the render block, restore any state that recovery modified. Add after the `interaction_quality_override` restore block (around line 1063-1066):

The recovery block already would have modified renderer state (scale, display_mode, HDR). We need to save/restore properly within the recovery block. Let me refine — the recovery block should save state before modifying and restore after:

The cleanest approach: recovery overrides are SET before the render call, and the restoration happens in the existing cleanup block. But since recovery may span multiple frames and each frame goes through the RedrawRequested handler fresh, we don't need to restore — the next frame's handler sets fresh values.

However, one concern: if recovery modifies `interaction_render_scale` and `hdr_post_processing` and the render completes, then the post-render code at line 1063 restores saved_hdr/saved_display. But those savings happened BEFORE recovery, so they restore correctly. The `interaction_render_scale` set by recovery should be reset after recovery ends (which happens automatically since `app.state.recovery = None` and the last step sets scale=1.0).

Add a final cleanup: reset `interaction_render_scale` to 1.0 when recovery completes. This goes right after `app.state.recovery = None`:

```rust
                        if rec.step >= total {
                            app.state.recovery = None;
                            renderer.set_interaction_render_scale(1.0);
                        }
```

- [ ] **Step 4: Verify full compilation**

```bash
cargo check -p rc3d-app
```
Expected: compiles with no new errors.

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-app/src/app/event_handler.rs
git commit -m "feat: implement graduated recovery rendering in RedrawRequested

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Integration test — run stl_diagnostic with large model

**Files:**
- No code changes — manual verification

- [ ] **Step 1: Build release**

```bash
cargo build -p rc3d-app --example stl_diagnostic --release
```
Expected: builds successfully.

- [ ] **Step 2: Run with Car engine.stl and verify smooth recovery**

```bash
cargo run -p rc3d-app --example stl_diagnostic --release "test_data/Car engine.stl"
```

Expected behavior:
- Orbit/pan during interaction: smooth (5-18ms frames at 0.5x Flat)
- Release mouse: 3 recovery frames fire in sequence:
  1. 0.67x Flat, ~8-12ms
  2. 1.0x Flat, ~10-15ms  
  3. Full quality Shaded, ~15-26ms
- No single frame exceeds 30ms
- Final state: normal rendering at full quality
- Re-interrupt recovery: orbit during recovery should abort recovery and resume inline interaction rendering

- [ ] **Step 3: Run full test suite**

```bash
cargo test --workspace
```
Expected: all 275+ tests pass.

- [ ] **Step 4: Commit if any final tweaks**

```bash
git status
```

---

## Self-Review Checklist

- [x] Spec coverage: RecoveryState struct (+), recovery trigger (+), recovery rendering (+), edge cases (re-interaction +)
- [x] Placeholder scan: No TBD/TODO, every step has concrete code
- [x] Type consistency: `RecoveryState { step: u32, total_steps: u32 }` used consistently across all tasks
