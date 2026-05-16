# Architecture Refactor: Decouple, Split, and Restructure

**Date:** 2026-05-16
**Status:** Draft
**Goal:** Restructure the rustcoin3d engine so users access it through a clean facade
(`rc3d-engine-api` + `rc3d-scene-api`) instead of the `rc3d-app` god object.
Extract the editor into its own crate, migrate examples to a dedicated crate,
and establish regression tests at every layer.

## Problem

`rc3d-app` depends on 12 internal crates and mixes orthogonal concerns:
- Window creation and event loop
- Editor command dispatch (813 lines)
- Editor UI (904 lines)
- Camera control, viewport management
- Streaming LOD, box select, measurement, gizmo support
- Hosts 46 example binaries

Users bypass `rc3d-scene-api` (the intended facade) and code directly against
`rc3d-app` internals. No regression tests exist in `rc3d-app`. The render pass
files `pass_markup.rs` (790 lines) and `pass_effects.rs` (695 lines) are
internally monolithic.

## Target Architecture

### Crate-level changes

```
BEFORE (18 crates)                    AFTER (21 crates)
─────────────────────                ──────────────────
rc3d-core                             rc3d-core            unchanged
rc3d-fields                           rc3d-fields           unchanged
rc3d-mesh                             rc3d-mesh             unchanged
rc3d-scene                            rc3d-scene            unchanged
rc3d-actions                          rc3d-actions          unchanged
rc3d-engine                           rc3d-engine           unchanged
rc3d-io                               rc3d-io               unchanged
rc3d-render                           rc3d-render           internal split only
rc3d-gizmo                            rc3d-gizmo            unchanged
rc3d-nodes/nurbs/pdf/pointcloud       (same)                unchanged
rc3d-scene-api   (build DSL)          rc3d-scene-api        build DSL only
rc3d-effects                          rc3d-effects          unchanged
rc3d-script                           rc3d-script           unchanged
                                      rc3d-engine-api ⭐     NEW: runtime facade
rc3d-app         (12 deps, 46 ex)     rc3d-app              THIN: ~150 lines
                                      rc3d-editor ⭐         NEW: extracted from app
                                      rc3d-examples ⭐       NEW: 46 examples
rc3d-cli-editor                       rc3d-cli-editor       updated consumers
```

### Dependency flow

(All crates ultimately depend on `rc3d-core`; omitted for clarity.)

```
rc3d-examples
  └─ rc3d-engine-api  (runtime facade)
       ├─ rc3d-render
       ├─ rc3d-io
       ├─ rc3d-engine
       ├─ rc3d-scene
       └─ rc3d-scene-api  (build DSL, re-exported)

rc3d-cli-editor
  └─ rc3d-editor
       ├─ rc3d-engine-api
       └─ rc3d-gizmo
```

## Crate Internals

### rc3d-engine-api (NEW)

The single runtime facade. Users never touch `rc3d-app` internals.

```
rc3d-engine-api/src/
├── lib.rs              re-exports + Engine struct
├── engine.rs           Engine::new(window) -> setup -> render() -> run()
├── import.rs           Engine::import(path) delegates to rc3d_io
├── camera.rs           CameraController (moved from rc3d-app)
├── viewport.rs         ViewportCameraSet (moved from rc3d-app)
├── world.rs            World (moved from rc3d-app)
├── settings.rs         re-exports DisplaySettings, PostEffectSettings from rc3d-render
└── background.rs       Background mode config
```

Public API surface:
```rust
pub struct Engine { /* private fields */ }
impl Engine {
    pub fn new(window: &Window) -> Self;
    pub fn set_display_mode(&mut self, mode: DisplayMode);
    pub fn set_background(&mut self, bg: BgSettings);
    pub fn set_post_effects(&mut self, params: PostEffectParams);
    pub fn load_scene(&mut self, graph: SceneGraph);
    pub fn scene_mut(&mut self) -> &mut SceneGraph;
    pub fn import(&mut self, path: impl AsRef<Path>) -> Result<NodeId>;
    pub fn camera_mut(&mut self) -> &mut CameraController;
    pub fn viewport_layout(&mut self, layout: LayoutMode);
    pub fn render(&mut self) -> FrameStats;
    pub fn resize(&mut self, width: u32, height: u32);
    pub fn run(self, event_loop: EventLoop<()>);
}
```

Depends on: rc3d-render, rc3d-io, rc3d-engine, rc3d-scene, rc3d-scene-api, rc3d-core.

Engine exposes minimal wgpu internals needed by editor consumers:
```rust
impl Engine {
    pub fn wgpu_device(&self) -> &wgpu::Device;
    pub fn wgpu_queue(&self) -> &wgpu::Queue;
    pub fn surface_format(&self) -> wgpu::TextureFormat;
}
```

### rc3d-editor (NEW)

All editor logic extracted from rc3d-app.

**Key refactoring note:** The existing editor code accesses `AppState` fields directly
(`app.state.renderer`, `app.state.camera_controller`, `app.state.world`). After
extraction, these must go through the `Engine` facade. An `EditorContext` struct wraps
the Engine reference + editor-local state, and all UI/interaction/command code receives
`&mut EditorContext` instead of `&mut App`. This is the main non-trivial work in Phase 2.

```
rc3d-editor/src/
├── lib.rs             re-exports Editor, EditorCommand, PanelConfig, PanelPreset, etc.
├── editor.rs          Editor struct: owns EditorContext + command dispatch
├── context.rs         EditorContext: wraps &mut Engine + editor-local state
├── commands.rs        EditorCommand enum + execute (from editor_commands.rs)
├── interaction.rs     Mouse/keyboard interaction (from editor_interaction.rs)
├── ui/
│   ├── mod.rs
│   ├── draw.rs        egui drawing (from editor_ui/draw.rs)
│   ├── panel.rs       Control panels (from control_panel.rs; re-exports PanelConfig, etc.)
│   └── types.rs       EditorDisplayMode etc.
├── gizmo.rs           Gizmo mode bridge (from gizmo_support.rs)
├── box_select.rs      Box selection
├── measurement.rs     Measurement tools
└── selection.rs       Selection management
```

Public re-exports from `rc3d-editor::lib.rs`: `Editor`, `EditorCommand`, `EditorDisplayMode`,
`PanelConfig`, `PanelPreset`, `RenderFeaturePanelHandle`, `PanelSections`, `FeatureChannelId`,
`RenderFeaturePanelState`.

Depends on: rc3d-engine-api, rc3d-gizmo, egui, egui-wgpu, egui-winit, wgpu.
Editor rendering requires `wgpu::Device` and `wgpu::Queue` from Engine (see Engine API below).

### rc3d-app (THINNED)

Reduced to a minimal runner. No public API surface for engine usage.

`rc3d-app::app.rs` (~100 lines) retains the `ApplicationHandler` impl — it is the only
place that talks to winit directly. Every event method immediately delegates:

- `resumed(event_loop)` → create window, init `Engine`, init optional `Editor`
- `window_event(WindowEvent::RedrawRequested)` → `engine.render()`
- `window_event(WindowEvent::Resized(size))` → `engine.resize(size.width, size.height)`
- `window_event(WindowEvent::MouseInput { .. })` → `editor.handle_mouse_input(..)` (if editor active)
- `window_event(WindowEvent::KeyboardInput { .. })` → `editor.handle_keyboard(..)`
- All other events → no-op or forward to `editor.handle_event(event)`

The App struct owns `Engine` and `Option<Editor>`. It does NOT expose them publicly —
they are internal implementation detail of the runner.

Modules that must find a home before deletion from rc3d-app:
- `streaming_lod.rs` (370 lines) → `rc3d-render` (it's a rendering concern: GPU mesh LOD budget management)
- `fps_tracker.rs` (85 lines) → `rc3d-engine-api` (runtime diagnostic, useful for examples)
- `adaptive_quality.rs` → `rc3d-render` (already has render-side adaptive_quality module; merge or reconcile)
- `scene_bridge.rs` → `rc3d-engine-api` (DynamicSurface bridge between scene and render)
- `input_state.rs` (5 lines) → `rc3d-engine-api` (both Engine and Editor need it; access via Engine facade)
- `lod_state.rs` (9 lines) → `rc3d-render` (pure rendering concern, alongside streaming_lod)

```
rc3d-app/src/
├── lib.rs             minimal: only re-exports needed for downstream binary crates
├── app.rs             ~100 lines: ApplicationHandler impl, delegates to Engine + Editor
```

### rc3d-examples (NEW)

All 46 examples migrated from rc3d-app/examples/.

```
rc3d-examples/
├── Cargo.toml         depends only on rc3d-engine-api (+ optional rc3d-editor)
└── examples/
    ├── cube.rs
    ├── import_viewer.rs
    ├── ... (all 46 examples)
    └── common/
        └── mod.rs     shared helpers (setup, import utilities)
```

Each example rewritten to use only the facade:
```rust
// Before: direct app internals
let mut app = App::new(graph);
app.state.renderer.as_mut().unwrap().set_display_mode(Shaded);

// After: facade only
let engine = Engine::new(&window)?;
engine.set_display_mode(Shaded);
```

### rc3d-render (INTERNAL SPLIT)

No public API change — purely internal decomposition:
- `pass_markup.rs` (790 lines) → `pass_markup/mod.rs` + `projection.rs` + `primitives.rs`
- `pass_effects.rs` (695 lines) → `pass_effects/mod.rs` + `collect.rs` + `render.rs`

## Regression Test Architecture

Three layers, each verifiable independently.

### Layer 1: Unit tests per crate (fast, ms)

| Crate | Target tests | Key coverage |
|-------|-------------|--------------|
| rc3d-engine-api | 8+ | load_scene, import, display mode, resize, post effects, background, camera, wgpu accessors |
| rc3d-editor | 6+ | undo/redo, box select, gizmo snap, hotkey dispatch, measurement |
| rc3d-app (thinned) | 3+ | runner delegates to engine, runner delegates to editor |
| rc3d-render (split) | 4+ | markup projection, effects collection — golden file snapshots |

### Layer 2: Integration scenario tests (medium, ~s)

```
tests/
├── scenes/
│   ├── empty_scene.rs       minimal sandbox
│   ├── cube_scene.rs        basic primitive
│   ├── imported_stl.rs      IO + render
│   ├── multi_object.rs      complex scene graph
│   └── animated_scene.rs    engine + time
├── editor/
│   ├── select_move_undo.rs  editor command roundtrip
│   ├── multi_viewport.rs    viewport layout
│   └── gizmo_interaction.rs gizmo state machine
└── render/
    ├── display_modes.rs     shaded/wireframe/points
    ├── post_effects.rs      effect chain
    └── background_modes.rs  background types
```

Each integration test: build scene → render frame → assert draw_calls > 0 and errors == 0.

### Layer 3: Example smoke tests (slow, ~min)

One test per migrated example — verifies construction without panic:
```rust
#[test]
fn smoke_test_cube_example_constructs() {
    let scene = build_cube_scene();
    assert!(scene.node_count() > 0);
}
```

## Execution Plan

6 phases, each produces a green `cargo check` / `cargo test` increment.

### Phase 0: Baseline + Infrastructure

- 0.1 — Audit all existing #[test] counts and pass/fail
- 0.2 — Write 5-8 characterization tests for current App behavior (may require headless-test scaffolding; `App` currently creates real windows via `resumed()` only)
- 0.3 — Write golden-file snapshot tests for pass_markup and pass_effects (snapshot the CPU-side geometry data — `MarkupVertex` arrays and `ProjectedAnnotation` structs — not GPU render output)
- 0.4 — Create integration test harness crate under tests/
- Verify: `cargo test --workspace` passes, baseline recorded

### Phase 1: Create rc3d-engine-api

- 1.1 — Scaffold crate with Engine struct skeleton + public API
- 1.2 — Move CameraController from rc3d-app → rc3d-engine-api::camera
- 1.3 — Move ViewportCamera/ViewportCameraSet → rc3d-engine-api::viewport
- 1.4 — Move World → rc3d-engine-api::world
- 1.5 — Implement Engine::import(path) delegating to rc3d_io
- 1.6 — Implement Engine::new(window) → render() → run() glue
- 1.7 — Write Layer 1 unit tests (6+ tests)
- Verify: rc3d-engine-api compiles, tests pass. rc3d-app still uses its original internals.

### Phase 2: Create rc3d-editor

- 2.1 — Scaffold crate, depends on rc3d-engine-api + rc3d-gizmo + egui
- 2.2 — Design `EditorContext` struct: wraps `&mut Engine` + editor-local state (replaces direct AppState field access)
- 2.3 — Move editor_commands.rs → rc3d-editor::commands, update AppState refs → EditorContext
- 2.4 — Move editor_interaction.rs → rc3d-editor::interaction, update AppState refs → EditorContext
- 2.5 — Move editor_ui/ → rc3d-editor::ui, update AppState refs → EditorContext
- 2.6 — Move gizmo_support.rs → rc3d-editor::gizmo
- 2.7 — Move box_select.rs, measurement.rs → rc3d-editor
- 2.8 — Move control_panel.rs → rc3d-editor::ui::panel; re-export public types from lib.rs
- 2.9 — Write Layer 1 unit tests (undo/redo, box select, gizmo snap, hotkey dispatch)
- Verify: rc3d-editor compiles, tests pass. Public panel types accessible from rc3d-editor.

### Phase 3: Thin rc3d-app + Redirect

- 3.1 — Rewrite App struct to own Engine + Option<Editor>
- 3.2 — Reconnect event handling to Engine::render() and Editor::handle_event()
- 3.3 — Remove duplicated camera/viewport/world modules from rc3d-app (already in engine-api)
- 3.4 — Remove duplicated editor_ui/commands/interaction modules from rc3d-app (already in rc3d-editor)
- 3.5 — Move streaming_lod.rs → rc3d-render (GPU mesh LOD concern)
- 3.6 — Move fps_tracker.rs → rc3d-engine-api (runtime diagnostic)
- 3.7 — Move scene_bridge.rs → rc3d-engine-api (DynamicSurface bridge)
- 3.8 — Move input_state.rs → rc3d-engine-api, lod_state.rs → rc3d-render
- 3.9 — Merge/reconcile rc3d-app adaptive_quality.rs with existing rc3d-render adaptive_quality module
- 3.10 — Update rc3d-cli-editor to use rc3d-editor + rc3d-engine-api
- 3.11 — Ensure Phase 0 baseline tests still pass
- 3.12 — Write Layer 2 integration tests
- Verify: rc3d-app ~150 lines (only lib.rs + app.rs). Zero duplicated code. All 222+ tests pass.

### Phase 4: Migrate Examples

- 4.1 — Scaffold rc3d-examples crate
- 4.2 — Create examples/common/mod.rs with shared helpers
- 4.3–4.8 — Batch-migrate all 46 examples (6 batches of ~8 each)
- 4.9 — Write Layer 3 smoke tests (1 per example)
- 4.10 — Delete original examples from rc3d-app
- Verify: All 46 examples compile. Layer 3 smoke tests pass.

### Phase 5: Split Large Render Pass Files

- 5.1 — Split pass_markup.rs (790 lines) into mod.rs + projection.rs + primitives.rs
- 5.2 — Split pass_effects.rs (695 lines) into mod.rs + collect.rs + render.rs
- 5.3 — Write Layer 1 unit tests for markup projection and effects collection
- 5.4 — Verify Phase 0 golden-file snapshots still pass
- Verify: All files <500 lines. Snapshot tests unchanged.

### Phase 6: Hardening + Final Regression Suite

- 6.1 — Full `cargo test --workspace` (all layers, zero failures)
- 6.2 — `cargo clippy --workspace` (zero warnings)
- 6.3 — `cargo doc --no-deps --workspace` (zero broken doc links)
- 6.4 — Compare test counts against Phase 0 baseline
- 6.5 — `cargo build --workspace --examples` (all 46 examples link)
- Verify: Final passing run. Comparison table recorded.

### Parallelization

Phases 4 and 5 can run in parallel — independent concerns, no shared files.

## Success Criteria

1. Users access the engine only through `rc3d-engine-api` + `rc3d-scene-api`
2. Editor functionality accessed through `rc3d-editor` (not `rc3d-app`)
3. `rc3d-app` is ≤150 lines (lib.rs + app.rs), no public API surface for engine/editor usage
4. Zero duplicated code between rc3d-app and new crates
5. All 46 examples compile and run from rc3d-examples
6. Test count ≥ baseline + 65 (Layer 1: 21+, Layer 2: 10+, Layer 3: 46)
7. `cargo clippy --workspace` passes with zero warnings
8. `cargo doc --workspace` generates without broken links
