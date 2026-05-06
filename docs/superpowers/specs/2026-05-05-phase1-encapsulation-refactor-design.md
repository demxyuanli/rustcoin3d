# Phase 1: Encapsulation Refactor — Design Spec

> **Goal:** Bring Renderer and App encapsulation to Coin3D/HOOPS standards: all internal state private, settings-driven mutation, GPU resources fully hidden.

> **Architecture:** Introduce `RenderSettings` (PostEffect/Lighting/Display sub-structs), make all Renderer fields private, add setter/getter API. Split App into `AppState` / `EditorSession` / `InputState` / `LODState`.

> **Tech Stack:** Rust, wgpu 24, winit 0.30, glam 0.29, slotmap

---

## 1. RenderSettings — New File

**Create:** `crates/rc3d-render/src/settings.rs`

Three-tier settings mirroring HOOPS `HPS::RenderingMode` pattern. Each tier is a `Copy + Clone` struct so panels can hold independent snapshots.

```rust
#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub post_effect: PostEffectSettings,
    pub lighting: LightingSettings,
    pub display: DisplaySettings,
}

#[derive(Clone, Copy, Debug)]
pub struct PostEffectSettings {
    pub hdr: bool,
    pub taa: bool,
    pub motion_blur: bool,
    pub ssr: bool,
    pub color_grading: bool,
    pub dof: bool,
    pub volumetric_fog: bool,
    pub ldr_fxaa: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct LightingSettings {
    pub cluster_lights: bool,
    pub omni_shadows: bool,
    pub ibl_preset: IblPreset,
}

#[derive(Clone, Copy, Debug)]
pub struct DisplaySettings {
    pub display_mode: DisplayMode,
    pub grid_enabled: bool,
    pub hud_enabled: bool,
    pub outline_width: f32,
    pub outline_color: [f32; 4],
    pub xray_mode: bool,
    pub vsync_enabled: bool,
    pub screen_space_selection_outline: bool,
}
```

`Default` impl provides sensible defaults: HDR on, TAA on, SSR off, color grading off, grid on, etc.

---

## 2. Renderer Refactor

**Modify:** `crates/rc3d-render/src/renderer.rs`

### 2.1 Field Reorganization

All ~90 pub fields become private. Access goes through three tiers:

| Tier | Visibility | Contains |
|------|-----------|----------|
| `settings: RenderSettings` | `pub` via getter/setter | All runtime-toggleable settings |
| `frame: FrameState` | `pub(crate)` | Per-frame mutable state for render_passes |
| `gpu: GpuInternals` | `pub(crate)` | All GPU resources (pipelines, textures, buffers, pools) |

One-time config held directly on Renderer (no struct wrapper):

```rust
pub struct Renderer {
    // ── One-time config (immutable after new) ──
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    wireframe_supported: bool,

    // ── Settings tier ──
    settings: RenderSettings,

    // ── Frame state tier ──
    frame: FrameState,

    // ── GPU internals tier ──
    gpu: GpuInternals,
}
```

### 2.2 FrameState (pub(crate))

```rust
pub(crate) struct FrameState {
    pub markup_vertices: Vec<LineVertex>,
    pub clip_planes: Vec<[f32; 4]>,
    pub scene_vp: Mat4,
    pub scene_camera_pos: Vec3,
    pub animation_time_sec: f32,
    pub frame_counter: u64,
    pub performance_mode_active: bool,
    pub last_diagnostics: Option<FrameDiagnostics>,
    pub last_hud_update_frame: u64,
    pub viewport_layout: ViewportLayout,
}
```

### 2.3 GpuInternals (pub(crate))

Contains ALL GPU resources: pipelines, shader_cache, pipeline_cache, shader_reload, shadow_pool, csm_shadow, omni_shadow, shadow_compare_sampler, hzb, hzb_baker, cluster_renderer, cluster_pipeline_generation, cluster_lights, cluster_light_culler, post_fx_pipelines, post_fx, ssao_noise_tex/view, instance_buffer, ibl_instance_bind_group, ibl_diffuse/specular, gpu_skinning_pass, skinned_mesh_resources, phong/flat/outline pools, taa_pass/jitter, motion_blur, ssr_pass, color_grading, dof_pass, volumetric_fog, auto_exposure, texture_cache, materials, selection_outline_pipelines/targets, ldr_shade_tex/view, gpu_query_set/buffer/slots/period, timing_supported, depth_texture, adaptive_quality, adaptive_frame_time_ema_ms, adaptive_switch_cooldown_frames.

### 2.4 New Public API

**Gpu resource proxy methods** (replace direct `renderer.device` access):

```rust
impl Renderer {
    pub fn create_vertex_buffer<T: bytemuck::Pod>(&self, data: &[T], label: &str) -> wgpu::Buffer;
    pub fn create_uniform_buffer(&self, size: u64, label: &str) -> wgpu::Buffer;
    pub fn queue_write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]);
}
```

**Settings getter:**

```rust
impl Renderer {
    pub fn settings(&self) -> &RenderSettings;
    pub fn apply_settings(&mut self, settings: RenderSettings);
}
```

**Single-field setters** (convenience, internally call `apply_settings` for the relevant sub-struct):

```rust
impl Renderer {
    pub fn set_taa(&mut self, enabled: bool);
    pub fn set_ssr(&mut self, enabled: bool);
    pub fn set_motion_blur(&mut self, enabled: bool);
    pub fn set_color_grading(&mut self, enabled: bool);
    pub fn set_dof(&mut self, enabled: bool);
    pub fn set_volumetric_fog(&mut self, enabled: bool);
    pub fn set_cluster_lights(&mut self, enabled: bool);
    pub fn set_omni_shadows(&mut self, enabled: bool);
    pub fn set_grid_enabled(&mut self, enabled: bool);
    pub fn set_hud_enabled(&mut self, enabled: bool);
    pub fn set_xray_mode(&mut self, enabled: bool);
    // Existing setters kept: set_hdr_post_processing, set_display_mode, set_outline_color,
    // set_ibl_preset, cycle_ibl_preset, set_vsync, set_screen_space_selection_outline,
    // set_ldr_fxaa
}
```

**Query methods:**

```rust
impl Renderer {
    pub fn last_frame_stats(&self) -> &FrameStats;
    pub fn last_diagnostics(&self) -> Option<&FrameDiagnostics>;
    pub fn frame_counter(&self) -> u64;
    // Existing queries: display_mode, performance_mode_active, adaptive_quality_name,
    // adaptive_is_low, ibl_preset_name, clip_planes
}
```

**Internal accessors** (for render_passes modules):

```rust
impl Renderer {
    pub(crate) fn gpu(&self) -> &GpuInternals;
    pub(crate) fn frame(&self) -> &FrameState;
    pub(crate) fn frame_mut(&mut self) -> &mut FrameState;
}
```

### 2.5 Existing API Kept Unchanged

`resize()`, `render_frame()`, `collect_markup_vertices()`, `report_frame_time_ms()`, `set_clip_planes()`, `offscreen_target_rgba8()`, `toggle_clip_plane()`, `invalidate_mesh_cache()`.

---

## 3. App Split

**Modify:** `crates/rc3d-app/src/app/mod.rs`
**Create:** `crates/rc3d-app/src/app/app_state.rs`, `editor_session.rs`, `input_state.rs`, `lod_state.rs`

### 3.1 AppState

Persistent state that survives across frames:

```rust
pub struct AppState {
    pub world: World,
    pub renderer: Option<Renderer>,
    pub window: Option<winit::window::Window>,
    /// Legacy camera controller; prefer viewport_cameras
    pub camera_controller: Option<CameraController>,
    pub viewport_cameras: ViewportCameraSet,
    pub initial_display_mode: DisplayMode,
    pub enable_hdr_post_processing: bool,
    pub adaptive_quality_mode: AdaptiveQualityMode,
    pub adaptive_last_interaction: Instant,
    pub continuous_redraw: bool,
    pub last_frame_time: Instant,
    pub last_frame_time_ms: f32,
    pub fps_tracker: FpsTracker,
    pub last_render_stats: FrameStats,
}
```

### 3.2 EditorSession

All interactive editing state:

```rust
pub struct EditorSession {
    pub ui: EditorUi,
    pub gizmo: Gizmo,
    pub gizmo_dragging: bool,
    pub gizmo_pending_transform: Option<(NodeId, Mat4)>,
    pub command_history: CommandHistory,
    pub markup_action: MarkupAction,
    pub measurement_mode: bool,
    pub measurement_type: Option<MeasurementType>,
    pub measurement_first_point: Option<Vec3>,
    pub measurements: Vec<(Vec3, Vec3, f32)>,
    pub axis_clip: [bool; 3],
}
```

### 3.3 InputState

Per-frame input snapshot:

```rust
pub struct InputState {
    pub cursor_pos: (f64, f64),
    pub shift_pressed: bool,
    pub ctrl_pressed: bool,
}
```

### 3.4 LODState

Streaming LOD state:

```rust
pub struct LODState {
    pub full_res_patches: Vec<FullResPatch>,
    pub preview_mode_active: bool,
    pub stream_next_tick: Option<Instant>,
}
```

### 3.5 App (simplified)

```rust
pub struct App {
    state: AppState,
    editor: EditorSession,
    input: InputState,
    lod: LODState,
    // Cross-cutting hooks kept at App level
    on_pick: Option<PickCallback>,
    pending_graph_rx: Option<std::sync::mpsc::Receiver<Result<SceneGraph, String>>>,
    graph_load_hook: Option<Box<dyn FnOnce(&mut App) + 'static>>,
    panel_overlay_text_hook: Option<Box<dyn Fn() -> String>>,
    panel_overlay_key_hook: Option<Box<dyn FnMut(winit::keyboard::KeyCode)>>,
    panel_overlay_mouse_hook: Option<Box<dyn FnMut(f32, f32, u32, u32) -> bool>>,
}
```

---

## 4. Caller Updates

### 4.1 rc3d-render internal modules

All `renderer.field_name` accesses within `render_passes/`, `render_action.rs`, `gpu_skinning.rs` etc. update to use `renderer.frame()` / `renderer.gpu()` accessors. Purely mechanical — no logic changes.

### 4.2 rc3d-gizmo

- `renderer.device.create_buffer_init(...)` → `renderer.create_vertex_buffer(data, "gizmo vb")`
- Other `renderer.device` calls similarly routed through GPU proxy methods

### 4.3 rc3d-app

- `renderer.enable_taa = true` → `renderer.set_taa(true)`
- `renderer.grid_enabled = false` → `renderer.set_grid_enabled(false)`
- Similar replacements for all direct field writes
- App sub-modules (`event_handler`, `editor_commands`, etc.) import from new sub-modules: `super::app_state::AppState`, `super::editor_session::EditorSession`, etc.

### 4.4 Examples

10 example files: mechanical replacement of `renderer.field = val` → `renderer.set_field(val)`.

---

## 5. Verification

```bash
rtk cargo check -p rc3d-render    # 0 errors
rtk cargo check -p rc3d-gizmo     # 0 errors
rtk cargo check -p rc3d-app       # 0 errors
rtk cargo test                    # all tests pass (>=113)
rtk cargo run -p rc3d-app --example editor  # UI functional, render toggles work
```

---

## 6. Non-Goals (deferred to later phases)

- Serialization (Phase 2)
- Resource lifecycle / LRU (Phase 3)
- NodeData componentization / plugin system (Phase 4)
- Multi-threaded traversal
- Script bindings
