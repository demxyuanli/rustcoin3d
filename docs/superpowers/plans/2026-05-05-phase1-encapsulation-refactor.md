# Phase 1: Encapsulation Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring Renderer and App encapsulation to Coin3D/HOOPS standards — all internal state private, settings-driven mutation, GPU resources fully hidden, App split into focused sub-structs.

**Architecture:** Introduce `RenderSettings` (PostEffect/Lighting/Display sub-structs), move Renderer fields into private settings + `pub(crate)` FrameState + `pub(crate)` GpuInternals, add setter/getter API. Split App into `AppState` / `EditorSession` / `InputState` / `LODState` sub-modules.

**Tech Stack:** Rust, wgpu 24, winit 0.30, glam 0.29, slotmap

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/rc3d-render/src/settings.rs` | **Create** | RenderSettings + PostEffect/Lighting/Display sub-structs |
| `crates/rc3d-render/src/renderer.rs` | **Modify** | Restructure fields into settings/frame/gpu tiers, add setters |
| `crates/rc3d-render/src/renderer_helpers.rs` | **Modify** | Update internal field paths |
| `crates/rc3d-render/src/renderer_render.rs` | **Modify** | Update internal field paths |
| `crates/rc3d-render/src/renderer_skinning.rs` | **Modify** | Update internal field paths |
| `crates/rc3d-render/src/render_passes.rs` | **Modify** | Update internal field paths via frame()/gpu() accessors |
| `crates/rc3d-render/src/selection_outline.rs` | **Modify** | Update internal field paths |
| `crates/rc3d-render/src/lib.rs` | **Modify** | Re-export new types |
| `crates/rc3d-gizmo/src/gizmo.rs` | **Modify** | `renderer.device` → `renderer.create_vertex_buffer()` or gpu proxy |
| `crates/rc3d-app/src/app/mod.rs` | **Modify** | Split into sub-modules, update renderer field access |
| `crates/rc3d-app/src/app/app_state.rs` | **Create** | AppState struct |
| `crates/rc3d-app/src/app/editor_session.rs` | **Create** | EditorSession struct |
| `crates/rc3d-app/src/app/input_state.rs` | **Create** | InputState struct |
| `crates/rc3d-app/src/app/lod_state.rs` | **Create** | LODState struct |
| `crates/rc3d-app/src/app/event_handler.rs` | **Modify** | Update renderer field access, use sub-module types |
| `crates/rc3d-app/src/app/editor_commands.rs` | **Modify** | Update renderer field access (direct write → setter) |
| `crates/rc3d-app/src/editor_ui/mod.rs` | **Modify** | `renderer.device` → `renderer.device_limits()` + stop direct egui init on renderer.device |
| `crates/rc3d-app/examples/*.rs` | **Modify** | Mechanical: field writes → setter calls |

---

## Task 1: Create RenderSettings

**Files:**
- Create: `crates/rc3d-render/src/settings.rs`
- Modify: `crates/rc3d-render/src/lib.rs`

- [ ] **Step 1: Write the types**

```rust
// crates/rc3d-render/src/settings.rs
use crate::ibl::IblPreset;
use rc3d_core::DisplayMode;

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

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            post_effect: PostEffectSettings::default(),
            lighting: LightingSettings::default(),
            display: DisplaySettings::default(),
        }
    }
}

impl Default for PostEffectSettings {
    fn default() -> Self {
        Self {
            hdr: false,
            taa: false,
            motion_blur: false,
            ssr: false,
            color_grading: false,
            dof: false,
            volumetric_fog: false,
            ldr_fxaa: true,
        }
    }
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            cluster_lights: true,
            omni_shadows: true,
            ibl_preset: IblPreset::Studio,
        }
    }
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            display_mode: DisplayMode::ShadedWithEdges,
            grid_enabled: false,
            hud_enabled: true,
            outline_width: 0.022,
            outline_color: [1.0, 0.5, 0.0, 1.0],
            xray_mode: false,
            vsync_enabled: true,
            screen_space_selection_outline: true,
        }
    }
}
```

- [ ] **Step 2: Add to lib.rs**

Add after the existing re-exports in `crates/rc3d-render/src/lib.rs`:

```rust
mod settings;
pub use settings::{DisplaySettings, LightingSettings, PostEffectSettings, RenderSettings};
```

- [ ] **Step 3: Compile check**

```bash
rtk cargo check -p rc3d-render
```

Expected: 0 errors. New types compile clean. No callers yet.

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-render/src/settings.rs crates/rc3d-render/src/lib.rs
rtk git commit -m "feat: add RenderSettings types (PostEffect/Lighting/Display tiers)"
```

---

## Task 2: Add settings field + apply/get methods to Renderer

**Files:**
- Modify: `crates/rc3d-render/src/renderer.rs`

- [ ] **Step 1: Add `settings` field to Renderer struct**

Insert after the `surface_config` field (currently `pub config: wgpu::SurfaceConfiguration`):

```rust
pub struct Renderer {
    // ... existing pub fields remain for now ...
    /// Runtime render settings (replaces scattered feature toggles).
    settings: RenderSettings,
    // ...
}
```

- [ ] **Step 2: Add `apply_settings` and `settings()` methods to Renderer impl**

```rust
impl Renderer {
    /// Batch-apply render settings (HOOPS-style).
    pub fn apply_settings(&mut self, settings: RenderSettings) {
        // PostEffect -> individual toggles
        self.hdr_post_processing = settings.post_effect.hdr;
        self.enable_taa = settings.post_effect.taa;
        self.enable_motion_blur = settings.post_effect.motion_blur;
        self.enable_ssr = settings.post_effect.ssr;
        self.enable_color_grading = settings.post_effect.color_grading;
        self.enable_dof = settings.post_effect.dof;
        self.enable_volumetric_fog = settings.post_effect.volumetric_fog;
        self.enable_ldr_fxaa = settings.post_effect.ldr_fxaa;
        // Lighting
        self.enable_cluster_lights = settings.lighting.cluster_lights;
        self.enable_omni_shadows = settings.lighting.omni_shadows;
        self.set_ibl_preset(settings.lighting.ibl_preset);
        // Display
        self.global_display_mode = settings.display.display_mode;
        self.grid_enabled = settings.display.grid_enabled;
        self.hud_enabled = settings.display.hud_enabled;
        self.outline_width = settings.display.outline_width;
        self.outline_color = settings.display.outline_color;
        self.xray_mode = settings.display.xray_mode;
        self.screen_space_selection_outline = settings.display.screen_space_selection_outline;
        if settings.display.vsync_enabled != self.is_vsync_enabled() {
            self.set_vsync(settings.display.vsync_enabled);
        }
        // Sync side-effects
        if !self.hdr_post_processing {
            self.post_fx = None;
        } else {
            self.ensure_post_fx_targets();
        }
        if !self.enable_ldr_fxaa {
            self.ldr_shade_tex = None;
            self.ldr_shade_view = None;
        }
    }

    /// Read-only access to current settings snapshot.
    pub fn settings(&self) -> RenderSettings {
        RenderSettings {
            post_effect: PostEffectSettings {
                hdr: self.hdr_post_processing,
                taa: self.enable_taa,
                motion_blur: self.enable_motion_blur,
                ssr: self.enable_ssr,
                color_grading: self.enable_color_grading,
                dof: self.enable_dof,
                volumetric_fog: self.enable_volumetric_fog,
                ldr_fxaa: self.enable_ldr_fxaa,
            },
            lighting: LightingSettings {
                cluster_lights: self.enable_cluster_lights,
                omni_shadows: self.enable_omni_shadows,
                ibl_preset: self.ibl_preset,
            },
            display: DisplaySettings {
                display_mode: self.global_display_mode,
                grid_enabled: self.grid_enabled,
                hud_enabled: self.hud_enabled,
                outline_width: self.outline_width,
                outline_color: self.outline_color,
                xray_mode: self.xray_mode,
                vsync_enabled: self.is_vsync_enabled(),
                screen_space_selection_outline: self.screen_space_selection_outline,
            },
        }
    }

    fn is_vsync_enabled(&self) -> bool {
        matches!(self.config.present_mode, wgpu::PresentMode::AutoVsync)
    }
}
```

- [ ] **Step 3: Initialize settings in Renderer::new()**

In the `Renderer::new()` constructor, add after the `surface_config` is set and before the `Self { .. }` literal:

```rust
// Build initial settings from the individual field values being set below
let _initial_settings = RenderSettings::default();
```

Then add `settings: RenderSettings::default(),` to the struct literal (after `surface_config` field).

- [ ] **Step 4: Compile check**

```bash
rtk cargo check -p rc3d-render
```

Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-render/src/renderer.rs
rtk git commit -m "feat: add settings field + apply_settings/settings() accessors to Renderer"
```

---

## Task 3: Add new single-field setter methods to Renderer

**Files:**
- Modify: `crates/rc3d-render/src/renderer.rs`

- [ ] **Step 1: Add setters for feature toggles currently written directly**

Add to `impl Renderer`:

```rust
pub fn set_taa(&mut self, enabled: bool) {
    self.enable_taa = enabled;
}

pub fn set_motion_blur(&mut self, enabled: bool) {
    self.enable_motion_blur = enabled;
}

pub fn set_ssr(&mut self, enabled: bool) {
    self.enable_ssr = enabled;
}

pub fn set_color_grading(&mut self, enabled: bool) {
    self.enable_color_grading = enabled;
}

pub fn set_dof(&mut self, enabled: bool) {
    self.enable_dof = enabled;
}

pub fn set_volumetric_fog(&mut self, enabled: bool) {
    self.enable_volumetric_fog = enabled;
}

pub fn set_cluster_lights(&mut self, enabled: bool) {
    self.enable_cluster_lights = enabled;
}

pub fn set_omni_shadows(&mut self, enabled: bool) {
    self.enable_omni_shadows = enabled;
}

pub fn set_grid_enabled(&mut self, enabled: bool) {
    self.grid_enabled = enabled;
}

pub fn set_hud_enabled(&mut self, enabled: bool) {
    self.hud_enabled = enabled;
}

pub fn set_xray_mode(&mut self, enabled: bool) {
    self.xray_mode = enabled;
}

pub fn set_outline_width(&mut self, width: f32) {
    self.outline_width = width;
}
```

- [ ] **Step 2: Compile check**

```bash
rtk cargo check -p rc3d-render
```

Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-render/src/renderer.rs
rtk git commit -m "feat: add single-field setters for all feature toggles on Renderer"
```

---

## Task 4: Add GPU proxy methods and query accessors

**Files:**
- Modify: `crates/rc3d-render/src/renderer.rs`

- [ ] **Step 1: Add GPU proxy methods**

```rust
impl Renderer {
    /// Create a vertex buffer suitable for the renderer's device.
    /// Replaces direct `renderer.device.create_buffer_init()` calls from external crates.
    pub fn create_vertex_buffer<T: bytemuck::Pod>(&self, data: &[T], label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    /// Maximum texture dimension for the current device.
    pub fn device_limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// The surface texture format (for egui renderer init).
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Clone of device handle (for subsystems that need their own ref, e.g. egui_wgpu).
    /// Returns Arc internally — wgpu types are cheap-clone ref-counted.
    pub fn device_clone(&self) -> wgpu::Device {
        self.device.clone()
    }

    /// Clone of queue handle.
    pub fn queue_clone(&self) -> wgpu::Queue {
        self.queue.clone()
    }

    /// Surface dimensions.
    pub fn surface_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}
```

- [ ] **Step 2: Add query accessors**

```rust
impl Renderer {
    pub fn last_frame_stats(&self) -> &FrameStats {
        // FrameStats is computed in render_draw_calls_with_overlay.
        // We need a stored field for this. Add to FrameState if not already.
        // Actually: last_render_stats is currently on App, not Renderer.
        // We'll add a cache field later; for now, return a reference to a stored field.
        // See Task 6 where we add frame_stats to FrameState.
        &self.frame.frame_stats
    }

    pub fn last_diagnostics(&self) -> Option<&FrameDiagnostics> {
        self.frame.last_diagnostics.as_ref()
    }

    pub fn frame_counter(&self) -> u64 {
        self.frame.frame_counter
    }
}
```

- [ ] **Step 3: Compile check**

```bash
rtk cargo check -p rc3d-render
```

Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-render/src/renderer.rs
rtk git commit -m "feat: add GPU proxy methods and query accessors to Renderer"
```

---

## Task 5: Extract FrameState and GpuInternals

**Files:**
- Modify: `crates/rc3d-render/src/renderer.rs`

- [ ] **Step 1: Define FrameState struct (before Renderer)**

```rust
/// Per-frame mutable state accessed by render_passes (pub(crate)).
pub(crate) struct FrameState {
    pub markup_vertices: Vec<crate::vertex::LineVertex>,
    pub clip_planes: Vec<[f32; 4]>,
    pub scene_vp: Mat4,
    pub scene_camera_pos: Vec3,
    pub animation_time_sec: f32,
    pub frame_counter: u64,
    pub performance_mode_active: bool,
    pub last_diagnostics: Option<FrameDiagnostics>,
    pub last_hud_update_frame: u64,
    pub viewport_layout: ViewportLayout,
    pub frame_stats: FrameStats,
}
```

- [ ] **Step 2: Define GpuInternals struct (before Renderer)**

```rust
/// All GPU resources (pub(crate) — accessible by render_passes, hidden from external crates).
pub(crate) struct GpuInternals {
    pub pipelines: PipelineSet,
    pub phong_pool: GpuUniformPool,
    pub flat_pool: GpuUniformPool,
    pub outline_pool: GpuUniformPool,
    pub shadow_pool: GpuUniformPool,
    pub gpu_meshes: GpuResourceManager,
    pub gpu_skinning_pass: Option<GpuSkinningPass>,
    pub skinned_mesh_resources: std::collections::HashMap<crate::gpu_resource::MeshId, GpuSkinningResources>,
    pub depth_texture: Option<(wgpu::Texture, wgpu::TextureView, wgpu::TextureView)>,
    pub assets: GpuAssetManager,
    pub materials: MaterialLibrary,
    pub hud: Option<HudRenderer>,
    pub adaptive_quality: AdaptiveQuality,
    pub adaptive_frame_time_ema_ms: f32,
    pub adaptive_switch_cooldown_frames: u8,
    pub cluster_renderer: Option<ClusterRenderer>,
    pub hzb: Option<HzbPyramids>,
    pub hzb_baker: Option<HzbBaker>,
    pub cluster_pipeline_generation: u32,
    pub depth_reversed_z_mismatch_warned: bool,
    pub texture_cache: TextureCache,
    pub ibl_diffuse: [f32; 4],
    pub ibl_specular: [f32; 4],
    pub shadow_compare_sampler: wgpu::Sampler,
    pub csm_shadow: Option<CsmShadowResources>,
    pub post_fx_pipelines: PostFxPipelines,
    pub post_fx: Option<PostFxTextures>,
    pub ssao_noise_tex: wgpu::Texture,
    pub ssao_noise_view: wgpu::TextureView,
    pub instance_buffer: wgpu::Buffer,
    pub ibl_instance_bind_group: wgpu::BindGroup,
    pub timing_supported: bool,
    pub gpu_query_set: Option<wgpu::QuerySet>,
    pub gpu_query_buffer: Option<wgpu::Buffer>,
    pub gpu_query_slots: u32,
    pub gpu_query_period: f32,
    pub pipeline_cache: Option<PipelineCacheManager>,
    pub shader_cache: ShaderVariantCache,
    pub shader_reload: ShaderHotReload,
    pub auto_exposure: AutoExposure,
    pub taa_pass: Option<TaaPass>,
    pub taa_jitter: TaaJitter,
    pub motion_blur: Option<MotionBlurPass>,
    pub ssr_pass: Option<SsrPass>,
    pub color_grading: Option<ColorGradingPass>,
    pub dof_pass: Option<DofPass>,
    pub volumetric_fog: Option<VolumetricFogPass>,
    pub cluster_lights: Option<ClusterLightResources>,
    pub cluster_light_culler: Option<ClusterLightCuller>,
    pub omni_shadow: Option<OmniShadowRenderer>,
    pub selection_outline_pipelines: Option<crate::selection_outline::SelectionOutlinePipelines>,
    pub selection_outline_targets: Option<crate::selection_outline::SelectionOutlineTargets>,
    pub ldr_shade_tex: Option<wgpu::Texture>,
    pub ldr_shade_view: Option<wgpu::TextureView>,
}
```

- [ ] **Step 3: Refactor Renderer struct**

Replace all the fields being moved into FrameState/GpuInternals with two fields:

```rust
pub struct Renderer {
    // ── One-time config (private) ──
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    wireframe_supported: bool,

    // ── Settings tier (private) ──
    settings: RenderSettings,

    // Feature toggles (migrated to settings, kept as convenience aliases for now)
    // These will be removed in Task 7, but kept for transitional compatibility:
    enable_taa: bool,
    enable_motion_blur: bool,
    enable_ssr: bool,
    enable_color_grading: bool,
    enable_dof: bool,
    enable_volumetric_fog: bool,
    enable_cluster_lights: bool,
    enable_omni_shadows: bool,
    enable_ldr_fxaa: bool,
    hdr_post_processing: bool,
    global_display_mode: DisplayMode,
    grid_enabled: bool,
    hud_enabled: bool,
    outline_width: f32,
    outline_color: [f32; 4],
    xray_mode: bool,
    screen_space_selection_outline: bool,
    ibl_preset: IblPreset,

    // ── Frame state tier (pub(crate)) ──
    frame: FrameState,

    // ── GPU internals tier (pub(crate)) ──
    gpu: GpuInternals,
}
```

- [ ] **Step 4: Add internal accessors**

```rust
impl Renderer {
    pub(crate) fn gpu(&self) -> &GpuInternals { &self.gpu }
    pub(crate) fn gpu_mut(&mut self) -> &mut GpuInternals { &mut self.gpu }
    pub(crate) fn frame(&self) -> &FrameState { &self.frame }
    pub(crate) fn frame_mut(&mut self) -> &mut FrameState { &mut self.frame }
    pub(crate) fn dev(&self) -> &wgpu::Device { &self.device }
    pub(crate) fn que(&self) -> &wgpu::Queue { &self.queue }
    pub(crate) fn surf(&self) -> &wgpu::Surface<'static> { &self.surface }
    pub(crate) fn surface_config_ref(&self) -> &wgpu::SurfaceConfiguration { &self.surface_config }
    pub(crate) fn surface_config_mut(&mut self) -> &mut wgpu::SurfaceConfiguration { &mut self.surface_config }
}
```

- [ ] **Step 5: Update Renderer::new() to initialize FrameState and GpuInternals**

In the constructor, after the existing field initializations, restructure to populate `frame:` and `gpu:` sub-structs from the same values. This is a large mechanical rewrite of the `Self { .. }` literal.

Key mapping (current field → new location):
- `markup_vertices, clip_planes, scene_vp, scene_camera_pos, animation_time_sec, frame_counter, performance_mode_active, last_diagnostics, last_hud_update_frame, viewport_layout` → `frame: FrameState { .. }`
- `pipelines, phong_pool, flat_pool, outline_pool, shadow_pool, gpu_meshes, gpu_skinning_pass, skinned_mesh_resources, depth_texture, assets, materials, hud, adaptive_quality, adaptive_frame_time_ema_ms, adaptive_switch_cooldown_frames, cluster_renderer, hzb, hzb_baker, cluster_pipeline_generation, depth_reversed_z_mismatch_warned, texture_cache, ibl_diffuse, ibl_specular, shadow_compare_sampler, csm_shadow, post_fx_pipelines, post_fx, ssao_noise_tex, ssao_noise_view, instance_buffer, ibl_instance_bind_group, timing_supported, gpu_query_set, gpu_query_buffer, gpu_query_slots, gpu_query_period, pipeline_cache, shader_cache, shader_reload, auto_exposure, taa_pass, taa_jitter, motion_blur, ssr_pass, color_grading, dof_pass, volumetric_fog, cluster_lights, cluster_light_culler, omni_shadow, selection_outline_pipelines, selection_outline_targets, ldr_shade_tex, ldr_shade_view` → `gpu: GpuInternals { .. }`

Also update the post-construction `renderer.hud = Some(...)` etc. to use `renderer.gpu.hud = Some(...)` and similar.

- [ ] **Step 6: Update internal module access patterns (same-crate)**

In `renderer_render.rs`, `renderer_helpers.rs`, `renderer_skinning.rs`, `render_passes.rs`, `selection_outline.rs`:

Mechanically replace:
- `self.field_name` → `self.gpu.field_name` (for GPU fields)
- `self.field_name` → `self.frame.field_name` (for frame fields)
- `self.device` → `self.dev` (use the accessor) — OR keep as `self.device` since it's on Renderer directly
- `renderer.field_name` → `renderer.gpu().field_name` or `renderer.frame().field_name`

For settings-tier fields that remain directly on Renderer (feature toggles), keep as `self.field_name` until Task 7.

- [ ] **Step 7: Compile check**

```bash
rtk cargo check -p rc3d-render
```

Expected: may have errors from missed field renames. Fix each one. Repeat until 0 errors.

- [ ] **Step 8: Run tests**

```bash
rtk cargo test -p rc3d-render
```

Expected: all tests pass (pass_markup tests don't touch Renderer).

- [ ] **Step 9: Commit**

```bash
rtk git add crates/rc3d-render/src/renderer.rs crates/rc3d-render/src/renderer_render.rs crates/rc3d-render/src/renderer_helpers.rs crates/rc3d-render/src/renderer_skinning.rs crates/rc3d-render/src/render_passes.rs crates/rc3d-render/src/selection_outline.rs
rtk git commit -m "refactor: extract FrameState and GpuInternals from Renderer"
```

---

## Task 6: Update external callers — rc3d-gizmo

**Files:**
- Modify: `crates/rc3d-gizmo/src/gizmo.rs`

- [ ] **Step 1: Find all direct renderer.device/queue access in gizmo**

Read `crates/rc3d-gizmo/src/gizmo.rs` and identify where it accesses renderer internals. Based on the grep, no direct `renderer.device` calls exist. But it may use `renderer.flat_pool` or `renderer.pipelines`.

Check and update:
- If `renderer.flat_pool` → needs to go through a `pub(crate)` accessor (but gizmo is a different crate!)
- Solution: expose `pub fn flat_pool(&self) -> &GpuUniformPool` from Renderer (GpuUniformPool is already pub)

Actually, let me check what gizmo actually accesses:

- [ ] **Step 2: If gizmo accesses renderer fields directly, add necessary pub accessors to Renderer**

Based on the grep results showing no matches for `renderer.device`, `renderer.queue`, etc. in gizmo, this task may be a no-op. Verify by reading the file.

- [ ] **Step 3: Compile check**

```bash
rtk cargo check -p rc3d-gizmo
```

Expected: 0 errors.

---

## Task 7: Update external callers — rc3d-app

**Files:**
- Modify: `crates/rc3d-app/src/app/editor_commands.rs`
- Modify: `crates/rc3d-app/src/app/event_handler.rs`
- Modify: `crates/rc3d-app/src/editor_ui/mod.rs`
- Possibly: `crates/rc3d-app/src/app/mod.rs`

- [ ] **Step 1: Update editor_commands.rs — replace direct field writes with setters**

Change all direct field writes to setter calls:

```rust
// Before:
r.enable_taa = enabled;
r.enable_motion_blur = enabled;
r.enable_ssr = enabled;
r.enable_color_grading = enabled;
r.enable_dof = enabled;
r.enable_volumetric_fog = enabled;
r.enable_cluster_lights = enabled;
r.enable_omni_shadows = enabled;
r.hdr_post_processing = enabled;
r.outline_width = w;
r.xray_mode = enabled;
r.hud_enabled = enabled;
r.grid_enabled = enabled;
r.ibl_preset = preset;

// After:
r.set_taa(enabled);
r.set_motion_blur(enabled);
r.set_ssr(enabled);
r.set_color_grading(enabled);
r.set_dof(enabled);
r.set_volumetric_fog(enabled);
r.set_cluster_lights(enabled);
r.set_omni_shadows(enabled);
r.set_hdr_post_processing(enabled);
r.set_outline_width(w);
r.set_xray_mode(enabled);
r.set_hud_enabled(enabled);
r.set_grid_enabled(enabled);
r.set_ibl_preset(preset);
```

Also update:
```rust
// Before:
r.config.width
r.config.height
r.viewport_layout.layout_mode = lm;
r.viewport_layout.rebuild(w, h);

// After:
let (w, h) = r.surface_size();
r.viewport_layout_mut().layout_mode = lm; // OR add set_layout_mode
r.viewport_layout_mut().rebuild(w, h);
```

For `viewport_layout`, since it's now in `frame: FrameState` which is `pub(crate)`, and editor_commands is in `rc3d-app` (a different crate), we need to expose `viewport_layout` access. Add to Renderer in Task 5:

```rust
pub fn viewport_layout(&self) -> &ViewportLayout { &self.frame.viewport_layout }
pub fn viewport_layout_mut(&mut self) -> &mut ViewportLayout { &mut self.frame.viewport_layout }
```

Similarly for `config.width`/`config.height`:

```rust
pub fn surface_width(&self) -> u32 { self.surface_config.width }
pub fn surface_height(&self) -> u32 { self.surface_config.height }
```

- [ ] **Step 2: Update event_handler.rs**

Replace all direct `renderer.field` reads with getter calls:

```rust
// Before:
renderer.enable_taa
renderer.enable_motion_blur
renderer.enable_ssr
renderer.enable_color_grading
renderer.enable_dof
renderer.enable_volumetric_fog
renderer.enable_cluster_lights
renderer.enable_omni_shadows
renderer.xray_mode
renderer.hdr_post_processing
renderer.hud_enabled
renderer.outline_width
renderer.outline_color
renderer.grid_enabled
renderer.ibl_preset
renderer.config.present_mode

// After: use renderer.settings() snapshot
let s = renderer.settings();
s.post_effect.taa
s.post_effect.motion_blur
// ... etc
s.lighting.ibl_preset
s.display.hud_enabled
s.display.outline_width
s.display.outline_color
s.display.grid_enabled
s.display.vsync_enabled
```

```rust
// Before:
renderer.hud_enabled = false;
renderer.grid_enabled = app.grid_enabled;

// After:
renderer.set_hud_enabled(false);
renderer.set_grid_enabled(app.grid_enabled);
```

```rust
// Before:
let w = renderer.config.width;
let h = renderer.config.height;

// After:
let (w, h) = renderer.surface_size();
```

```rust
// Before:
let device = renderer.device.clone();
let queue = renderer.queue.clone();

// After:
let device = renderer.device_clone();
let queue = renderer.queue_clone();
```

```rust
// Before:
renderer.surface.get_current_texture()

// After: add a pub method on Renderer
renderer.acquire_surface_texture()
```

Add to Renderer:
```rust
pub fn acquire_surface_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
    self.surface.get_current_texture()
}
```

- [ ] **Step 3: Update editor_ui/mod.rs**

```rust
// Before:
let max_texture_side = renderer.device.limits().max_texture_dimension_2d as usize;
let format = renderer.config.format;
let egui_renderer = egui_wgpu::Renderer::new(&renderer.device, format, None, 1, false);
// ...
.update_texture(&renderer.device, &renderer.queue, *id, image_delta);

// After:
let max_texture_side = renderer.device_limits().max_texture_dimension_2d as usize;
let format = renderer.surface_format();
let egui_renderer = egui_wgpu::Renderer::new(&renderer.device_clone(), format, None, 1, false);
// ...
.update_texture(&renderer.device_clone(), &renderer.queue_clone(), *id, image_delta);
```

- [ ] **Step 4: Compile check**

```bash
rtk cargo check -p rc3d-app
```

Expected: Fix errors from missed renames. Iterate.

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-app/src/app/editor_commands.rs crates/rc3d-app/src/app/event_handler.rs crates/rc3d-app/src/editor_ui/mod.rs crates/rc3d-render/src/renderer.rs
rtk git commit -m "refactor: update rc3d-app callers to use Renderer setters/getters"
```

---

## Task 8: Make remaining settings fields private

**Files:**
- Modify: `crates/rc3d-render/src/renderer.rs`

- [ ] **Step 1: Change pub to private for feature toggle fields**

The feature toggle fields that now have setters should become private:

```rust
// Change from pub to private:
enable_taa: bool,          // was pub
enable_motion_blur: bool,  // was pub
enable_ssr: bool,          // was pub
enable_color_grading: bool,// was pub
enable_dof: bool,          // was pub
enable_volumetric_fog: bool,// was pub
enable_cluster_lights: bool,// was pub
enable_omni_shadows: bool, // was pub
enable_ldr_fxaa: bool,     // was pub
hdr_post_processing: bool, // was pub
global_display_mode: DisplayMode, // was pub
grid_enabled: bool,        // was pub
hud_enabled: bool,         // was pub
outline_width: f32,        // was pub
xray_mode: bool,           // was pub
screen_space_selection_outline: bool, // was pub
```

`outline_color` stays `pub` for now (used in render passes for the line color uniform).

- [ ] **Step 2: Compile check**

```bash
rtk cargo check --workspace
```

Expected: compiler errors for any remaining direct field access. Fix by using setters. Iterate until 0 errors.

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-render/src/renderer.rs
rtk git commit -m "refactor: make feature toggle fields private on Renderer"
```

---

## Task 9: Update examples

**Files:**
- Modify: `crates/rc3d-app/examples/*.rs` (10 example files)

- [ ] **Step 1: Find all direct renderer field writes in examples**

Read each example file, find patterns like:
```rust
renderer.enable_taa = true;
renderer.grid_enabled = true;
renderer.hdr_post_processing = true;
renderer.global_display_mode = ...;
```

Replace with setter calls:
```rust
renderer.set_taa(true);
renderer.set_grid_enabled(true);
renderer.set_hdr_post_processing(true);
renderer.set_display_mode(...);
```

- [ ] **Step 2: Compile check**

```bash
rtk cargo check --examples -p rc3d-app
```

- [ ] **Step 3: Commit**

```bash
rtk git add crates/rc3d-app/examples/
rtk git commit -m "refactor: update examples to use Renderer setters"
```

---

## Task 10: Split App into sub-modules

**Files:**
- Create: `crates/rc3d-app/src/app/app_state.rs`
- Create: `crates/rc3d-app/src/app/editor_session.rs`
- Create: `crates/rc3d-app/src/app/input_state.rs`
- Create: `crates/rc3d-app/src/app/lod_state.rs`
- Modify: `crates/rc3d-app/src/app/mod.rs`

- [ ] **Step 1: Create app_state.rs**

```rust
// crates/rc3d-app/src/app/app_state.rs
use std::time::Instant;
use rc3d_core::DisplayMode;
use rc3d_render::{FrameStats, Renderer};
use crate::adaptive_quality::AdaptiveQualityMode;
use crate::camera_controller::CameraController;
use crate::fps_tracker::FpsTracker;
use crate::viewport_camera::ViewportCameraSet;
use crate::world::World;

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

- [ ] **Step 2: Create editor_session.rs**

```rust
// crates/rc3d-app/src/app/editor_session.rs
use rc3d_actions::{CommandHistory, MarkupAction};
use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_gizmo::Gizmo;
use rc3d_scene::node_data::MeasurementType;
use crate::editor_ui::EditorUi;

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

Note: `Vec3` import from `rc3d_core::math::Vec3`.

- [ ] **Step 3: Create input_state.rs**

```rust
// crates/rc3d-app/src/app/input_state.rs
pub struct InputState {
    pub cursor_pos: (f64, f64),
    pub shift_pressed: bool,
    pub ctrl_pressed: bool,
}
```

- [ ] **Step 4: Create lod_state.rs**

```rust
// crates/rc3d-app/src/app/lod_state.rs
use std::time::Instant;
use crate::streaming_lod::FullResPatch;

pub struct LODState {
    pub full_res_patches: Vec<FullResPatch>,
    pub preview_mode_active: bool,
    pub stream_next_tick: Option<Instant>,
}
```

- [ ] **Step 5: Refactor App struct in mod.rs**

Replace the large App struct with:

```rust
pub struct App {
    pub state: AppState,
    pub editor: EditorSession,
    pub input: InputState,
    pub lod: LODState,
    // Cross-cutting hooks
    pub on_pick: Option<PickCallback>,
    pub pending_graph_rx: Option<std::sync::mpsc::Receiver<Result<SceneGraph, String>>>,
    pub graph_load_hook: Option<Box<dyn FnOnce(&mut App) + 'static>>,
    pub panel_overlay_text_hook: Option<Box<dyn Fn() -> String>>,
    pub panel_overlay_key_hook: Option<Box<dyn FnMut(winit::keyboard::KeyCode)>>,
    pub panel_overlay_mouse_hook: Option<Box<dyn FnMut(f32, f32, u32, u32) -> bool>>,
}
```

Update all `self.field_name` references in mod.rs, event_handler.rs, editor_commands.rs, and other app/ modules to use the new sub-struct paths:
- `self.world` → `self.state.world`
- `app.renderer` → `app.state.renderer`
- `app.gizmo` → `app.editor.gizmo`
- `app.command_history` → `app.editor.command_history`
- `app.markup_action` → `app.editor.markup_action`
- `app.cursor_pos` → `app.input.cursor_pos`
- `app.shift_pressed` → `app.input.shift_pressed`
- `app.ctrl_pressed` → `app.input.ctrl_pressed`
- `app.full_res_patches` → `app.lod.full_res_patches`
- etc.

- [ ] **Step 6: Compile check**

```bash
rtk cargo check -p rc3d-app
```

Expected: many errors from missed renames. Fix systematically. Iterate until 0 errors.

- [ ] **Step 7: Run all tests**

```bash
rtk cargo test
```

Expected: all tests pass (113+).

- [ ] **Step 8: Commit**

```bash
rtk git add crates/rc3d-app/src/app/
rtk git commit -m "refactor: split App into AppState/EditorSession/InputState/LODState"
```

---

## Task 11: Final integration verification

**Files:** None (verification only)

- [ ] **Step 1: Full workspace check**

```bash
rtk cargo check --workspace
```

Expected: 0 errors, 0 warnings (except pre-existing dead_code/unused).

- [ ] **Step 2: Full test suite**

```bash
rtk cargo test
```

Expected: all tests pass.

- [ ] **Step 3: Run editor example**

```bash
rtk cargo run -p rc3d-app --example editor
```

Expected: editor opens, viewport renders, UI panels show, render toggles work (HDR, TAA, grid, HUD, etc.), gizmo works, markup tools work.

- [ ] **Step 4: Commit**

```bash
rtk git add -A
rtk git commit -m "chore: final integration verification — Phase 1 encapsulation refactor complete"
```

---

## Summary

| Task | Description | Files | Estimated LoC change |
|------|-------------|-------|---------------------|
| 1 | Create RenderSettings | 2 new | +120 |
| 2 | Add settings field + apply/settings() | renderer.rs | +90 |
| 3 | Add single-field setters | renderer.rs | +60 |
| 4 | Add GPU proxy + query methods | renderer.rs | +50 |
| 5 | Extract FrameState/GpuInternals | renderer.rs + 5 internal | +80 / -80 refactor |
| 6 | Update gizmo callers | gizmo.rs | ~0 (verify) |
| 7 | Update app callers (setters) | 3 files in app | +30 / -30 |
| 8 | Make fields private | renderer.rs | +5 / -5 |
| 9 | Update examples | ~10 files | +30 / -30 |
| 10 | Split App | 4 new + 3 modify | +120 / -60 |
| 11 | Final verification | none | 0 |

Total: ~11 commits, ~665 lines new, ~205 lines removed.
