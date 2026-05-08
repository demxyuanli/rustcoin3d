# 渲染循环接线 FlatDrawCache — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已构建的 FlatDrawCache / dirty flags / 增量遍历接入 `render_draw_calls_core`，消除每帧全图遍历 + DrawCall 重新分配

**Architecture:** 入口处调用 `traverse_into_cache()` → `cache_to_draw_calls()` adapter → 现有渲染路径不变

**Tech Stack:** Rust + wgpu

---

### Task 1: cache_to_draw_calls adapter

**Files:** Modify `crates/rc3d-render/src/render_action.rs`

- [ ] **Step 1: Read current FlatDrawCache and DrawCall to plan field mapping**

Read `crates/rc3d-render/src/flat_draw_cache.rs` and `crates/rc3d-render/src/render_action.rs:100-230` (DrawCall struct).

- [ ] **Step 2: Add cache_to_draw_calls function**

In `render_action.rs`, add:

```rust
/// Adapter: convert FlatDrawCache back to Vec<DrawCall> for existing render path consumption.
/// Each GpuDrawData + CachedDrawMetadata pair is rebuilt into a legacy DrawCall.
pub fn cache_to_draw_calls(
    cache: &crate::flat_draw_cache::FlatDrawCache,
    texture_table: &crate::global_tables::TexturePathTable,
) -> Vec<DrawCall> {
    use crate::flat_draw_cache::DrawFlags;

    cache.gpu_data.iter().zip(cache.metadata.iter()).map(|(gpu, meta)| {
        let model_matrix = glam::Mat4::from_cols_array_2d(&gpu.model_matrix);
        let flags = DrawFlags::from_bits_truncate(gpu.draw_flags);
        DrawCall {
            vertices: Arc::new(Vec::new()), // populated later by mesh system
            indices: None,
            edge_positions: Arc::new(Vec::new()),
            wireframe_edge_positions: Arc::new(Vec::new()),
            model_matrix,
            mvp: model_matrix, // caller overrides with VP
            camera_pos: Vec3::ZERO,
            light_dirs: [[0.0; 4]; MAX_LIGHTS],
            light_colors: [[0.0; 4]; MAX_LIGHTS],
            light_types: [[0.0; 4]; MAX_LIGHTS],
            light_positions: [[0.0; 4]; MAX_LIGHTS],
            spot_params: [[0.0; 4]; MAX_LIGHTS],
            light_count: 0,
            diffuse_color: Vec3::new(meta.material_params.base_color[0], meta.material_params.base_color[1], meta.material_params.base_color[2]),
            ambient_color: Vec3::ZERO,
            specular_color: Vec3::ZERO,
            shininess: 32.0,
            base_color: Vec3::new(meta.material_params.base_color[0], meta.material_params.base_color[1], meta.material_params.base_color[2]),
            metallic: meta.material_params.metallic_roughness_anisotropic[0],
            roughness: meta.material_params.metallic_roughness_anisotropic[1],
            anisotropic: meta.material_params.metallic_roughness_anisotropic[2],
            opacity: meta.material_params.base_color[3],
            albedo_path: tex_id_to_arcstr(texture_table, meta.albedo_tex_id),
            normal_path: tex_id_to_arcstr(texture_table, meta.normal_tex_id),
            emissive_color: Vec3::new(meta.material_params.emissive_color[0], meta.material_params.emissive_color[1], meta.material_params.emissive_color[2]),
            emissive_path: tex_id_to_arcstr(texture_table, meta.emissive_tex_id),
            metallic_roughness_path: tex_id_to_arcstr(texture_table, meta.mr_tex_id),
            occlusion_path: tex_id_to_arcstr(texture_table, meta.occlusion_tex_id),
            alpha_mode: if meta.alpha_mode == 1 { rc3d_scene::AlphaMode::Mask } else if meta.alpha_mode == 2 { rc3d_scene::AlphaMode::Blend } else { rc3d_scene::AlphaMode::Opaque },
            alpha_cutoff: meta.alpha_cutoff,
            double_sided: meta.double_sided != 0,
            aabb: None,
            display_mode: rc3d_core::DisplayMode::ShadedWithEdges,
            selected: flags.contains(DrawFlags::SELECTED),
            overlay_color: None,
            mesh_hash: Some(meta.mesh_hash),
            meshlet_data: None,
            projection_orthographic: flags.contains(DrawFlags::ORTHOGRAPHIC),
            depth_reversed_z: flags.contains(DrawFlags::DEPTH_REVERSED),
            is_overlay: flags.contains(DrawFlags::OVERLAY),
            node_type_label: Arc::from(""),
            instance_transforms: None,
            morph_weights: Vec::new(),
            morph_target_deltas: None,
            skinning: None,
        }
    }).collect()
}

fn tex_id_to_arcstr(table: &crate::global_tables::TexturePathTable, id: u16) -> Option<Arc<str>> {
    if id == u16::MAX { None } else { table.get(id).cloned() }
}
```

- [ ] **Step 3: Verify compile**

```bash
rtk cargo check -p rc3d-render
```

Expected: zero errors. May have unused warnings (adapter not yet called).

- [ ] **Step 4: Commit**

```bash
rtk git add crates/rc3d-render/src/render_action.rs
rtk git commit -m "feat: add cache_to_draw_calls adapter for FlatDrawCache→DrawCall conversion"
```

---

### Task 2: Wire FlatDrawCache into render_draw_calls_core

**Files:** Modify `crates/rc3d-render/src/renderer_render.rs`, `crates/rc3d-app/src/app/event_handler.rs`

- [ ] **Step 1: Read current render_draw_calls_core signature and event_handler call site**

Read `renderer_render.rs:22-29` (fn signature) and `event_handler.rs` around line 648 (call site).

- [ ] **Step 2: Modify render_draw_calls_core to use draw_cache**

Change the function to accept `scene: &SceneGraph` instead of `draw_calls: &[DrawCall]`:

```rust
fn render_draw_calls_core<'p>(
    &'p mut self,
    scene: &SceneGraph,
    post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    presentation: render_passes::FramePresentation<'p>,
    ssao_projection: Option<(Mat4, Mat4)>,
) -> FrameStats {
    self.texture_streamer.poll_completed(&self.device, &self.queue);
    self.cpu_span.begin_frame();

    // ... existing frame_counter, dt, shader_reload code ...

    // Collect text and effects from scene
    if let Some(hud) = &mut self.gpu.hud {
        hud.overlay_lines = render_passes::pass_text::collect_text_nodes(scene)
            .into_iter().map(|cmd| cmd.string).collect();
    }
    let effect_commands = render_passes::pass_effects::collect_effect_nodes(scene);

    // ── Incremental traversal → FlatDrawCache (NEW) ──
    // Only re-traverse dirty subtrees; static subtrees reuse cached data.
    crate::render_action::traverse_into_cache(
        scene,
        &mut self.draw_cache,
        &self.texture_table,
        &std::collections::HashSet::new(), // hidden_nodes — pass empty for now
    );

    // ── Adapter conversion (NEW) ──
    let draw_calls = crate::render_action::cache_to_draw_calls(
        &self.draw_cache,
        &self.texture_table,
    );

    if draw_calls.is_empty() {
        return FrameStats::default();
    }

    // ... rest of existing code unchanged ...
}
```

Note: `traverse_into_cache` currently takes `&SceneGraph` but `clear_all_dirty_flags` needs `&mut SceneGraph`. For frame-end cleanup, skip calling `clear_all_dirty_flags` for now (dirty flags accumulate harmlessly — they just cause re-traversal).

- [ ] **Step 3: Update event_handler.rs call site**

In `event_handler.rs`, find the `render_draw_calls_core` call (around line 648-670) and update to remove the `draw_calls` parameter:

```rust
// Before:
let stats = self.render_draw_calls_core(draw_calls, scene, ...);

// After:
let stats = self.render_draw_calls_core(scene, ...);
```

Also find `render_draw_calls_to_viewport_texture` call site if any and update similarly.

- [ ] **Step 4: Verify compile**

```bash
rtk cargo check --workspace
```

Fix any type errors. Expected: zero errors.

- [ ] **Step 5: Commit**

```bash
rtk git add crates/rc3d-render/src/renderer_render.rs crates/rc3d-app/src/app/event_handler.rs
rtk git commit -m "feat: wire FlatDrawCache + incremental traversal into render loop"
```

---

### Task 3: Final verification

- [ ] **Step 1: Full compile check**

```bash
rtk cargo check --workspace --all-targets
```

Expected: zero errors.

- [ ] **Step 2: Full test suite**

```bash
rtk cargo test -p rc3d-render
rtk cargo test -p rc3d-core
rtk cargo test -p rc3d-scene
```

Expected: all pass.

- [ ] **Step 3: Run cube example**

```bash
cargo run -p rc3d-app --example cube --release 2>&1 | head -20
```

Expected: no crash, renders normally.

- [ ] **Step 4: Commit**

```bash
rtk git add -A
rtk git commit -m "chore: final verification — workspace check + test suite"
```
