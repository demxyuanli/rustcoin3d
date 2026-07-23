//! Pass orchestration: sort/batch, CSM setup, PassContext, execute_passes, diagnostics.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use slotmap::Key;
use rc3d_core::DisplayMode;
use rc3d_scene::AlphaMode;
use rc3d_scene::SceneGraph;
use crate::adaptive_quality::AdaptiveQuality;
use crate::cluster::{ClusterRenderer, ClusterSet};
use crate::render_action::DrawCall;
use crate::render_passes;
use crate::render_passes::PassContext;
use crate::render_passes::pass_effects::EffectCommands;
use crate::shadow_map::{aabb_from_scene, csm_light_view_projs, compute_csm_splits, primary_directional_light_dir, union_draw_call_aabbs};
use crate::sort_keys;
use crate::vertex::CSM_CASCADE_COUNT;
use super::internals::CadDisplayTier;
use super::types::FrameStats;

impl super::Renderer {
    pub(crate) fn orchestrate_frame_passes<'p>(
        &'p mut self,
        visible: Vec<&DrawCall>,
        mesh_handles: Vec<Option<crate::gpu_resource::MeshId>>,
        draw_calls: &[DrawCall],
        first: &DrawCall,
        scene: &SceneGraph,
        post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
        presentation: render_passes::FramePresentation<'p>,
        ssao_projection: Option<(Mat4, Mat4)>,
        effect_commands: EffectCommands,
        prev_vp: Mat4,
    ) -> FrameStats {
        // Use pre-computed light_key from DrawCall (computed during traversal).
        let mut light_hashes = std::mem::take(&mut self.frame.light_hashes_buf);
        light_hashes.clear();
        light_hashes.extend(visible.iter().map(|dc| dc.light_key));

        self.frame.scene_camera_pos = draw_calls
            .first()
            .map(|dc| dc.camera_pos)
            .unwrap_or(Vec3::ZERO);

        let (mut solid_order, mut edge_order, mut selected_order) = if self.frame.bvh_fully_static && self.frame.static_frame_count >= 2 {
            (std::mem::take(&mut self.frame.solid_order_buf),
             std::mem::take(&mut self.frame.edge_order_buf),
             std::mem::take(&mut self.frame.selected_order_buf))
        } else {
            self.cpu_span.measure("sorting", || {
                let mut solid_order = std::mem::take(&mut self.frame.solid_order_buf);
                solid_order.clear();
                solid_order.extend((0..visible.len())
                    .filter(|&i| !visible[i].vertices.is_empty() || visible[i].meshlet_data.is_some()));
                solid_order.sort_unstable_by_key(|&i| light_hashes[i]);
                let mut edge_order = std::mem::take(&mut self.frame.edge_order_buf);
                edge_order.clear();
                edge_order.extend((0..visible.len())
                    .filter(|&i| !visible[i].edge_positions.is_empty()));
                edge_order.sort_unstable_by_key(|&i| {
                    let dc = visible[i];
                    (
                        sort_keys::display_mode_sort_key(dc.display_mode),
                        sort_keys::color_sort_key(dc.overlay_color.unwrap_or([0.0, 0.0, 0.0, 0.5])),
                        mesh_handles[i].map(|m| m.data().as_ffi()).unwrap_or(0),
                    )
                });
                let mut selected_order = std::mem::take(&mut self.frame.selected_order_buf);
                selected_order.clear();
                selected_order.extend((0..visible.len())
                    .filter(|&i| visible[i].selected && (!visible[i].vertices.is_empty() || visible[i].meshlet_data.is_some())));
                selected_order.sort_unstable_by_key(|&i| {
                    let dc = visible[i];
                    (
                        sort_keys::display_mode_sort_key(dc.display_mode),
                        mesh_handles[i].map(|m| m.data().as_ffi()).unwrap_or(0),
                    )
                });
                (solid_order, edge_order, selected_order)
            })
        };
        let mut transparent_order: Vec<usize> = Vec::new();
        transparent_order.extend((0..visible.len())
            .filter(|&i| visible[i].alpha_mode != AlphaMode::Opaque
                && (!visible[i].vertices.is_empty() || visible[i].meshlet_data.is_some())));
        // Remove transparent objects from solid order (avoids double-draw)
        solid_order.retain(|&i| visible[i].alpha_mode == AlphaMode::Opaque
            || visible[i].vertices.is_empty() && visible[i].meshlet_data.is_none());

        let mut meshlet_indices = std::mem::take(&mut self.frame.meshlet_indices_buf);
        meshlet_indices.clear();
        for (i, dc) in visible.iter().enumerate() {
            if dc.meshlet_data.is_some() {
                meshlet_indices.push(i);
            }
        }
        if !meshlet_indices.is_empty() {
            if self.gpu.cluster_pipeline_generation != super::CLUSTER_PIPELINE_GENERATION {
                self.gpu.cluster_renderer = None;
                self.gpu.cluster_pipeline_generation = super::CLUSTER_PIPELINE_GENERATION;
            }
            if self.gpu.cluster_renderer.is_none() {
                self.gpu.cluster_renderer = Some(ClusterRenderer::new(&self.device));
            }
        }

        let bgls = self.gpu.cluster_renderer.as_ref().map(|cr| cr.bind_group_layouts());
        for &idx in &meshlet_indices {
            let dc = visible[idx];
            let md = dc.meshlet_data.as_ref().unwrap();
            let ptr = Arc::as_ptr(md) as u64;
            if !self.gpu.assets.cluster_contains(&ptr) {
                if let Some((cs_bgl, cmp_bgl, fin_bgl)) = bgls {
                    let cs = ClusterSet::from_meshlet_data(
                        &self.device, md, cs_bgl, cmp_bgl, fin_bgl,
                    );
                    self.gpu.assets.cluster_insert(ptr, cs);
                }
            }
        }

        let base_mode = if self.frame.performance_mode_active {
            match self.global_display_mode {
                DisplayMode::HiddenLine | DisplayMode::Wireframe => DisplayMode::Shaded,
                // DesignCreation tier uses FlatWithEdge — degrade to Flat, not Shaded,
                // to preserve flat-shading appearance during interaction.
                // Flat-shading tiers (DesignCreation): keep FlatWithEdge as-is.
                // Flat rendering is already fast — removing edges doesn't help.
                DisplayMode::FlatWithEdge => {
                    
                    if self.gpu.effective_tier == CadDisplayTier::DesignCreation {
                        DisplayMode::FlatWithEdge
                    } else {
                        DisplayMode::Shaded
                    }
                }
                _ => self.global_display_mode,
            }
        } else {
            self.global_display_mode
        };
        let mode = if self.gpu.adaptive_quality == AdaptiveQuality::Low && base_mode == DisplayMode::Wireframe {
            DisplayMode::Shaded
        } else {
            base_mode
        };
        let run_outline = !self.frame.performance_mode_active
            && self.gpu.adaptive_quality == AdaptiveQuality::High
            && self.tier_wants_edges
            && (mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine);

        let solid_wants_shadow = !self.frame.performance_mode_active
            && matches!(
                mode,
                DisplayMode::Shaded | DisplayMode::ShadedWithEdges | DisplayMode::HiddenLine
            )
            && mode != DisplayMode::Flat
            && mode != DisplayMode::FlatWithEdge
            && self.tier_wants_shadow;

        let (camera_proj, camera_inv_proj) = if let Some((p, ip)) = ssao_projection {
            (p, ip)
        } else {
            let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
            let near = 0.1f32;
            let far = 1000.0f32;
            let fov = 60.0f32.to_radians();
            let f = 1.0 / (fov / 2.0).tan();
            let camera_proj = Mat4::from_cols_array_2d(&[
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, far / (far - near), 1.0],
                [0.0, 0.0, -near * far / (far - near), 0.0],
            ]);
            let camera_inv_proj = camera_proj.inverse();
            (camera_proj, camera_inv_proj)
        };

        let mut csm_view_proj = [Mat4::IDENTITY; CSM_CASCADE_COUNT];
        let mut csm_split_depths = [0.0f32; CSM_CASCADE_COUNT];
        let mut shadow_params = [0.0_f32, 0.0004, 0.0, 0.0];
        let mut run_shadow_pass = false;

        if solid_wants_shadow {
            #[cfg(feature = "profiler")]
            let _span_csm = tracy_client::span!("csm_setup");
            if let Some(dir) = primary_directional_light_dir(scene) {
                let aabb = aabb_from_scene(scene).or_else(|| union_draw_call_aabbs(visible.iter().copied()));
                if let Some(_aabb) = aabb {
                    let sm_size = match self.gpu.adaptive_quality {
                        AdaptiveQuality::High => 2048,
                        AdaptiveQuality::Medium => 1024,
                        AdaptiveQuality::Low => 512,
                    };
                    let cascade_count = match self.gpu.adaptive_quality {
                        AdaptiveQuality::High => 4,
                        AdaptiveQuality::Medium => 3,
                        AdaptiveQuality::Low => 1,
                    };
                    self.set_csm_shadow(sm_size, cascade_count);

                    let vp = first.mvp * first.model_matrix.inverse();
                    let camera_near = 0.1f32;
                    let camera_far = 1000.0f32;
                    let splits = compute_csm_splits(camera_near, camera_far, cascade_count, 0.5);
                    let inv_vp = vp.inverse();
                    let light_vps = csm_light_view_projs(dir, inv_vp, &splits, 8.0);

                    for (i, lvp) in light_vps.iter().enumerate() {
                        if i < CSM_CASCADE_COUNT {
                            csm_view_proj[i] = *lvp;
                        }
                    }
                    // Shader compares csm_split_depths against view-space depth
                    // (abs(view_pos.z), in world units) — write the splits as
                    // view-space distances, NOT normalized [0,1] values.
                    // Slots at/beyond the last real cascade get a huge sentinel
                    // so select_cascade_blended() clamps to the last valid
                    // cascade instead of falling through to an IDENTITY one.
                    const SPLIT_SENTINEL: f32 = 1.0e30;
                    for i in 0..cascade_count.saturating_sub(1) as usize {
                        if i + 1 < splits.len() {
                            csm_split_depths[i] = splits[i + 1];
                        }
                    }
                    for s in &mut csm_split_depths[cascade_count.saturating_sub(1) as usize..CSM_CASCADE_COUNT] {
                        *s = SPLIT_SENTINEL;
                    }

                    let inv = 1.0 / sm_size as f32;
                    let (bias, pcf) = match self.gpu.adaptive_quality {
                        AdaptiveQuality::High => (0.00015_f32, 2.0_f32),
                        AdaptiveQuality::Medium => (0.00028, 1.0),
                        AdaptiveQuality::Low => (0.00045, 0.0),
                    };
                    shadow_params = [inv, bias, pcf, 1.0];
                    run_shadow_pass = true;
                }
            }
        }

        // Upload global frame uniforms (lights, CSM, IBL) once per frame.
        if let Some(ref buf) = self.gpu.global_frame_buffer {
            let primary = *self.light_sets.get(0);
            let global = crate::vertex::GlobalFrameUniforms {
                light_dirs: primary.0,
                light_colors: primary.1,
                light_types: primary.2,
                light_positions: primary.3,
                spot_params: primary.4,
                light_count: [primary.5 as f32, 0.0, 0.0, 0.0],
                ibl_diffuse: self.gpu.ibl_diffuse,
                ibl_specular: self.gpu.ibl_specular,
                csm_view_proj: crate::render_passes::draw_opaque::csm_to_uniform(&csm_view_proj),
                csm_split_depths,
                shadow_params,
            };
            self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&global));
        }

        // Clone the light-set table (typically 1-5 entries) so ctx doesn't borrow self.
        let light_sets_snapshot = self.light_sets.clone();
        let ctx = PassContext {
            visible: &visible,
            solid_order: &solid_order,
            edge_order: &edge_order,
            selected_order: &selected_order,
            transparent_order: &transparent_order,
            mesh_handles: &mesh_handles,
            mode,
            run_outline,
            bg_color: wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 },
            performance_mode_active: self.frame.performance_mode_active,
            wireframe_supported: self.wireframe_supported,
            adaptive_quality: self.gpu.adaptive_quality,
            outline_width: self.outline_width,
            outline_color: self.outline_color,
            meshlet_indices: &meshlet_indices,
            camera_pos: [first.camera_pos.x, first.camera_pos.y, first.camera_pos.z],
            depth_reversed_z: first.depth_reversed_z,
            csm_view_proj,
            run_shadow_pass,
            camera_proj,
            camera_inv_proj,
            scene_vp: self.frame.scene_vp,
            prev_vp,
            effect_commands: &effect_commands,
            light_sets: &light_sets_snapshot,
        };

        let mut stats = render_passes::execute_passes(
            self,
            &ctx,
            draw_calls,
            self.frame.frame_counter,
            post_swapchain_overlay,
            presentation,
        );
        // ctx is dropped here — restore reusable Vecs to FrameState
        self.frame.light_hashes_buf = std::mem::take(&mut light_hashes);
        self.frame.meshlet_indices_buf = std::mem::take(&mut meshlet_indices);
        self.frame.solid_order_buf = std::mem::take(&mut solid_order);
        self.frame.edge_order_buf = std::mem::take(&mut edge_order);
        self.frame.selected_order_buf = std::mem::take(&mut selected_order);
        self.frame.transparent_order_buf = std::mem::take(&mut transparent_order);

        let gpu_pass = {
            let timestamps = &self.gpu_timer.last_timestamps;
            let labels = &self.gpu_timer.labels;
            let period_ns = self.gpu_timer.timestamp_period_ns as f64;
            let mut shadow = 0.0f64;
            let mut solid = 0.0f64;
            let mut post = 0.0f64;
            let mut total = 0.0f64;
            for i in (0..timestamps.len()).step_by(2) {
                if i + 1 < timestamps.len() {
                    let dur_us =
                        (timestamps[i + 1].saturating_sub(timestamps[i]) as f64) * period_ns
                            / 1_000.0;
                    total += dur_us;
                    let label = labels.get(i / 2).copied().unwrap_or("?");
                    match label {
                        "CSM Shadow" => shadow = dur_us,
                        "Solid+Outline" => solid = dur_us,
                        "PostProcess" => post = dur_us,
                        _ => {}
                    }
                }
            }
            if timestamps.len() >= 2 {
                Some([shadow, solid, post, total])
            } else {
                None
            }
        };
        stats.gpu_pass_times_us = gpu_pass;
        if self.frame.frame_counter % 10 == 0 {
            if let Some([shadow, solid, post, total]) = gpu_pass {
                log::debug!(
                    "GPU pass timings (us): shadow={:.0} solid={:.0} post={:.0} total={:.0}",
                    shadow, solid, post, total
                );
            }
        }
        // NOTE: solid_order/transparent_order were moved into self.frame above;
        // read them from there (the locals are empty after mem::take).
        let diagnostics = self.build_frame_diagnostics(
            &visible,
            &mesh_handles,
            &self.frame.solid_order_buf,
            &self.frame.transparent_order_buf,
            gpu_pass,
        );
        stats.diagnostics = Some(diagnostics.clone());
        self.frame.last_diagnostics = Some(diagnostics);
        stats

    }
}
