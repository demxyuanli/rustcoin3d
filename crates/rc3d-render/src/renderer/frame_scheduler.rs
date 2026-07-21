//! Frame scheduler: per-frame begin, visibility (BVH/GPU cull), performance mode.
//! Delegates mesh upload to `resource_lifecycle` and pass execution to `pass_orchestration`.

use glam::Mat4;
use rc3d_scene::SceneGraph;
use crate::frustum::Frustum;
use crate::render_action::DrawCall;
use crate::render_passes;

use super::types::FrameStats;
use super::PERFORMANCE_MODE_TRIANGLE_THRESHOLD;

impl super::Renderer {
    pub(crate) fn render_draw_calls_core<'p>(
        &'p mut self,
        draw_calls: &[DrawCall],
        scene: &SceneGraph,
        post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
        presentation: render_passes::FramePresentation<'p>,
        ssao_projection: Option<(Mat4, Mat4)>,
    ) -> FrameStats {
        // Poll completed texture streaming loads from background threads
        self.texture_streamer.poll_completed(&self.device, &self.queue);

        self.cpu_span.begin_frame();

        self.begin_frame_gpu_resources();

        // ── GPU cull: read back previous frame's instance count (non-blocking) ──
        let mut gpu_visible: Option<Vec<usize>> = None;
        if self.gpu.gpu_cull_enabled
            && self.frame.frame_counter > 2
            && self.frame.gpu_cull_ready
        {
            if let Some(ref staging) = self.gpu.gpu_cull_staging {
                let buf_slice = staging.slice(..);
                let mapping_ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let ready_clone = std::sync::Arc::clone(&mapping_ready);
                buf_slice.map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        ready_clone.store(true, std::sync::atomic::Ordering::Release);
                    }
                });
                self.device.poll(wgpu::Maintain::Poll);
                if mapping_ready.load(std::sync::atomic::Ordering::Acquire) {
                    let mapped = buf_slice.get_mapped_range();
                    let byte_count = mapped.len();
                    if byte_count >= 4 {
                        let count = u32::from_le_bytes([mapped[0], mapped[1], mapped[2], mapped[3]]);
                        let count = (count as usize).min(draw_calls.len());
                        let mut indices = Vec::with_capacity(count);
                        for i in 0..count {
                            let off = 4 + i * 4;
                            if off + 4 <= byte_count {
                                let idx = u32::from_le_bytes([
                                    mapped[off], mapped[off+1], mapped[off+2], mapped[off+3]
                                ]) as usize;
                                if idx < draw_calls.len() {
                                    indices.push(idx);
                                }
                            }
                        }
                        if !indices.is_empty() {
                            log::debug!("GPU cull: {} visible (CPU: {})",
                                indices.len(), self.frame.visible_indices.len());
                            gpu_visible = Some(indices);
                        }
                    }
                    drop(mapped);
                    staging.unmap();
                }
                self.frame.gpu_cull_ready = false;
            }
        }

        // ── Parallel traversal (when enabled) before cache update ──
        if self.frame.parallel_traversal_enabled {
            crate::parallel_traversal::parallel_traverse_into_cache(
                scene,
                &mut self.draw_cache,
                &self.texture_table,
                &std::collections::HashSet::new(),
            );
        }

        // ── Populate FlatDrawCache as side effect (for future incremental traversal) ──
        self.cpu_span.measure("cache_update", || {
            crate::render_action::populate_cache_from_draw_calls(
                &mut self.draw_cache,
                draw_calls,
                &mut self.texture_table,
            );
        });
        self.frame.frame_counter = self.frame.frame_counter.wrapping_add(1);
        // Resolve interaction degradation/recovery before computing passes so shadow/outline
        // and tier-driven flags match this frame's PassContext.
        self.update_tier();
        if self.cad_tier_authoritative {
            self.reapply_cad_tier_constraints();
        }
        let dt_sec = (self.gpu.adaptive_frame_time_ema_ms / 1000.0).clamp(0.0, 0.25);
        self.frame.animation_time_sec += dt_sec;

        let _changed = self.gpu.shader_reload.check_and_reload();

        if self.frame.frame_counter % 300 == 0 {
            if let Some(ref mut pc) = self.gpu.pipeline_cache {
                pc.save_to_disk();
            }
            if let Some(ref pool) = self.gpu.assets.mesh_pool {
                log::debug!(
                    "MeshPool: {}/{} slots, {} MB / {} MB, {} uploads/frame",
                    pool.len(), crate::mesh_pool::DEFAULT_POOL_SIZE,
                    pool.total_bytes() / (1024 * 1024),
                    crate::mesh_pool::DEFAULT_MAX_POOL_BYTES / (1024 * 1024),
                    pool.uploads_this_frame(),
                );
            }
        }

        // Compute VP + camera_pos early for Text3 world labels
        let text_vp = draw_calls.first().map(|dc| dc.mvp * dc.model_matrix.inverse());
        let text_cam = draw_calls.first().map(|dc| dc.camera_pos);
        let text_viewport = self.gpu.hud.as_ref().map(|h| (h.width, h.height));
        self.frame.annotation_world_labels.clear();

        // Fast-path: auto-learn whether scene has text/effect nodes.
        // First 2 frames always traverse to detect; after that, skip if empty.
        if let Some(hud) = &mut self.gpu.hud {
            if self.frame.has_text_nodes || self.frame.frame_counter < 2 {
                let world_labels = &mut self.frame.annotation_world_labels;
                let depth_rz = self.frame.scene_depth_reversed_z;
                let t = self.cpu_span.measure("text_collect", || {
                    render_passes::pass_text::collect_text_nodes(
                        scene,
                        text_vp,
                        text_viewport,
                        text_cam,
                        depth_rz,
                        world_labels,
                    )
                });
                hud.overlay_lines = t.overlay_lines;
                hud.scene_positioned_texts = t.positioned;
                if hud.overlay_lines.is_empty()
                    && hud.scene_positioned_texts.is_empty()
                    && self.frame.annotation_world_labels.is_empty()
                    && self.frame.frame_counter >= 2
                {
                    self.frame.has_text_nodes = false;
                }
            } else {
                hud.overlay_lines = Vec::new();
                hud.scene_positioned_texts.clear();
            };
        }
        let mut effect_commands = std::mem::take(&mut self.frame.effect_commands);
        if effect_commands.is_empty()
            && (self.frame.has_effect_nodes || self.frame.frame_counter < 2)
        {
            effect_commands = self.cpu_span.measure("effect_collect", || {
                render_passes::pass_effects::collect_effect_nodes(scene)
            });
            if effect_commands.is_empty() && self.frame.frame_counter >= 2 {
                self.frame.has_effect_nodes = false;
            }
        }

        if draw_calls.is_empty() {
            return render_passes::render_overlay_only_frame(
                self,
                presentation,
                post_swapchain_overlay,
                self.frame.frame_counter,
                &effect_commands,
            );
        }

        let first = &draw_calls[0];
        // Always prefer VP from current draw calls (updated each frame via apply_world_camera).
        // Cached frame.scene_vp must not win — it freezes annotations/grid when the camera moves.
        let vp_from_draws = crate::render_action::view_projection_from_draw_call(first);
        let vp = if vp_from_draws != Mat4::IDENTITY {
            vp_from_draws
        } else {
            self.frame.scene_vp
        };
        // Save previous VP for velocity buffer before overwriting last_vp
        let prev_vp = self.frame.last_vp;
        self.frame.scene_vp = vp;
        self.frame.scene_depth_reversed_z = first.depth_reversed_z;
        // Camera moved? Force re-cull even if AABBs are static.
        let camera_moved = vp != prev_vp;
        if camera_moved {
            self.frame.static_frame_count = 0;
        }
        self.frame.last_vp = vp;
        let frustum = Frustum::from_view_projection(vp, first.depth_reversed_z);

        let visible: Vec<&DrawCall> = self.cpu_span.measure("bvh_frustum_cull", || {
            // Build BVH items: reuse cached Vec when count matches (skip allocation),
            // otherwise allocate a fresh one.
            let bvh_items = match self.frame.cached_bvh {
                Some((_, ref mut cached_items)) if cached_items.len() == draw_calls.len() => {
                for (i, dc) in draw_calls.iter().enumerate() {
                    if let Some(ref aabb) = dc.aabb {
                        cached_items[i] = (aabb.clone(), i as u32);
                    }
                }
                std::mem::take(cached_items)
                }
                _ => {
                draw_calls
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dc)| dc.aabb.as_ref().map(|a| (a.clone(), i as u32)))
                    .collect()
                }
            };

            let bvh = match self.frame.cached_bvh {
                Some((ref mut cached, ref mut _items)) if _items.len() == bvh_items.len() => {
                    let updates: Vec<(usize, rc3d_core::Aabb)> = bvh_items.iter()
                        .enumerate()
                        .filter(|(i, (aabb, _))| {
                            _items.get(*i).map_or(true, |(old, _)| {
                                old.min != aabb.min || old.max != aabb.max
                            })
                        })
                        .map(|(i, (aabb, _))| (i, aabb.clone()))
                        .collect();
                    let fully_static = updates.is_empty();
                    cached.incremental_update(&updates, &bvh_items);
                    *_items = bvh_items;
                    if fully_static {
                        self.frame.bvh_fully_static = true;
                        self.frame.static_frame_count += 1;
                    } else {
                        self.frame.bvh_fully_static = false;
                        self.frame.static_frame_count = 0;
                    }
                    &*cached
                }
                _ => {
                    self.frame.bvh_fully_static = false;
                    self.frame.static_frame_count = 0;
                    let bvh = rc3d_core::Bvh::build(&bvh_items);
                    self.frame.cached_bvh = Some((bvh, bvh_items));
                    &self.frame.cached_bvh.as_ref().unwrap().0
                }
            };

            // Static frame fast path: if BVH is fully static for 2+ consecutive frames,
            // reuse cached visible indices and skip the BVH query entirely.
            if self.frame.bvh_fully_static && self.frame.static_frame_count >= 2 {
                self.frame.visible_indices.clear();
                self.frame.visible_indices.extend_from_slice(&self.frame.static_visible_indices);
            } else {
                self.frame.visible_indices.clear();
                if bvh.is_empty() {
                    self.frame.visible_indices.extend(0..draw_calls.len());
                } else {
                    self.frame.bvh_out.clear();
                    bvh.query_filter(|aabb| frustum.intersects_aabb(aabb), &mut self.frame.bvh_out);
                    for &idx in &self.frame.bvh_out {
                        self.frame.visible_indices.push(idx as usize);
                    }
                    for (i, dc) in draw_calls.iter().enumerate() {
                        if dc.aabb.is_none() {
                            self.frame.visible_indices.push(i);
                        }
                    }
                    self.frame.visible_indices.sort_unstable();
                    self.frame.visible_indices.dedup();
                }
                // Cache visible indices for static frame reuse
                self.frame.static_visible_indices.clear();
                self.frame.static_visible_indices.extend_from_slice(&self.frame.visible_indices);
            }
            self.frame.visible_indices.iter().map(|&i| &draw_calls[i]).collect()
        });

        // ── GPU cull replacement: if readback has fresh indices, replace CPU cull ──
        let visible = if let Some(ref gpu_indices) = gpu_visible {
            self.frame.visible_indices.clear();
            self.frame.visible_indices.extend(gpu_indices.iter().copied());
            self.frame.visible_indices.iter().map(|&i| &draw_calls[i]).collect()
        } else {
            visible
        };

        // ── GPU compute culling (runs alongside CPU culling for now) ──
        if self.gpu.gpu_cull_enabled {
            if let (Some(cull_pass), Some(transform_buf), Some(indirect_buf),
                    Some(_instance_buf), Some(frustum_buf), Some(_bg)) = (
                self.gpu.gpu_cull_pass.as_ref(),
                self.gpu.transform_buffer.as_ref(),
                self.gpu.indirect_args_buffer.as_ref(),
                self.gpu.instance_indices_buffer.as_ref(),
                self.gpu.frustum_uniform.as_ref(),
                self.gpu.gpu_cull_bg.as_ref(),
            ) {
                // Upload transforms (all objects for now; dirty-tracking TBD).
                // Cap at the GPU buffer capacity to avoid out-of-bounds writes.
                let transforms: Vec<crate::vertex::GpuObjectTransform> = draw_calls.iter()
                    .take(self.gpu.max_gpu_cull_objects as usize)
                    .map(|dc| crate::vertex::GpuObjectTransform {
                        model_matrix: dc.model_matrix.to_cols_array_2d(),
                        aabb_min: dc.aabb.as_ref().map_or([0.0f32; 3], |a| a.min.to_array()),
                        flags: 0,
                        aabb_max: dc.aabb.as_ref().map_or([0.0f32; 3], |a| a.max.to_array()),
                        mesh_id: 0,
                        material_id: 0,
                        _pad: [0; 7],
                    })
                    .collect();
                cull_pass.write_transforms(&self.queue, transform_buf, &transforms);
                self.frame.gpu_cull_object_count = transforms.len() as u32;

                // Write frustum planes + object count (consumed by the cull shader)
                let planes = frustum.plane_array();
                cull_pass.write_frustum(&self.queue, frustum_buf, &planes, transforms.len() as u32);

                // Reset indirect args
                cull_pass.reset_indirect_args(&self.queue, indirect_buf, 4096);

                // Dispatch (fires asynchronously — GPU consumes it in subsequent draws)
                // Note: encoder is created below; this upload goes via queue.write_buffer
                // which is immediate. The actual cull compute dispatch happens before
                // the render pass when gpu_cull_pass.dispatch() is called.
            }
        }

        if let Some(head) = visible.first() {
            let dz = head.depth_reversed_z;
            let inconsistent = visible.iter().any(|dc| dc.depth_reversed_z != dz);
            if inconsistent {
                if !self.gpu.depth_reversed_z_mismatch_warned {
                    log::warn!(
                        "visible draw calls disagree on depth_reversed_z; using first visible ({}) for pipelines and depth clears",
                        dz
                    );
                    self.gpu.depth_reversed_z_mismatch_warned = true;
                }
            } else {
                self.gpu.depth_reversed_z_mismatch_warned = false;
            }
        }

        let total_visible_triangles: u64 = visible
            .iter()
            .map(|dc| {
                if let Some(md) = dc.meshlet_data.as_ref() {
                    md.total_triangles as u64
                } else if let Some(indices) = dc.indices.as_ref() {
                    (indices.len() / 3) as u64
                } else {
                    (dc.vertices.len() / 3) as u64
                }
            })
            .sum();
        let triangle_over_budget = total_visible_triangles > PERFORMANCE_MODE_TRIANGLE_THRESHOLD;
        let want_perf_mode = triangle_over_budget
            || (self.interaction_active && total_visible_triangles > 500_000);

        // Hysteresis: prevent rapid toggling. Enable immediately, disable only
        // after cooldown frames (or immediately if triangle count drops below threshold).
        if want_perf_mode {
            self.frame.perf_mode_cooldown = 0;
            if !self.frame.performance_mode_active {
                self.frame.performance_mode_active = true;
                if triangle_over_budget {
                    log::warn!("Performance mode enabled: triangle_count={}", total_visible_triangles);
                }
            }
        } else if self.frame.performance_mode_active {
            const PERF_COOLDOWN_FRAMES: u8 = 30; // ~0.5s at 60fps
            self.frame.perf_mode_cooldown += 1;
            if self.frame.perf_mode_cooldown >= PERF_COOLDOWN_FRAMES {
                self.frame.performance_mode_active = false;
                log::info!("Performance mode disabled after cooldown");
            }
        }

        let mesh_handles = self.upload_visible_meshes(&visible);

        self.orchestrate_frame_passes(
            visible,
            mesh_handles,
            draw_calls,
            first,
            scene,
            post_swapchain_overlay,
            presentation,
            ssao_projection,
            effect_commands,
            prev_vp,
        )
    }
}
