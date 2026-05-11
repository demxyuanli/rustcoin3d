use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use glam::{Mat4, Vec3};
use slotmap::Key;
use rc3d_core::DisplayMode;

// #region agent log
fn debug_log_620b84_render(location: &str, message: &str, data: &str) {
    use std::io::Write;
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("debug-620b84.log");
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let line = format!(
        r#"{{"sessionId":"620b84","location":"{}","message":"{}","data":{},"timestamp":{}}}"#,
        location, message, data, ts
    );
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
        let _ = writeln!(f, "{}", line);
    }
}
// #endregion
use rc3d_scene::SceneGraph;
use crate::adaptive_quality::AdaptiveQuality;
use crate::cluster::{ClusterRenderer, ClusterSet};
use crate::frustum::Frustum;
use crate::gpu_skinning::GpuSkinningPass;
use crate::render_action::DrawCall;
use crate::render_passes;
use crate::render_passes::PassContext;
use crate::shadow_map::{aabb_from_scene, csm_light_view_projs, compute_csm_splits, primary_directional_light_dir, union_draw_call_aabbs};
use crate::sort_keys;
use crate::vertex::CSM_CASCADE_COUNT;

use super::renderer_types::FrameStats;
use super::PERFORMANCE_MODE_TRIANGLE_THRESHOLD;

impl super::Renderer {
    fn render_draw_calls_core<'p>(
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

        // ── Streaming mesh pool: reset per-frame upload counter ──
        if let Some(ref mut pool) = self.gpu.assets.mesh_pool {
            pool.begin_frame();
        }

        // ── GPU cull: read back previous frame's instance count + indices ──
        let mut gpu_visible: Option<Vec<usize>> = None;
        if self.gpu.gpu_cull_enabled
            && self.frame.frame_counter > 2
            && self.frame.gpu_cull_ready
        {
            if let Some(ref staging) = self.gpu.gpu_cull_staging {
                let buf_slice = staging.slice(..);
                buf_slice.map_async(wgpu::MapMode::Read, |_| {});
                self.device.poll(wgpu::Maintain::Wait);
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
                        log::info!("GPU cull: {} visible (CPU: {})",
                            indices.len(), self.frame.visible_indices.len());
                        gpu_visible = Some(indices);
                    }
                }
                drop(mapped);
                staging.unmap();
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
        let dt_sec = (self.gpu.adaptive_frame_time_ema_ms / 1000.0).clamp(0.0, 0.25);
        self.frame.animation_time_sec += dt_sec;

        let _changed = self.gpu.shader_reload.check_and_reload();

        if self.frame.frame_counter % 300 == 0 {
            if let Some(ref mut pc) = self.gpu.pipeline_cache {
                pc.save_to_disk();
            }
            if let Some(ref pool) = self.gpu.assets.mesh_pool {
                log::info!(
                    "MeshPool: {}/{} slots, {} MB / {} MB, {} uploads/frame",
                    pool.len(), crate::mesh_pool::DEFAULT_POOL_SIZE,
                    pool.total_bytes() / (1024 * 1024),
                    crate::mesh_pool::DEFAULT_MAX_POOL_BYTES / (1024 * 1024),
                    pool.uploads_this_frame(),
                );
            }
        }

        // Compute VP + camera_pos early for text billboard projection
        let text_vp = draw_calls.first().map(|dc| dc.mvp * dc.model_matrix.inverse());
        let text_cam = draw_calls.first().map(|dc| dc.camera_pos);
        let text_viewport = self.gpu.hud.as_ref().map(|h| (h.width, h.height));

        // Fast-path: auto-learn whether scene has text/effect nodes.
        // First 2 frames always traverse to detect; after that, skip if empty.
        if let Some(hud) = &mut self.gpu.hud {
            if self.frame.has_text_nodes || self.frame.frame_counter < 2 {
                let t = self.cpu_span.measure("text_collect", || {
                    render_passes::pass_text::collect_text_nodes(scene, text_vp, text_viewport, text_cam)
                });
                hud.overlay_lines = t.overlay_lines;
                hud.positioned_texts = t.positioned;
                if hud.overlay_lines.is_empty()
                    && hud.positioned_texts.is_empty()
                    && self.frame.frame_counter >= 2
                {
                    self.frame.has_text_nodes = false;
                }
            } else {
                hud.overlay_lines = Vec::new();
                hud.positioned_texts = Vec::new();
            };
        }
        let effect_commands = if self.frame.has_effect_nodes || self.frame.frame_counter < 2 {
            let cmds = self.cpu_span.measure("effect_collect", || {
                render_passes::pass_effects::collect_effect_nodes(scene)
            });
            if cmds.is_empty() && self.frame.frame_counter >= 2 {
                self.frame.has_effect_nodes = false;
            }
            cmds
        } else {
            crate::render_passes::pass_effects::EffectCommands::default()
        };

        if draw_calls.is_empty() {
            return FrameStats::default();
        }

        let first = &draw_calls[0];
        let vp = first.mvp * first.model_matrix.inverse();
        self.frame.scene_vp = vp;
        // Camera moved? Force re-cull even if AABBs are static.
        let camera_moved = vp != self.frame.last_vp;
        if camera_moved {
            self.frame.static_frame_count = 0;
        }
        self.frame.last_vp = vp;
        let frustum = Frustum::from_view_projection(vp);

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

        // #region agent log
        {
            static RENDER_SEQ: AtomicU64 = AtomicU64::new(0);
            let seq = RENDER_SEQ.fetch_add(1, AtomicOrdering::Relaxed);
            let total_dc = draw_calls.len();
            let total_tris: u64 = visible.iter().map(|dc| {
                if let Some(ref md) = dc.meshlet_data { md.total_triangles as u64 }
                else if let Some(ref idx) = dc.indices { (idx.len() / 3) as u64 }
                else { (dc.vertices.len() / 3) as u64 }
            }).sum();
            if (seq < 5 || self.interaction_active) && total_tris > 100 {
                let vis = visible.len();
                debug_log_620b84_render(
                    "renderer_render.rs:frustum_cull",
                    "cull_result",
                    &format!(r#"{{"hypothesisId":"H","frame":{},"total_dc":{},"visible":{},"total_tris":{},"perf_mode":{},"interaction_active":{},"hdr":{}}}"#,
                        seq, total_dc, vis, total_tris, self.frame.performance_mode_active, self.interaction_active, self.hdr_post_processing),
                );
            }
        }
        // #endregion
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
            if let (Some(ref cull_pass), Some(ref transform_buf), Some(ref indirect_buf),
                    Some(ref _instance_buf), Some(ref frustum_buf), Some(ref _bg)) = (
                self.gpu.gpu_cull_pass.as_ref(),
                self.gpu.transform_buffer.as_ref(),
                self.gpu.indirect_args_buffer.as_ref(),
                self.gpu.instance_indices_buffer.as_ref(),
                self.gpu.frustum_uniform.as_ref(),
                self.gpu.gpu_cull_bg.as_ref(),
            ) {
                // Upload transforms (all objects for now; dirty-tracking TBD)
                let transforms: Vec<crate::vertex::GpuObjectTransform> = draw_calls.iter()
                    .enumerate()
                    .map(|(_i, dc)| crate::vertex::GpuObjectTransform {
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

                // Write frustum planes
                let planes = frustum.plane_array();
                cull_pass.write_frustum(&self.queue, frustum_buf, &planes);

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

        let mesh_upload_start = std::time::Instant::now();
        let mut mesh_handles: Vec<Option<crate::gpu_resource::MeshId>> = Vec::with_capacity(visible.len());
        for dc in &visible {
            let handle = if dc.vertices.is_empty() {
                if let Some(md) = dc.meshlet_data.as_ref() {
                    let ptr = Arc::as_ptr(md) as u64;
                    if let Some(mesh_id) = self.gpu.assets.mesh_touch(&ptr, self.frame.frame_counter) {
                        Some(mesh_id)
                    } else {
                        let verts: Vec<crate::vertex::Vertex> =
                            md.vertices
                                .iter()
                                .map(|mv| crate::vertex::Vertex {
                                    position: mv.position,
                                    normal: mv.normal,
                                    texcoord: mv.texcoord,
                                    tangent: mv.tangent,
                                })
                                .collect();
                        let mesh_id = self.gpu.gpu_meshes.upload_mesh(
                            &self.device,
                            &verts,
                            Some(&md.indices),
                            &[],
                            &[],
                        );
                        self.gpu.assets.mesh_insert(ptr, mesh_id, self.frame.frame_counter, Some(&mut self.gpu.gpu_meshes));
                        Some(mesh_id)
                    }
                } else {
                    None
                }
            } else {
                let base_hash = dc.mesh_hash.unwrap_or_else(|| {
                    let ptr_key = (
                        Arc::as_ptr(&dc.vertices) as u64,
                        dc.indices.as_ref().map_or(0u64, |a| Arc::as_ptr(a) as u64),
                    );
                    let mut h = twox_hash::XxHash64::with_seed(0);
                    std::hash::Hasher::write_u64(&mut h, ptr_key.0);
                    std::hash::Hasher::write_u64(&mut h, ptr_key.1);
                    std::hash::Hasher::finish(&h)
                });
                let hash = if let Some(ref skin) = dc.skinning {
                    let mut h = twox_hash::XxHash64::with_seed(0);
                    std::hash::Hasher::write_u64(&mut h, base_hash);
                    std::hash::Hasher::write_u64(&mut h, Arc::as_ptr(skin) as u64);
                    std::hash::Hasher::finish(&h)
                } else {
                    base_hash
                };
                if let Some(mesh_id) = self.gpu.assets.mesh_touch(&hash, self.frame.frame_counter) {
                    Some(mesh_id)
                } else {
                    if let Some(ref skin) = dc.skinning {
                        let verts = &*dc.vertices;
                        let skin_slice = skin.skin_data.as_slice();
                        if verts.len() != skin_slice.len() {
                            log::warn!(
                                "skinning data len {} != mesh verts {}; using rigid mesh",
                                skin_slice.len(),
                                verts.len()
                            );
                            let mesh_id = self.gpu.gpu_meshes.upload_mesh(
                                &self.device,
                                &dc.vertices,
                                dc.indices.as_ref().map(|a| a.as_slice()),
                                &dc.edge_positions,
                                &dc.wireframe_edge_positions,
                            );
                            self.gpu.assets.mesh_insert(hash, mesh_id, self.frame.frame_counter, Some(&mut self.gpu.gpu_meshes));
                            Some(mesh_id)
                        } else {
                            let pass = self
                                .gpu.gpu_skinning_pass
                                .get_or_insert_with(|| GpuSkinningPass::new(&self.device));
                            let positions: Vec<[f32; 3]> = verts.iter().map(|v| v.position).collect();
                            let normals: Vec<[f32; 3]> = verts.iter().map(|v| v.normal).collect();
                            let texcoords: Vec<[f32; 2]> = verts.iter().map(|v| v.texcoord).collect();
                            let tangents: Vec<[f32; 4]> = verts.iter().map(|v| v.tangent).collect();
                            let max_bones = skin.skeleton.joint_count().max(1) as u32;
                            let resources = pass.create_skinned_mesh(
                                &self.device,
                                &positions,
                                &normals,
                                &texcoords,
                                &tangents,
                                skin_slice,
                                max_bones,
                            );
                            let dst_vb = resources.dst_vertex_buffer.clone();
                            let mesh_id = self.gpu.gpu_meshes.insert_skinned_mesh(
                                &self.device,
                                dst_vb,
                                verts.len() as u32,
                                dc.indices.as_ref().map(|a| a.as_slice()),
                                &dc.edge_positions,
                                &dc.wireframe_edge_positions,
                            );
                            self.gpu.skinned_mesh_resources.insert(mesh_id, resources);
                            self.gpu.assets.mesh_insert(hash, mesh_id, self.frame.frame_counter, Some(&mut self.gpu.gpu_meshes));
                            Some(mesh_id)
                        }
                    } else {
                        let mesh_id = self.gpu.gpu_meshes.upload_mesh(
                            &self.device,
                            &dc.vertices,
                            dc.indices.as_ref().map(|a| a.as_slice()),
                            &dc.edge_positions,
                            &dc.wireframe_edge_positions,
                        );
                        self.gpu.assets.mesh_insert(hash, mesh_id, self.frame.frame_counter, Some(&mut self.gpu.gpu_meshes));
                        Some(mesh_id)
                    }
                }
            };
            mesh_handles.push(handle);
        }
        self.cpu_span.record("mesh_upload", mesh_upload_start.elapsed().as_secs_f64() * 1000.0);

        // Use pre-computed light_key from DrawCall (computed during traversal).
        let mut light_hashes = std::mem::take(&mut self.frame.light_hashes_buf);
        light_hashes.clear();
        light_hashes.extend(visible.iter().map(|dc| dc.light_key));

        let (mut solid_order, mut edge_order, mut selected_order, mut transparent_order) = self.cpu_span.measure("sorting", || {
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

            let camera_pos_vec: Vec3 = draw_calls.first().map(|dc| dc.camera_pos).unwrap_or(Vec3::ZERO);
            self.frame.scene_camera_pos = camera_pos_vec;
            let mut transparent_order = std::mem::take(&mut self.frame.transparent_order_buf);
            transparent_order.clear();
            transparent_order.extend((0..visible.len())
                .filter(|&i| visible[i].opacity < 1.0 && visible[i].opacity > 0.0));
            transparent_order.sort_unstable_by(|&a, &b| {
                let pos_a = visible[a].model_matrix.w_axis.truncate();
                let pos_b = visible[b].model_matrix.w_axis.truncate();
                let da = pos_a.distance(camera_pos_vec);
                let db = pos_b.distance(camera_pos_vec);
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            });
            (solid_order, edge_order, selected_order, transparent_order)
        });

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

        self.gpu.phong_pool.reset();
        self.gpu.shadow_pool.reset();
        self.gpu.flat_pool.reset();
        self.gpu.section_cap_pool.reset();
        self.gpu.outline_pool.reset();

        let base_mode = if self.frame.performance_mode_active {
            match self.global_display_mode {
                DisplayMode::HiddenLine | DisplayMode::Wireframe => DisplayMode::Shaded,
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
            && (mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine);

        let solid_wants_shadow = !self.frame.performance_mode_active
            && matches!(
                mode,
                DisplayMode::Shaded | DisplayMode::ShadedWithEdges | DisplayMode::HiddenLine
            )
            && mode != DisplayMode::Flat;

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
                    self.ensure_csm_shadow(sm_size, cascade_count);

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
                    let far_range = camera_far - camera_near;
                    for i in 0..cascade_count as usize {
                        if i + 1 < splits.len() {
                            csm_split_depths[i] = (splits[i + 1] - camera_near) / far_range;
                        }
                    }
                    for i in cascade_count as usize..CSM_CASCADE_COUNT {
                        csm_split_depths[i] = 1.0;
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
            csm_split_depths,
            shadow_params,
            run_shadow_pass,
            camera_proj,
            camera_inv_proj,
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
                log::info!(
                    "GPU pass timings (us): shadow={:.0} solid={:.0} post={:.0} total={:.0}",
                    shadow, solid, post, total
                );
            }
        }
        let diagnostics = self.build_frame_diagnostics(
            &visible,
            &mesh_handles,
            &solid_order,
            &transparent_order,
            gpu_pass,
        );
        stats.diagnostics = Some(diagnostics.clone());
        self.frame.last_diagnostics = Some(diagnostics);
        stats
    }

    pub fn render_draw_calls_with_overlay(
        &mut self,
        draw_calls: &[DrawCall],
        scene: &SceneGraph,
        post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    ) -> FrameStats {
        self.render_draw_calls_core(
            draw_calls,
            scene,
            post_swapchain_overlay,
            render_passes::FramePresentation::Swapchain,
            None,
        )
    }

    pub fn render_draw_calls_to_viewport_texture<'t>(
        &'t mut self,
        draw_calls: &[DrawCall],
        scene: &SceneGraph,
        viewport_texture: &'t wgpu::Texture,
        viewport_view: &'t wgpu::TextureView,
        viewport_width_px: u32,
        viewport_height_px: u32,
        projection: Mat4,
        inverse_projection: Mat4,
    ) -> FrameStats {
        let vw = viewport_width_px.max(1);
        let vh = viewport_height_px.max(1);

        let saved_depth = self.gpu.depth_texture.take();
        self.create_depth_texture_at(vw, vh);

        let saved_hdr = self.hdr_post_processing;
        let saved_fxaa = self.enable_ldr_fxaa;
        let saved_hud = self.hud_enabled;
        let saved_hzb = self.gpu.hzb.take();

        self.hdr_post_processing = false;
        self.enable_ldr_fxaa = false;
        self.hud_enabled = false;

        if let Some(ref baker) = self.gpu.hzb_baker {
            let bgl = &baker.downsample_bgl;
            self.gpu.hzb = Some(crate::hzb::HzbPyramids::new(&self.device, bgl, vw, vh));
        }

        let stats = self.render_draw_calls_core(
            draw_calls,
            scene,
            None,
            render_passes::FramePresentation::OffscreenSurface {
                output_texture: viewport_texture,
                output_view: viewport_view,
                width_px: vw,
                height_px: vh,
            },
            Some((projection, inverse_projection)),
        );

        self.gpu.depth_texture = saved_depth;
        self.hdr_post_processing = saved_hdr;
        self.enable_ldr_fxaa = saved_fxaa;
        self.hud_enabled = saved_hud;
        self.gpu.hzb = saved_hzb;
        stats
    }

    pub fn render_draw_calls(&mut self, draw_calls: &[DrawCall], scene: &SceneGraph) -> FrameStats {
        self.render_draw_calls_with_overlay(draw_calls, scene, None)
    }

    pub fn update_hud(&mut self, fps: f32, frame_time_ms: f32, stats: &FrameStats, mode_name: &str) {
        if !self.hud_enabled {
            return;
        }
        let interval = match self.gpu.adaptive_quality {
            AdaptiveQuality::High => 1,
            AdaptiveQuality::Medium => 2,
            AdaptiveQuality::Low => 6,
        };
        if self.frame.frame_counter.saturating_sub(self.frame.last_hud_update_frame) < interval {
            return;
        }
        self.frame.last_hud_update_frame = self.frame.frame_counter;
        let quality_name = self.adaptive_quality_name();
        let hud_mode_name = format!("{mode_name} [{quality_name}]");
        if let Some(hud) = &mut self.gpu.hud {
            hud.update_text(&self.device, &self.queue, fps, frame_time_ms, stats, &hud_mode_name);
        }
    }

    /// Render section caps from scene context (called from render_passes).
    pub fn render_section_caps_from_ctx(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        draw_calls: &[&crate::render_action::DrawCall],
        solid_order: &[usize],
        mesh_handles: &[Option<crate::gpu_resource::MeshId>],
        scene_pl: &crate::pipelines::DepthModePipelines,
    ) {
        let cap_specs: Vec<([f32; 4], [f32; 4])> = self
            .frame
            .section_cap_tints
            .iter()
            .zip(self.frame.clip_planes.iter())
            .filter_map(|(t, p)| t.map(|c| (*p, c)))
            .collect();
        if cap_specs.is_empty() {
            return;
        }
        let entries: Vec<(usize, crate::gpu_resource::MeshId)> =
            solid_order.iter().filter_map(|&i| mesh_handles[i].map(|m| (i, m))).collect();
        self.render_section_caps(
            encoder,
            shade_view,
            depth_view,
            scene_pl,
            &cap_specs,
            &entries,
            draw_calls,
        );
    }

    /// Fills the open cross section with flat color by rasterizing mesh triangles and
    /// keeping fragments within a narrow band of each cap plane (`section_cap.wgsl`).
    pub fn render_section_caps(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        scene_pl: &crate::pipelines::DepthModePipelines,
        cap_specs: &[([f32; 4], [f32; 4])],
        entries: &[(usize, crate::gpu_resource::MeshId)],
        draw_calls: &[&crate::render_action::DrawCall],
    ) {
        if cap_specs.is_empty() || entries.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Section cap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&scene_pl.section_cap_fill);
        const MIN_BAND: f32 = 5e-4;
        for &(plane, cap_color) in cap_specs {
            for &(idx, mesh_id) in entries {
                let dc = draw_calls[idx];
                let cap_uniforms = crate::vertex::SectionCapUniforms {
                    mvp: dc.mvp.to_cols_array_2d(),
                    model: dc.model_matrix.to_cols_array_2d(),
                    color: cap_color,
                    plane,
                    params: [MIN_BAND, 0.0, 0.0, 0.0],
                };
                let Some(offset) = self.gpu.section_cap_pool.push_section_cap(&cap_uniforms) else {
                    continue;
                };
                pass.set_bind_group(0, self.gpu.section_cap_pool.bind_group(), &[offset]);
                if let Some(mesh) = self.gpu.gpu_meshes.get(mesh_id) {
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    if let Some(ref ib) = mesh.index_buffer {
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    } else {
                        pass.draw(0..mesh.vertex_count as u32, 0..1);
                    }
                }
            }
        }
    }
}
