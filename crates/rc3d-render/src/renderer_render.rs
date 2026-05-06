use std::sync::Arc;

use glam::{Mat4, Vec3};
use slotmap::Key;
use rc3d_core::DisplayMode;
use rc3d_scene::SceneGraph;
use crate::adaptive_quality::AdaptiveQuality;
use crate::asset_manager::MESH_CACHE_MAX;
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
    pub fn render_draw_calls_with_overlay(
        &mut self,
        draw_calls: &[DrawCall],
        scene: &SceneGraph,
        post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    ) -> FrameStats {
        self.frame.frame_counter = self.frame.frame_counter.wrapping_add(1);
        let dt_sec = (self.gpu.adaptive_frame_time_ema_ms / 1000.0).clamp(0.0, 0.25);
        self.frame.animation_time_sec += dt_sec;

        let _changed = self.gpu.shader_reload.check_and_reload();

        if self.frame.frame_counter % 300 == 0 {
            if let Some(ref mut pc) = self.gpu.pipeline_cache {
                pc.save_to_disk();
            }
        }

        if draw_calls.is_empty() {
            return FrameStats::default();
        }

        let first = &draw_calls[0];
        let vp = first.mvp * first.model_matrix.inverse();
        self.frame.scene_vp = vp;
        let frustum = Frustum::from_view_projection(vp);

        let bvh_items: Vec<(rc3d_core::Aabb, u32)> = draw_calls
            .iter()
            .enumerate()
            .filter_map(|(i, dc)| dc.aabb.as_ref().map(|a| (a.clone(), i as u32)))
            .collect();
        let bvh = rc3d_core::Bvh::build(&bvh_items);

        let mut visible_indices: Vec<usize> = Vec::with_capacity(draw_calls.len());
        if bvh.is_empty() {
            visible_indices.extend(0..draw_calls.len());
        } else {
            let mut bvh_out = Vec::new();
            bvh.query_filter(|aabb| frustum.intersects_aabb(aabb), &mut bvh_out);
            for &idx in &bvh_out {
                visible_indices.push(idx as usize);
            }
            for (i, dc) in draw_calls.iter().enumerate() {
                if dc.aabb.is_none() {
                    visible_indices.push(i);
                }
            }
            visible_indices.sort_unstable();
            visible_indices.dedup();
        }
        let visible: Vec<&DrawCall> = visible_indices.iter().map(|&i| &draw_calls[i]).collect();

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
        let enable_perf_mode = total_visible_triangles > PERFORMANCE_MODE_TRIANGLE_THRESHOLD;
        if enable_perf_mode != self.frame.performance_mode_active {
            self.frame.performance_mode_active = enable_perf_mode;
            if enable_perf_mode {
                log::warn!("Performance mode enabled: triangle_count={}", total_visible_triangles);
            } else {
                log::info!("Performance mode disabled");
            }
        }

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
                        self.gpu.assets.mesh_insert(ptr, mesh_id, self.frame.frame_counter);
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
                            self.gpu.assets.mesh_insert(hash, mesh_id, self.frame.frame_counter);
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
                        self.gpu.assets.mesh_insert(hash, mesh_id, self.frame.frame_counter);
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
                        self.gpu.assets.mesh_insert(hash, mesh_id, self.frame.frame_counter);
                        Some(mesh_id)
                    }
                }
            };
            mesh_handles.push(handle);
        }

        let mut solid_order: Vec<usize> = (0..visible.len())
            .filter(|&i| !visible[i].vertices.is_empty() || visible[i].meshlet_data.is_some())
            .collect();
        solid_order.sort_by_key(|&i| {
            let dc = visible[i];
            (
                sort_keys::vec4_array_sort_key(dc.light_dirs),
                sort_keys::vec4_array_sort_key(dc.light_colors),
                sort_keys::vec4_array_sort_key(dc.light_types),
                sort_keys::vec4_array_sort_key(dc.light_positions),
                sort_keys::vec4_array_sort_key(dc.spot_params),
                dc.light_count,
                sort_keys::display_mode_sort_key(dc.display_mode),
                sort_keys::color_sort_key([dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]),
                sort_keys::color_sort_key([dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0]),
                sort_keys::color_sort_key([dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0]),
                dc.shininess.to_bits(),
                mesh_handles[i].map(|m| m.data().as_ffi()).unwrap_or(0),
            )
        });
        let mut edge_order: Vec<usize> = (0..visible.len())
            .filter(|&i| !visible[i].edge_positions.is_empty())
            .collect();
        edge_order.sort_by_key(|&i| {
            let dc = visible[i];
            (
                sort_keys::display_mode_sort_key(dc.display_mode),
                sort_keys::color_sort_key(dc.overlay_color.unwrap_or([0.0, 0.0, 0.0, 0.5])),
                mesh_handles[i].map(|m| m.data().as_ffi()).unwrap_or(0),
            )
        });
        let mut selected_order: Vec<usize> = (0..visible.len())
            .filter(|&i| visible[i].selected && (!visible[i].vertices.is_empty() || visible[i].meshlet_data.is_some()))
            .collect();
        selected_order.sort_by_key(|&i| {
            let dc = visible[i];
            (
                sort_keys::display_mode_sort_key(dc.display_mode),
                mesh_handles[i].map(|m| m.data().as_ffi()).unwrap_or(0),
            )
        });

        let camera_pos_vec: Vec3 = draw_calls.first().map(|dc| dc.camera_pos).unwrap_or(Vec3::ZERO);
        self.frame.scene_camera_pos = camera_pos_vec;
        let mut transparent_order: Vec<usize> = (0..visible.len())
            .filter(|&i| visible[i].opacity < 1.0 && visible[i].opacity > 0.0)
            .collect();
        transparent_order.sort_unstable_by(|&a, &b| {
            let pos_a = visible[a].model_matrix.w_axis.truncate();
            let pos_b = visible[b].model_matrix.w_axis.truncate();
            let da = pos_a.distance(camera_pos_vec);
            let db = pos_b.distance(camera_pos_vec);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut meshlet_indices: Vec<usize> = Vec::new();
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
        self.gpu.outline_pool.reset();

        let base_mode = if self.frame.performance_mode_active {
            DisplayMode::Shaded
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
            );

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

        let mut csm_view_proj = [Mat4::IDENTITY; CSM_CASCADE_COUNT];
        let mut csm_split_depths = [0.0f32; CSM_CASCADE_COUNT];
        let mut shadow_params = [0.0_f32, 0.0004, 0.0, 0.0];
        let mut run_shadow_pass = false;

        if solid_wants_shadow {
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
        };

        let mut stats = render_passes::execute_passes(
            self,
            &ctx,
            draw_calls,
            self.frame.frame_counter,
            post_swapchain_overlay,
        );
        let gpu_pass = self.read_gpu_timestamps();
        stats.gpu_pass_times_us = gpu_pass;
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

    /// Render section plane cap surfaces for visible draw calls.
    /// Renders back-faces of clipped geometry to fill the cut boundary.
    pub fn render_section_caps(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        cap_color: [f32; 4],
        draw_calls: &[(usize, crate::gpu_resource::MeshId)],
    ) {
        if draw_calls.is_empty() {
            return;
        }
        let ident = glam::Mat4::IDENTITY.to_cols_array_2d();
        let cap_uniforms = crate::vertex::FlatUniforms {
            mvp: ident,
            color: cap_color,
        };
        if let Some(offset) = self.gpu.flat_pool.push_flat(&cap_uniforms) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Section Cap"),
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
            // Render back-faces with cap color to fill clip boundary
            pass.set_pipeline(&self.gpu.pipelines.forward.edge_overlay);
            pass.set_bind_group(0, self.gpu.flat_pool.bind_group(), &[offset]);
            for &(_, mesh_id) in draw_calls {
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
