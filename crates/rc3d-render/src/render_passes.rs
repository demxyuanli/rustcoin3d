use crate::render_action::DrawCall;
use crate::render_graph::{declaration_order_is_valid, RenderGraph};
use crate::vertex::{FlatUniforms, OutlineUniforms, SceneUniforms, ShadowDrawUniforms};
use crate::FrameStats;
use crate::renderer::AdaptiveQuality;
use glam::{Mat4, Vec3};
use rc3d_core::DisplayMode;
use std::sync::OnceLock;

static RC3D_RENDER_GRAPH_OK: OnceLock<()> = OnceLock::new();

fn albedo_material_bind_group<'a>(
    texture_cache: &'a mut crate::texture_cache::TextureCache,
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    queue: &wgpu::Queue,
    dc: &DrawCall,
) -> &'a wgpu::BindGroup {
    let handle = match &dc.albedo_path {
        None => texture_cache.white_handle(),
        Some(p) => texture_cache.load_path(device, queue, p.as_ref()),
    };
    texture_cache.albedo_bind_group(device, layout, handle)
}

pub(super) struct PassContext<'a> {
    pub visible: &'a [&'a DrawCall],
    pub solid_order: &'a [usize],
    pub edge_order: &'a [usize],
    pub selected_order: &'a [usize],
    pub mesh_handles: &'a [Option<crate::gpu_resource::MeshId>],
    pub mode: DisplayMode,
    pub run_outline: bool,
    pub bg_color: wgpu::Color,
    pub performance_mode_active: bool,
    pub wireframe_supported: bool,
    pub adaptive_quality: AdaptiveQuality,
    pub outline_width: f32,
    pub outline_color: [f32; 4],
    pub meshlet_indices: &'a [usize],
    pub camera_pos: [f32; 3],
    /// Matches `DrawCall::depth_reversed_z` / projection (frame-wide depth buffer convention).
    pub depth_reversed_z: bool,
    pub light_view_proj: Mat4,
    pub shadow_params: [f32; 4],
    pub run_shadow_pass: bool,
}

pub(super) fn execute_passes(
    renderer: &mut crate::renderer::Renderer,
    ctx: &PassContext<'_>,
    draw_calls: &[DrawCall],
    _frame_counter: u64,
) -> FrameStats {
    RC3D_RENDER_GRAPH_OK.get_or_init(|| {
        let g = RenderGraph::rc3d_forward_default();
        g.topological_sort()
            .expect("rc3d default render graph must be acyclic");
        assert!(
            declaration_order_is_valid(&g.passes),
            "rc3d_forward_default pass vec order must match encoder dependency order; update passes or CHAIN_* edges"
        );
    });

    let output = match renderer.surface.get_current_texture() {
        Ok(o) => o,
        Err(_) => return FrameStats::default(),
    };
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    if renderer.depth_texture.is_none() {
        renderer.create_depth_texture();
    }
    let Some((_, depth_view, depth_read_view)) = renderer.depth_texture.as_ref() else {
        return FrameStats::default();
    };
    let depth_view = depth_view.clone();
    let depth_read_view = depth_read_view.clone();

    if renderer.hdr_post_processing {
        renderer.ensure_hdr_scene_target();
    }
    let scene_pl = renderer
        .pipelines
        .for_shaded_target(ctx.depth_reversed_z, renderer.hdr_post_processing)
        .clone();
    let shade_color_view = if renderer.hdr_post_processing {
        renderer
            .hdr_scene_view
            .as_ref()
            .expect("HDR scene view")
            .clone()
    } else {
        view.clone()
    };
    let shade_view: &wgpu::TextureView = &shade_color_view;

    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

    if ctx.run_shadow_pass {
        pass_shadow_depth(renderer, &mut encoder, ctx);
    }

    let mode = ctx.mode;

    let solid_mode = matches!(
        mode,
        DisplayMode::Shaded | DisplayMode::ShadedWithEdges | DisplayMode::HiddenLine
    );

    let hzb_need_max = !ctx.meshlet_indices.is_empty()
        && ctx
            .meshlet_indices
            .iter()
            .any(|&i| ctx.visible[i].depth_reversed_z);
    let hzb_need_min = !ctx.meshlet_indices.is_empty()
        && ctx
            .meshlet_indices
            .iter()
            .any(|&i| !ctx.visible[i].depth_reversed_z);

    let run_meshlet_hzb_prepass = solid_mode
        && !ctx.meshlet_indices.is_empty()
        && renderer.cluster_renderer.is_some()
        && renderer.hzb.is_some()
        && renderer.hzb_baker.is_some();

    let mut meshlet_hzb_prepass_done = false;

    if run_meshlet_hzb_prepass {
        let mip_max;
        let hzb_dims_xy: (u32, u32);
        let max_bind_owned: wgpu::TextureView;
        let min_bind_owned: wgpu::TextureView;
        {
            let hzb = renderer.hzb.as_ref().unwrap();
            mip_max = hzb.max_pyramid.mip_count.saturating_sub(1);
            hzb_dims_xy = (hzb.max_pyramid.width, hzb.max_pyramid.height);
            max_bind_owned = if hzb_need_max {
                hzb.max_pyramid.full_view.clone()
            } else {
                hzb.min_pyramid.full_view.clone()
            };
            min_bind_owned = if hzb_need_min {
                hzb.min_pyramid.full_view.clone()
            } else {
                hzb.max_pyramid.full_view.clone()
            };
        }
        let lod_stride = 1u32;
        let meshlet_phase = 0u32;
        let meshlet_stride_spatial = false;

        {
            let cluster_renderer = renderer.cluster_renderer.as_ref().unwrap();
            let max_bind: &wgpu::TextureView = &max_bind_owned;
            let min_bind: &wgpu::TextureView = &min_bind_owned;
            for &vis_idx in ctx.meshlet_indices {
                let dc = ctx.visible[vis_idx];
                let md = dc.meshlet_data.as_ref().unwrap();
                let ptr = std::sync::Arc::as_ptr(md) as u64;
                if let Some(cluster_set) = renderer.cluster_cache.get(&ptr) {
                    let model_inv = dc.model_matrix.inverse();
                    let cam_model = model_inv.transform_point3(Vec3::from(ctx.camera_pos));
                    cluster_renderer.cull_and_compact(
                        &renderer.device,
                        &renderer.queue,
                        &mut encoder,
                        cluster_set,
                        dc.mvp.to_cols_array_2d(),
                        [cam_model.x, cam_model.y, cam_model.z],
                        lod_stride,
                        meshlet_phase,
                        meshlet_stride_spatial,
                        max_bind,
                        min_bind,
                        hzb_dims_xy,
                        mip_max,
                        false,
                        dc.depth_reversed_z,
                        dc.projection_orthographic,
                    );
                }
            }
        }

        pass_depth_prepass(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);

        {
            let baker = renderer.hzb_baker.as_ref().unwrap();
            let hzb = renderer.hzb.as_ref().unwrap();
            baker.build_from_depth(
                &renderer.device,
                &mut encoder,
                &depth_read_view,
                hzb_need_max,
                hzb_need_min,
                &hzb.max_pyramid,
                &hzb.min_pyramid,
            );
        }

        {
            let cluster_renderer = renderer.cluster_renderer.as_ref().unwrap();
            let max_bind: &wgpu::TextureView = &max_bind_owned;
            let min_bind: &wgpu::TextureView = &min_bind_owned;
            for &vis_idx in ctx.meshlet_indices {
                let dc = ctx.visible[vis_idx];
                let md = dc.meshlet_data.as_ref().unwrap();
                let ptr = std::sync::Arc::as_ptr(md) as u64;
                if let Some(cluster_set) = renderer.cluster_cache.get(&ptr) {
                    let model_inv = dc.model_matrix.inverse();
                    let cam_model = model_inv.transform_point3(Vec3::from(ctx.camera_pos));
                    cluster_renderer.cull_and_compact(
                        &renderer.device,
                        &renderer.queue,
                        &mut encoder,
                        cluster_set,
                        dc.mvp.to_cols_array_2d(),
                        [cam_model.x, cam_model.y, cam_model.z],
                        lod_stride,
                        meshlet_phase,
                        meshlet_stride_spatial,
                        max_bind,
                        min_bind,
                        hzb_dims_xy,
                        mip_max,
                        true,
                        dc.depth_reversed_z,
                        dc.projection_orthographic,
                    );
                }
            }
        }
        meshlet_hzb_prepass_done = true;
    } else if !ctx.meshlet_indices.is_empty() {
        if let (Some(cluster_renderer), Some(hzb)) =
            (renderer.cluster_renderer.as_ref(), renderer.hzb.as_ref())
        {
            let mip_max = hzb.max_pyramid.mip_count.saturating_sub(1);
            let max_bind: &wgpu::TextureView = if hzb_need_max {
                &hzb.max_pyramid.full_view
            } else {
                &hzb.min_pyramid.full_view
            };
            let min_bind: &wgpu::TextureView = if hzb_need_min {
                &hzb.min_pyramid.full_view
            } else {
                &hzb.max_pyramid.full_view
            };
            let lod_stride = 1u32;
            let meshlet_phase = 0u32;
            let meshlet_stride_spatial = false;
            for &vis_idx in ctx.meshlet_indices {
                let dc = ctx.visible[vis_idx];
                let md = dc.meshlet_data.as_ref().unwrap();
                let ptr = std::sync::Arc::as_ptr(md) as u64;
                if let Some(cluster_set) = renderer.cluster_cache.get(&ptr) {
                    let model_inv = dc.model_matrix.inverse();
                    let cam_model = model_inv.transform_point3(Vec3::from(ctx.camera_pos));
                    cluster_renderer.cull_and_compact(
                        &renderer.device,
                        &renderer.queue,
                        &mut encoder,
                        cluster_set,
                        dc.mvp.to_cols_array_2d(),
                        [cam_model.x, cam_model.y, cam_model.z],
                        lod_stride,
                        meshlet_phase,
                        meshlet_stride_spatial,
                        max_bind,
                        min_bind,
                        (hzb.max_pyramid.width, hzb.max_pyramid.height),
                        mip_max,
                        false,
                        dc.depth_reversed_z,
                        dc.projection_orthographic,
                    );
                }
            }
        }
    }

    if solid_mode {
        pass_solid_and_outline(
            renderer,
            &mut encoder,
            shade_view,
            &depth_view,
            ctx,
            meshlet_hzb_prepass_done,
            &scene_pl,
        );
    }

    if !ctx.performance_mode_active && ctx.wireframe_supported && mode == DisplayMode::Wireframe {
        pass_wireframe(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
    }

    let edge_worthy = !ctx.performance_mode_active
        && ctx.adaptive_quality != AdaptiveQuality::Low
        && (mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine);
    let has_overlay = ctx.visible.iter().any(|dc| dc.overlay_color.is_some());
    if edge_worthy || has_overlay {
        pass_edge_overlay(renderer, &mut encoder, shade_view, &depth_view, ctx, edge_worthy, &scene_pl);
    }

    let has_selection = ctx.visible.iter().any(|dc| dc.selected);
    if has_selection && !ctx.performance_mode_active {
        pass_selection_fill(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
        if ctx.wireframe_supported && ctx.adaptive_quality != AdaptiveQuality::Low {
            pass_selection_edge(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
        }
    }

    if renderer.hdr_post_processing {
        pass_tonemap_hdr_to_post_ldr(renderer, &mut encoder, ctx.bg_color);
        pass_blit_post_ldr_to_swapchain(renderer, &mut encoder, &view, ctx.bg_color);
    }

    renderer.outline_pool.flush(&renderer.queue);
    renderer.phong_pool.flush(&renderer.queue);
    renderer.shadow_pool.flush(&renderer.queue);
    renderer.flat_pool.flush(&renderer.queue);

    if renderer.hud_enabled {
        if let Some(hud) = renderer.hud.as_ref() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("HUD Overlay Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            hud.render(&mut pass);
        }
    }

    renderer.prune_mesh_cache();
    renderer.queue.submit(std::iter::once(encoder.finish()));
    output.present();

    let total_visible_triangles: u64 = ctx.visible
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
    FrameStats {
        visible_triangles: total_visible_triangles,
        visible_draw_calls: ctx.visible.len(),
        culled_draw_calls: draw_calls.len().saturating_sub(ctx.visible.len()),
    }
}



fn pass_tonemap_hdr_to_post_ldr(
    renderer: &crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    bg_color: wgpu::Color,
) {
    let Some(hdr_bg) = renderer.post_process_bind_group.as_ref() else {
        return;
    };
    let Some(post_ldr_view) = renderer.post_ldr_view.as_ref() else {
        return;
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Tonemap + FXAA to post LDR"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: post_ldr_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(bg_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(&renderer.post_process_pipeline);
    pass.set_bind_group(0, hdr_bg, &[]);
    pass.draw(0..3, 0..1);
}

fn pass_blit_post_ldr_to_swapchain(
    renderer: &crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    swap_view: &wgpu::TextureView,
    bg_color: wgpu::Color,
) {
    let Some(blit_bg) = renderer.blit_ldr_bind_group.as_ref() else {
        return;
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Blit post LDR to swapchain"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: swap_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(bg_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(&renderer.blit_ldr_pipeline);
    pass.set_bind_group(0, blit_bg, &[]);
    pass.draw(0..3, 0..1);
}

fn pass_shadow_depth(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &PassContext<'_>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Shadow depth"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &renderer.shadow_depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_pipeline(&renderer.pipelines.shadow_depth);
    let light_vp = ctx.light_view_proj;
    let mut last_bound_mesh = None;
    for &i in ctx.solid_order {
        let dc = ctx.visible[i];
        let shadow_mvp = light_vp * dc.model_matrix;
        let uniforms = ShadowDrawUniforms {
            shadow_mvp: shadow_mvp.to_cols_array_2d(),
        };
        if let Some(offset) = renderer.shadow_pool.push_shadow(&uniforms) {
            if let Some(mesh_id) = ctx.mesh_handles[i] {
                pass.set_bind_group(0, renderer.shadow_pool.bind_group(), &[offset]);
                renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
            }
        }
    }
}

fn pass_depth_prepass(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let prepass_pipeline = scene_pl.solid_depth_prepass.clone();
    let depth_clear = if ctx.depth_reversed_z { 0.0 } else { 1.0 };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Depth prepass (HZB)"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(ctx.bg_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(depth_clear),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0u32),
                store: wgpu::StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    // Omit meshlets here so HZB encodes only other occluders; otherwise meshlets self-occlude in the cull pass.
    draw_opaque_triangle_batches(renderer, &mut pass, ctx, &prepass_pipeline, false);
}

fn draw_opaque_triangle_batches(
    renderer: &mut crate::renderer::Renderer,
    pass: &mut wgpu::RenderPass<'_>,
    ctx: &PassContext<'_>,
    solid_pipeline: &wgpu::RenderPipeline,
    draw_meshlets: bool,
) {
    pass.set_pipeline(solid_pipeline);
    pass.set_stencil_reference(1);

    let mut clip_arr = [[0.0f32; 4]; 6];
    for (i, cp) in renderer.clip_planes.iter().enumerate() {
        if i < 6 {
            clip_arr[i] = *cp;
        }
    }
    let clip_count = [renderer.clip_planes.len().min(6) as f32, 0.0, 0.0, 0.0];

    let meshlet_set: std::collections::HashSet<usize> = ctx.meshlet_indices.iter().copied().collect();

    let mut last_bound_mesh = None;
    let mut start = 0usize;
    while start < ctx.solid_order.len() {
        let head_idx = ctx.solid_order[start];
        let head_dc = ctx.visible[head_idx];
        let light_key = crate::sort_keys::light_sort_key(head_dc);
        let mut end = start + 1;
        while end < ctx.solid_order.len() {
            let idx = ctx.solid_order[end];
            let dc = ctx.visible[idx];
            if crate::sort_keys::light_sort_key(dc) != light_key {
                break;
            }
            end += 1;
        }

        for &i in &ctx.solid_order[start..end] {
            let dc = ctx.visible[i];

            if meshlet_set.contains(&i) {
                if draw_meshlets {
                    if let Some(md) = dc.meshlet_data.as_ref() {
                        let ptr = std::sync::Arc::as_ptr(md) as u64;
                        if renderer.cluster_renderer.is_some() && renderer.cluster_cache.contains_key(&ptr)
                        {
                            let diffuse_color = if ctx.mode == DisplayMode::HiddenLine {
                                [0.08, 0.08, 0.08, 1.0]
                            } else {
                                [dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]
                            };
                            let uniforms = SceneUniforms {
                                mvp: dc.mvp.to_cols_array_2d(),
                                model: dc.model_matrix.to_cols_array_2d(),
                                camera_pos: [dc.camera_pos.x, dc.camera_pos.y, dc.camera_pos.z, 1.0],
                                light_dirs: head_dc.light_dirs,
                                light_colors: head_dc.light_colors,
                                light_types: head_dc.light_types,
                                light_positions: head_dc.light_positions,
                                spot_params: head_dc.spot_params,
                                light_count: [head_dc.light_count as f32, 0.0, 0.0, 0.0],
                                diffuse_color,
                                ambient_color: [dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0],
                                specular_color: [dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0],
                                shininess: [dc.shininess, 0.0, 0.0, 0.0],
                                clip_planes: clip_arr,
                                clip_count,
                                pbr_base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
                                pbr_metallic_roughness: [dc.metallic, dc.roughness, 0.0, 0.0],
                                ibl_diffuse: renderer.ibl_diffuse,
                                ibl_specular: renderer.ibl_specular,
                                light_view_proj: ctx.light_view_proj.to_cols_array_2d(),
                                shadow_params: ctx.shadow_params,
                            };
                            if let Some(offset) = renderer.phong_pool.push_scene(&uniforms) {
                                let mat_bg = albedo_material_bind_group(
                                    &mut renderer.texture_cache,
                                    &renderer.device,
                                    &renderer.pipelines.pbr_material_bgl,
                                    &renderer.queue,
                                    dc,
                                );
                                let pool_bg = renderer.phong_pool.bind_group();
                                pass.set_bind_group(0, pool_bg, &[offset]);
                                pass.set_bind_group(1, mat_bg, &[]);
                                pass.set_bind_group(2, &renderer.shadow_bind_group, &[]);
                                let cluster_set = renderer.cluster_cache.get(&ptr).expect("cluster set");
                                renderer
                                    .cluster_renderer
                                    .as_ref()
                                    .expect("cluster renderer")
                                    .draw_clustered(pass, cluster_set);
                            }
                        }
                    }
                }
                continue;
            }

            let diffuse_color = if ctx.mode == DisplayMode::HiddenLine {
                [0.08, 0.08, 0.08, 1.0]
            } else {
                [dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]
            };
            let uniforms = SceneUniforms {
                mvp: dc.mvp.to_cols_array_2d(),
                model: dc.model_matrix.to_cols_array_2d(),
                camera_pos: [dc.camera_pos.x, dc.camera_pos.y, dc.camera_pos.z, 1.0],
                light_dirs: head_dc.light_dirs,
                light_colors: head_dc.light_colors,
                light_types: head_dc.light_types,
                light_positions: head_dc.light_positions,
                spot_params: head_dc.spot_params,
                light_count: [head_dc.light_count as f32, 0.0, 0.0, 0.0],
                diffuse_color,
                ambient_color: [dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0],
                specular_color: [dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0],
                shininess: [dc.shininess, 0.0, 0.0, 0.0],
                clip_planes: clip_arr,
                clip_count,
                pbr_base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
                pbr_metallic_roughness: [dc.metallic, dc.roughness, 0.0, 0.0],
                ibl_diffuse: renderer.ibl_diffuse,
                ibl_specular: renderer.ibl_specular,
                light_view_proj: ctx.light_view_proj.to_cols_array_2d(),
                shadow_params: ctx.shadow_params,
            };
            if let Some(offset) = renderer.phong_pool.push_scene(&uniforms) {
                let mat_bg = albedo_material_bind_group(
                    &mut renderer.texture_cache,
                    &renderer.device,
                    &renderer.pipelines.pbr_material_bgl,
                    &renderer.queue,
                    dc,
                );
                let pool_bg = renderer.phong_pool.bind_group();
                pass.set_bind_group(0, pool_bg, &[offset]);
                pass.set_bind_group(1, mat_bg, &[]);
                pass.set_bind_group(2, &renderer.shadow_bind_group, &[]);
                if let Some(mesh_id) = ctx.mesh_handles[i] {
                    renderer.draw_mesh_batched(pass, mesh_id, &mut last_bound_mesh);
                }
            }
        }
        start = end;
    }
}

fn pass_solid_and_outline(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    reuse_prepass_depth: bool,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let solid_pipeline = scene_pl.solid.clone();
    let outline_pipeline = scene_pl.outline.clone();
    let depth_clear = if ctx.depth_reversed_z { 0.0 } else { 1.0 };
    let depth_load = if reuse_prepass_depth {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(depth_clear)
    };
    let stencil_load = if reuse_prepass_depth {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(0u32)
    };
    let pass_label = if ctx.run_outline { "Solid+Outline Pass" } else { "Solid Pass" };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(pass_label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(ctx.bg_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: depth_load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: Some(wgpu::Operations {
                load: stencil_load,
                store: wgpu::StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    draw_opaque_triangle_batches(renderer, &mut pass, ctx, &solid_pipeline, true);

    if ctx.run_outline {
        let meshlet_set: std::collections::HashSet<usize> =
            ctx.meshlet_indices.iter().copied().collect();
        pass.set_pipeline(&outline_pipeline);
        pass.set_stencil_reference(0);
        let outline_color = if ctx.mode == DisplayMode::HiddenLine {
            [0.5, 0.7, 1.0, 1.0]
        } else {
            ctx.outline_color
        };
        let mut last_bound_mesh = None;
        for &i in ctx.solid_order {
            if meshlet_set.contains(&i) {
                continue;
            }
            let dc = ctx.visible[i];
            let uniforms = OutlineUniforms {
                mvp: dc.mvp.to_cols_array_2d(),
                outline_width: ctx.outline_width,
                _pad: [0.0; 3],
                color: outline_color,
            };
            if let Some(offset) = renderer.outline_pool.push_outline(&uniforms) {
                pass.set_bind_group(0, renderer.outline_pool.bind_group(), &[offset]);
                if let Some(mesh_id) = ctx.mesh_handles[i] {
                    renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
                }
            }
        }
    }
}

fn pass_wireframe(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let pl = scene_pl;
    let depth_clear = if ctx.depth_reversed_z { 0.0 } else { 1.0 };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Wireframe Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(ctx.bg_color), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(depth_clear),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0u32),
                store: wgpu::StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_pipeline(&pl.wireframe);
    let line_color = [0.2, 1.0, 0.4, 1.0];
    let mut last_bound_mesh = None;
    for &i in ctx.solid_order {
        let dc = ctx.visible[i];
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: line_color,
        };
        if let Some(offset) = renderer.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.flat_pool.bind_group(), &[offset]);
            if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
            }
        }
    }
}

fn pass_edge_overlay(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    edge_worthy: bool,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let pl = scene_pl;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Edge Overlay Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
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
            stencil_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_stencil_reference(0);
    pass.set_pipeline(&pl.edge_overlay);

    let default_edge_color = if ctx.mode == DisplayMode::HiddenLine {
        [0.6, 0.8, 1.0, 0.8]
    } else {
        [0.0, 0.0, 0.0, 0.5]
    };

    let mut last_bound_edge_mesh = None;
    for &i in ctx.edge_order {
        let dc = ctx.visible[i];
        if !edge_worthy && dc.overlay_color.is_none() {
            continue;
        }
        let edge_color = dc.overlay_color.unwrap_or(default_edge_color);
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: edge_color,
        };
        if let Some(offset) = renderer.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.flat_pool.bind_group(), &[offset]);
            let drawn_from_cache = if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_edges_batched(&mut pass, mesh_id, &mut last_bound_edge_mesh)
            } else {
                false
            };
            if !drawn_from_cache {
                renderer.bind_and_draw_edges(&mut pass, dc, ctx.mesh_handles[i]);
            }
        }
    }
}

fn pass_selection_fill(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let pl = scene_pl;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Selection Fill Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
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
            stencil_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_stencil_reference(0);
    pass.set_pipeline(&pl.selection_fill);
    let mut last_bound_mesh = None;
    for &i in ctx.selected_order {
        let dc = ctx.visible[i];
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: [1.0, 0.6, 0.0, 0.35],
        };
        if let Some(offset) = renderer.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.flat_pool.bind_group(), &[offset]);
            if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
            }
        }
    }
}

fn pass_selection_edge(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    ctx: &PassContext<'_>,
    scene_pl: &crate::pipelines::DepthModePipelines,
) {
    let pl = scene_pl;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Selection Edge Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
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
            stencil_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_stencil_reference(0);
    pass.set_pipeline(&pl.wireframe);
    let sel_color = [1.0, 0.5, 0.0, 1.0];
    let mut last_bound_mesh = None;
    for &i in ctx.selected_order {
        let dc = ctx.visible[i];
        let uniforms = FlatUniforms {
            mvp: dc.mvp.to_cols_array_2d(),
            color: sel_color,
        };
        if let Some(offset) = renderer.flat_pool.push_flat(&uniforms) {
            pass.set_bind_group(0, renderer.flat_pool.bind_group(), &[offset]);
            if let Some(mesh_id) = ctx.mesh_handles[i] {
                renderer.draw_mesh_batched(&mut pass, mesh_id, &mut last_bound_mesh);
            }
        }
    }
}
