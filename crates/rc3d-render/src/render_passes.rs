use crate::adaptive_quality::AdaptiveQuality;
use crate::render_action::DrawCall;
use crate::render_graph::{declaration_order_is_valid, RenderGraph};
use crate::vertex::{InstanceData, SceneUniforms, CSM_CASCADE_COUNT};
use crate::FrameStats;
use glam::{Mat4, Vec3};
use rc3d_core::DisplayMode;
use std::sync::OnceLock;

mod pass_edge;
mod pass_post;
mod pass_selection;
mod pass_shadow;
mod pass_solid;
mod pass_text;
mod pass_wireframe;

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
    // Normal map: use default flat normal if no explicit normal path
    let normal_handle = texture_cache.default_normal_handle();
    texture_cache.pbr_material_bind_group(device, layout, handle, normal_handle)
}

pub(super) struct PassContext<'a> {
    pub visible: &'a [&'a DrawCall],
    pub solid_order: &'a [usize],
    pub edge_order: &'a [usize],
    pub selected_order: &'a [usize],
    pub transparent_order: &'a [usize],
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
    pub depth_reversed_z: bool,
    /// CSM cascade view-projection matrices (one per cascade)
    pub csm_view_proj: [Mat4; CSM_CASCADE_COUNT],
    /// CSM split depths in view space: [near, split1, split2, far] (4 values)
    pub csm_split_depths: [f32; CSM_CASCADE_COUNT],
    pub shadow_params: [f32; 4],
    pub run_shadow_pass: bool,
    /// Camera projection matrix (for SSAO depth reconstruction)
    pub camera_proj: Mat4,
    /// Camera inverse projection matrix
    pub camera_inv_proj: Mat4,
}

pub(super) fn execute_passes(
    renderer: &mut crate::renderer::Renderer,
    ctx: &PassContext<'_>,
    draw_calls: &[DrawCall],
    _frame_counter: u64,
) -> FrameStats {
    RC3D_RENDER_GRAPH_OK.get_or_init(|| {
        let g = RenderGraph::rc3d_forward_default();
        if g.topological_sort().is_err() {
            log::error!("rc3d default render graph is cyclic; execution will continue with best-effort pass order");
        }
        if !declaration_order_is_valid(&g.passes) {
            log::error!(
                "rc3d_forward_default pass declaration order mismatches dependency order; execution continues with degraded safety"
            );
        }
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
        renderer.ensure_post_fx_targets();
    }
    let scene_pl = renderer
        .pipelines
        .for_shaded_target(ctx.depth_reversed_z, renderer.hdr_post_processing)
        .clone();
    let shade_color_view = if renderer.hdr_post_processing {
        if let Some(fx) = renderer.post_fx.as_ref() {
            fx.hdr_view.clone()
        } else {
            log::warn!("hdr_post_processing is enabled but post_fx is missing; falling back to swapchain view");
            view.clone()
        }
    } else {
        view.clone()
    };
    let shade_view: &wgpu::TextureView = &shade_color_view;

    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

    // GPU timestamp: frame start
    renderer.write_gpu_timestamp(&mut encoder);

    if ctx.run_shadow_pass {
        pass_shadow::pass_shadow_depth(renderer, &mut encoder, ctx);
        renderer.write_gpu_timestamp(&mut encoder); // shadow end
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

    let requested_meshlet_hzb_prepass = solid_mode
        && !ctx.meshlet_indices.is_empty()
        && renderer.cluster_renderer.is_some()
        && renderer.hzb.is_some()
        && renderer.hzb_baker.is_some();
    let run_meshlet_hzb_prepass = if requested_meshlet_hzb_prepass {
        true
    } else {
        if solid_mode
            && !ctx.meshlet_indices.is_empty()
            && (renderer.cluster_renderer.is_none()
                || renderer.hzb.is_none()
                || renderer.hzb_baker.is_none())
        {
            log::warn!("meshlet HZB prepass requested but resources are incomplete; using fallback meshlet cull path");
        }
        false
    };

    let mut meshlet_hzb_prepass_done = false;

    'hzb_prepass: {
        if run_meshlet_hzb_prepass {
        let hzb = renderer.hzb.as_ref().expect("HZB missing despite pre-check");
        let cluster_renderer = renderer.cluster_renderer.as_ref().expect("cluster renderer missing despite pre-check");

        let mip_max = hzb.max_pyramid.mip_count.saturating_sub(1);
        let hzb_dims_xy = (hzb.max_pyramid.width, hzb.max_pyramid.height);
        let max_bind_owned: wgpu::TextureView = if hzb_need_max {
            hzb.max_pyramid.full_view.clone()
        } else {
            hzb.min_pyramid.full_view.clone()
        };
        let min_bind_owned: wgpu::TextureView = if hzb_need_min {
            hzb.min_pyramid.full_view.clone()
        } else {
            hzb.max_pyramid.full_view.clone()
        };

        let lod_stride = 1u32;
        let meshlet_phase = 0u32;
        let meshlet_stride_spatial = false;

        // First cull pass (no HZB - uses full mip0 as coarse cull)
        {
            let max_bind: &wgpu::TextureView = &max_bind_owned;
            let min_bind: &wgpu::TextureView = &min_bind_owned;
            for &vis_idx in ctx.meshlet_indices {
                let dc = ctx.visible[vis_idx];
                let Some(md) = dc.meshlet_data.as_ref() else {
                    log::warn!("meshlet index {} has no meshlet_data; skipping", vis_idx);
                    continue;
                };
                let ptr = std::sync::Arc::as_ptr(md) as u64;
                if let Some(cluster_set) = renderer.assets.cluster_cache.get(&ptr) {
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

        // Depth prepass
        pass_solid::pass_depth_prepass(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);

        // Build HZB from depth
        if let (Some(baker), Some(hzb)) = (renderer.hzb_baker.as_mut(), renderer.hzb.as_ref()) {
            baker.build_from_depth(
                &renderer.device,
                &mut encoder,
                &depth_read_view,
                hzb_need_max,
                hzb_need_min,
                &hzb.max_pyramid,
                &hzb.min_pyramid,
            );
        } else {
            meshlet_hzb_prepass_done = false;
            break 'hzb_prepass;
        }

        // Second cull pass (with HZB)
        if let Some(cluster_renderer) = renderer.cluster_renderer.as_ref() {
            let max_bind: &wgpu::TextureView = &max_bind_owned;
            let min_bind: &wgpu::TextureView = &min_bind_owned;
            for &vis_idx in ctx.meshlet_indices {
                let dc = ctx.visible[vis_idx];
                let Some(md) = dc.meshlet_data.as_ref() else {
                    continue;
                };
                let ptr = std::sync::Arc::as_ptr(md) as u64;
                if let Some(cluster_set) = renderer.assets.cluster_cache.get(&ptr) {
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
        } else {
            meshlet_hzb_prepass_done = false;
            break 'hzb_prepass;
        }

        meshlet_hzb_prepass_done = true;
        }
    }

    if !meshlet_hzb_prepass_done && !ctx.meshlet_indices.is_empty() {
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
                let Some(md) = dc.meshlet_data.as_ref() else {
                    log::warn!("meshlet index {} has no meshlet_data; skipping", vis_idx);
                    continue;
                };
                let ptr = std::sync::Arc::as_ptr(md) as u64;
                if let Some(cluster_set) = renderer.assets.cluster_cache.get(&ptr) {
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
        pass_solid::pass_solid_and_outline(
            renderer,
            &mut encoder,
            shade_view,
            &depth_view,
            ctx,
            meshlet_hzb_prepass_done,
            &scene_pl,
        );
        renderer.write_gpu_timestamp(&mut encoder); // solid end
    }

    if !ctx.performance_mode_active && ctx.wireframe_supported && mode == DisplayMode::Wireframe {
        pass_wireframe::pass_wireframe(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
    }

    let edge_worthy = !ctx.performance_mode_active
        && ctx.adaptive_quality != AdaptiveQuality::Low
        && (mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine);
    let has_overlay = ctx.visible.iter().any(|dc| dc.overlay_color.is_some());
    if edge_worthy || has_overlay {
        pass_edge::pass_edge_overlay(renderer, &mut encoder, shade_view, &depth_view, ctx, edge_worthy, &scene_pl);
    }

    let has_selection = ctx.visible.iter().any(|dc| dc.selected);
    if has_selection && !ctx.performance_mode_active {
        pass_selection::pass_selection_fill(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
        if ctx.wireframe_supported && ctx.adaptive_quality != AdaptiveQuality::Low {
            pass_selection::pass_selection_edge(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
        }
        pass_selection::pass_selection_bbox(&mut encoder, shade_view, &depth_view, ctx, &scene_pl, &mut renderer.flat_pool, ctx.wireframe_supported);
    }

    // Transparent pass: iterate sorted draw calls with alpha blend pipeline
    if !ctx.transparent_order.is_empty() {
        for &idx in ctx.transparent_order {
            let dc = &draw_calls[idx];
            if dc.vertices.is_empty() && dc.meshlet_data.is_none() {
                continue;
            }
            // Transparent rendering infrastructure ready; full draw-call dispatch
            // will be integrated when draw_call_mesh() wiring is complete.
            // For now this validates compilation of the pipeline/sort/pass chain.
            let _ = (dc, &scene_pl.solid_alpha);
        }
    }

    if renderer.hdr_post_processing {
        if let Some(ref fx) = renderer.post_fx {
            let pl = &renderer.post_fx_pipelines;
            let w = renderer.config.width.max(1);
            let h = renderer.config.height.max(1);
            let proj = ctx.camera_proj.to_cols_array_2d();
            let inv_proj = ctx.camera_inv_proj.to_cols_array_2d();

            // ── Auto Exposure ──
            let _exposure = renderer.auto_exposure.update(
                &renderer.device, &renderer.queue, &mut encoder,
                &fx.hdr_view, w, h, 0.016,
            );

            // ── SSR (screen-space reflections) ──
            if renderer.enable_ssr {
                if let Some(ref ssr) = renderer.ssr_pass {
                    if let Some(ref hzb) = renderer.hzb {
                        ssr.trace(
                            &renderer.device, &renderer.queue, &mut encoder,
                            &fx.hdr_view, &depth_read_view,
                            &hzb.max_pyramid.full_view, &fx.hdr_view,
                            w, h,
                            ctx.camera_inv_proj, Mat4::IDENTITY,
                        );
                    }
                }
            }

            // ── Volumetric Fog ──
            if renderer.enable_volumetric_fog {
                if let Some(ref fog) = renderer.volumetric_fog {
                    fog.compute(
                        &renderer.device, &renderer.queue, &mut encoder,
                        &depth_read_view, &fx.hdr_view, w, h,
                        ctx.camera_inv_proj,
                        Vec3::from(ctx.camera_pos),
                        Vec3::new(0.5, -0.8, 0.3),
                        Vec3::new(1.0, 0.9, 0.7),
                        Vec3::new(0.6, 0.7, 0.8),
                        0.02, 0.5, 100.0, 32,
                    );
                }
            }

            // ── Motion Blur ──
            if renderer.enable_motion_blur {
                if let Some(ref mb) = renderer.motion_blur {
                    mb.apply(
                        &renderer.device, &renderer.queue, &mut encoder,
                        &fx.hdr_view, &depth_read_view, &depth_read_view,
                        &fx.hdr_view, w, h, 16, 0.5,
                    );
                }
            }

            // ── DOF ──
            if renderer.enable_dof {
                if let Some(ref dof) = renderer.dof_pass {
                    dof.apply(
                        &renderer.device, &renderer.queue, &mut encoder,
                        &fx.hdr_view, &depth_read_view, &fx.hdr_view,
                        w, h, 5.0, 2.0,
                    );
                }
            }

            // ── Color Grading ──
            if renderer.enable_color_grading {
                if let Some(ref cg) = renderer.color_grading {
                    cg.apply(
                        &renderer.device, &renderer.queue, &mut encoder,
                        &fx.hdr_view, &fx.hdr_view, w, h, 1.0,
                    );
                }
            }

            // Bloom prefilter (compute dispatch: read HDR, write half-res bloom)
            pass_post::pass_bloom_prefilter(&renderer.device, &mut encoder, pl, fx);
            // SSAO (read depth, write AO)
            pass_post::pass_ssao(&renderer.device, &mut encoder, pl, fx,
                &depth_read_view, &renderer.ssao_noise_view, &proj, &inv_proj);
            // SSAO blur (read AO + depth, write blurred AO)
            pass_post::pass_ssao_blur(&renderer.device, &mut encoder, pl, fx, &depth_read_view);

            // ── TAA (temporal anti-aliasing) ──
            if renderer.enable_taa {
                if let Some(ref mut taa) = renderer.taa_pass {
                    taa.ensure_history(&renderer.device, w, h);
                    taa.resolve(
                        &renderer.device, &renderer.queue, &mut encoder,
                        &fx.hdr_view, &depth_read_view, &depth_read_view,
                        &fx.hdr_view,
                        0.05, 1.0,
                    );
                }
            }

            // Tonemap + FXAA + Bloom + SSAO
            pass_post::pass_tonemap_hdr_to_post_ldr(&mut encoder, pl, fx, ctx.bg_color);
            // Blit to swapchain
            pass_post::pass_blit_post_ldr_to_swapchain(&mut encoder, pl, fx, &view, ctx.bg_color);
            renderer.write_gpu_timestamp(&mut encoder); // post end
        }
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
    renderer.resolve_gpu_timestamps(&mut encoder);
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
        gpu_pass_times_us: None,
    }
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

        // Split into meshlet and standard draws within this light group
        let mut meshlet_draws: Vec<usize> = Vec::new();
        let mut standard_draws: Vec<usize> = Vec::new();
        for &i in &ctx.solid_order[start..end] {
            if meshlet_set.contains(&i) {
                if draw_meshlets {
                    meshlet_draws.push(i);
                } else {
                    // Meshlet path disabled: fall back to stable per-object drawing.
                    standard_draws.push(i);
                }
            } else {
                standard_draws.push(i);
            }
        }

        // Meshlet path (unchanged: uses draw_clustered)
        for &i in &meshlet_draws {
            if !draw_meshlets { continue; }
            let dc = ctx.visible[i];
            let md = match dc.meshlet_data.as_ref() { Some(md) => md, None => continue };
            let ptr = std::sync::Arc::as_ptr(md) as u64;
            if !(renderer.cluster_renderer.is_some() && renderer.assets.cluster_cache.contains_key(&ptr)) { continue; }
            let diffuse_color = if ctx.mode == DisplayMode::HiddenLine {
                [0.08, 0.08, 0.08, 1.0]
            } else {
                [dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]
            };
            let uniforms = SceneUniforms {
                mvp: dc.mvp.to_cols_array_2d(),
                model: dc.model_matrix.to_cols_array_2d(),
                camera_pos: [dc.camera_pos.x, dc.camera_pos.y, dc.camera_pos.z, 1.0],
                light_dirs: head_dc.light_dirs, light_colors: head_dc.light_colors,
                light_types: head_dc.light_types, light_positions: head_dc.light_positions,
                spot_params: head_dc.spot_params,
                light_count: [head_dc.light_count as f32, 0.0, 0.0, 0.0],
                diffuse_color,
                ambient_color: [dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0],
                specular_color: [dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0],
                shininess: [dc.shininess, 0.0, 0.0, 0.0],
                clip_planes: clip_arr, clip_count,
                pbr_base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
                pbr_metallic_roughness: [dc.metallic, dc.roughness, 0.0, 0.0],
                ibl_diffuse: renderer.ibl_diffuse, ibl_specular: renderer.ibl_specular,
                csm_view_proj: csm_to_uniform(&ctx.csm_view_proj),
                csm_split_depths: ctx.csm_split_depths,
                shadow_params: ctx.shadow_params,
            };
            if let Some(offset) = renderer.phong_pool.push_scene(&uniforms) {
                let mat_bg = albedo_material_bind_group(
                    &mut renderer.texture_cache, &renderer.device,
                    &renderer.pipelines.pbr_material_bgl, &renderer.queue, dc,
                );
                pass.set_bind_group(0, renderer.phong_pool.bind_group(), &[offset]);
                pass.set_bind_group(1, mat_bg, &[]);
                if let Some(ref csm) = renderer.csm_shadow { pass.set_bind_group(2, &csm.bind_group, &[]); }
                pass.set_bind_group(3, &renderer.ibl_instance_bind_group, &[]);
                if let Some(cluster_set) = renderer.assets.cluster_cache.get(&ptr) {
                    if let Some(cluster_renderer) = renderer.cluster_renderer.as_ref() {
                        cluster_renderer.draw_clustered(pass, cluster_set);
                    }
                }
            }
        }

        // Standard mesh path: draw per object to preserve exact transforms.
        // The previous instanced path can collapse placements when instance payload
        // and per-draw state drift apart; correctness is prioritized here.
        if !standard_draws.is_empty() {
            for &i in &standard_draws {
                let dc = ctx.visible[i];
                let diffuse = if ctx.mode == DisplayMode::HiddenLine {
                    [0.08, 0.08, 0.08, 1.0]
                } else {
                    [dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0]
                };
                let uniforms = SceneUniforms {
                    mvp: dc.mvp.to_cols_array_2d(),
                    model: dc.model_matrix.to_cols_array_2d(),
                    camera_pos: [dc.camera_pos.x, dc.camera_pos.y, dc.camera_pos.z, 1.0],
                    light_dirs: head_dc.light_dirs, light_colors: head_dc.light_colors,
                    light_types: head_dc.light_types, light_positions: head_dc.light_positions,
                    spot_params: head_dc.spot_params,
                    light_count: [head_dc.light_count as f32, 0.0, 0.0, 0.0],
                    diffuse_color: diffuse,
                    ambient_color: [dc.ambient_color.x, dc.ambient_color.y, dc.ambient_color.z, 1.0],
                    specular_color: [dc.specular_color.x, dc.specular_color.y, dc.specular_color.z, 1.0],
                    shininess: [dc.shininess, 0.0, 0.0, 0.0],
                    clip_planes: clip_arr, clip_count,
                    pbr_base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
                    pbr_metallic_roughness: [dc.metallic, dc.roughness, 0.0, 0.0],
                    ibl_diffuse: renderer.ibl_diffuse, ibl_specular: renderer.ibl_specular,
                    csm_view_proj: csm_to_uniform(&ctx.csm_view_proj),
                    csm_split_depths: ctx.csm_split_depths,
                    shadow_params: ctx.shadow_params,
                };
                let Some(offset) = renderer.phong_pool.push_scene(&uniforms) else { continue };

                // Keep storage instance slot 0 in sync for shaders that source per-instance fields.
                let one = [InstanceData {
                    model: dc.model_matrix.to_cols_array_2d(),
                    mvp: dc.mvp.to_cols_array_2d(),
                    diffuse_color: diffuse,
                    base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
                    metallic_roughness: [dc.metallic, dc.roughness, 0.0, 0.0],
                }];
                renderer.queue.write_buffer(&renderer.instance_buffer, 0, bytemuck::cast_slice(&one));

                let mat_bg = albedo_material_bind_group(
                    &mut renderer.texture_cache, &renderer.device,
                    &renderer.pipelines.pbr_material_bgl, &renderer.queue, dc,
                );
                pass.set_bind_group(0, renderer.phong_pool.bind_group(), &[offset]);
                pass.set_bind_group(1, mat_bg, &[]);
                if let Some(ref csm) = renderer.csm_shadow { pass.set_bind_group(2, &csm.bind_group, &[]); }
                pass.set_bind_group(3, &renderer.ibl_instance_bind_group, &[]);

                if let Some(mesh_id) = ctx.mesh_handles[i] {
                    renderer.draw_mesh_batched(pass, mesh_id, &mut last_bound_mesh);
                }
            }
        }
        start = end;
    }
}

fn csm_to_uniform(vps: &[Mat4; CSM_CASCADE_COUNT]) -> [[f32; 4]; 16] {
    let mut arr = [[0.0f32; 4]; 16];
    for (i, vp) in vps.iter().enumerate() {
        let cols = vp.to_cols_array_2d();
        for r in 0..4 {
            arr[i * 4 + r] = cols[r];
        }
    }
    arr
}
