use crate::adaptive_quality::AdaptiveQuality;
use crate::render_action::DrawCall;
use crate::vertex::CSM_CASCADE_COUNT;
use crate::FrameStats;
use glam::Mat4;
use rc3d_core::DisplayMode;

mod pass_edge;
pub(crate) mod pass_effects;
mod pass_grid;
pub(crate) mod pass_markup;
mod meshlet_cull;
mod pass_post;
mod pass_selection;
mod pass_shadow;
mod pass_solid;
mod pass_transparent;
pub(crate) mod pass_text;
mod pass_viewport;
mod ss_edge;
pub(crate) mod pass_wireframe;
mod pass_hud;
mod pass_shared;

pub(crate) mod draw_opaque;

#[cfg(test)]
mod pass_markup_tests;
use draw_opaque::draw_opaque_triangle_batches;
pub(crate) use meshlet_cull::submit_meshlet_cull;

pub(crate) struct PassContext<'a> {
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
    pub run_shadow_pass: bool,
    /// Camera projection matrix (for SSAO depth reconstruction)
    pub camera_proj: Mat4,
    /// Camera inverse projection matrix
    pub camera_inv_proj: Mat4,
    /// Combined view-projection matrix (proj * view).
    pub scene_vp: Mat4,
    /// Previous frame's view-projection matrix (for velocity buffer).
    pub prev_vp: Mat4,
    pub effect_commands: &'a pass_effects::EffectCommands,
    pub light_sets: &'a crate::light_set::LightSetTable,
}

/// Final color target for the frame (swapchain or an application-owned render target).
pub(crate) enum FramePresentation<'a> {
    Swapchain,
    /// Same render path as the swapchain (`hdr_off`, `ldr_fxaa_off` enforced by callers for correct resolve).
    OffscreenSurface {
        output_texture: &'a wgpu::Texture,
        output_view: &'a wgpu::TextureView,
        width_px: u32,
        height_px: u32,
    },
}

pub(super) fn execute_passes(
    renderer: &mut crate::renderer::Renderer,
    ctx: &PassContext<'_>,
    draw_calls: &[DrawCall],
    _frame_counter: u64,
    mut post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    presentation: FramePresentation<'_>,
) -> FrameStats {
    let t_entry = std::time::Instant::now();

    let t_surface_start = std::time::Instant::now();
    let ((scene_tex_raw, eff_width, eff_height), mut acquired_swapchain, _w, _h) = pass_shared::acquire_surface(
        renderer,
        &presentation,
        |tex_ptr, w, h| (tex_ptr, w, h),
    );

    let view: &wgpu::TextureView = match &acquired_swapchain {
        Some((_s, vw)) => vw,
        None => match &presentation {
            FramePresentation::OffscreenSurface { output_view, .. } => output_view,
            FramePresentation::Swapchain => return FrameStats::default(),
        },
    };

    let ew = eff_width.max(1);
    let eh = eff_height.max(1);

    if renderer.gpu.depth_texture.is_none() {
        renderer.create_depth_texture();
    }
    let Some((_, depth_view, depth_read_view)) = renderer.gpu.depth_texture.as_ref() else {
        return FrameStats::default();
    };
    let depth_view = depth_view.clone();
    let depth_read_view = depth_read_view.clone();

    if renderer.hdr_post_processing {
        renderer.ensure_post_fx_targets();
    }
    let use_ldr_fxaa = !renderer.hdr_post_processing && renderer.enable_ldr_fxaa;
    if use_ldr_fxaa {
        renderer.ensure_ldr_shade_target();
    }
    let scene_pl = renderer
        .gpu.pipelines
        .for_shaded_target(ctx.depth_reversed_z, renderer.hdr_post_processing)
        .clone();
    let shade_color_view = if renderer.hdr_post_processing {
        if let Some(fx) = renderer.gpu.post_fx.as_ref() {
            fx.hdr_view.clone()
        } else {
            log::warn!("hdr_post_processing is enabled but post_fx is missing; falling back to swapchain view");
            view.clone()
        }
    } else if use_ldr_fxaa {
        renderer
            .gpu.ldr_shade_view
            .as_ref()
            .expect("LDR shade view: enable_taa or enable_fxaa requires post-processing to be initialized")
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

    renderer.encode_skinning_compute(&mut encoder, ctx.visible, ctx.mesh_handles);

    // ── Background pass (gradient / image / solid) ──
    if let Some(ref bg) = renderer.gpu.bg_pass {
        bg.encode(&renderer.device, &renderer.queue, &mut encoder, shade_view, &renderer.gpu.bg_settings, ew, eh, ctx.camera_inv_proj);
    }

    // ── GPU compute culling dispatch ──
    if renderer.gpu.gpu_cull_enabled {
        if let (Some(ref cull_pass), Some(ref bg)) =
            (&renderer.gpu.gpu_cull_pass, &renderer.gpu.gpu_cull_bg)
        {
            // Must match the transform upload count (the cull replaces CPU
            // culling, so it runs over ALL uploaded draw calls, not just the
            // CPU-visible subset).
            let obj_count = renderer.frame.gpu_cull_object_count as usize;
            if obj_count > 0 {
                #[cfg(feature = "profiler")]
                let _span_cull = tracy_client::span!("gpu_cull");
                cull_pass.dispatch(&mut encoder, bg, obj_count as u32);
            }

            // Copy visible count + instance indices to staging for frame-delayed readback
            if let (Some(ref indirect_buf), Some(ref instance_buf), Some(ref staging)) =
                (&renderer.gpu.indirect_args_buffer,
                 &renderer.gpu.instance_indices_buffer,
                 &renderer.gpu.gpu_cull_staging)
            {
                // Copy instance_count from indirect_args[0] → staging[0..4]
                encoder.copy_buffer_to_buffer(indirect_buf, 4, staging, 0, 4);
                // Copy instance_indices → staging[4..]
                let idx_size = (obj_count as u64 * 4).min(staging.size() - 4);
                if idx_size > 0 {
                    encoder.copy_buffer_to_buffer(instance_buf, 0, staging, 4, idx_size);
                }
            }
            renderer.frame.gpu_cull_ready = true;
        }
    }

    let ti_shadow = renderer.gpu_timer.begin(&mut encoder, "CSM Shadow");
    if ctx.run_shadow_pass {
        #[cfg(feature = "profiler")]
        let _span_shadow = tracy_client::span!("shadow");
        pass_shadow::pass_shadow_depth(renderer, &mut encoder, ctx);
    }
    renderer.gpu_timer.end(&mut encoder, ti_shadow);

    // ── Omni Shadow Pass ──
    let ti_omni = renderer.gpu_timer.begin(&mut encoder, "OmniShadow");
    if renderer.enable_omni_shadows {
        crate::shadow_omni::render_omni_shadow_pass(renderer, &mut encoder, ctx);
    }
    renderer.gpu_timer.end(&mut encoder, ti_omni);

    let mode = ctx.mode;

    let solid_mode = matches!(
        mode,
        DisplayMode::Shaded | DisplayMode::ShadedWithEdges | DisplayMode::HiddenLine | DisplayMode::Flat | DisplayMode::FlatWithEdge
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
        && mode != DisplayMode::Flat
        && mode != DisplayMode::FlatWithEdge
        && !ctx.meshlet_indices.is_empty()
        && renderer.gpu.cluster_renderer.is_some()
        && renderer.gpu.hzb.is_some()
        && renderer.gpu.hzb_baker.is_some()
        && renderer.gpu.gpu_capability.meshlet_gpu_cull_enabled
        && !(renderer.interaction_active && renderer.skip_prepass_interaction);
    let run_meshlet_hzb_prepass = if requested_meshlet_hzb_prepass {
        true
    } else {
        if solid_mode
            && !ctx.meshlet_indices.is_empty()
            && (renderer.gpu.cluster_renderer.is_none()
                || renderer.gpu.hzb.is_none()
                || renderer.gpu.hzb_baker.is_none())
        {
            log::warn!("meshlet HZB prepass requested but resources are incomplete; using fallback meshlet cull path");
        }
        false
    };

    let mut meshlet_hzb_prepass_done = false;

    'hzb_prepass: {
        if run_meshlet_hzb_prepass {
        let (mip_max, hzb_dims_xy) = match (
            renderer.gpu.hzb.as_ref(),
            renderer.gpu.cluster_renderer.as_ref(),
        ) {
            (Some(hzb), Some(_)) => (
                hzb.max_pyramid.mip_count.saturating_sub(1),
                (hzb.max_pyramid.width, hzb.max_pyramid.height),
            ),
            _ => break 'hzb_prepass,
        };

        // First cull pass (no HZB - uses full mip0 as coarse cull)
        submit_meshlet_cull(renderer, &mut encoder, ctx, false, hzb_dims_xy, mip_max, hzb_need_max, hzb_need_min);

        // Depth prepass
        pass_solid::pass_depth_prepass(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);

        // Build HZB from depth
        if let (Some(baker), Some(hzb)) = (renderer.gpu.hzb_baker.as_mut(), renderer.gpu.hzb.as_ref()) {
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
        submit_meshlet_cull(renderer, &mut encoder, ctx, true, hzb_dims_xy, mip_max, hzb_need_max, hzb_need_min);

        meshlet_hzb_prepass_done = true;
        }
    }

    if !meshlet_hzb_prepass_done && !ctx.meshlet_indices.is_empty() && mode != DisplayMode::Flat && mode != DisplayMode::FlatWithEdge && renderer.gpu.gpu_capability.meshlet_gpu_cull_enabled {
        let (fallback_dims, fallback_mip) = if let Some(hzb) = renderer.gpu.hzb.as_ref() {
            ((hzb.max_pyramid.width, hzb.max_pyramid.height), hzb.max_pyramid.mip_count.saturating_sub(1))
        } else {
            ((1, 1), 0)
        };
        submit_meshlet_cull(renderer, &mut encoder, ctx, false, fallback_dims, fallback_mip, hzb_need_max, hzb_need_min);
    }

    if solid_mode && renderer.enable_cluster_lights {
        crate::cluster_lighting::dispatch_cluster_light_cull(
            renderer,
            &mut encoder,
            ctx.visible,
            ctx.light_sets,
            ctx.camera_inv_proj,
            ew,
            eh,
        );
    }

    let ti_solid = renderer.gpu_timer.begin(&mut encoder, "Solid+Outline");
    if solid_mode {
        #[cfg(feature = "profiler")]
        let _span_solid = tracy_client::span!("solid");
        pass_solid::pass_solid_and_outline(
            renderer,
            &mut encoder,
            shade_view,
            &depth_view,
            ctx,
            meshlet_hzb_prepass_done,
            &scene_pl,
        );
        let want_section_caps = !renderer.frame.clip_planes.is_empty()
            && renderer.frame.section_cap_tints.iter().any(|c| c.is_some());
        if want_section_caps {
            renderer.render_section_caps_from_ctx(
                &mut encoder,
                shade_view,
                &depth_view,
                ctx.visible,
                ctx.solid_order,
                ctx.mesh_handles,
                &scene_pl,
            );
        }
    }
    renderer.gpu_timer.end(&mut encoder, ti_solid);

    // ── Transparent pass: alpha-blended draw calls ──
    if !ctx.transparent_order.is_empty() {
        if renderer.enable_wboit && renderer.hdr_post_processing {
            // WBOIT path: accumulate into MRT buffers, then composite onto scene.
            // Clone views upfront to avoid borrowing renderer.gpu.post_fx while
            // passing renderer mutably to the pass functions.
            let wboit_views = renderer.gpu.post_fx.as_ref().map(|fx| {
                (fx.wboit_accum_view.clone(), fx.wboit_revealage_view.clone())
            });
            if let Some((accum_view, revealage_view)) = wboit_views {
                let ti_transparent = renderer.gpu_timer.begin(&mut encoder, "WBOIT Accumulate");
                pass_transparent::pass_transparent_wboit(
                    renderer,
                    &mut encoder,
                    &accum_view,
                    &revealage_view,
                    &depth_view,
                    ctx,
                    &scene_pl,
                );
                renderer.gpu_timer.end(&mut encoder, ti_transparent);

                let ti_composite = renderer.gpu_timer.begin(&mut encoder, "WBOIT Composite");
                pass_transparent::pass_wboit_composite(
                    renderer,
                    &mut encoder,
                    shade_view,
                    &accum_view,
                    &revealage_view,
                );
                renderer.gpu_timer.end(&mut encoder, ti_composite);
            }
        } else {
            // Traditional back-to-front painter's algorithm
            let ti_transparent = renderer.gpu_timer.begin(&mut encoder, "Transparent");
            pass_transparent::pass_transparent(
                renderer,
                &mut encoder,
                shade_view,
                &depth_view,
                ctx,
                &scene_pl,
            );
            renderer.gpu_timer.end(&mut encoder, ti_transparent);
        }
    }

    let ti_effects = renderer.gpu_timer.begin(&mut encoder, "Effects");
    // ── Effect passes (Decal, Volume, PointCloud) ──
    if !ctx.effect_commands.is_empty() {
        if !ctx.effect_commands.decals.is_empty() {
            renderer.ensure_decal_pass();
            if let Some(ref pass) = renderer.gpu.decal_pass {
                pass.encode(
                    &renderer.device, &renderer.queue, &mut encoder,
                    shade_view, &depth_view, &ctx.effect_commands.decals, ew, eh,
                );
            }
        }
        if !ctx.effect_commands.volumes.is_empty() {
            renderer.ensure_volume_pass();
            if let Some(ref pass) = renderer.gpu.volume_pass {
                pass.encode(
                    &renderer.device, &renderer.queue, &mut encoder,
                    shade_view, &depth_view, &ctx.effect_commands.volumes, ew, eh,
                );
            }
        }
        if !ctx.effect_commands.point_clouds.is_empty() {
            renderer.ensure_point_cloud_pass();
            if let Some(ref pass) = renderer.gpu.point_cloud_pass {
                pass.encode(
                    &renderer.device, &renderer.queue, &mut encoder,
                    shade_view, &depth_view, &ctx.effect_commands.point_clouds,
                    ctx.camera_proj, ctx.camera_inv_proj, ew, eh,
                );
            }
        }
    }
    renderer.gpu_timer.end(&mut encoder, ti_effects);

    if !ctx.performance_mode_active && ctx.wireframe_supported && mode == DisplayMode::Wireframe && mode != DisplayMode::Flat {
        #[cfg(feature = "profiler")]
        let _span_wireframe = tracy_client::span!("wireframe");
        pass_wireframe::pass_wireframe(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
    }

    let has_selection = ctx.visible.iter().any(|dc| dc.selected);
    let ss_outline = renderer.screen_space_selection_outline
        && ctx.adaptive_quality != AdaptiveQuality::Low;
    let run_geom_sel_edge = has_selection
        && !ctx.performance_mode_active
        && !ss_outline
        && ctx.wireframe_supported
        && ctx.adaptive_quality != AdaptiveQuality::Low;

    let edge_worthy = !ctx.performance_mode_active
        && (mode == DisplayMode::ShadedWithEdges || mode == DisplayMode::HiddenLine || mode == DisplayMode::FlatWithEdge)
        && mode != DisplayMode::Flat;
    let has_overlay = ctx.visible.iter().any(|dc| dc.overlay_color.is_some());
    let defer_line_overlays = use_ldr_fxaa;
    if (edge_worthy || has_overlay) && !defer_line_overlays {
        #[cfg(feature = "profiler")]
        let _span_edge = tracy_client::span!("edge");
        pass_edge::pass_edge_overlay(renderer, &mut encoder, shade_view, &depth_view, ctx, edge_worthy, &scene_pl);
    }

    if has_selection && !ctx.performance_mode_active && mode != DisplayMode::Flat && mode != DisplayMode::FlatWithEdge {
        pass_selection::pass_selection_fill(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
        if ss_outline {
            let shade_fmt = if renderer.hdr_post_processing {
                wgpu::TextureFormat::Rgba16Float
            } else {
                renderer.config.format
            };
            let scene_tex_ptr: *const wgpu::Texture = if renderer.hdr_post_processing {
                std::ptr::from_ref(&renderer.gpu.post_fx.as_ref().unwrap().hdr_tex)
            } else if use_ldr_fxaa {
                std::ptr::from_ref(renderer.gpu.ldr_shade_tex.as_ref().expect("LDR shade texture: enable_taa or enable_fxaa requires post-processing to be initialized"))
            } else {
                scene_tex_raw
            };
            crate::selection_outline::encode_selection_outline_pass(
                renderer,
                &mut encoder,
                ctx,
                shade_view,
                shade_fmt,
                scene_tex_ptr,
                ew,
                eh,
            );
        } else if run_geom_sel_edge && !defer_line_overlays {
            pass_selection::pass_selection_edge(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
        }
        if !defer_line_overlays {
            pass_selection::pass_selection_bbox(&mut encoder, shade_view, &depth_view, ctx, &scene_pl, &mut renderer.gpu.flat_pool, ctx.wireframe_supported);
        }
    }

    // Transparent pass: sorted draw calls with alpha blend pipeline.
    // Deferred until draw_call_mesh() wiring is complete.

    if use_ldr_fxaa {
        pass_post::pass_fxaa_ldr_to_swapchain(
            &renderer.device,
            &mut encoder,
            &renderer.gpu.post_fx_pipelines,
            renderer.gpu.ldr_shade_view.as_ref().expect("LDR shade view"),
            view,
            ctx.bg_color,
        );
    }

    // Screen-space edge detection: replaces geometry edges during interaction
    {
        let ss_resources = if renderer.screen_space_edges && renderer.interaction_active {
            let gpu = &renderer.gpu;
            match (
                gpu.ss_edge_pipeline.as_ref(),
                gpu.ss_edge_bgl.as_ref(),
                gpu.ss_edge_uniform.as_ref(),
                gpu.ss_edge_sampler.as_ref(),
            ) {
                (Some(pl), Some(bgl), Some(ub), Some(samp)) => Some(ss_edge::SsEdgeResources {
                    pipeline: pl,
                    bind_group_layout: bgl,
                    uniform_buf: ub,
                    sampler: samp,
                }),
                _ => None,
            }
        } else {
            None
        };
        let ss_params = ss_edge::SsEdgeParams {
            edge_color: renderer.feature_edge_color,
            threshold: renderer.ss_edge_threshold,
        };
        ss_edge::encode_ss_edge_pass(
            &renderer.device, &renderer.queue,
            ss_resources.as_ref(), &ss_params,
            &mut encoder, view, &depth_read_view, ew, eh,
        );
    }

    if defer_line_overlays {
        if edge_worthy || has_overlay {
            pass_edge::pass_edge_overlay(renderer, &mut encoder, view, &depth_view, ctx, edge_worthy, &scene_pl);
        }
        if has_selection && !ctx.performance_mode_active && mode != DisplayMode::Flat && mode != DisplayMode::FlatWithEdge {
            if run_geom_sel_edge {
                pass_selection::pass_selection_edge(renderer, &mut encoder, view, &depth_view, ctx, &scene_pl);
            }
            pass_selection::pass_selection_bbox(
                &mut encoder,
                view,
                &depth_view,
                ctx,
                &scene_pl,
                &mut renderer.gpu.flat_pool,
                ctx.wireframe_supported,
            );
        }
    }

    let ti_post = renderer.gpu_timer.begin(&mut encoder, "PostProcess");
    if renderer.hdr_post_processing {
        pass_post::encode_post_processing(
            renderer,
            &mut encoder,
            &depth_read_view,
            view,
            ew,
            eh,
            ctx,
        );
    }
    renderer.gpu_timer.end(&mut encoder, ti_post);

    renderer.gpu.outline_pool.flush(&renderer.queue);
    renderer.gpu.phong_pool.flush(&renderer.queue);
    renderer.gpu.shadow_pool.flush(&renderer.queue);
    renderer.gpu.flat_pool.flush(&renderer.queue);
    renderer.gpu.line_pool.flush(&renderer.queue);
    renderer.gpu.section_cap_pool.flush(&renderer.queue);

    let ti_hud = renderer.gpu_timer.begin(&mut encoder, "HUD+Overlay");

    // Ground plane grid overlay
    if renderer.grid_enabled {
        pass_grid::pass_grid(
            renderer,
            &mut encoder,
            view,
            &depth_view,
            renderer.frame.scene_vp,
            renderer.frame.scene_camera_pos,
            ctx.depth_reversed_z,
        );
    }

    // Viewport border overlay
    pass_viewport::encode_viewport_borders(
        renderer,
        &mut encoder,
        view,
        ew,
        eh,
    );

    // Markup overlay
    #[cfg(feature = "profiler")]
    let _span_markup = tracy_client::span!("markup");
    // Annotation occlusion depth: harvest last frame's async readback (if any),
    // then encode a fresh whole-screen downsample + copy for the next frame.
    const ALIGN: u32 = 256; // COPY_BYTES_PER_ROW_ALIGNMENT
    let ds = 4u32;
    let dw = ew.div_ceil(ds);
    let dh = eh.div_ceil(ds);
    let row_bytes = (dw * 4).div_ceil(ALIGN) * ALIGN;
    let dims_changed = renderer.frame.occlusion_dims.0 != dw || renderer.frame.occlusion_dims.1 != dh;

    // 1) Harvest a completed (non-blocking) readback from a previous frame.
    if let Some(pending) = renderer.frame.occlusion_map_pending.take() {
        match pending.load(std::sync::atomic::Ordering::Acquire) {
            1 => {
                if let Some(ref buf) = renderer.frame.occlusion_capture_buf {
                    let (bw, bh, brow) = renderer.frame.occlusion_dims;
                    let padded_w = (brow / 4) as usize;
                    {
                        let mapped = buf.slice(..).get_mapped_range();
                        let raw: &[f32] = bytemuck::cast_slice(&mapped);
                        let mut data = Vec::with_capacity((bw * bh) as usize);
                        for row in 0..bh as usize {
                            let start = row * padded_w;
                            data.extend_from_slice(&raw[start..start + bw as usize]);
                        }
                        renderer.frame.occlusion_data = Some((data, bw, bh));
                    }
                    buf.unmap();
                }
            }
            2 => {} // mapping failed; buffer is back to unmapped state, retry below
            _ => {
                // Mapping still in flight: keep waiting and skip this frame's
                // capture (the buffer must not be written while mapped).
                renderer.frame.occlusion_map_pending = Some(pending);
            }
        }
    }

    let occlusion_buf_free = renderer.frame.occlusion_map_pending.is_none();
    if dims_changed && occlusion_buf_free {
        let size = (row_bytes * dh) as u64;
        renderer.frame.occlusion_capture_buf = Some(renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion depth capture"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        renderer.frame.occlusion_dims = (dw, dh, row_bytes);
    }

    // 2) Downsample the full depth buffer into a small R32Float grid covering
    //    the WHOLE screen, then copy it into the staging buffer. A direct
    //    depth-to-buffer copy would only capture the top-left dw×dh pixels.
    let mut occlusion_captured = false;
    if occlusion_buf_free && !(dims_changed && renderer.frame.occlusion_capture_buf.is_none()) {
        ensure_occlusion_downsample_resources(renderer, dw, dh);
        if let (Some(pipeline), Some(bgl), Some((ds_tex, ds_view)), Some(buf)) = (
            renderer.gpu.occlusion_downsample_pipeline.as_ref(),
            renderer.gpu.occlusion_downsample_bgl.as_ref(),
            renderer.gpu.occlusion_downsample_tex.as_ref(),
            renderer.frame.occlusion_capture_buf.as_ref(),
        ) {
            let bg = renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Occlusion Downsample BG"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&depth_read_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(ds_view) },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Occlusion Downsample"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
            }
            encoder.copy_texture_to_buffer(
                ds_tex.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row_bytes),
                        rows_per_image: Some(dh),
                    },
                },
                wgpu::Extent3d { width: dw, height: dh, depth_or_array_layers: 1 },
            );
            occlusion_captured = true;
        }
    }

    // Borrow occlusion data without cloning (~0.5 MB/frame at 1080p).
    let occlusion = renderer.frame.occlusion_data.take();
    // Static frame: reuse cached projection output.
    let is_static = renderer.frame.bvh_fully_static && renderer.frame.static_frame_count >= 2;
    if !is_static {
        let (projected, wl) = pass_markup::compute_projected_markup(
            renderer,
            ctx.effect_commands,
            ctx.scene_vp,
            ew as f32,
            eh as f32,
            ctx.depth_reversed_z,
            occlusion.as_ref().map(|(b, w, h)| (&b[..], *w, *h)),
        );
        renderer.frame.cached_projected_markup = projected;
        renderer.frame.cached_projected_labels = wl;
    }
    renderer.frame.occlusion_data = occlusion;
    // Take cached fields to avoid borrow conflict with pass_markup's &mut renderer.
    let cache_markup = std::mem::take(&mut renderer.frame.cached_projected_markup);
    let cache_labels = std::mem::take(&mut renderer.frame.cached_projected_labels);
    pass_markup::pass_markup(
        renderer,
        &mut encoder,
        view,
        &depth_view,
        ew,
        eh,
        ctx.scene_vp,
        ctx.depth_reversed_z,
        &cache_markup,
        &cache_labels,
    );
    renderer.frame.cached_projected_markup = cache_markup;
    renderer.frame.cached_projected_labels = cache_labels;
    renderer.gpu.flat_pool.flush(&renderer.queue);
    renderer.gpu.line_pool.flush(&renderer.queue);

    pass_hud::encode_hud_overlay(
        renderer,
        &mut encoder,
        view,
        &depth_view,
        ctx.depth_reversed_z,
    );
    if let Some(cb) = &mut post_swapchain_overlay {
        cb(&mut encoder, view);
    }
    renderer.gpu_timer.end(&mut encoder, ti_hud);

    renderer.prune_mesh_cache();
    renderer.gpu_timer.resolve(&mut encoder);
    let t0 = std::time::Instant::now();
    renderer.queue.submit(std::iter::once(encoder.finish()));
    let t_submit = t0.elapsed().as_secs_f64() * 1000.0;

    // Kick off the async occlusion-depth readback (harvested non-blockingly
    // at the start of a later frame's markup pass — no GPU stall here).
    if occlusion_captured {
        if let Some(ref buf) = renderer.frame.occlusion_capture_buf {
            let status = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
            let status_cb = std::sync::Arc::clone(&status);
            buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                let v = if result.is_ok() { 1 } else { 2 };
                status_cb.store(v, std::sync::atomic::Ordering::Release);
            });
            renderer.frame.occlusion_map_pending = Some(status);
            renderer.device.poll(wgpu::Maintain::Poll);
        }
    }

    if let Some((surface_tex, vw)) = acquired_swapchain.take() {
        drop(vw);
        surface_tex.present();
    }
    let t_present = t0.elapsed().as_secs_f64() * 1000.0 - t_submit;

    let t_collect = std::time::Instant::now();
    renderer.gpu_timer.collect(&renderer.device);
    let t_collect = t_collect.elapsed().as_secs_f64() * 1000.0;

    if renderer.frame.frame_counter % 10 == 0 {
        log::debug!(
            "CPU submit/present/collect: submit={:.1}ms present={:.1}ms collect={:.1}ms",
            t_submit, t_present, t_collect
        );
    }
    let gpu_timestamps = &renderer.gpu_timer.last_timestamps;

    let mut gpu_sections_data = Vec::new();
    for i in (0..gpu_timestamps.len()).step_by(2) {
        if i + 1 < gpu_timestamps.len() {
            let label = renderer.gpu_timer.labels.get(i / 2).copied().unwrap_or("?");
            let dur_ticks = gpu_timestamps[i + 1].saturating_sub(gpu_timestamps[i]);
            let dur_us = dur_ticks as f64 * renderer.gpu_timer.timestamp_period_ns as f64 / 1_000.0;
            gpu_sections_data.push((label, dur_us));
        }
    }

    let stats = build_frame_stats(
        ctx.visible,
        draw_calls.len(),
        gpu_sections_data,
        renderer.cpu_span.total_ms(),
        renderer.cpu_span.spans().to_vec(),
    );

    log_frame_stats(&stats, renderer.frame.frame_counter);

    let t_total = t_entry.elapsed().as_secs_f64() * 1000.0;
    let t_surface = t_surface_start.elapsed().as_secs_f64() * 1000.0;
    if renderer.frame.frame_counter % 10 == 0 {
        log::debug!(
            "execute_passes wall={:.1}ms (surface_acquire={:.1}ms) draws={}",
            t_total, t_surface, draw_calls.len()
        );
    }

    stats
}

/// Build frame statistics from rendering results.
fn build_frame_stats(
    visible: &[&DrawCall],
    total_draw_calls: usize,
    gpu_sections: Vec<(&'static str, f64)>,
    frame_time_ms: f64,
    cpu_sections: Vec<(&'static str, f64)>,
) -> FrameStats {
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
    FrameStats {
        visible_triangles: total_visible_triangles,
        visible_draw_calls: visible.len(),
        culled_draw_calls: total_draw_calls.saturating_sub(visible.len()),
        gpu_pass_times_us: None,
        diagnostics: None,
        frame_time_ms,
        cpu_sections,
        gpu_sections,
    }
}

/// Periodic per-pass GPU/CPU timing report (every 10 frames).
fn log_frame_stats(stats: &FrameStats, frame_counter: u64) {
    if frame_counter % 10 != 0 || stats.gpu_sections.is_empty() {
        return;
    }
    let parts: Vec<String> = stats.gpu_sections.iter()
        .map(|(label, us)| format!("{}={:.0}us", label, us))
        .collect();
    let cpu_parts: Vec<String> = stats.cpu_sections.iter()
        .map(|(label, ms)| format!("{}={:.2}ms", label, ms))
        .collect();
    log::debug!(
        "GPU: {} | CPU: {} | frame={:.2}ms draws={} tris={}",
        parts.join(" "),
        cpu_parts.join(" "),
        stats.frame_time_ms,
        stats.visible_draw_calls,
        stats.visible_triangles,
    );
}

/// Render a frame with only overlay elements (markup, HUD) — no 3D geometry.
/// Used when draw_calls is empty but markup vertices exist.
pub(super) fn render_overlay_only_frame(
    renderer: &mut crate::renderer::Renderer,
    presentation: FramePresentation<'_>,
    post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    _frame_counter: u64,
    effect_commands: &pass_effects::EffectCommands,
) -> FrameStats {
    let t_entry = std::time::Instant::now();
    let (_tex_ptr, mut acquired_swapchain, eff_width, eff_height) = pass_shared::acquire_surface(
        renderer,
        &presentation,
        |_tex_ptr, w, h| (w, h),
    );

    let view: &wgpu::TextureView = match &acquired_swapchain {
        Some((_s, vw)) => vw,
        None => match &presentation {
            FramePresentation::OffscreenSurface { output_view, .. } => output_view,
            FramePresentation::Swapchain => return FrameStats::default(),
        },
    };

    let ew = eff_width.max(1);
    let eh = eff_height.max(1);

    let bg_color = wgpu::Color {
        r: 0.02,
        g: 0.02,
        b: 0.02,
        a: 1.0,
    };

    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Overlay-only Encoder"),
        });

    // Clear color to bg_color (no depth needed for overlays)
    {
        let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Overlay Clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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
    }

    // Markup overlay — skip occlusion when geometry pass is skipped (no depth data).
    let occlusion: Option<(Vec<f32>, u32, u32)> = None;
    let scene_vp = renderer.frame.scene_vp;
    let depth_reversed_z = renderer
        .frame
        .scene_depth_reversed_z;
    if renderer.gpu.depth_texture.is_none() {
        renderer.create_depth_texture();
    }
    let depth_view = renderer
        .gpu
        .depth_texture
        .as_ref()
        .map(|(_, v, _)| v.clone());

    // Static frame: reuse cached projection output.
    let is_static = renderer.frame.bvh_fully_static && renderer.frame.static_frame_count >= 2;
    if !is_static {
        let (projected, wl) = pass_markup::compute_projected_markup(
            renderer,
            effect_commands,
            scene_vp,
            ew as f32,
            eh as f32,
            depth_reversed_z,
            occlusion.as_ref().map(|(d, w, h)| (&d[..], *w, *h)),
        );
        renderer.frame.cached_projected_markup = projected;
        renderer.frame.cached_projected_labels = wl;
    }
    // Take cached fields to avoid borrow conflict with pass_markup's &mut renderer.
    let cache_markup = std::mem::take(&mut renderer.frame.cached_projected_markup);
    let cache_labels = std::mem::take(&mut renderer.frame.cached_projected_labels);
    pass_markup::pass_markup(
        renderer,
        &mut encoder,
        view,
        depth_view.as_ref().expect("depth texture"),
        ew,
        eh,
        scene_vp,
        depth_reversed_z,
        &cache_markup,
        &cache_labels,
    );
    renderer.frame.cached_projected_markup = cache_markup;
    renderer.frame.cached_projected_labels = cache_labels;
    renderer.gpu.flat_pool.flush(&renderer.queue);
    renderer.gpu.line_pool.flush(&renderer.queue);

    if let Some(depth_view) = depth_view.as_ref() {
        pass_hud::encode_hud_overlay(
            renderer,
            &mut encoder,
            view,
            depth_view,
            depth_reversed_z,
        );
    }

    if let Some(cb) = post_swapchain_overlay {
        cb(&mut encoder, view);
    }

    renderer
        .queue
        .submit(std::iter::once(encoder.finish()));

    if let Some((surface_tex, vw)) = acquired_swapchain.take() {
        drop(vw);
        surface_tex.present();
    }

    let t_total = t_entry.elapsed().as_secs_f64() * 1000.0;
    FrameStats {
        frame_time_ms: t_total,
        cpu_sections: Vec::new(),
        gpu_sections: Vec::new(),
        ..FrameStats::default()
    }
}

/// Lazily create the occlusion downsample compute pipeline and (re)create its
/// small R32Float output texture when the target dimensions change.
fn ensure_occlusion_downsample_resources(
    renderer: &mut crate::renderer::Renderer,
    dw: u32,
    dh: u32,
) {
    if renderer.gpu.occlusion_downsample_pipeline.is_none() {
        let shader = renderer.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("occlusion_downsample.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/occlusion_downsample.wgsl").into(),
            ),
        });
        let bgl = renderer.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Occlusion Downsample BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let pll = renderer.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Occlusion Downsample PLL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = renderer.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Occlusion Downsample Pipe"),
            layout: Some(&pll),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        renderer.gpu.occlusion_downsample_pipeline = Some(pipeline);
        renderer.gpu.occlusion_downsample_bgl = Some(bgl);
    }

    let needs_tex = renderer
        .gpu
        .occlusion_downsample_tex
        .as_ref()
        .map_or(true, |(t, _)| t.width() != dw || t.height() != dh);
    if needs_tex {
        let tex = renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Occlusion Downsample"),
            size: wgpu::Extent3d { width: dw, height: dh, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        renderer.gpu.occlusion_downsample_tex = Some((tex, view));
    }
}
