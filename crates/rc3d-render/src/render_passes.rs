use crate::adaptive_quality::AdaptiveQuality;
use crate::render_action::DrawCall;
use crate::render_graph::{declaration_order_is_valid, RenderGraph};
use crate::renderer::GpuTier;
use crate::viewport::LayoutMode;
use crate::vertex::{FlatUniforms, CSM_CASCADE_COUNT};
use crate::FrameStats;
use glam::{Mat4, Vec3};
use rc3d_core::DisplayMode;
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

mod pass_edge;
pub(crate) mod pass_effects;
mod pass_grid;
pub(crate) mod pass_markup;
mod pass_post;
mod pass_selection;
mod pass_shadow;
mod pass_solid;
pub(crate) mod pass_text;
mod pass_viewport;
mod pass_wireframe;

pub(crate) mod draw_opaque;
use draw_opaque::draw_opaque_triangle_batches;

static RC3D_RENDER_GRAPH_OK: OnceLock<()> = OnceLock::new();

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
    /// CSM split depths in view space: [near, split1, split2, far] (4 values)
    pub csm_split_depths: [f32; CSM_CASCADE_COUNT],
    pub shadow_params: [f32; 4],
    pub run_shadow_pass: bool,
    /// Camera projection matrix (for SSAO depth reconstruction)
    pub camera_proj: Mat4,
    /// Camera inverse projection matrix
    pub camera_inv_proj: Mat4,
    /// Combined view-projection matrix (proj * view).
    pub scene_vp: Mat4,
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

/// Run meshlet cull compute passes in the given encoder, then copy meshlet
/// InstanceData from staging to instance_buffer slot 0 via encoder copy.
/// The copy is on the encoder timeline, guaranteeing proper ordering with
/// the subsequent render pass (meshlet draw reads instances[0]).
fn submit_meshlet_cull(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &PassContext<'_>,
    hzb_enabled: bool,
    hzb_dims: (u32, u32),
    mip_max: u32,
    hzb_need_max: bool,
    hzb_need_min: bool,
) {
    let Some(cluster_renderer) = renderer.gpu.cluster_renderer.as_ref() else {
        return;
    };
    let Some(hzb) = renderer.gpu.hzb.as_ref() else {
        return;
    };

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

    // Collect meshlet InstanceData and write to staging buffer.
    // The staging buffer is later copied to instance_buffer slot 0 via
    // encoder.copy_buffer_to_buffer, putting the write on the encoder timeline
    // (after cull, before render pass).
    let mut wrote_staging = false;
    for &vis_idx in ctx.meshlet_indices {
        let dc = ctx.visible[vis_idx];
        let Some(md) = dc.meshlet_data.as_ref() else {
            continue;
        };
        let ptr = std::sync::Arc::as_ptr(md) as u64;
        if let Some(cluster_set) = renderer.gpu.assets.cluster_get(&ptr) {
            // Write meshlet InstanceData to staging before cull
            if !wrote_staging {
                if let Some(ref staging) = renderer.gpu.meshlet_instance_staging {
                    let inst = crate::vertex::InstanceData {
                        model: dc.model_matrix.to_cols_array_2d(),
                        mvp: dc.mvp.to_cols_array_2d(),
                        diffuse_color: [dc.diffuse_color.x, dc.diffuse_color.y, dc.diffuse_color.z, 1.0],
                        base_color: [dc.base_color.x, dc.base_color.y, dc.base_color.z, 1.0],
                        metallic_roughness: [dc.metallic, dc.roughness, 0.0, 0.0],
                        emissive_alpha: [dc.emissive_color.x, dc.emissive_color.y, dc.emissive_color.z, dc.alpha_cutoff],
                        morph_weights: [0.0f32; 8],
                        morph_count: [0.0f32; 4],
                    };
                    renderer.queue.write_buffer(staging, 0, bytemuck::bytes_of(&inst));
                    wrote_staging = true;
                }
            }

            let model_inv = dc.model_matrix.inverse();
            let cam_model = model_inv.transform_point3(Vec3::from(ctx.camera_pos));
            cluster_renderer.cull_and_compact(
                &renderer.device,
                &renderer.queue,
                encoder,
                cluster_set,
                dc.mvp.to_cols_array_2d(),
                [cam_model.x, cam_model.y, cam_model.z],
                1,
                0,
                false,
                max_bind,
                min_bind,
                hzb_dims,
                mip_max,
                hzb_enabled,
                dc.depth_reversed_z,
                dc.projection_orthographic,
            );
        }
    }

    // Copy meshlet InstanceData from staging to instance_buffer slot 0.
    // This is on the encoder timeline — after cull passes, before render pass.
    if wrote_staging {
        if let Some(ref staging) = renderer.gpu.meshlet_instance_staging {
            let stride = std::mem::size_of::<crate::vertex::InstanceData>() as u64;
            encoder.copy_buffer_to_buffer(staging, 0, &renderer.gpu.instance_buffer, 0, stride);
        }
    }
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

    let mut acquired_swapchain: Option<(wgpu::SurfaceTexture, wgpu::TextureView)> = None;

    let t_surface_start = std::time::Instant::now();
    let (eff_width, eff_height, scene_tex_raw): (u32, u32, *const wgpu::Texture) = match &presentation {
        FramePresentation::Swapchain => match renderer.surface.get_current_texture() {
            Ok(output) => {
                let tex_ptr = std::ptr::from_ref(&output.texture);
                let v = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                acquired_swapchain = Some((output, v));
                (renderer.config.width, renderer.config.height, tex_ptr)
            }
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                renderer.surface.configure(&renderer.device, &renderer.config);
                return FrameStats::default();
            }
            Err(_) => return FrameStats::default(),
        },
        FramePresentation::OffscreenSurface {
            output_texture,
            width_px,
            height_px,
            ..
        } => (*width_px, *height_px, std::ptr::from_ref(output_texture)),
    };

    let view: &wgpu::TextureView = match acquired_swapchain.as_ref() {
        Some((_s, vw)) => vw,
        None => match &presentation {
            FramePresentation::OffscreenSurface { output_view, .. } => output_view,
            FramePresentation::Swapchain => unreachable!(),
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
            .expect("LDR shade view")
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
            let obj_count = ctx.visible.len().min(renderer.gpu.max_gpu_cull_objects as usize);
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

    let tier_allows_hzb = renderer.gpu.gpu_capability.tier != GpuTier::Basic;
    let requested_meshlet_hzb_prepass = solid_mode
        && mode != DisplayMode::Flat
        && mode != DisplayMode::FlatWithEdge
        && !ctx.meshlet_indices.is_empty()
        && renderer.gpu.cluster_renderer.is_some()
        && renderer.gpu.hzb.is_some()
        && renderer.gpu.hzb_baker.is_some()
        && tier_allows_hzb;
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

    if !meshlet_hzb_prepass_done && !ctx.meshlet_indices.is_empty() && mode != DisplayMode::Flat && mode != DisplayMode::FlatWithEdge {
        let (fallback_dims, fallback_mip) = if let Some(hzb) = renderer.gpu.hzb.as_ref() {
            ((hzb.max_pyramid.width, hzb.max_pyramid.height), hzb.max_pyramid.mip_count.saturating_sub(1))
        } else {
            ((1, 1), 0)
        };
        submit_meshlet_cull(renderer, &mut encoder, ctx, false, fallback_dims, fallback_mip, hzb_need_max, hzb_need_min);
    }

    if solid_mode && renderer.enable_cluster_lights {
        if let (Some(ref culler), Some(ref resources)) =
            (renderer.gpu.cluster_light_culler.as_ref(), renderer.gpu.cluster_lights.as_ref())
        {
            use crate::cluster_lighting::{GpuPointLight, GpuSpotLight};

            let mut point_lights: Vec<GpuPointLight> = Vec::new();
            let mut spot_lights: Vec<GpuSpotLight> = Vec::new();

            let mut seen_light_sets: std::collections::HashSet<u32> = std::collections::HashSet::new();
            for dc in ctx.visible.iter() {
                if !seen_light_sets.insert(dc.light_set_id) {
                    continue; // already processed this light set
                }
                let lights = ctx.light_sets.get(dc.light_set_id);
                let (ref light_dirs, ref light_colors, ref light_types, ref light_positions, ref spot_params, light_count) = *lights;
                for i in 0..(light_count as usize).min(crate::vertex::MAX_LIGHTS) {
                    let lt = light_types[i][0];
                    let pos = light_positions[i];
                    let col = light_colors[i];
                    let intensity = light_colors[i][3];

                    if (lt - 1.0).abs() < 0.5 {
                        if point_lights.len() < 256 {
                            point_lights.push(GpuPointLight {
                                position: [pos[0], pos[1], pos[2]],
                                radius: pos[3].max(1.0),
                                color: [col[0], col[1], col[2]],
                                intensity,
                            });
                        }
                    } else if (lt - 3.0).abs() < 0.5 {
                        let dir = light_dirs[i];
                        let sp = spot_params[i];
                        if spot_lights.len() < 256 {
                            spot_lights.push(GpuSpotLight {
                                position: [pos[0], pos[1], pos[2]],
                                direction: [dir[0], dir[1], dir[2]],
                                radius: pos[3].max(1.0),
                                cos_inner: sp[0],
                                cos_outer: sp[1],
                                color: [col[0], col[1], col[2]],
                                intensity,
                                _pad: 0.0,
                            });
                        }
                    }
                }
            }

            if !point_lights.is_empty() || !spot_lights.is_empty() {
                let w = ew;
                let h = eh;
                culler.cull_lights(
                    &renderer.device,
                    &renderer.queue,
                    &mut encoder,
                    resources,
                    ctx.camera_inv_proj,
                    w, h,
                    0.1, 1000.0,
                    &point_lights,
                    &spot_lights,
                );
            }
        }
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
        && ctx.adaptive_quality != AdaptiveQuality::Low
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
                std::ptr::from_ref(renderer.gpu.ldr_shade_tex.as_ref().expect("LDR shade texture"))
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
            &view,
            ctx.bg_color,
        );
    }

    if defer_line_overlays {
        if edge_worthy || has_overlay {
            pass_edge::pass_edge_overlay(renderer, &mut encoder, &view, &depth_view, ctx, edge_worthy, &scene_pl);
        }
        if has_selection && !ctx.performance_mode_active && mode != DisplayMode::Flat && mode != DisplayMode::FlatWithEdge {
            if run_geom_sel_edge {
                pass_selection::pass_selection_edge(renderer, &mut encoder, &view, &depth_view, ctx, &scene_pl);
            }
            pass_selection::pass_selection_bbox(
                &mut encoder,
                &view,
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
        #[cfg(feature = "profiler")]
        let _span_post = tracy_client::span!("post");
        if let Some(ref fx) = renderer.gpu.post_fx {
            let pl = &renderer.gpu.post_fx_pipelines;
            let w = ew;
            let h = eh;
            let proj = ctx.camera_proj.to_cols_array_2d();
            let inv_proj = ctx.camera_inv_proj.to_cols_array_2d();

            // ── SSR (screen-space reflections) ──
            if renderer.enable_ssr {
                if let Some(ref ssr) = renderer.gpu.ssr_pass {
                    if let Some(ref hzb) = renderer.gpu.hzb {
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
                if let Some(ref fog) = renderer.gpu.volumetric_fog {
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
                if let Some(ref mb) = renderer.gpu.motion_blur {
                    mb.apply(
                        &renderer.device, &renderer.queue, &mut encoder,
                        &fx.hdr_view, &depth_read_view, &depth_read_view,
                        &fx.hdr_view, w, h, 16, 0.5,
                    );
                }
            }

            // ── DOF ──
            if renderer.enable_dof {
                if let Some(ref dof) = renderer.gpu.dof_pass {
                    dof.apply(
                        &renderer.device, &renderer.queue, &mut encoder,
                        &fx.hdr_view, &depth_read_view, &fx.hdr_view,
                        w, h, 5.0, 2.0,
                    );
                }
            }

            // ── Color Grading ──
            if renderer.enable_color_grading {
                if let Some(ref cg) = renderer.gpu.color_grading {
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
                &depth_read_view, &renderer.gpu.ssao_noise_view, &proj, &inv_proj);
            // SSAO blur (read AO + depth, write blurred AO)
            pass_post::pass_ssao_blur(&renderer.device, &mut encoder, pl, fx, &depth_read_view);

            // ── TAA (temporal anti-aliasing) ──
            if renderer.enable_taa {
                if let Some(ref mut taa) = renderer.gpu.taa_pass {
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
        }
    }
    renderer.gpu_timer.end(&mut encoder, ti_post);

    renderer.gpu.outline_pool.flush(&renderer.queue);
    renderer.gpu.phong_pool.flush(&renderer.queue);
    renderer.gpu.shadow_pool.flush(&renderer.queue);
    renderer.gpu.flat_pool.flush(&renderer.queue);
    renderer.gpu.section_cap_pool.flush(&renderer.queue);

    let ti_hud = renderer.gpu_timer.begin(&mut encoder, "HUD+Overlay");

    // Ground plane grid overlay
    if renderer.grid_enabled {
        pass_grid::pass_grid(
            renderer,
            &mut encoder,
            &view,
            &depth_view,
            renderer.frame.scene_vp,
            renderer.frame.scene_camera_pos,
            ctx.depth_reversed_z,
        );
    }

    // Viewport border overlay
    {
        let geom = pass_viewport::ViewportBorderGeometry::build(
            &renderer.frame.viewport_layout,
            ew,
            eh,
        );
        let has_splits = !geom.split_lines.is_empty();
        let has_active = !geom.active_lines.is_empty();
        if has_splits || has_active {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Viewport Borders"),
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
            pass.set_pipeline(&renderer.gpu.pipelines.viewport_border_lines);
            let wpx = ew as f32;
            let hpx = eh as f32;
            // `border_lines` uses window pixel coords (origin top-left, +y down).
            let screen_mvp = Mat4::orthographic_rh_gl(0.0, wpx, hpx, 0.0, -1.0, 1.0).to_cols_array_2d();
            // Split borders
            if has_splits {
                let uniforms = FlatUniforms {
                    mvp: screen_mvp,
                    color: [0.4, 0.4, 0.4, 1.0],
                };
                if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
                    for chunk in geom.split_lines.chunks(2) {
                        if chunk.len() < 2 { break; }
                        let verts = [chunk[0], chunk[1]];
                        let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("split border vb"),
                            contents: bytemuck::cast_slice(&verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                        pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.draw(0..2, 0..1);
                    }
                }
            }
            // Active viewport highlight (hidden for single full-window viewport — no editor benefit).
            if has_active && renderer.frame.viewport_layout.layout_mode != LayoutMode::Single {
                let uniforms = FlatUniforms {
                    mvp: screen_mvp,
                    color: [1.0, 0.85, 0.1, 1.0],
                };
                if let Some(offset) = renderer.gpu.flat_pool.push_flat(&uniforms) {
                    for chunk in geom.active_lines.chunks(2) {
                        if chunk.len() < 2 { break; }
                        let verts = [chunk[0], chunk[1]];
                        let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("active border vb"),
                            contents: bytemuck::cast_slice(&verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                        pass.set_bind_group(0, renderer.gpu.flat_pool.bind_group(), &[offset]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.draw(0..2, 0..1);
                    }
                }
            }
        }
    }

    // Markup overlay
    #[cfg(feature = "profiler")]
    let _span_markup = tracy_client::span!("markup");
    let view_matrix = ctx.camera_inv_proj * ctx.scene_vp;
    pass_markup::pass_markup(
        renderer, &mut encoder, &view, ew, eh,
        view_matrix, ctx.camera_proj,
        ctx.depth_reversed_z, ctx.effect_commands,
    );
    renderer.gpu.flat_pool.flush(&renderer.queue);

    if renderer.hud_enabled {
        if let Some(hud) = renderer.gpu.hud.as_ref() {
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
    if let Some(cb) = &mut post_swapchain_overlay {
        cb(&mut encoder, view);
    }
    renderer.gpu_timer.end(&mut encoder, ti_hud);

    renderer.prune_mesh_cache();
    renderer.gpu_timer.resolve(&mut encoder);
    let t0 = std::time::Instant::now();
    renderer.queue.submit(std::iter::once(encoder.finish()));
    let t_submit = t0.elapsed().as_secs_f64() * 1000.0;
    if let Some((surface_tex, vw)) = acquired_swapchain.take() {
        drop(vw);
        surface_tex.present();
    }
    let t_present = t0.elapsed().as_secs_f64() * 1000.0 - t_submit;

    let t_collect = std::time::Instant::now();
    renderer.gpu_timer.collect(&renderer.device);
    let t_collect = t_collect.elapsed().as_secs_f64() * 1000.0;

    if renderer.frame.frame_counter % 10 == 0 {
        log::info!(
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
    let stats = FrameStats {
        visible_triangles: total_visible_triangles,
        visible_draw_calls: ctx.visible.len(),
        culled_draw_calls: draw_calls.len().saturating_sub(ctx.visible.len()),
        gpu_pass_times_us: None,
        diagnostics: None,
        frame_time_ms: renderer.cpu_span.total_ms(),
        cpu_sections: renderer.cpu_span.spans().to_vec(),
        gpu_sections: gpu_sections_data,
    };

    // Periodic per-pass GPU timing report (every 10 frames)
    if renderer.frame.frame_counter % 10 == 0 && !stats.gpu_sections.is_empty() {
        let parts: Vec<String> = stats.gpu_sections.iter()
            .map(|(label, us)| format!("{}={:.0}us", label, us))
            .collect();
        let cpu_parts: Vec<String> = stats.cpu_sections.iter()
            .map(|(label, ms)| format!("{}={:.2}ms", label, ms))
            .collect();
        log::info!(
            "GPU: {} | CPU: {} | frame={:.2}ms draws={} tris={}",
            parts.join(" "),
            cpu_parts.join(" "),
            stats.frame_time_ms,
            stats.visible_draw_calls,
            stats.visible_triangles,
        );
    }

    let t_total = t_entry.elapsed().as_secs_f64() * 1000.0;
    let t_surface = t_surface_start.elapsed().as_secs_f64() * 1000.0;
    if renderer.frame.frame_counter % 10 == 0 {
        log::info!(
            "execute_passes wall={:.1}ms (surface_acquire={:.1}ms) draws={}",
            t_total, t_surface, draw_calls.len()
        );
    }

    stats
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
    let mut acquired_swapchain: Option<(wgpu::SurfaceTexture, wgpu::TextureView)> = None;

    let (eff_width, eff_height) = match &presentation {
        FramePresentation::Swapchain => match renderer.surface.get_current_texture() {
            Ok(output) => {
                let v = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                acquired_swapchain = Some((output, v));
                (renderer.config.width, renderer.config.height)
            }
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                renderer.surface.configure(&renderer.device, &renderer.config);
                return FrameStats::default();
            }
            Err(_) => return FrameStats::default(),
        },
        FramePresentation::OffscreenSurface {
            width_px,
            height_px,
            ..
        } => (*width_px, *height_px),
    };

    let view: &wgpu::TextureView = match acquired_swapchain.as_ref() {
        Some((_s, vw)) => vw,
        None => match &presentation {
            FramePresentation::OffscreenSurface { output_view, .. } => output_view,
            FramePresentation::Swapchain => unreachable!(),
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

    // Markup overlay — push_flat inside, then flush before submit
    pass_markup::pass_markup(
        renderer, &mut encoder, view, ew, eh,
        Mat4::IDENTITY, Mat4::IDENTITY, false,
        effect_commands,
    );
    renderer.gpu.flat_pool.flush(&renderer.queue);

    // HUD overlay
    if renderer.hud_enabled {
        if let Some(hud) = renderer.gpu.hud.as_ref() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("HUD Overlay Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
