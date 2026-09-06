//! 3D film path: shade, present LDR/HDR, compositor.
use crate::adaptive_quality::AdaptiveQuality;
use crate::pipelines::DepthModePipelines;
use super::PassContext;
use super::{
    pass_edge, pass_hidden, pass_post, pass_shadow, pass_solid, pass_transparent, pass_wireframe,
    submit_meshlet_cull,
};

pub(super) struct FilmEncodeResult {
    pub scene_pl: DepthModePipelines,
    pub defer_line_overlays: bool,
    pub run_hidden: bool,
    pub run_wireframe: bool,
    pub edge_worthy: bool,
    pub has_overlay: bool,
}

pub(super) fn encode_film(
    renderer: &mut crate::renderer::Renderer,
    ctx: &PassContext<'_>,
    mut encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    shade_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    depth_read_view: &wgpu::TextureView,
    ew: u32,
    eh: u32,
    pass_vp: crate::viewport::ViewportRect,
    scene_pl: DepthModePipelines,
    use_ldr_fxaa: bool,
    use_ldr_film: bool,
    need_comp_film: bool,
) -> FilmEncodeResult {
    renderer.encode_skinning_compute(&mut encoder, ctx.visible, ctx.mesh_handles);

    // ── Background pass (gradient / image / solid) ──
    // Transparent overlay tiles: Clear only (no fullscreen bg draw) so unused
    // pixels stay chroma-key black / a=0 for the overlay blit.
    let skip_bg_draw = renderer.overlay_pass && renderer.gpu.bg_settings.top_color[3] < 0.5;
    if skip_bg_draw {
        let c = renderer.gpu.bg_settings.top_color;
        let mut clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Overlay transparent clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: c[0] as f64,
                        g: c[1] as f64,
                        b: c[2] as f64,
                        a: c[3] as f64,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass_vp.apply_to_pass(&mut clear_pass);
    } else if let Some(ref bg) = renderer.gpu.bg_pass {
        bg.encode(&renderer.device, &renderer.queue, &mut encoder, shade_view, &renderer.gpu.bg_settings, ew, eh, ctx.camera_inv_proj, pass_vp);
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

    let any_filled = ctx.visible.iter().any(|dc| dc.appearance().wants_filled());
    let any_lit = ctx.visible.iter().any(|dc| dc.appearance().wants_lit_solid());
    let any_wireframe = ctx.visible.iter().any(|dc| dc.appearance().wants_full_edges());
    let any_feature_edges = ctx.visible.iter().any(|dc| dc.appearance().wants_edge_overlay());
    let solid_mode = any_filled;
    let defer_line_overlays = use_ldr_fxaa && !need_comp_film;

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
        && any_lit
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

    if !meshlet_hzb_prepass_done && !ctx.meshlet_indices.is_empty() && any_lit && renderer.gpu.gpu_capability.meshlet_gpu_cull_enabled {
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

    let ti_solid = renderer.gpu_timer.begin(&mut encoder, "Solid");
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

    if ctx.visible.iter().any(|dc| crate::custom_shader::is_custom_shader_draw(dc)) {
        renderer.ensure_custom_shader_pass();
        if let Some(ref mut pass) = renderer.gpu.custom_shader_pass {
            pass.encode(
                &renderer.device,
                &mut encoder,
                shade_view,
                &depth_view,
                ctx.visible,
                0.0,
                pass_vp,
            );
        }
    }

    // ── Transparent pass: WBOIT (LDR or HDR) or painter's algorithm ──
    if !ctx.transparent_order.is_empty() {
        if renderer.enable_wboit {
            renderer.ensure_wboit_targets(ew, eh);
            let wboit_views = renderer.gpu.wboit_targets.as_ref().map(|t| {
                (t.accum_view.clone(), t.revealage_view.clone())
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
                    pass_vp,
                );
            }
        }
        if !ctx.effect_commands.volumes.is_empty() {
            renderer.ensure_volume_pass();
            if let Some(ref pass) = renderer.gpu.volume_pass {
                pass.encode(
                    &renderer.device, &renderer.queue, &mut encoder,
                    shade_view, &depth_view, &ctx.effect_commands.volumes, ew, eh,
                    pass_vp,
                );
            }
        }
        if !ctx.effect_commands.point_clouds.is_empty() {
            renderer.ensure_point_cloud_pass();
            if let Some(ref mut pass) = renderer.gpu.point_cloud_pass {
                pass.encode(
                    &renderer.device, &renderer.queue, &mut encoder,
                    shade_view, &depth_view, &ctx.effect_commands.point_clouds,
                    ctx.scene_vp, ctx.camera_inv_proj, ew, eh,
                    pass_vp,
                );
            }
        }
    }
    renderer.gpu_timer.end(&mut encoder, ti_effects);

    let run_hidden = !ctx.performance_mode_active
        && ctx.visible.iter().any(|dc| dc.appearance().wants_hidden_dashes());
    let run_wireframe = !ctx.performance_mode_active && ctx.wireframe_supported && any_wireframe;
    // FXAA blits shade -> swapchain with a whole-target Clear. Draw lines after that
    // blit so mixed filled+line frames keep the solid pass (pass_wireframe Loads).
    if run_hidden && !defer_line_overlays {
        pass_hidden::pass_hidden_edges(
            renderer,
            &mut encoder,
            shade_view,
            &depth_view,
            ctx,
            &scene_pl,
        );
    }
    if run_wireframe && !defer_line_overlays {
        #[cfg(feature = "profiler")]
        let _span_wireframe = tracy_client::span!("wireframe");
        pass_wireframe::pass_wireframe(renderer, &mut encoder, shade_view, &depth_view, ctx, &scene_pl);
    }

    let run_selection_outline = ctx.visible.iter().any(|dc| dc.selected)
        && !ctx.performance_mode_active
        && renderer.screen_space_selection_outline
        && ctx.adaptive_quality != AdaptiveQuality::Low;

    let edge_worthy = !ctx.performance_mode_active && any_feature_edges;
    let has_overlay = ctx.visible.iter().any(|dc| dc.overlay_color.is_some());
    if (edge_worthy || has_overlay) && !defer_line_overlays {
        #[cfg(feature = "profiler")]
        let _span_edge = tracy_client::span!("edge");
        pass_edge::pass_edge_overlay(renderer, &mut encoder, shade_view, &depth_view, ctx, edge_worthy, &scene_pl);
    }

    if run_selection_outline {
        let shade_fmt = if renderer.hdr_post_processing {
            wgpu::TextureFormat::Rgba16Float
        } else {
            renderer.config.format
        };
        crate::selection_outline::encode_selection_outline_pass(
            renderer,
            &mut encoder,
            ctx,
            shade_view,
            shade_fmt,
            ew,
            eh,
        );
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
    } else if use_ldr_film {
        pass_post::pass_blit_ldr_to_swapchain(
            &renderer.device,
            &mut encoder,
            &renderer.gpu.post_fx_pipelines,
            renderer.gpu.ldr_shade_view.as_ref().expect("LDR shade view"),
            view,
            ctx.bg_color,
        );
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

    renderer.encode_compositor(&mut encoder, view);

    FilmEncodeResult {
        scene_pl,
        defer_line_overlays,
        run_hidden,
        run_wireframe,
        edge_worthy,
        has_overlay,
    }
}
