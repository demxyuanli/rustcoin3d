use crate::post_processor::{PostFxPipelines, PostFxTextures, SsaoParamsUniform};
use crate::render_passes::PassContext;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Encode the full HDR post-processing pipeline: X-Ray → SSR → VolFog → Velocity →
/// MotionBlur → DOF → ColorGrading → Bloom → SSAO → TAA → Tonemap → Blit.
pub(super) fn encode_post_processing(
    renderer: &mut crate::renderer::Renderer,
    encoder: &mut wgpu::CommandEncoder,
    depth_read_view: &wgpu::TextureView,
    view: &wgpu::TextureView,
    ew: u32,
    eh: u32,
    ctx: &PassContext<'_>,
) {
    #[cfg(feature = "profiler")]
    let _span_post = tracy_client::span!("post");
    if let Some(ref fx) = renderer.gpu.post_fx {
        let pl = &renderer.gpu.post_fx_pipelines;
        let w = ew;
        let h = eh;
        let proj = ctx.camera_proj.to_cols_array_2d();
        let inv_proj = ctx.camera_inv_proj.to_cols_array_2d();

        // ── Ping-pong buffers: avoid read-write conflict on same texture ──
        // hdr_is_src: true → rendered image is in hdr_view, scratch is free
        //             false → rendered image is in scratch_view, hdr is free
        let mut hdr_is_src = true;
        let hdr: &wgpu::TextureView = &fx.hdr_view;
        let alt: &wgpu::TextureView = &fx.scratch_view;

        // ── X-Ray (depth edge detection) ──
        if renderer.xray_mode {
            let ti = renderer.gpu_timer.begin(encoder, "PP XRay");
            let xray_bg = renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("X-Ray BG"),
                layout: &pl.xray_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_read_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&pl.ssao_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(alt) },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("X-Ray"), timestamp_writes: None,
            });
            pass.set_pipeline(&pl.xray_pipeline);
            pass.set_bind_group(0, &xray_bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
            drop(pass);
            hdr_is_src = false; // x-ray wrote to alt; alt is now current
            renderer.gpu_timer.end(encoder, ti);
        }

        // ── SSR (screen-space reflections) ──
        if renderer.enable_ssr {
            let ti = renderer.gpu_timer.begin(encoder, "PP SSR");
            let (src, dst) = if hdr_is_src { (hdr, alt) } else { (alt, hdr) };
            if let Some(ref ssr) = renderer.gpu.ssr_pass {
                if let Some(ref hzb) = renderer.gpu.hzb {
                    ssr.trace(
                        &renderer.device, &renderer.queue, encoder,
                        src, depth_read_view,
                        &hzb.max_pyramid.full_view, dst,
                        w, h,
                        ctx.camera_inv_proj, Mat4::IDENTITY,
                    );
                    hdr_is_src = !hdr_is_src;
                }
            }
            renderer.gpu_timer.end(encoder, ti);
        }

        // ── Volumetric Fog (reads depth + scene color, writes composited dst) ──
        if renderer.enable_volumetric_fog {
            let ti = renderer.gpu_timer.begin(encoder, "PP VolFog");
            let (src, dst) = if hdr_is_src { (hdr, alt) } else { (alt, hdr) };
            if let Some(ref fog) = renderer.gpu.volumetric_fog {
                fog.compute(
                    &renderer.device, &renderer.queue, encoder,
                    depth_read_view, src, dst, w, h,
                    ctx.camera_inv_proj,
                    Vec3::from(ctx.camera_pos),
                    Vec3::new(0.5, -0.8, 0.3),
                    Vec3::new(1.0, 0.9, 0.7),
                    Vec3::new(0.6, 0.7, 0.8),
                    0.02, 0.5, 100.0, 32,
                );
                hdr_is_src = !hdr_is_src;
            }
            renderer.gpu_timer.end(encoder, ti);
        }

        // ── Velocity buffer (for motion blur + TAA) ──
        if renderer.enable_motion_blur || renderer.enable_taa {
            let ti = renderer.gpu_timer.begin(encoder, "PP Velocity");
            let inv_vp = (ctx.scene_vp).inverse();
            let vp_prev = ctx.prev_vp;
            // VelocityParams uniform: mat4x4 + mat4x4 + vec2 + vec2 = 144 bytes
            let mut vel_data: Vec<u8> = Vec::with_capacity(144);
            for row in &inv_vp.to_cols_array_2d() { vel_data.extend_from_slice(bytemuck::bytes_of(row)); }
            for row in &vp_prev.to_cols_array_2d() { vel_data.extend_from_slice(bytemuck::bytes_of(row)); }
            vel_data.extend_from_slice(bytemuck::bytes_of(&[0.0f32; 4])); // _pad0 + _pad1
            // Use pre-allocated velocity buffer instead of creating new one each frame
            if let Some(ref vel_buf) = renderer.gpu.velocity_buffer {
                renderer.queue.write_buffer(vel_buf, 0, &vel_data);
            }
            let vel_bg = renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Velocity BG"),
                layout: &pl.velocity_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_read_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&pl.ssao_sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: renderer.gpu.velocity_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&fx.velocity_view) },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Velocity"), timestamp_writes: None,
            });
            pass.set_pipeline(&pl.velocity_pipeline);
            pass.set_bind_group(0, &vel_bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
            drop(pass);
            renderer.gpu_timer.end(encoder, ti);
        }

        // ── Motion Blur ──
        if renderer.enable_motion_blur {
            let ti = renderer.gpu_timer.begin(encoder, "PP MotionBlur");
            let (src, dst) = if hdr_is_src { (hdr, alt) } else { (alt, hdr) };
            if let Some(ref mb) = renderer.gpu.motion_blur {
                mb.apply(
                    &renderer.device, &renderer.queue, encoder,
                    src, &fx.velocity_view, depth_read_view,
                    dst, w, h, 16, 0.25,
                );
                hdr_is_src = !hdr_is_src;
            }
            renderer.gpu_timer.end(encoder, ti);
        }

        // ── DOF ──
        if renderer.enable_dof {
            let ti = renderer.gpu_timer.begin(encoder, "PP DOF");
            let (src, dst) = if hdr_is_src { (hdr, alt) } else { (alt, hdr) };
            if let Some(ref dof) = renderer.gpu.dof_pass {
                dof.apply(
                    &renderer.device, &renderer.queue, encoder,
                    src, depth_read_view, dst,
                    w, h, renderer.dof_focus_distance, renderer.dof_aperture,
                );
                hdr_is_src = !hdr_is_src;
            }
            renderer.gpu_timer.end(encoder, ti);
        }

        // ── Color Grading ──
        if renderer.enable_color_grading {
            let ti = renderer.gpu_timer.begin(encoder, "PP ColorGrading");
            let (src, dst) = if hdr_is_src { (hdr, alt) } else { (alt, hdr) };
            if let Some(ref cg) = renderer.gpu.color_grading {
                cg.apply(
                    &renderer.device, &renderer.queue, encoder,
                    src, dst, w, h, 1.0,
                );
                hdr_is_src = !hdr_is_src;
            }
            renderer.gpu_timer.end(encoder, ti);
        }

        // ── Ensure HDR has the current accumulated image for Bloom/SSAO ──
        // Bloom reads from hdr_view; SSAO reads from depth only.
        // If the accumulated image is in alt, copy it back to hdr for bloom+tonemap.
        if !hdr_is_src {
            let ti = renderer.gpu_timer.begin(encoder, "PP PingPongBlit");
            pass_copy_texture(
                &renderer.device, encoder, pl,
                alt, hdr, w, h,
            );
            hdr_is_src = true;
            renderer.gpu_timer.end(encoder, ti);
        }

        // Bloom prefilter (compute dispatch: read HDR, write half-res bloom)
        let ti = renderer.gpu_timer.begin(encoder, "PP Bloom");
        pass_bloom_prefilter(&renderer.device, encoder, pl, fx);
        renderer.gpu_timer.end(encoder, ti);
        // SSAO (read depth, write AO) + blur
        let ti = renderer.gpu_timer.begin(encoder, "PP SSAO");
        pass_ssao(&renderer.device, encoder, pl, fx,
            depth_read_view, &renderer.gpu.ssao_noise_view, &proj, &inv_proj);
        // SSAO blur (read AO + depth, write blurred AO)
        pass_ssao_blur(&renderer.device, encoder, pl, fx, depth_read_view);
        renderer.gpu_timer.end(encoder, ti);

        // ── TAA (temporal anti-aliasing) ──
        if renderer.enable_taa {
            let ti = renderer.gpu_timer.begin(encoder, "PP TAA");
            let (src, dst) = if hdr_is_src { (hdr, alt) } else { (alt, hdr) };
            let (src_tex, dst_tex) = if hdr_is_src {
                (&fx.hdr_tex, &fx.scratch_tex)
            } else {
                (&fx.scratch_tex, &fx.hdr_tex)
            };
            if let Some(ref mut taa) = renderer.gpu.taa_pass {
                taa.ensure_history(&renderer.device, w, h);
                taa.resolve(
                    &renderer.device, &renderer.queue, encoder,
                    src, src_tex, &fx.velocity_view, depth_read_view,
                    dst, dst_tex,
                    0.05, 1.0,
                );
                hdr_is_src = !hdr_is_src;
            }
            renderer.gpu_timer.end(encoder, ti);
        }

        // ── Final: ensure hdr_view has the accumulated result for tonemap ──
        if !hdr_is_src {
            let ti = renderer.gpu_timer.begin(encoder, "PP FinalBlit");
            pass_copy_texture(
                &renderer.device, encoder, pl,
                alt, hdr, w, h,
            );
            renderer.gpu_timer.end(encoder, ti);
        }

        // Tonemap + FXAA + Bloom + SSAO
        let ti = renderer.gpu_timer.begin(encoder, "PP Tonemap");
        pass_tonemap_hdr_to_post_ldr(encoder, pl, fx, ctx.bg_color);
        // Blit to swapchain
        pass_blit_post_ldr_to_swapchain(encoder, pl, fx, view, ctx.bg_color);
        renderer.gpu_timer.end(encoder, ti);
    }
}

pub(super) fn pass_bloom_prefilter(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    fx: &PostFxTextures,
) {
    let bloom_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bloom prefilter BG"),
        layout: &pl.bloom_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&fx.hdr_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&pl.tonemap_sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&fx.bloom_view) },
        ],
    });
    let bloom_size = fx.bloom_tex.size();
    let wg_x = bloom_size.width.div_ceil(8);
    let wg_y = bloom_size.height.div_ceil(8);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Bloom prefilter"), timestamp_writes: None,
        });
        pass.set_pipeline(&pl.bloom_prefilter);
        pass.set_bind_group(0, &bloom_bg, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

pub(super) fn pass_ssao(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    fx: &PostFxTextures,
    depth_view: &wgpu::TextureView,
    noise_view: &wgpu::TextureView,
    proj: &[[f32; 4]; 4],
    inv_proj: &[[f32; 4]; 4],
) {
    let params = SsaoParamsUniform {
        // Keep AO conservative so streamed LOD/full-res transitions do not cause visible brightness jumps.
        proj: *proj, inv_proj: *inv_proj, radius: 0.8, bias: 0.02, power: 1.0, _pad: [0.0, 0.0], _tail_pad: [0.0, 0.0, 0.0],
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SSAO params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let ssao_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SSAO BG"), layout: &pl.ssao_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(noise_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&pl.ssao_sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("SSAO"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &fx.ssao_view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
    });
    pass.set_pipeline(&pl.ssao_pipeline);
    pass.set_bind_group(0, &ssao_bg, &[]);
    pass.draw(0..3, 0..1);
}

pub(super) fn pass_ssao_blur(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    fx: &PostFxTextures,
    depth_view: &wgpu::TextureView,
) {
    let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SSAO Blur BG"), layout: &pl.ssao_blur_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&fx.ssao_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&pl.ssao_sampler) },
        ],
    });

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("SSAO Blur"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &fx.ssao_blur_view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
    });
    pass.set_pipeline(&pl.ssao_blur_pipeline);
    pass.set_bind_group(0, &blur_bg, &[]);
    pass.draw(0..3, 0..1);
}

pub(super) fn pass_tonemap_hdr_to_post_ldr(
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    fx: &PostFxTextures,
    bg_color: wgpu::Color,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("ACES+FXAA+Bloom+SSAO"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &fx.post_ldr_view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(bg_color), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
    });
    pass.set_pipeline(&pl.tonemap_pipeline);
    pass.set_bind_group(0, &fx.tonemap_bg, &[]);
    pass.draw(0..3, 0..1);
}

/// Full-resolution texture copy (src → dst) via compute dispatch.
/// Used for ping-pong buffer resolution in post-processing chain.
pub(super) fn pass_copy_texture(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pl: &crate::post_processor::PostFxPipelines,
    src: &wgpu::TextureView,
    dst: &wgpu::TextureView,
    width: u32,
    height: u32,
) {
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Copy BG"),
        layout: &pl.bloom_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&pl.tonemap_sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(dst) },
        ],
    });
    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Copy texture"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pl.copy_pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(wg_x, wg_y, 1);
}

pub(super) fn pass_fxaa_ldr_to_swapchain(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    ldr_scene_view: &wgpu::TextureView,
    swap_view: &wgpu::TextureView,
    bg_color: wgpu::Color,
) {
    let fxaa_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FXAA LDR BG"),
        layout: &pl.fxaa_ldr_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(ldr_scene_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&pl.tonemap_sampler),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("FXAA LDR to swapchain"),
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
    pass.set_pipeline(&pl.fxaa_ldr_pipeline);
    pass.set_bind_group(0, &fxaa_bg, &[]);
    pass.draw(0..3, 0..1);
}

pub(super) fn pass_blit_post_ldr_to_swapchain(
    encoder: &mut wgpu::CommandEncoder,
    pl: &PostFxPipelines,
    fx: &PostFxTextures,
    swap_view: &wgpu::TextureView,
    bg_color: wgpu::Color,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Blit LDR to swapchain"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: swap_view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(bg_color), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
    });
    pass.set_pipeline(&pl.blit_pipeline);
    pass.set_bind_group(0, &fx.blit_bg, &[]);
    pass.draw(0..3, 0..1);
}
