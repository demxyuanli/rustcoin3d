//! Swapchain overlays after the film: SS edges, lines, grid, markup, HUD.
use crate::pipelines::DepthModePipelines;
use super::PassContext;
use super::{
    pass_edge, pass_grid, pass_hidden, pass_hud, pass_markup, pass_viewport, pass_wireframe, ss_edge,
};

pub(super) fn encode_swapchain_overlays(
    renderer: &mut crate::renderer::Renderer,
    ctx: &PassContext<'_>,
    mut encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    depth_read_view: &wgpu::TextureView,
    ew: u32,
    eh: u32,
    scene_pl: &DepthModePipelines,
    defer_line_overlays: bool,
    run_hidden: bool,
    run_wireframe: bool,
    edge_worthy: bool,
    has_overlay: bool,
    mut post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    // When set, snapshot scene film (pre-egui) into the UI retain buffer.
    retain_src: Option<&wgpu::Texture>,
) -> bool {
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

    if defer_line_overlays {
        if run_hidden {
            pass_hidden::pass_hidden_edges(
                renderer,
                &mut encoder,
                view,
                &depth_view,
                ctx,
                &scene_pl,
            );
        }
        if run_wireframe {
            #[cfg(feature = "profiler")]
            let _span_wireframe = tracy_client::span!("wireframe");
            pass_wireframe::pass_wireframe(renderer, &mut encoder, view, &depth_view, ctx, &scene_pl);
        }
        if edge_worthy || has_overlay {
            pass_edge::pass_edge_overlay(renderer, &mut encoder, view, &depth_view, ctx, edge_worthy, &scene_pl);
        }
    }

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

    pass_grid::pass_gizmo_lines(
        renderer,
        &mut encoder,
        view,
        &depth_view,
        renderer.frame.scene_vp,
        ctx.depth_reversed_z,
    );

    // Viewport border overlay (not for independent overlay world tiles).
    if !renderer.overlay_pass {
        pass_viewport::encode_viewport_borders(
            renderer,
            &mut encoder,
            view,
            ew,
            eh,
        );
    }

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
                        let mapped = buf.slice(..).get_mapped_range().expect("occlusion map");
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

    renderer.encode_overlay_composite(&mut encoder, view);
    pass_hud::encode_hud_overlay(
        renderer,
        &mut encoder,
        view,
        &depth_view,
        ctx.depth_reversed_z,
    );
    // Retain scene-only film before egui so UiOnly = blit + paint once (no stacked UI ghosts).
    if let Some(tex) = retain_src {
        renderer.capture_ui_retain(&mut encoder, tex);
    }
    if let Some(cb) = &mut post_swapchain_overlay {
        cb(&mut encoder, view);
    }
    renderer.gpu_timer.end(&mut encoder, ti_hud);

    occlusion_captured
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
                include_str!("../shaders/occlusion_downsample.wgsl").into(),
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
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
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

