use glam::Mat4;
use rc3d_scene::SceneGraph;
use crate::adaptive_quality::AdaptiveQuality;
use crate::render_action::{apply_world_camera_ex, DrawCall};
use crate::render_passes;
use crate::viewport::{QuadViewEye, ViewportRect};
use super::internals::QuadViewTile;
use super::types::FrameStats;

impl super::Renderer {
    pub fn render_draw_calls_with_overlay(
        &mut self,
        draw_calls: &[DrawCall],
        scene: &SceneGraph,
        mut post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    ) -> FrameStats {
        let scale_active = self.interaction_active
            && self.interaction_render_scale < 0.999
            && self.gpu.upscale_pipeline.is_some();
        if !scale_active {
            return self.render_draw_calls_core(
                draw_calls, scene, post_swapchain_overlay,
                render_passes::FramePresentation::Swapchain, None,
            );
        }

        // Dynamic resolution: render to downscaled intermediate, then upscale to swapchain.
        let (ew, eh) = (self.config.width, self.config.height);
        self.ensure_interaction_downscale_targets(ew, eh);

        // Take intermediate textures out to avoid borrow conflicts with render_draw_calls_core
        let down_tex = self.gpu.interaction_downscale_tex.take();
        let down_view = self.gpu.interaction_downscale_view.take();
        let down_depth = self.gpu.interaction_downscale_depth.take();
        let down_depth_view = self.gpu.interaction_downscale_depth_view.take();
        let down_depth_read_view = self.gpu.interaction_downscale_depth_read_view.take();

        let (down_tex, down_view, down_depth, down_depth_view, down_depth_read_view) =
            match (down_tex, down_view, down_depth, down_depth_view, down_depth_read_view) {
                (Some(t), Some(v), Some(d), Some(dv), Some(drv)) => (t, v, d, dv, drv),
                _ => return self.render_draw_calls_core(
                    draw_calls, scene, post_swapchain_overlay,
                    render_passes::FramePresentation::Swapchain, None,
                ),
            };
        let down_w = down_tex.width();
        let down_h = down_tex.height();

        // Swap in the cached downscaled depth texture for the offscreen render
        // (avoids creating/destroying a depth texture every interaction frame).
        let saved_depth = self.gpu.depth_texture.take();
        self.gpu.depth_texture = Some((down_depth, down_depth_view, down_depth_read_view));

        // Downscaled depth mismatches full-res intermediate targets (LDR shade, HDR post-fx).
        // Save and disable both so execute_passes renders directly to the offscreen surface.
        let saved_ldr_fxaa = self.enable_ldr_fxaa;
        let saved_hdr = self.hdr_post_processing;
        self.enable_ldr_fxaa = false;
        self.hdr_post_processing = false;

        let stats = self.render_draw_calls_core(
            draw_calls, scene, None,
            render_passes::FramePresentation::OffscreenSurface {
                output_texture: &down_tex,
                output_view: &down_view,
                width_px: down_w,
                height_px: down_h,
            },
            None,
        );

        self.hdr_post_processing = saved_hdr;
        self.enable_ldr_fxaa = saved_ldr_fxaa;

        // Restore original depth + intermediate textures (the downscaled depth
        // and its views go back into the interaction cache for reuse).
        if let Some((d, dv, drv)) = self.gpu.depth_texture.take() {
            // Guard against the core render having replaced the depth texture.
            if d.width() == down_w && d.height() == down_h {
                self.gpu.interaction_downscale_depth = Some(d);
                self.gpu.interaction_downscale_depth_view = Some(dv);
                self.gpu.interaction_downscale_depth_read_view = Some(drv);
            }
        }
        self.gpu.depth_texture = saved_depth;
        self.gpu.interaction_downscale_tex = Some(down_tex);
        self.gpu.interaction_downscale_view = Some(down_view);

        // Upscale from intermediate to swapchain
        let Some(swapchain_frame) = self.acquire_surface_texture() else {
            return stats;
        };
        let swapchain_view = swapchain_frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let intermediate_view = self
            .gpu
            .interaction_downscale_view
            .as_ref()
            .expect("Intermediate view must exist");
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Upscale Encoder"),
        });
        let _ = self.blit_texture_view_to_target(
            &mut encoder,
            intermediate_view,
            &swapchain_view,
        );
        // Scene film + nav-cube tiles, then retain, then egui (UiOnly reuses retain).
        self.encode_overlay_composite(&mut encoder, &swapchain_view);
        self.capture_ui_retain(&mut encoder, &swapchain_frame.texture);
        if let Some(ref mut hook) = post_swapchain_overlay {
            hook(&mut encoder, &swapchain_view);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        // Occlusion depth readback is handled non-blockingly inside
        // execute_passes (async map kicked after submit, harvested next frame).
        self.queue.present(swapchain_frame);

        stats
    }

    /// Present by blitting the scene-only retain film, then painting UI once.
    /// Returns false when blit cannot run (caller should Full-render).
    pub fn present_post_overlay_only(
        &mut self,
        mut post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    ) -> bool {
        if !self.has_ui_retain() || self.gpu.ui_retain_blit_bg.is_none() {
            return false;
        }
        let Some(output) = self.acquire_surface_texture() else {
            return false;
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("post_overlay_only"),
            });
        if !self.blit_ui_retain_to_view(&mut encoder, &view) {
            return false;
        }
        if let Some(hook) = post_swapchain_overlay.as_mut() {
            hook(&mut encoder, &view);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(output);
        true
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
        let cad = self.cad_status_line();
        let cad_changed = self.frame.last_cad_hud != cad;
        let interval = match self.gpu.adaptive_quality {
            AdaptiveQuality::High => 1,
            AdaptiveQuality::Medium => 2,
            AdaptiveQuality::Low => 6,
        };
        if !cad_changed
            && self.frame.frame_counter.saturating_sub(self.frame.last_hud_update_frame) < interval
        {
            return;
        }
        self.frame.last_hud_update_frame = self.frame.frame_counter;
        self.frame.last_cad_hud = cad.clone();
        let quality_name = self.adaptive_quality_name();
        let hud_mode_name = format!("{mode_name} [{quality_name}]\n{cad}");
        if let Some(hud) = &mut self.gpu.hud {
            hud.set_fps_buffer_text(fps, frame_time_ms, stats, &hud_mode_name);
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
        let cap_specs: Vec<([f32; 4], rc3d_scene::SectionCapStyle)> = self
            .frame
            .section_cap_tints
            .iter()
            .zip(self.frame.clip_planes.iter())
            .filter_map(|(t, p)| t.map(|c| (*p, c)))
            .collect();
        if cap_specs.is_empty() {
            return;
        }
        let entries: Vec<(usize, crate::gpu_resource::MeshId)> = solid_order
            .iter()
            .filter_map(|&i| {
                if !draw_calls[i].appearance().wants_filled() {
                    return None;
                }
                mesh_handles[i].map(|m| (i, m))
            })
            .collect();
        if entries.is_empty() {
            return;
        }
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

    /// Fills the open cross section with flat color using the mesh's back faces
    /// with clip plane. Since cull=Front, only back faces render; the clip plane
    /// discards fragments above the plane, leaving only the cross-section disc.
    pub fn render_section_caps(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        shade_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        scene_pl: &crate::pipelines::DepthModePipelines,
        cap_specs: &[([f32; 4], rc3d_scene::SectionCapStyle)],
        entries: &[(usize, crate::gpu_resource::MeshId)],
        draw_calls: &[&crate::render_action::DrawCall],
    ) {
        if cap_specs.is_empty() || entries.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Section cap fill (back faces)"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: shade_view,
                resolve_target: None,
                depth_slice: None,
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
            occlusion_query_set: None, multiview_mask: None,
        });
        self.apply_scene_viewport(&mut pass);
        pass.set_pipeline(&scene_pl.section_cap_fill);
        let mut last_bound_mesh = None;
        for &(plane, style) in cap_specs {
            for &(i, mesh_id) in entries {
                let dc = draw_calls[i];
                let mut clip_planes = [[0.0f32; 4]; 6];
                clip_planes[0] = plane;
                let uniforms = crate::vertex::FlatUniforms {
                    mvp: dc.mvp.to_cols_array_2d(),
                    color: style.color,
                    model: dc.model_matrix.to_cols_array_2d(),
                    clip_planes,
                    clip_count: [1.0, 0.0, 0.0, 0.0],
                    hatch_color: style.hatch_color,
                    hatch_params: style.hatch_params_gpu(),
                    hatch_extra: style.hatch_extra_gpu(),
                };
                if let Some(offset) = self.gpu.flat_pool.push_flat(&uniforms) {
                    pass.set_bind_group(0, self.gpu.flat_pool.bind_group(), &[offset]);
                    self.draw_mesh_instanced(
                        &mut pass,
                        mesh_id,
                        0,
                        1,
                        dc.index_draw_range(),
                        &mut last_bound_mesh,
                    );
                }
            }
        }
    }

    /// Render a scene to an offscreen RGBA8 image. Returns (width, height, pixels).
    /// Useful for headless rendering, thumbnails, and automated testing.
    pub fn render_to_image(
        &mut self,
        draw_calls: &[DrawCall],
        scene: &SceneGraph,
        width: u32,
        height: u32,
    ) -> (u32, u32, Vec<u8>) {
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        // Match swapchain / post-process pipeline color format (typically Bgra8UnormSrgb).
        let format = self.config.format;
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[format],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let pres = render_passes::FramePresentation::OffscreenSurface {
            output_texture: &tex,
            output_view: &view,
            width_px: width,
            height_px: height,
        };
        self.render_draw_calls_core(draw_calls, scene, None, pres, None);

        let bytes_per_row = width * 4;
        let padded_bytes_per_row = bytes_per_row.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback buffer"),
            size: padded_bytes_per_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_bytes_per_row), rows_per_image: Some(height) } },
            size,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = slice.get_mapped_range().expect("readback map");
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        for row in 0..height as usize {
            let src = row * padded_bytes_per_row as usize;
            let dst = row * bytes_per_row as usize;
            pixels[dst..dst + bytes_per_row as usize].copy_from_slice(&data[src..src + bytes_per_row as usize]);
        }
        drop(data);
        buffer.unmap();
        if matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for px in pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        (width, height, pixels)
    }

    pub(crate) fn ensure_quad_tiles(&mut self, rects: &[ViewportRect]) {
        let reuse = self.gpu.quad_tiles.len() == rects.len()
            && self
                .gpu
                .quad_tiles
                .iter()
                .zip(rects)
                .all(|(t, r)| t.width == r.width.max(1) && t.height == r.height.max(1));
        if reuse {
            return;
        }
        self.gpu.quad_tiles.clear();
        for rect in rects {
            let w = rect.width.max(1);
            let h = rect.height.max(1);
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Quad view tile"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.gpu.quad_tiles.push(QuadViewTile {
                width: w,
                height: h,
                texture,
                view,
                depth: None,
            });
        }
    }

    pub(crate) fn render_quad_tiles(
        &mut self,
        draw_calls: &mut [DrawCall],
        scene: &SceneGraph,
        eyes: &[QuadViewEye],
        rects: &[ViewportRect],
    ) -> FrameStats {
        self.ensure_quad_tiles(rects);
        let tiles = std::mem::take(&mut self.gpu.quad_tiles);
        let n = tiles.len().min(eyes.len());
        let mut stats = FrameStats::default();
        for i in 0..n {
            apply_world_camera_ex(
                draw_calls,
                eyes[i].view,
                eyes[i].projection,
                eyes[i].camera_pos,
                eyes[i].orthographic,
            );
            let tile = &tiles[i];
            let inv = eyes[i].projection.inverse();
            let tile_stats = self.render_draw_calls_to_viewport_texture(
                draw_calls,
                scene,
                &tile.texture,
                &tile.view,
                tile.width,
                tile.height,
                eyes[i].projection,
                inv,
            );
            stats.visible_triangles += tile_stats.visible_triangles;
            stats.visible_draw_calls += tile_stats.visible_draw_calls;
            stats.culled_draw_calls += tile_stats.culled_draw_calls;
            stats.frame_time_ms += tile_stats.frame_time_ms;
        }
        self.gpu.quad_tiles = tiles;
        stats
    }

    pub(crate) fn blit_quad_tiles_to_view(&self, target: &wgpu::TextureView, rects: &[ViewportRect]) {
        let Some(pipeline) = self.gpu.upscale_pipeline.as_ref() else {
            return;
        };
        let Some(bgl) = self.gpu.upscale_bgl.as_ref() else {
            return;
        };
        let Some(sampler) = self.gpu.upscale_sampler.as_ref() else {
            return;
        };
        let mut bind_groups = Vec::with_capacity(self.gpu.quad_tiles.len());
        for tile in &self.gpu.quad_tiles {
            bind_groups.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Quad tile blit"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&tile.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            }));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Quad view composite"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit quad tiles"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.06,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None, multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            for (i, bg) in bind_groups.iter().enumerate() {
                let Some(rect) = rects.get(i) else {
                    break;
                };
                // Clamp to the destination target: swapchain size can lag
                // `config` during resize; out-of-bounds scissor is fatal.
                let rect = rect.clamped_to(self.pass_target_size.0, self.pass_target_size.1);
                let w = rect.width.max(1);
                let h = rect.height.max(1);
                pass.set_viewport(
                    rect.x as f32,
                    rect.y as f32,
                    w as f32,
                    h as f32,
                    0.0,
                    1.0,
                );
                pass.set_scissor_rect(rect.x, rect.y, w, h);
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Render the standard four-view pack to the swapchain (tiles + blit + borders).
    pub fn render_standard_quad_views(
        &mut self,
        draw_calls: &mut [DrawCall],
        scene: &SceneGraph,
        eyes: &[QuadViewEye],
    ) -> FrameStats {
        self.render_standard_quad_views_with_overlay(draw_calls, scene, eyes, None)
    }

    pub fn render_standard_quad_views_with_overlay(
        &mut self,
        draw_calls: &mut [DrawCall],
        scene: &SceneGraph,
        eyes: &[QuadViewEye],
        mut post_swapchain_overlay: Option<&mut dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView)>,
    ) -> FrameStats {
        if eyes.len() < 2 || self.gpu.upscale_pipeline.is_none() {
            apply_world_camera_ex(
                draw_calls,
                eyes.first().map(|e| e.view).unwrap_or(Mat4::IDENTITY),
                eyes.first().map(|e| e.projection).unwrap_or(Mat4::IDENTITY),
                eyes.first().map(|e| e.camera_pos).unwrap_or(glam::Vec3::ZERO),
                eyes.first().map(|e| e.orthographic).unwrap_or(false),
            );
            return self.render_draw_calls_with_overlay(draw_calls, scene, post_swapchain_overlay);
        }
        let rects: Vec<ViewportRect> = self
            .frame
            .viewport_layout
            .viewports
            .iter()
            .map(|v| v.rect)
            .collect();
        let stats = self.render_quad_tiles(draw_calls, scene, eyes, &rects);
        let Some(frame) = self.acquire_surface_texture() else {
            return stats;
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (sw, sh) = (self.config.width, self.config.height);
        self.blit_quad_tiles_to_view(&view, &rects);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Quad view borders"),
            });
        crate::render_passes::pass_viewport::encode_viewport_borders(
            self, &mut encoder, &view, sw, sh,
        );
        self.encode_overlay_composite(&mut encoder, &view);
        self.capture_ui_retain(&mut encoder, &frame.texture);
        if let Some(ref mut hook) = post_swapchain_overlay {
            hook(&mut encoder, &view);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
        stats
    }

    /// Offscreen composite of the four-view pack (for screenshots).
    pub fn render_standard_quad_to_image(
        &mut self,
        draw_calls: &mut [DrawCall],
        scene: &SceneGraph,
        eyes: &[QuadViewEye],
        width: u32,
        height: u32,
    ) -> (u32, u32, Vec<u8>) {
        let w = width.max(1);
        let h = height.max(1);
        let rects: Vec<ViewportRect> = self
            .frame
            .viewport_layout
            .viewports
            .iter()
            .map(|v| v.rect)
            .collect();
        let _ = self.render_quad_tiles(draw_calls, scene, eyes, &rects);
        let format = self.config.format;
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Quad composite"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.blit_quad_tiles_to_view(&view, &rects);
        let bytes_per_row = w * 4;
        let padded_bytes_per_row = bytes_per_row.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad readback"),
            size: padded_bytes_per_row as u64 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = slice.get_mapped_range().expect("readback map");
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for row in 0..h as usize {
            let src = row * padded_bytes_per_row as usize;
            let dst = row * bytes_per_row as usize;
            pixels[dst..dst + bytes_per_row as usize]
                .copy_from_slice(&data[src..src + bytes_per_row as usize]);
        }
        drop(data);
        buffer.unmap();
        if matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for px in pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        (w, h, pixels)
    }
}
