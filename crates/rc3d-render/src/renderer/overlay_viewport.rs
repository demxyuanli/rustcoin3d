//! Independent overlay viewports: offscreen world tiles composited onto the swapchain.

use rc3d_scene::SceneGraph;

use crate::background::BgMode;
use crate::render_action::DrawCall;
use crate::viewport::ViewportRect;
use super::internals::QuadViewTile;
use super::types::FrameStats;

impl super::Renderer {
    pub(crate) fn ensure_overlay_tiles(&mut self, rects: &[ViewportRect]) {
        let reuse = self.gpu.overlay_tiles.len() == rects.len()
            && self
                .gpu
                .overlay_tiles
                .iter()
                .zip(rects)
                .all(|(t, r)| t.width == r.width.max(1) && t.height == r.height.max(1));
        if reuse {
            return;
        }
        self.gpu.overlay_tiles.clear();
        for rect in rects {
            let w = rect.width.max(1);
            let h = rect.height.max(1);
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Overlay viewport tile"),
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
            self.gpu.overlay_tiles.push(QuadViewTile {
                width: w,
                height: h,
                texture,
                view,
                depth: None,
            });
        }
    }

    /// Render one overlay world into `gpu.overlay_tiles[tile_index]`.
    ///
    /// Isolates renderer flags so the main film is not affected. Does not blit.
    pub fn render_overlay_world_tile(
        &mut self,
        tile_index: usize,
        draw_calls: &[DrawCall],
        scene: &SceneGraph,
        projection: glam::Mat4,
        clear_color: [f32; 4],
    ) -> FrameStats {
        if tile_index >= self.gpu.overlay_tiles.len() {
            return FrameStats::default();
        }

        let saved_hdr = self.hdr_post_processing;
        let saved_fxaa = self.enable_ldr_fxaa;
        let saved_smaa = self.enable_ldr_smaa;
        let saved_hud = self.hud_enabled;
        let saved_grid = self.grid_enabled;
        let saved_omni = self.enable_omni_shadows;
        let saved_cluster = self.enable_cluster_lights;
        let saved_taa = self.enable_taa;
        let saved_ssr = self.enable_ssr;
        let saved_shadow = self.tier_wants_shadow;
        let saved_edges = self.tier_wants_edges;
        let saved_cull = self.gpu.gpu_cull_enabled;
        let saved_wboit = self.enable_wboit;
        let saved_xray = self.xray_mode;
        let saved_ghost = self.ghost_unselected;
        let saved_ss_edges = self.screen_space_edges;
        let saved_ss_sel = self.screen_space_selection_outline;
        let saved_gizmo = std::mem::take(&mut self.gizmo_line_batches);
        let saved_bg = self.gpu.bg_settings.clone();
        let saved_hzb = self.gpu.hzb.take();
        let saved_depth = self.gpu.depth_texture.take();
        let saved_vp = self.frame.scene_vp;
        let saved_last_vp = self.frame.last_vp;
        let saved_effects = std::mem::take(&mut self.frame.effect_commands);
        let saved_markup = std::mem::take(&mut self.frame.markup_vertices);
        let saved_labels = std::mem::take(&mut self.frame.annotation_world_labels);
        let saved_clip = std::mem::take(&mut self.frame.clip_planes);
        let saved_caps = std::mem::take(&mut self.frame.section_cap_tints);

        self.overlay_pass = true;
        self.hdr_post_processing = false;
        self.enable_ldr_fxaa = false;
        self.enable_ldr_smaa = false;
        self.hud_enabled = false;
        self.grid_enabled = false;
        self.enable_omni_shadows = false;
        self.enable_cluster_lights = false;
        self.enable_taa = false;
        self.enable_ssr = false;
        self.tier_wants_shadow = false;
        self.tier_wants_edges = false;
        self.gpu.gpu_cull_enabled = false;
        self.enable_wboit = false;
        self.xray_mode = false;
        self.ghost_unselected = false;
        self.screen_space_edges = false;
        self.screen_space_selection_outline = false;
        // Transparent overlays: solid black clear (RGB) so chroma-key blit works
        // even when the surface format drops alpha; a=0 when the format keeps it.
        let transparent = clear_color[3] < 0.5;
        let tile_clear = if transparent {
            [0.0, 0.0, 0.0, 0.0]
        } else {
            clear_color
        };
        self.gpu.bg_settings.mode = BgMode::Solid;
        self.gpu.bg_settings.top_color = tile_clear;
        self.gpu.bg_settings.bot_color = tile_clear;

        let tiles = std::mem::take(&mut self.gpu.overlay_tiles);
        let stats = {
            let tile = &tiles[tile_index];
            let vw = tile.width.max(1);
            let vh = tile.height.max(1);
            self.create_depth_texture_at(vw, vh);
            self.render_draw_calls_core(
                draw_calls,
                scene,
                None,
                crate::render_passes::FramePresentation::OffscreenSurface {
                    output_texture: &tile.texture,
                    output_view: &tile.view,
                    width_px: vw,
                    height_px: vh,
                },
                Some((projection, projection.inverse())),
            )
        };
        // Keep this tile's depth for transparent composite (discard uncleared pixels).
        let tile_depth = self.gpu.depth_texture.take();
        self.gpu.overlay_tiles = tiles;
        if let Some(tile) = self.gpu.overlay_tiles.get_mut(tile_index) {
            tile.depth = tile_depth;
        }

        self.overlay_pass = false;
        self.hdr_post_processing = saved_hdr;
        self.enable_ldr_fxaa = saved_fxaa;
        self.enable_ldr_smaa = saved_smaa;
        self.hud_enabled = saved_hud;
        self.grid_enabled = saved_grid;
        self.enable_omni_shadows = saved_omni;
        self.enable_cluster_lights = saved_cluster;
        self.enable_taa = saved_taa;
        self.enable_ssr = saved_ssr;
        self.tier_wants_shadow = saved_shadow;
        self.tier_wants_edges = saved_edges;
        self.gpu.gpu_cull_enabled = saved_cull;
        self.enable_wboit = saved_wboit;
        self.xray_mode = saved_xray;
        self.ghost_unselected = saved_ghost;
        self.screen_space_edges = saved_ss_edges;
        self.screen_space_selection_outline = saved_ss_sel;
        self.gizmo_line_batches = saved_gizmo;
        self.gpu.bg_settings = saved_bg;
        self.gpu.hzb = saved_hzb;
        self.gpu.depth_texture = saved_depth;
        self.frame.scene_vp = saved_vp;
        self.frame.last_vp = saved_last_vp;
        self.frame.effect_commands = saved_effects;
        self.frame.markup_vertices = saved_markup;
        self.frame.annotation_world_labels = saved_labels;
        self.frame.clip_planes = saved_clip;
        self.frame.section_cap_tints = saved_caps;
        stats
    }

    /// Composite overlay tiles onto `target` with Load (does not clear the main film).
    /// Discards pixels whose overlay depth is still at the clear value so the main
    /// film shows through (transparent nav cube background).
    pub(crate) fn encode_overlay_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        if self.overlay_pass || self.overlay_blit_rects.is_empty() {
            return;
        }
        let Some(pipeline) = self.gpu.overlay_blit_pipeline.as_ref() else {
            return;
        };
        let Some(bgl) = self.gpu.overlay_blit_bgl.as_ref() else {
            return;
        };
        let Some(color_sampler) = self.gpu.overlay_blit_sampler.as_ref() else {
            return;
        };
        let Some(depth_sampler) = self.gpu.overlay_depth_sampler.as_ref() else {
            return;
        };
        let n = self
            .gpu
            .overlay_tiles
            .len()
            .min(self.overlay_blit_rects.len());
        if n == 0 {
            return;
        }
        let mut indexed = Vec::with_capacity(n);
        for (i, tile) in self.gpu.overlay_tiles.iter().take(n).enumerate() {
            let Some((_, _, depth_read)) = tile.depth.as_ref() else {
                continue;
            };
            indexed.push((
                i,
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Overlay tile blit"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tile.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(color_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(depth_read),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(depth_sampler),
                        },
                    ],
                }),
            ));
        }
        if indexed.is_empty() {
            return;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit overlay viewports"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            for (i, bg) in indexed.iter() {
                let Some(rect) = self.overlay_blit_rects.get(*i) else {
                    break;
                };
        // Clamp to the destination target: swapchain texture can lag `config`
        // during resize; out-of-bounds scissor is a fatal wgpu error.
        let rect = rect.clamped_to(self.pass_target_size.0, self.pass_target_size.1);
        let w = rect.width.max(1);
        let h = rect.height.max(1);
        pass.set_viewport(rect.x as f32, rect.y as f32, w as f32, h as f32, 0.0, 1.0);
        pass.set_scissor_rect(rect.x, rect.y, w, h);
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }

    /// Allocate overlay tiles for this frame. Empty `rects` drops previous tiles.
    pub fn prepare_overlay_tiles(&mut self, rects: &[ViewportRect]) {
        self.overlay_blit_rects.clear();
        if rects.is_empty() {
            self.gpu.overlay_tiles.clear();
            return;
        }
        self.ensure_overlay_tiles(rects);
    }

    pub fn push_overlay_blit_rect(&mut self, rect: ViewportRect) {
        self.overlay_blit_rects.push(rect);
    }
}
