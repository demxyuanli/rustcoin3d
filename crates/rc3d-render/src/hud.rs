use std::collections::HashMap;

use glyphon::{
    Attrs, Buffer, Cache, Color, CustomGlyph, Family, FontSystem, Metrics,
    RasterizeCustomGlyphRequest, Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds,
    TextRenderer, Viewport,
};

use crate::render_passes::pass_text::TextDrawCommand;
use crate::renderer::FrameStats;

fn compose_hud_text(
    overlay_lines: &[String],
    fps: f32,
    frame_time_ms: f32,
    stats: &FrameStats,
    mode_name: &str,
) -> String {
    let mut text = String::new();
    if !overlay_lines.is_empty() {
        text.push_str(&overlay_lines.join("\n"));
        text.push('\n');
    }
    text.push_str(&format!(
        "FPS: {fps:.1} | Frame: {frame_time_ms:.2}ms\nTri: {} | Draws: {} | Culled: {}\nMode: {mode_name}",
        stats.visible_triangles, stats.visible_draw_calls, stats.culled_draw_calls
    ));
    if let Some(times) = stats.gpu_pass_times_us {
        text.push_str(&format!(
            "\nGPU: shadow={:.0}us solid={:.0}us post={:.0}us total={:.0}us",
            times[0], times[1], times[2], times[3]
        ));
    }
    text
}

fn glyphon_depth_stencil(compare: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32FloatStencil8,
        depth_write_enabled: Some(false),
        depth_compare: Some(compare),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn depth_from_glyph_metadata(metadata: usize) -> f32 {
    f32::from_bits(metadata as u32)
}

pub struct HudRenderer {
    pub(crate) font_system: FontSystem,
    pub(crate) swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    /// FPS / scene Text2 — always on top (no depth test).
    text_renderer: TextRenderer,
    /// Plane annotation labels — depth-tested (forward-Z).
    annotation_text_renderer_forward: TextRenderer,
    /// Plane annotation labels — depth-tested (reverse-Z).
    annotation_text_renderer_reverse: TextRenderer,
    buffer: Buffer,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// User-defined overlay text lines displayed above FPS stats.
    pub overlay_lines: Vec<String>,
    /// Positioned text entries from Billboards / Text3 nodes.
    pub(crate) scene_positioned_texts: Vec<TextDrawCommand>,
    /// Merged list (scene Text2/3 only) rebuilt each HUD pass.
    pub(crate) positioned_texts: Vec<TextDrawCommand>,
    positioned_buffers: Vec<Buffer>,
    empty_buffer: Buffer,
    plane_custom_glyphs: Vec<CustomGlyph>,
    /// Persistent mask data keyed by custom glyph id (glyphon atlas contract).
    plane_glyph_raster_by_id: HashMap<u16, crate::plane_text::PlaneGlyphCacheEntry>,
    plane_label_raster_cache: crate::plane_text::PlaneLabelRasterCache,
    label_attrs: Attrs<'static>,
}

impl HudRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let cache = Cache::new(device);
        let mut atlas = TextAtlas::new(device, queue, &cache, surface_format);
        let viewport = Viewport::new(device, &cache);
        let ms = wgpu::MultisampleState::default();
        let text_renderer = TextRenderer::new(&mut atlas, device, ms, None);
        let annotation_text_renderer_forward = TextRenderer::new(
            &mut atlas,
            device,
            ms,
            Some(glyphon_depth_stencil(wgpu::CompareFunction::Less)),
        );
        let annotation_text_renderer_reverse = TextRenderer::new(
            &mut atlas,
            device,
            ms,
            Some(glyphon_depth_stencil(wgpu::CompareFunction::Greater)),
        );
        let mut font_system = FontSystem::new();
        let _family = crate::font_loader::configure_font_system(&mut font_system);
        let label_attrs = Attrs::new().family(Family::SansSerif);
        let swash_cache = SwashCache::new();
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(16.0, 22.0));
        buffer.set_size(Some(width as f32), Some(height as f32));
        buffer.set_text("", &label_attrs, Shaping::Advanced, None);
        let mut empty_buffer = Buffer::new(&mut font_system, Metrics::new(14.0, 17.0));
        empty_buffer.set_size(Some(width as f32), Some(height as f32));
        empty_buffer.set_text("", &label_attrs, Shaping::Advanced, None);
        empty_buffer.shape_until_scroll(&mut font_system, false);
        let mut hud = Self {
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            annotation_text_renderer_forward,
            annotation_text_renderer_reverse,
            buffer,
            width,
            height,
            overlay_lines: Vec::new(),
            scene_positioned_texts: Vec::new(),
            positioned_texts: Vec::new(),
            positioned_buffers: Vec::new(),
            empty_buffer,
            plane_custom_glyphs: Vec::new(),
            plane_glyph_raster_by_id: HashMap::new(),
            plane_label_raster_cache: crate::plane_text::PlaneLabelRasterCache::new(),
            label_attrs,
        };
        hud.viewport.update(queue, Resolution { width, height });
        hud
    }

    pub fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.buffer
            .set_size(Some(width as f32), Some(height as f32));
        self.viewport.update(queue, Resolution { width, height });
    }

    pub fn has_plane_annotation_glyphs(&self) -> bool {
        !self.plane_custom_glyphs.is_empty()
    }

    /// Prepare positioned-text buffers from the current `positioned_texts` list.
    pub fn prepare_positioned_texts(&mut self) {
        self.positioned_buffers.clear();
        for cmd in &self.positioned_texts {
            crate::font_loader::ensure_named_font(&mut self.font_system, &cmd.font_name);
            let mut buf = Buffer::new(&mut self.font_system, Metrics::new(cmd.size, cmd.size * 1.2));
            buf.set_size(Some(self.width as f32), Some(self.height as f32));
            let attrs = crate::font_loader::attrs_from_font(&cmd.font_name, cmd.font_style);
            buf.set_text(&cmd.string, &attrs, Shaping::Advanced, None);
            buf.shape_until_scroll(&mut self.font_system, false);
            self.positioned_buffers.push(buf);
        }
    }

    /// Rasterize plane-aligned annotation labels as rotated custom glyphs.
    pub fn prepare_plane_annotation_glyphs(&mut self, labels: &[TextDrawCommand]) {
        self.plane_custom_glyphs = crate::plane_text::build_plane_label_custom_glyphs(
            &mut self.plane_label_raster_cache,
            &mut self.plane_glyph_raster_by_id,
            &mut self.font_system,
            &mut self.swash_cache,
            labels,
        );
    }

    fn text_bounds(&self) -> TextBounds {
        TextBounds {
            left: 0,
            top: 0,
            right: self.width as i32,
            bottom: self.height as i32,
        }
    }

    /// Upload plane annotation glyphs with per-label NDC depth (call before depth-tested render pass).
    pub fn prepare_plane_annotation_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth_reversed_z: bool,
    ) {
        if self.plane_custom_glyphs.is_empty() {
            return;
        }
        let bounds = self.text_bounds();
        let areas = [TextArea {
            buffer: &self.empty_buffer,
            left: 0.0,
            top: 0.0,
            scale: 1.0,
            bounds,
            default_color: Color::rgb(255, 255, 255),
            custom_glyphs: &self.plane_custom_glyphs,
        }];
        let rasters = self.plane_glyph_raster_by_id.clone();
        let renderer = if depth_reversed_z {
            &mut self.annotation_text_renderer_reverse
        } else {
            &mut self.annotation_text_renderer_forward
        };
        if let Err(e) = renderer.prepare_with_depth_and_custom(
            device,
            queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            areas,
            &mut self.swash_cache,
            depth_from_glyph_metadata,
            move |req: RasterizeCustomGlyphRequest| {
                Some(
                    rasters
                        .get(&req.id)
                        .map(|entry| crate::plane_text::glyph_for_request(entry, &req))
                        .unwrap_or_else(|| crate::plane_text::fallback_custom_glyph_raster(&req)),
                )
            },
        ) {
            log::error!("Annotation text prepare: {:?}", e);
        }
    }

    /// Upload FPS / scene Text2 buffers (no depth; drawn on top).
    pub fn prepare_hud_chrome_atlas(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.prepare_positioned_texts();
        let bounds = self.text_bounds();
        let mut areas = Vec::with_capacity(1 + self.positioned_texts.len());
        areas.push(TextArea {
            buffer: &self.buffer,
            left: 12.0,
            top: 12.0,
            scale: 1.0,
            bounds,
            default_color: Color::rgb(240, 240, 240),
            custom_glyphs: &[],
        });
        for (i, cmd) in self.positioned_texts.iter().enumerate() {
            let col = cmd.color;
            areas.push(TextArea {
                buffer: &self.positioned_buffers[i],
                left: cmd.screen_pos[0],
                top: cmd.screen_pos[1],
                scale: 1.0,
                bounds,
                default_color: Color::rgba(
                    (col[0] * 255.0) as u8,
                    (col[1] * 255.0) as u8,
                    (col[2] * 255.0) as u8,
                    (col[3] * 255.0) as u8,
                ),
                custom_glyphs: &[],
            });
        }
        if let Err(e) = self.text_renderer.prepare(
            device,
            queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            areas,
            &mut self.swash_cache,
        ) {
            log::error!("HUD chrome prepare: {:?}", e);
        }
    }

    pub fn prepare_gpu_atlas_for_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth_reversed_z: bool,
    ) {
        self.prepare_plane_annotation_atlas(device, queue, depth_reversed_z);
        self.prepare_hud_chrome_atlas(device, queue);
    }

    pub fn set_fps_buffer_text(
        &mut self,
        fps: f32,
        frame_time_ms: f32,
        stats: &FrameStats,
        mode_name: &str,
    ) {
        let text = compose_hud_text(&self.overlay_lines, fps, frame_time_ms, stats, mode_name);
        self.buffer.set_text(&text, &self.label_attrs, Shaping::Advanced, None);
        self.buffer.shape_until_scroll(&mut self.font_system, false);
    }

    pub fn render_plane_annotations(&self, pass: &mut wgpu::RenderPass<'_>, depth_reversed_z: bool) {
        let renderer = if depth_reversed_z {
            &self.annotation_text_renderer_reverse
        } else {
            &self.annotation_text_renderer_forward
        };
        if let Err(e) = renderer.render(&self.atlas, &self.viewport, pass) {
            log::error!("Annotation text render: {:?}", e);
        }
    }

    pub fn render_hud_chrome(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Err(e) = self.text_renderer.render(&self.atlas, &self.viewport, pass) {
            log::error!("HUD chrome render: {:?}", e);
        }
    }

    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.render_plane_annotations(pass, false);
        self.render_hud_chrome(pass);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_hud_text_prepends_overlay_lines() {
        let stats = FrameStats {
            visible_triangles: 12,
            visible_draw_calls: 3,
            culled_draw_calls: 1,
            ..Default::default()
        };
        let text = compose_hud_text(
            &["hello".to_string(), "world".to_string()],
            60.0,
            16.67,
            &stats,
            "Shaded",
        );

        assert!(text.starts_with("hello\nworld\nFPS: 60.0"));
        assert!(text.contains("Tri: 12 | Draws: 3 | Culled: 1"));
        assert!(text.contains("Mode: Shaded"));
    }
}
