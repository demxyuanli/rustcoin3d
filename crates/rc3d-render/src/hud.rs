use std::collections::HashMap;

use glyphon::{
    Attrs, Buffer, Cache, Color, CustomGlyph, Family, FontSystem, Metrics, RasterizeCustomGlyphRequest,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds,
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

pub struct HudRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,
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
    plane_glyph_cache: HashMap<u16, crate::plane_text::PlaneGlyphCacheEntry>,
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
        let text_renderer = TextRenderer::new(
            &mut atlas,
            device,
            wgpu::MultisampleState::default(),
            None,
        );
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(16.0, 22.0));
        buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        buffer.set_text(
            &mut font_system,
            "",
            Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
        );
        let mut empty_buffer = Buffer::new(&mut font_system, Metrics::new(14.0, 17.0));
        empty_buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        empty_buffer.set_text(
            &mut font_system,
            "",
            Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
        );
        empty_buffer.shape_until_scroll(&mut font_system, false);
        let mut hud = Self {
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            buffer,
            width,
            height,
            overlay_lines: Vec::new(),
            scene_positioned_texts: Vec::new(),
            positioned_texts: Vec::new(),
            positioned_buffers: Vec::new(),
            empty_buffer,
            plane_custom_glyphs: Vec::new(),
            plane_glyph_cache: HashMap::new(),
        };
        hud.viewport.update(queue, Resolution { width, height });
        hud
    }

    pub fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.buffer
            .set_size(&mut self.font_system, Some(width as f32), Some(height as f32));
        self.viewport.update(queue, Resolution { width, height });
    }

    /// Prepare positioned-text buffers from the current `positioned_texts` list.
    /// Must be called before the HUD render pass if `positioned_texts` were modified.
    pub fn prepare_positioned_texts(&mut self) {
        self.positioned_buffers.clear();
        for cmd in &self.positioned_texts {
            let mut buf = Buffer::new(&mut self.font_system, Metrics::new(cmd.size, cmd.size * 1.2));
            buf.set_size(&mut self.font_system, Some(self.width as f32), Some(self.height as f32));
            buf.set_text(
                &mut self.font_system,
                &cmd.string,
                Attrs::new().family(Family::SansSerif),
                Shaping::Advanced,
            );
            buf.shape_until_scroll(&mut self.font_system, false);
            self.positioned_buffers.push(buf);
        }
    }

    /// Rasterize plane-aligned annotation labels as rotated custom glyphs.
    pub fn prepare_plane_annotation_glyphs(&mut self, labels: &[TextDrawCommand]) {
        let (glyphs, cache) = crate::plane_text::build_plane_label_custom_glyphs(
            &mut self.font_system,
            &mut self.swash_cache,
            labels,
        );
        self.plane_custom_glyphs = glyphs;
        self.plane_glyph_cache = cache;
    }

    /// Upload current FPS + positioned text to the glyphon atlas (call before HUD render pass).
    pub fn prepare_gpu_atlas_for_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.prepare_positioned_texts();
        let bounds = TextBounds {
            left: 0,
            top: 0,
            right: self.width as i32,
            bottom: self.height as i32,
        };
        let mut areas: Vec<TextArea> =
            Vec::with_capacity(2 + self.positioned_texts.len() + usize::from(!self.plane_custom_glyphs.is_empty()));
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
        if !self.plane_custom_glyphs.is_empty() {
            areas.push(TextArea {
                buffer: &self.empty_buffer,
                left: 0.0,
                top: 0.0,
                scale: 1.0,
                bounds,
                default_color: Color::rgb(255, 255, 255),
                custom_glyphs: &self.plane_custom_glyphs,
            });
        }
        let cache = self.plane_glyph_cache.clone();
        if let Err(e) = self.text_renderer.prepare_with_custom(
            device,
            queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            areas,
            &mut self.swash_cache,
            move |req: RasterizeCustomGlyphRequest| {
                cache
                    .get(&req.id)
                    .map(|entry| crate::plane_text::glyph_for_request(entry, &req))
            },
        ) {
            log::error!("HUD prepare (render pass): {:?}", e);
        }
    }

    /// Update the FPS / stats block only (glyph upload happens in `prepare_gpu_atlas_for_render`).
    pub fn set_fps_buffer_text(
        &mut self,
        fps: f32,
        frame_time_ms: f32,
        stats: &FrameStats,
        mode_name: &str,
    ) {
        let text = compose_hud_text(&self.overlay_lines, fps, frame_time_ms, stats, mode_name);
        self.buffer.set_text(
            &mut self.font_system,
            &text,
            Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
        );
        self.buffer.shape_until_scroll(&mut self.font_system, false);
    }

    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Err(e) = self.text_renderer.render(&self.atlas, &self.viewport, pass) {
            log::error!("HUD render: {:?}", e);
        }
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
