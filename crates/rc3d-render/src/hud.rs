use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

use crate::renderer::FrameStats;

pub struct HudRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,
    buffer: Buffer,
    width: u32,
    height: u32,
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
        let mut hud = Self {
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            buffer,
            width,
            height,
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

    pub fn update_text(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fps: f32,
        frame_time_ms: f32,
        stats: FrameStats,
        mode_name: &str,
    ) {
        let text = format!(
            "FPS: {fps:.1} | Frame: {frame_time_ms:.2}ms\nTriangles: {} | Draw Calls: {} | Culled: {}\nMode: {mode_name}",
            stats.visible_triangles, stats.visible_draw_calls, stats.culled_draw_calls
        );
        self.buffer.set_text(
            &mut self.font_system,
            &text,
            Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
        );
        self.buffer.shape_until_scroll(&mut self.font_system, false);

        let areas = [TextArea {
            buffer: &self.buffer,
            left: 12.0,
            top: 12.0,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: self.width as i32,
                bottom: self.height as i32,
            },
            default_color: Color::rgb(240, 240, 240),
            custom_glyphs: &[],
        }];

        let _ = self.text_renderer.prepare(
            device,
            queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            areas,
            &mut self.swash_cache,
        );
    }

    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>) {
        let _ = self.text_renderer.render(&self.atlas, &self.viewport, pass);
    }
}
