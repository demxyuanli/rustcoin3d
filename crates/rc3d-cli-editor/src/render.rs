use rc3d_render::Renderer;

use crate::egui_paint::EguiPainter;

pub struct RenderContext {
    pub renderer: Renderer,
    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,
    pub egui_painter: EguiPainter,
    pub pixels_per_point: f32,
}

impl RenderContext {
    pub fn new(window: &winit::window::Window, renderer: Renderer) -> Self {
        let egui_ctx = egui::Context::default();
        let max_side = renderer.device.limits().max_texture_dimension_2d as usize;
        let winit_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(max_side),
        );
        let egui_painter = EguiPainter::new(&renderer.device, renderer.config.format);

        Self {
            renderer,
            egui_ctx,
            egui_winit: winit_state,
            egui_painter,
            pixels_per_point: window.scale_factor() as f32,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f32) {
        self.renderer.resize(width, height);
        self.pixels_per_point = scale_factor;
    }

    pub fn on_window_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        self.egui_winit.on_window_event(window, event).consumed
    }
}
