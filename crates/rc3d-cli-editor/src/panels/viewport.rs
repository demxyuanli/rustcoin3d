use rc3d_render::{DrawCall, FrameStats, Renderer};

use crate::engine::state::EngineState;

pub(crate) const VIEWPORT_EGUI_TEX_ID_U64: u64 = 0x_8e30_551c_0091;

pub struct ViewportPanel {
    pub(crate) viewport_texture_id: egui::TextureId,
    viewport_px: [u32; 2],
    pub draw_calls: Vec<DrawCall>,
    pub frame_stats: Option<FrameStats>,
}

impl ViewportPanel {
    pub fn new() -> Self {
        Self {
            viewport_texture_id: egui::TextureId::User(VIEWPORT_EGUI_TEX_ID_U64),
            viewport_px: [800, 480],
            draw_calls: Vec::new(),
            frame_stats: None,
        }
    }

    pub fn viewport_pixel_extent(&self) -> [u32; 2] {
        self.viewport_px
    }

    pub fn collect(&mut self, state: &EngineState) {
        self.draw_calls.clear();
        let roots = state.scene.roots();
        if roots.is_empty() {
            return;
        }
        use rc3d_render::render_action::RenderCollector;
        let mut collector = RenderCollector::new();
        for &root in roots {
            collector.traverse(&state.scene, root);
        }
        self.draw_calls = collector.draw_calls;
    }

    #[allow(dead_code)]
    pub fn render_offscreen_placeholder(&mut self, renderer: &mut Renderer) {
        if self.draw_calls.is_empty() {
            return;
        }
        self.frame_stats = Some(renderer.render_draw_calls(
            &self.draw_calls,
            &rc3d_scene::SceneGraph::new(),
        ));
    }

    pub fn ui(&mut self, ctx: &egui::Context, state: &EngineState, pixels_per_point: f32) {
        let inner = egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Viewport");
            ui.separator();

            let available = ui.available_size();
            if state.scene.roots().is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("No scene loaded. Use the terminal TUI: scene load <path>");
                });
            } else {
                let st = egui::load::SizedTexture::new(
                    self.viewport_texture_id,
                    egui::vec2(
                        available.x.max(1.0),
                        available.y.max(1.0),
                    ),
                );
                let r = ui.add(egui::Image::new(st));
                let rr = r.rect;
                let ppp = pixels_per_point;
                self.viewport_px = [
                    (rr.width() * ppp).floor().clamp(1.0, 16384.0) as u32,
                    (rr.height() * ppp).floor().clamp(1.0, 16384.0) as u32,
                ];
            }

            if let Some(ref stats) = self.frame_stats {
                ui.label(format!(
                    "Visible DC: {} | Culled: {} | Tris: {}",
                    stats.visible_draw_calls, stats.culled_draw_calls, stats.visible_triangles
                ));
            } else if !state.scene.roots().is_empty() {
                ui.label("Rt: allocating…");
            }
        });

        if state.scene.roots().is_empty() {
            let r = inner.response.rect;
            let ppp = pixels_per_point;
            self.viewport_px = [
                (r.width() * ppp).floor().clamp(1.0, 16384.0) as u32,
                (r.height() * ppp).floor().clamp(1.0, 16384.0) as u32,
            ];
        }
    }
}
