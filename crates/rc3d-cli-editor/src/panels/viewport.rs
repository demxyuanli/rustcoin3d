use rc3d_render::render_action::RenderCollector;
use rc3d_render::{DrawCall, FrameStats, Renderer};

use crate::engine::state::EngineState;

pub struct ViewportPanel {
    pub draw_calls: Vec<DrawCall>,
    pub frame_stats: Option<FrameStats>,
}

impl ViewportPanel {
    pub fn new() -> Self {
        Self {
            draw_calls: Vec::new(),
            frame_stats: None,
        }
    }

    pub fn collect(&mut self, state: &EngineState) {
        self.draw_calls.clear();
        let roots = state.scene.roots();
        if roots.is_empty() {
            return;
        }
        let mut collector = RenderCollector::new();
        for &root in roots {
            collector.traverse(&state.scene, root);
        }
        self.draw_calls = collector.draw_calls;
    }

    pub fn render(&mut self, renderer: &mut Renderer) {
        if self.draw_calls.is_empty() {
            return;
        }
        self.frame_stats = Some(renderer.render_draw_calls(
            &self.draw_calls,
            &rc3d_scene::SceneGraph::new(),
        ));
    }

    pub fn ui(&self, ctx: &egui::Context, _state: &EngineState) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("3D Viewport");
            ui.separator();
            if _state.scene.roots().is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("No scene loaded. Use 'scene load <path>' or 'test run' to begin.");
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("3D Viewport (rendering to surface)");
                });
            }
            if let Some(ref stats) = self.frame_stats {
                ui.label(format!(
                    "Visible DC: {} | Culled: {} | Tris: {}",
                    stats.visible_draw_calls, stats.culled_draw_calls, stats.visible_triangles
                ));
            }
        });
    }
}
