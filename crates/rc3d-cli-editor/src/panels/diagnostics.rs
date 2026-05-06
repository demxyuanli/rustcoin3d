use crate::engine::state::EngineState;

#[derive(PartialEq)]
enum DiagTab {
    Log,
    Tests,
    Perf,
}

pub struct DiagnosticsPanel {
    tab: DiagTab,
}

impl DiagnosticsPanel {
    pub fn new() -> Self {
        Self { tab: DiagTab::Log }
    }

    pub fn ui(&mut self, ctx: &egui::Context, state: &EngineState, fps: f64, frame_time_ms: f64) {
        egui::TopBottomPanel::bottom("diagnostics")
            .min_height(120.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tab, DiagTab::Log, "Log");
                    ui.selectable_value(&mut self.tab, DiagTab::Tests, "Tests");
                    ui.selectable_value(&mut self.tab, DiagTab::Perf, "Perf");
                });
                ui.separator();
                match self.tab {
                    DiagTab::Log => self.log_tab(ui, state),
                    DiagTab::Tests => self.test_tab(ui, state),
                    DiagTab::Perf => self.perf_tab(ui, fps, frame_time_ms),
                }
            });
    }

    fn log_tab(&self, ui: &mut egui::Ui, state: &EngineState) {
        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for entry in &state.log_entries {
                    let color = match entry.level.as_str() {
                        "error" => egui::Color32::RED,
                        "warn" => egui::Color32::YELLOW,
                        _ => egui::Color32::LIGHT_GRAY,
                    };
                    ui.colored_label(color, format!("[{}] {}", entry.level, entry.message));
                }
            });
    }

    fn test_tab(&self, ui: &mut egui::Ui, state: &EngineState) {
        if let Some(ref test) = state.active_test {
            ui.label(format!("Test suite: {}", test.name));
            ui.label(format!(
                "Total: {}  Passed: {}  Failed: {}",
                test.total, test.passed, test.failed
            ));
            let frac = if test.total > 0 {
                test.passed as f32 / test.total as f32
            } else {
                0.0
            };
            ui.add(egui::ProgressBar::new(frac).text(format!("{}/{}", test.passed, test.total)));
            if test.running {
                ui.spinner();
            }
        } else {
            ui.label("No active test run.");
        }
    }

    fn perf_tab(&self, ui: &mut egui::Ui, fps: f64, frame_time_ms: f64) {
        ui.label(format!("FPS: {fps:.1}"));
        ui.label(format!("Frame time: {frame_time_ms:.2} ms"));
    }
}
