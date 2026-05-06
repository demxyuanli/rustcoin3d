use crate::cli;
use crate::engine::state::{EngineEvent, EngineState};

pub struct CliPanel {
    input: String,
    history: Vec<String>,
    output_lines: Vec<String>,
}

impl CliPanel {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            history: Vec::new(),
            output_lines: Vec::with_capacity(200),
        }
    }

    pub fn ui(&mut self, ctx: &egui::Context, state: &mut EngineState) {
        egui::SidePanel::left("cli_panel")
            .min_width(280.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("CLI");
                ui.separator();

                // Output scroll area
                let output_height = ui.available_height() - 40.0;
                egui::ScrollArea::vertical()
                    .max_height(output_height)
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.output_lines {
                            ui.label(line.as_str());
                        }
                    });

                // Input line at bottom
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(">");
                    let resp = ui.add_sized(
                        [ui.available_width() - 50.0, 20.0],
                        egui::TextEdit::singleline(&mut self.input),
                    );

                    // Up arrow: recall last command
                    if resp.has_focus() && ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                        if let Some(last) = self.history.last() {
                            self.input = last.clone();
                        }
                    }

                    let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.button("Send").clicked() || (resp.has_focus() && enter) {
                        self.submit_command(state);
                    }
                });
            });
    }

    fn submit_command(&mut self, state: &mut EngineState) {
        let cmd_str = self.input.trim().to_string();
        if cmd_str.is_empty() {
            return;
        }
        self.output_lines.push(format!("> {cmd_str}"));
        self.history.push(cmd_str.clone());
        self.input.clear();

        match cli::parse(&cmd_str) {
            Ok(cmd) => {
                if matches!(cmd, cli::CliCommand::Quit) {
                    self.output_lines.push("Quitting...".into());
                    state.push_event(EngineEvent::Quit);
                    return;
                }
                crate::engine::execute(cmd, state);
            }
            Err(e) => {
                self.output_lines.push(format!("Error: {e}"));
            }
        }

        for event in state.take_events() {
            let msg = match event {
                EngineEvent::SceneLoaded => "Scene loaded.".into(),
                EngineEvent::SceneReset => "Scene reset.".into(),
                EngineEvent::TestStarted(name) => format!("Test started: {name}"),
                EngineEvent::TestPassed(name) => format!("PASS: {name}"),
                EngineEvent::TestFailed(name) => format!("FAIL: {name}"),
                EngineEvent::TestStopped => "Test stopped.".into(),
                EngineEvent::SelectionChanged => {
                    format!(
                        "Selection: {:?}",
                        state.selection.iter().collect::<Vec<_>>()
                    )
                }
                EngineEvent::PropertyChanged { node, field } => {
                    format!("Property changed: {node:?}.{field}")
                }
                EngineEvent::LogAppended => state
                    .log_entries
                    .back()
                    .map(|e| format!("[{}] {}", e.level, e.message))
                    .unwrap_or_default(),
                EngineEvent::LogCleared => "Log cleared.".into(),
                EngineEvent::DisplayModeChanged(mode) => format!("Display mode: {mode}"),
                EngineEvent::Quit => "Quitting...".into(),
            };
            self.output_lines.push(msg);
        }
        if self.output_lines.len() > 200 {
            let trim = self.output_lines.len() - 200;
            self.output_lines.drain(0..trim);
        }
    }
}
