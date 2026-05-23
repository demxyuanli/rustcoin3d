use std::sync::{Arc, Mutex};

use eframe::egui::{self, Slider};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureChannelId {
    ChannelA,
    ChannelB,
    ChannelC,
}

#[derive(Clone, Copy, Debug)]
pub struct PanelSections {
    pub visibility: bool,
    pub channel_enable: bool,
    pub update_control: bool,
    pub channel_transition: bool,
    pub channel_mix: bool,
    pub global_rate: bool,
}

impl Default for PanelSections {
    fn default() -> Self {
        Self {
            visibility: true,
            channel_enable: true,
            update_control: true,
            channel_transition: true,
            channel_mix: true,
            global_rate: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PanelConfig {
    pub title: String,
    pub sections: PanelSections,
}

impl Default for PanelConfig {
    fn default() -> Self {
        Self {
            title: "Render Feature Controls".to_string(),
            sections: PanelSections::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CrossfadeRequest {
    pub from: FeatureChannelId,
    pub to: FeatureChannelId,
    pub progress: f32,
}

#[derive(Clone, Debug)]
pub struct RenderFeaturePanelState {
    pub show_model: bool,
    pub show_debug_overlay: bool,
    pub channel_a_active: bool,
    pub channel_b_active: bool,
    pub channel_c_active: bool,
    pub paused: bool,
    pub step_size: f32,
    pub step_once: bool,
    pub use_default_duration: bool,
    pub custom_duration: f32,
    pub channel_a_weight: f32,
    pub channel_b_weight: f32,
    pub channel_c_weight: f32,
    pub time_scale: f32,
    pub crossfade: Option<CrossfadeRequest>,
}

impl Default for RenderFeaturePanelState {
    fn default() -> Self {
        Self {
            show_model: true,
            show_debug_overlay: false,
            channel_a_active: false,
            channel_b_active: true,
            channel_c_active: false,
            paused: false,
            step_size: 0.05,
            step_once: false,
            use_default_duration: true,
            custom_duration: 3.5,
            channel_a_weight: 0.0,
            channel_b_weight: 1.0,
            channel_c_weight: 0.0,
            time_scale: 1.0,
            crossfade: None,
        }
    }
}

impl RenderFeaturePanelState {
    pub fn request_crossfade(&mut self, from: FeatureChannelId, to: FeatureChannelId) {
        self.crossfade = Some(CrossfadeRequest {
            from,
            to,
            progress: 0.0,
        });
    }

    pub fn activate_all(&mut self) {
        self.channel_a_active = true;
        self.channel_b_active = true;
        self.channel_c_active = true;
    }

    pub fn deactivate_all(&mut self) {
        self.channel_a_active = false;
        self.channel_b_active = false;
        self.channel_c_active = false;
    }

    pub fn advance_time(&mut self, raw_dt: f32) -> f32 {
        let mut step_dt = if self.paused {
            0.0
        } else {
            raw_dt.max(0.0) * self.time_scale.max(0.0)
        };
        if self.step_once {
            step_dt += self.step_size.max(0.0);
            self.step_once = false;
        }
        self.update_crossfade(step_dt);
        self.normalize_weights_for_activation();
        step_dt
    }

    fn update_crossfade(&mut self, step_dt: f32) {
        let Some(cf) = self.crossfade else { return };
        let duration = if self.use_default_duration {
            0.35
        } else {
            self.custom_duration.max(0.01)
        };
        let new_progress = (cf.progress + step_dt / duration).clamp(0.0, 1.0);
        let t = new_progress;

        match (cf.from, cf.to) {
            (FeatureChannelId::ChannelB, FeatureChannelId::ChannelA) => {
                self.channel_b_weight = 1.0 - t;
                self.channel_a_weight = t;
                self.channel_c_weight = 0.0;
            }
            (FeatureChannelId::ChannelA, FeatureChannelId::ChannelB) => {
                self.channel_a_weight = 1.0 - t;
                self.channel_b_weight = t;
                self.channel_c_weight = 0.0;
            }
            (FeatureChannelId::ChannelB, FeatureChannelId::ChannelC) => {
                self.channel_b_weight = 1.0 - t;
                self.channel_c_weight = t;
                self.channel_a_weight = 0.0;
            }
            (FeatureChannelId::ChannelC, FeatureChannelId::ChannelB) => {
                self.channel_c_weight = 1.0 - t;
                self.channel_b_weight = t;
                self.channel_a_weight = 0.0;
            }
            _ => {}
        }

        if new_progress >= 1.0 {
            self.crossfade = None;
        } else {
            self.crossfade = Some(CrossfadeRequest {
                from: cf.from,
                to: cf.to,
                progress: new_progress,
            });
        }
    }

    pub fn normalize_weights_for_activation(&mut self) {
        if !self.channel_a_active {
            self.channel_a_weight = 0.0;
        }
        if !self.channel_b_active {
            self.channel_b_weight = 0.0;
        }
        if !self.channel_c_active {
            self.channel_c_weight = 0.0;
        }
        let sum = self.channel_a_weight + self.channel_b_weight + self.channel_c_weight;
        if sum > 1e-6 {
            self.channel_a_weight /= sum;
            self.channel_b_weight /= sum;
            self.channel_c_weight /= sum;
        }
    }
}

pub type RenderFeaturePanelHandle = Arc<Mutex<RenderFeaturePanelState>>;

pub fn spawn_render_feature_panel(state: RenderFeaturePanelHandle, config: PanelConfig) {
    std::thread::spawn(move || {
        let title = config.title.clone();
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([420.0, 760.0])
                .with_position([40.0, 40.0]),
            persist_window: false,
            ..Default::default()
        };
        eprintln!("[rc3d-panel] launching panel window: {title}");
        if let Err(err) = eframe::run_native(
            &title,
            options,
            Box::new(|_cc| Ok(Box::new(ControlPanelApp { state, config }))),
        ) {
            eprintln!("[rc3d-panel] panel window failed: {err}");
        }
    });
}

#[derive(Clone, Debug)]
pub struct PanelPreset {
    pub config: PanelConfig,
    pub state: RenderFeaturePanelState,
}

pub fn preset_for_import_viewer_panel() -> PanelPreset {
    let state = RenderFeaturePanelState {
        channel_a_active: false,
        channel_b_active: false,
        channel_c_active: false,
        channel_a_weight: 0.0,
        channel_b_weight: 0.0,
        channel_c_weight: 0.0,
        ..Default::default()
    };
    PanelPreset {
        config: PanelConfig {
            title: "Import Viewer Render Feature Panel".to_string(),
            sections: PanelSections::default(),
        },
        state,
    }
}

pub fn preset_for_render_features_panel() -> PanelPreset {
    let state = RenderFeaturePanelState {
        channel_a_active: true,
        channel_b_active: true,
        channel_c_active: true,
        channel_a_weight: 0.34,
        channel_b_weight: 0.33,
        channel_c_weight: 0.33,
        ..RenderFeaturePanelState::default()
    };
    PanelPreset {
        config: PanelConfig {
            title: "Render Features Panel (Shared)".to_string(),
            sections: PanelSections {
                channel_transition: false,
                ..PanelSections::default()
            },
        },
        state,
    }
}

struct ControlPanelApp {
    state: RenderFeaturePanelHandle,
    config: PanelConfig,
}

impl eframe::App for ControlPanelApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut s = self.state.lock().expect("panel state lock");
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Controls");

            if self.config.sections.visibility {
                ui.collapsing("Visibility", |ui| {
                    ui.checkbox(&mut s.show_model, "show model");
                    ui.checkbox(&mut s.show_debug_overlay, "show debug overlay");
                });
            }

            if self.config.sections.channel_enable {
                ui.collapsing("Channel Enable", |ui| {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut s.channel_a_active, "channel A");
                        ui.checkbox(&mut s.channel_b_active, "channel B");
                        ui.checkbox(&mut s.channel_c_active, "channel C");
                    });
                    if ui.button("disable all channels").clicked() {
                        s.deactivate_all();
                    }
                    if ui.button("enable all channels").clicked() {
                        s.activate_all();
                    }
                });
            }

            if self.config.sections.update_control {
                ui.collapsing("Update Control", |ui| {
                    if ui.button("pause/resume updates").clicked() {
                        s.paused = !s.paused;
                    }
                    if ui.button("apply single update step").clicked() {
                        s.step_once = true;
                    }
                    ui.add(Slider::new(&mut s.step_size, 0.001..=0.2).text("update step size"));
                });
            }

            if self.config.sections.channel_transition {
                ui.collapsing("Channel Transition", |ui| {
                    if ui.button("from B to A").clicked() {
                        s.request_crossfade(FeatureChannelId::ChannelB, FeatureChannelId::ChannelA);
                    }
                    if ui.button("from A to B").clicked() {
                        s.request_crossfade(FeatureChannelId::ChannelA, FeatureChannelId::ChannelB);
                    }
                    if ui.button("from B to C").clicked() {
                        s.request_crossfade(FeatureChannelId::ChannelB, FeatureChannelId::ChannelC);
                    }
                    if ui.button("from C to B").clicked() {
                        s.request_crossfade(FeatureChannelId::ChannelC, FeatureChannelId::ChannelB);
                    }
                    ui.checkbox(
                        &mut s.use_default_duration,
                        "use default transition duration",
                    );
                    ui.add_enabled(
                        !s.use_default_duration,
                        Slider::new(&mut s.custom_duration, 0.05..=10.0)
                            .text("custom transition duration"),
                    );
                });
            }

            if self.config.sections.channel_mix {
                ui.collapsing("Channel Mix", |ui| {
                    ui.add(
                        Slider::new(&mut s.channel_a_weight, 0.0..=1.0)
                            .text("channel A mix weight"),
                    );
                    ui.add(
                        Slider::new(&mut s.channel_b_weight, 0.0..=1.0)
                            .text("channel B mix weight"),
                    );
                    ui.add(
                        Slider::new(&mut s.channel_c_weight, 0.0..=1.0)
                            .text("channel C mix weight"),
                    );
                });
            }

            if self.config.sections.global_rate {
                ui.collapsing("Global Rate", |ui| {
                    ui.add(Slider::new(&mut s.time_scale, 0.0..=3.0).text("global update rate"));
                });
            }
        });
        drop(s);
        ctx.request_repaint();
    }
}
