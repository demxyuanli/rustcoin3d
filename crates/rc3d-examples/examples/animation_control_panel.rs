//! Standalone animation control panel — launches an eframe window with
//! shared render feature controls (visibility, channels, global rate).
//!
//! This panel runs independently of any 3D rendering. Connect it to a
//! running example by sharing the `Arc<Mutex<RenderFeaturePanelState>>`.
//!
//! Usage: cargo run -p rc3d-examples --example animation_control_panel

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use rc3d_editor::{
    spawn_render_feature_panel, PanelConfig, RenderFeaturePanelState,
};

fn main() {
    let state = Arc::new(Mutex::new(RenderFeaturePanelState::default()));
    spawn_render_feature_panel(
        state.clone(),
        PanelConfig {
            title: "Animation Control Panel (Shared Component)".to_string(),
            ..Default::default()
        },
    );
    loop {
        thread::sleep(Duration::from_millis(1000));
    }
}
