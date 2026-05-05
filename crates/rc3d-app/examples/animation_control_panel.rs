use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use rc3d_app::{spawn_render_feature_panel, PanelConfig, RenderFeaturePanelState};

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
