//! Product editor host: egui chrome + central 3D region + document panel.

mod app;
mod document;
mod prefs;
mod scene;
mod ui_ctx;
mod win_shell;

use app::StudioApp;
use winit::event_loop::EventLoop;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = StudioApp::new(scene::build_demo_scene());
    event_loop.run_app(&mut app).expect("run");
}
