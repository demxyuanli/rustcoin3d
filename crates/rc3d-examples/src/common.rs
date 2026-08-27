//! Shared event-loop wrapper for all examples.
//!
//! Every example calls [`run_example`] with a title and a setup closure.
//! Window input goes through [`Engine::handle_window_event`] (`HandleEventAction`
//! + camera + click pick).

use rc3d_engine_api::{Engine, EventRouteOpts};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowAttributes;

struct ExampleApp<F> {
    title: String,
    setup: Option<F>,
    with_hooks: bool,
    engine: Option<Engine>,
    window: Option<winit::window::Window>,
}

impl<F: FnOnce(&mut Engine)> ExampleApp<F> {
    fn new(title: impl Into<String>, setup: F, with_hooks: bool) -> Self {
        Self {
            title: title.into(),
            setup: Some(setup),
            with_hooks,
            engine: None,
            window: None,
        }
    }
}

impl<F: FnOnce(&mut Engine)> ApplicationHandler for ExampleApp<F> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = event_loop
            .create_window(WindowAttributes::default().with_title(&self.title))
            .expect("failed to create window");

        let mut engine = Engine::new(&window);
        if let Some(setup) = self.setup.take() {
            setup(&mut engine);
        }
        maybe_screenshot(&mut engine, &self.title);

        self.engine = Some(engine);
        self.window = Some(window);
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::RedrawRequested => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.render();
                }
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size.width, size.height);
                }
            }
            _ => {
                if let Some(engine) = self.engine.as_mut() {
                    let opts = EventRouteOpts {
                        pick_on_click: engine.on_pick.is_some()
                            || engine.on_pick_hit.is_some(),
                        ..EventRouteOpts::default()
                    };
                    let routed = engine.handle_window_event(&event, opts);
                    if routed.redraw {
                        if let Some(window) = self.window.as_ref() {
                            window.request_redraw();
                        }
                    }
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let should_redraw = if self.with_hooks {
            self.engine
                .as_ref()
                .is_some_and(|engine| engine.continuous_redraw)
        } else {
            true
        };
        if should_redraw {
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        }
    }
}

/// Run a custom winit application (ApplicationHandler + run_app).
pub fn run_app<A: ApplicationHandler>(mut app: A) {
    let _ = env_logger::try_init();
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.run_app(&mut app).expect("event loop failed");
}

/// Run a rustcoin3d example with a winit event loop.
pub fn run_example(title: &str, setup: impl FnOnce(&mut Engine)) {
    let _ = env_logger::try_init();
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = ExampleApp::new(title, setup, false);
    event_loop.run_app(&mut app).expect("event loop failed");
}

/// Run a rustcoin3d example with overlay-hook / idle-redraw support.
pub fn run_example_with_hooks(title: &str, setup: impl FnOnce(&mut Engine)) {
    let _ = env_logger::try_init();
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = ExampleApp::new(title, setup, true);
    event_loop.run_app(&mut app).expect("event loop failed");
}

fn arg_value(flag: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(a) = args.next() {
        if a == flag {
            return args.next();
        }
    }
    None
}

fn screenshot_default_path(title: &str) -> String {
    let slug: String = title
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("target/{slug}.png")
}

/// `--screenshot` writes an offscreen PNG; `--screenshot-path` overrides the
/// file; `--screenshot-exit` quits after the write.
fn maybe_screenshot(engine: &mut Engine, title: &str) {
    if !std::env::args().any(|a| a == "--screenshot") {
        return;
    }
    engine.render();
    let dcs = engine.world.cached_draw_calls.clone();
    let Some(renderer) = engine.renderer.as_mut() else {
        return;
    };
    let (w, h, pixels) = renderer.render_to_image(&dcs, &engine.world.graph, 800, 600);
    let path = arg_value("--screenshot-path").unwrap_or_else(|| screenshot_default_path(title));
    if let Err(err) = image::save_buffer(&path, &pixels, w, h, image::ExtendedColorType::Rgba8) {
        eprintln!("screenshot failed: {err}");
        return;
    }
    let n_sel = dcs.iter().filter(|dc| dc.selected).count();
    let n_blend = dcs
        .iter()
        .filter(|dc| dc.alpha_mode == rc3d_scene::AlphaMode::Blend)
        .count();
    eprintln!("wrote {path}  selected={n_sel} blend={n_blend}");
    if std::env::args().any(|a| a == "--screenshot-exit") {
        std::process::exit(0);
    }
}
