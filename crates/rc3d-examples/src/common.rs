//! Shared event-loop wrapper for all examples.
//!
//! Every example calls [`run_example`] with a title and a setup closure.
//! The closure receives `&mut Engine` so it can build the scene graph,
//! configure the renderer, and register engines before the event loop starts.

use rc3d_engine_api::Engine;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowAttributes;

struct ExampleApp<F> {
    title: String,
    setup: Option<F>,
    with_hooks: bool,
    engine: Option<Engine>,
    window: Option<winit::window::Window>,
    cursor_prev: (f64, f64),
    cursor_pos: Option<(f32, f32)>,
    window_size: (u32, u32),
}

impl<F: FnOnce(&mut Engine)> ExampleApp<F> {
    fn new(title: impl Into<String>, setup: F, with_hooks: bool) -> Self {
        Self {
            title: title.into(),
            setup: Some(setup),
            with_hooks,
            engine: None,
            window: None,
            cursor_prev: (0.0, 0.0),
            cursor_pos: None,
            window_size: (800, 600),
        }
    }

    fn dispatch_camera(&mut self, event: &WindowEvent) -> bool {
        let Some(engine) = self.engine.as_mut() else {
            return false;
        };
        let left_orbit = engine.on_pick.is_none();
        engine
            .controller
            .dispatch_window_event(event, self.cursor_prev, left_orbit)
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
                self.window_size = (size.width, size.height);
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size.width, size.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dispatch_camera(&event) {
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }
                self.cursor_prev = (position.x, position.y);
                self.cursor_pos = Some((position.x as f32, position.y as f32));
            }
            WindowEvent::MouseInput { .. } => {
                if self.with_hooks {
                    if let WindowEvent::MouseInput { state, .. } = &event {
                        if *state == ElementState::Pressed {
                            if let (Some(engine), Some((x, y))) =
                                (self.engine.as_mut(), self.cursor_pos)
                            {
                                if let Some(ref hook) = engine.panel_overlay_mouse_hook {
                                    let _ = hook(
                                        x,
                                        y,
                                        self.window_size.0,
                                        self.window_size.1,
                                    );
                                }
                            }
                        }
                    }
                }
                if self.dispatch_camera(&event) {
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::MouseWheel { .. } => {
                if self.dispatch_camera(&event) {
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if self.with_hooks && event.state == ElementState::Pressed {
                    let mut requested = false;
                    if let Some(engine) = self.engine.as_mut() {
                        if let Some(ref mut hook) = engine.panel_overlay_key_hook {
                            requested = hook(event.physical_key);
                        }
                    }
                    if requested {
                        if let Some(window) = self.window.as_ref() {
                            window.request_redraw();
                        }
                    }
                }
            }
            _ => {}
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

/// Run a rustcoin3d example with hook support.
pub fn run_example_with_hooks(title: &str, setup: impl FnOnce(&mut Engine)) {
    let _ = env_logger::try_init();
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = ExampleApp::new(title, setup, true);
    event_loop.run_app(&mut app).expect("event loop failed");
}
