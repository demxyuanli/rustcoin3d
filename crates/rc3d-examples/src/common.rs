//! Shared event-loop wrapper for all examples.
//!
//! Every example calls [`run_example`] with a title and a setup closure.
//! The closure receives `&mut Engine` so it can build the scene graph,
//! configure the renderer, and register engines before the event loop starts.

use rc3d_engine_api::Engine;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowAttributes;

/// Run a rustcoin3d example with a winit event loop.
///
/// # Example
/// ```ignore
/// run_example("hello", |engine| {
///     engine.scene_mut().add_root(/* ... */);
/// });
/// ```
pub fn run_example(title: &str, setup: impl FnOnce(&mut Engine)) {
    env_logger::init();

    let event_loop = EventLoop::new().expect("failed to create event loop");

    let window = event_loop
        .create_window(WindowAttributes::default().with_title(title))
        .expect("failed to create window");

    let mut engine = Engine::new(&window);
    setup(&mut engine);

    let mut cursor_prev: (f64, f64) = (0.0, 0.0);

    let _ = event_loop.run(move |event, elwt| {
        match &event {
            Event::WindowEvent { event: win_event, .. } => {
                match win_event {
                    WindowEvent::RedrawRequested => {
                        engine.render();
                        window.request_redraw();
                    }
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => {
                        engine.resize(size.width, size.height);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let left_orbit = engine.on_pick.is_none();
                        engine.controller.dispatch_window_event(
                            win_event,
                            cursor_prev,
                            left_orbit,
                        );
                        cursor_prev = (position.x, position.y);
                    }
                    WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                        let left_orbit = engine.on_pick.is_none();
                        engine.controller.dispatch_window_event(
                            win_event,
                            cursor_prev,
                            left_orbit,
                        );
                        if let WindowEvent::MouseWheel { .. } = win_event {
                            window.request_redraw();
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}

/// Run a rustcoin3d example with hook support.
///
/// Like [`run_example`], but conditionally requests redraw based on
/// [`Engine::continuous_redraw`] and dispatches keyboard events to
/// [`Engine::panel_overlay_key_hook`].
///
/// Use this for examples that need on-screen HUD overlays, animation-driven
/// redraw, or interactive panel overlays via keyboard.
pub fn run_example_with_hooks(title: &str, setup: impl FnOnce(&mut Engine)) {
    env_logger::init();

    let event_loop = EventLoop::new().expect("failed to create event loop");

    let window = event_loop
        .create_window(WindowAttributes::default().with_title(title))
        .expect("failed to create window");

    let mut engine = Engine::new(&window);
    setup(&mut engine);

    // Track cursor position for camera orbit/pan delta computation
    let mut cursor_prev: (f64, f64) = (0.0, 0.0);

    let _ = event_loop.run(move |event, elwt| {
        match &event {
            Event::WindowEvent { event: win_event, .. } => {
                match win_event {
                    WindowEvent::RedrawRequested => {
                        engine.render();
                        window.request_redraw();
                    }
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => {
                        engine.resize(size.width, size.height);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let left_orbit = engine.on_pick.is_none();
                        engine.controller.dispatch_window_event(
                            win_event,
                            cursor_prev,
                            left_orbit,
                        );
                        cursor_prev = (position.x, position.y);
                    }
                    WindowEvent::MouseInput { state, .. } => {
                        // Forward mouse clicks to panel overlay hook
                        if *state == ElementState::Pressed {
                            if let Some(ref hook) = engine.panel_overlay_mouse_hook {
                                let _ = hook(
                                    cursor_prev.0 as f32,
                                    cursor_prev.1 as f32,
                                    window.inner_size().width,
                                    window.inner_size().height,
                                );
                            }
                        }
                        // Camera orbit/pan attach/detach
                        let left_orbit = engine.on_pick.is_none();
                        engine.controller.dispatch_window_event(
                            win_event,
                            cursor_prev,
                            left_orbit,
                        );
                    }
                    WindowEvent::MouseWheel { .. } => {
                        let left_orbit = engine.on_pick.is_none();
                        engine.controller.dispatch_window_event(
                            win_event,
                            cursor_prev,
                            left_orbit,
                        );
                        window.request_redraw();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            if let Some(ref mut hook) = engine.panel_overlay_key_hook {
                                if hook(event.physical_key) {
                                    window.request_redraw();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                if engine.continuous_redraw {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    });
}
