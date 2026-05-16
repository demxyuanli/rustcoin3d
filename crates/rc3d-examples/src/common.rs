//! Shared event-loop wrapper for all examples.
//!
//! Every example calls [`run_example`] with a title and a setup closure.
//! The closure receives `&mut Engine` so it can build the scene graph,
//! configure the renderer, and register engines before the event loop starts.

use rc3d_engine_api::Engine;
use winit::event::{Event, WindowEvent};
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

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::RedrawRequested => {
                engine.render();
                window.request_redraw();
            }
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(size) => {
                engine.resize(size.width, size.height);
            }
            _ => {}
        },
        Event::AboutToWait => {
            window.request_redraw();
        }
        _ => {}
    });
}
