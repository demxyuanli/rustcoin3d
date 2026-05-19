use rc3d_engine_api::Engine;
use rc3d_editor::Editor;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::WindowAttributes,
};

pub struct App {
    engine: Option<Engine>,
    editor: Option<Editor>,
    window: Option<winit::window::Window>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            engine: None,
            editor: None,
            window: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("rustcoin3d")
                        .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
                )
                .expect("failed to create window");
            let engine = Engine::new(&window);
            let editor = Editor::new(&window, &engine);
            self.window = Some(window);
            self.engine = Some(engine);
            self.editor = Some(editor);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::RedrawRequested => {
                if let Some(ref mut engine) = self.engine {
                    engine.render();
                }
                if let Some(ref window) = self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(ref mut engine) = self.engine {
                    engine.resize(size.width, size.height);
                }
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {
                if let (Some(ref mut editor), Some(ref window)) =
                    (&mut self.editor, &self.window)
                {
                    editor.handle_event(window, &event);
                }
            }
        }
    }
}
