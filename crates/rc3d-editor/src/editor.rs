/// Thin Editor struct that wraps egui state and provides a clean public API.
///
/// The Editor owns the egui context, window state, and renderer. It receives
/// an `EditorContext` (which wraps `&mut Engine`) at render time so the UI
/// can read/write engine state without holding a long-lived reference.
pub struct Editor {
    pub ui: crate::ui::EditorUi,
}

impl Editor {
    pub fn new(window: &winit::window::Window, engine: &rc3d_engine_api::Engine) -> Self {
        let ui = crate::ui::EditorUi::new(window, engine);
        Self { ui }
    }

    /// Forward a window event to the editor UI. Returns `true` if the event
    /// was consumed (e.g., the user is interacting with an egui widget).
    pub fn handle_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        self.ui.handle_event(window, event)
    }

    /// Build egui UI for this frame. Call this each frame before `paint()`.
    pub fn render(
        &mut self,
        window: &winit::window::Window,
        engine: &rc3d_engine_api::Engine,
        ui_ctx: &crate::ui::types::EditorUiContext,
    ) {
        self.ui
            .render(window, engine.scene(), engine, ui_ctx);
    }

    /// Collect editor commands queued during the last `render()` call.
    pub fn take_commands(&mut self) -> std::collections::vec_deque::IntoIter<crate::commands::EditorCommand> {
        self.ui.take_commands()
    }

    /// Paint the egui overlay into the provided render pass.
    pub fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        self.ui.paint(device, queue, encoder, view);
    }

    /// Handle window resize.
    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f32) {
        self.ui.resize(width, height, scale_factor);
    }

    /// Push a log message to the console panel.
    pub fn push_log(&mut self, msg: &str) {
        self.ui.push_log(msg);
    }

    /// Toggle the console panel visibility.
    pub fn toggle_console(&mut self) {
        self.ui.toggle_console();
    }
}
