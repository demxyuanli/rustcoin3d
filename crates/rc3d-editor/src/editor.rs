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

    /// Host-driven splash lifecycle: set the loading stage / completion.
    pub fn set_splash(&mut self, splash: crate::ui::splash::SplashState) {
        self.ui.set_splash(splash);
    }

    pub fn splash(&self) -> crate::ui::splash::SplashState {
        self.ui.splash()
    }

    /// Build egui UI for this frame. Call this each frame before `paint()`.
    /// Returns `true` when egui needs another immediate redraw (menus/popups).
    pub fn render(
        &mut self,
        window: &winit::window::Window,
        engine: &rc3d_engine_api::Engine,
        ui_ctx: &crate::ui::types::EditorUiContext,
    ) -> bool {
        self.ui.render(window, engine.scene(), engine, ui_ctx)
    }

    /// Collect editor commands queued during the last `render()` call.
    pub fn take_commands(
        &mut self,
    ) -> std::collections::vec_deque::IntoIter<crate::commands::EditorCommand> {
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

    pub fn scene_pixel_rect(&self) -> Option<crate::ui::types::PixelRect> {
        self.ui.scene_pixel_rect()
    }

    pub fn document_open(&self) -> bool {
        self.ui.document_open()
    }

    pub fn nav_cube_blocks_scene_pointer(&self, px: f32, py: f32) -> bool {
        self.ui.nav_cube_blocks_scene_pointer(px, py)
    }

    pub fn nav_cube_hover_slot(&self) -> Option<u32> {
        self.ui.nav_cube_hover_slot()
    }

    pub fn egui_blocks_scene_pointer(&self) -> bool {
        self.ui.egui_blocks_scene_pointer()
    }

    pub fn compositor_open(&self) -> bool {
        self.ui.compositor_open()
    }

    pub fn compositor_graph(&self) -> &rc3d_render::CompositorGraph {
        self.ui.compositor_graph()
    }

    /// Frameless caption + Fluent dark visuals. Used by the studio host only.
    pub fn enable_studio_shell(&mut self, title: &str) {
        self.ui.enable_studio_shell(title);
    }

    pub fn set_window_maximized(&mut self, maximized: bool) {
        self.ui.set_window_maximized(maximized);
    }

    pub fn take_caption_action(&mut self) -> Option<crate::ui::types::CaptionAction> {
        self.ui.take_caption_action()
    }

    pub fn sync_document_chrome(&mut self, title: String, dirty: bool) {
        self.ui.sync_document_chrome(title, dirty);
    }

    pub fn close_after_save(&self) -> bool {
        self.ui.close_after_save()
    }

    pub fn clear_close_after_save(&mut self) {
        self.ui.clear_close_after_save();
    }

    pub fn request_close_prompt(&mut self) {
        self.ui.request_close_prompt();
    }

    pub fn chrome_mut(&mut self) -> &mut crate::ui::types::EditorChromeState {
        self.ui.chrome_mut()
    }

    pub fn chrome(&self) -> &crate::ui::types::EditorChromeState {
        self.ui.chrome()
    }
}
