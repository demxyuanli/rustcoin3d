//! In-viewport editor UI (egui) composited after the main scene pass.

pub mod assets;
pub mod caption;
pub mod compositor;
mod docks;
pub mod draw;
pub mod hierarchy;
pub mod history;
pub mod i18n;
pub mod icons;
pub mod inspector;
pub mod menus;
pub mod nav_cube;
pub mod panel;
mod shell;
pub mod splash;
pub mod theme;
pub mod tool_strip;
pub mod types;

use std::collections::VecDeque;

use egui_wgpu::ScreenDescriptor;
use rc3d_engine_api::Engine;
use rc3d_scene::SceneGraph;
use wgpu;

pub use types::{
    BottomTab, CaptionAction, CaptionBarState, CaseKind, CaseListItem, CaseParamView, CaseStepView,
    EditorChromeState, EditorDisplayMode, EditorUiContext, NodeDataType, PixelRect,
    RenderFeatureFlags, SideTab,
};

pub struct EditorUi {
    egui_ctx: egui::Context,
    winit_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    screen_descriptor: ScreenDescriptor,
    command_queue: VecDeque<crate::commands::EditorCommand>,
    clipped_primitives: Vec<egui::ClippedPrimitive>,
    pixels_per_point: f32,
    textures_to_free: Vec<egui::TextureId>,
    /// Right-click context menu position in egui coordinates, if visible.
    context_menu_pos: Option<egui::Pos2>,
    chrome: crate::ui::types::EditorChromeState,
    splash: crate::ui::splash::SplashState,
    fonts_ready: bool,
    applied_theme: Option<crate::ui::theme::UiTheme>,
    compositor: crate::ui::compositor::CompositorEditor,
    markdown_cache: egui_commonmark::CommonMarkCache,
}

impl EditorUi {
    pub fn new(window: &winit::window::Window, engine: &Engine) -> Self {
        let egui_ctx = egui::Context::default();
        let max_texture_side = engine.wgpu_device().limits().max_texture_dimension_2d as usize;
        let winit_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(max_texture_side),
        );
        let format = engine.surface_format();
        let egui_renderer = egui_wgpu::Renderer::new(
            engine.wgpu_device(),
            format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: false,
                predictable_texture_filtering: false,
            },
        );
        let size = window.inner_size();
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [size.width.max(1), size.height.max(1)],
            pixels_per_point: window.scale_factor() as f32,
        };
        Self {
            egui_ctx,
            winit_state,
            egui_renderer,
            screen_descriptor,
            command_queue: VecDeque::new(),
            clipped_primitives: Vec::new(),
            pixels_per_point: window.scale_factor() as f32,
            textures_to_free: Vec::new(),
            context_menu_pos: None,
            chrome: crate::ui::types::EditorChromeState::default(),
            splash: crate::ui::splash::SplashState::default(),
            fonts_ready: false,
            applied_theme: None,
            compositor: crate::ui::compositor::CompositorEditor::default(),
            markdown_cache: egui_commonmark::CommonMarkCache::default(),
        }
    }

    pub fn on_window_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let r = self.winit_state.on_window_event(window, event);
        r.consumed
    }

    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f32) {
        self.screen_descriptor.size_in_pixels = [width.max(1), height.max(1)];
        self.screen_descriptor.pixels_per_point = scale_factor;
        self.pixels_per_point = scale_factor;
    }

    pub fn take_commands(
        &mut self,
    ) -> std::collections::vec_deque::IntoIter<crate::commands::EditorCommand> {
        std::mem::take(&mut self.command_queue).into_iter()
    }

    pub fn handle_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        self.on_window_event(window, event)
    }

    pub fn enable_studio_shell(&mut self, title: &str) {
        self.chrome.caption.enabled = true;
        self.chrome.caption.title = title.to_string();
        crate::ui::theme::apply_theme(&self.egui_ctx, crate::ui::theme::UiTheme::Dark, true);
        self.fonts_ready = true;
        // First `render` applies session theme (Dark or Light) without reloading fonts.
        self.applied_theme = None;
    }

    pub fn set_window_maximized(&mut self, maximized: bool) {
        self.chrome.caption.maximized = maximized;
    }

    pub fn take_caption_action(&mut self) -> Option<CaptionAction> {
        self.chrome.caption.action.take()
    }

    pub fn sync_document_chrome(&mut self, title: String, dirty: bool) {
        if !self.chrome.caption.enabled {
            return;
        }
        self.chrome.caption.title = title;
        self.chrome.caption.dirty = dirty;
    }

    pub fn close_after_save(&self) -> bool {
        self.chrome.caption.close_after_save
    }

    pub fn clear_close_after_save(&mut self) {
        self.chrome.caption.close_after_save = false;
    }

    pub fn request_close_prompt(&mut self) {
        self.chrome.caption.close_prompt = true;
    }

    pub fn chrome_mut(&mut self) -> &mut crate::ui::types::EditorChromeState {
        &mut self.chrome
    }

    pub fn chrome(&self) -> &crate::ui::types::EditorChromeState {
        &self.chrome
    }

    /// Host-driven splash lifecycle: set the loading stage / completion.
    pub fn set_splash(&mut self, splash: crate::ui::splash::SplashState) {
        self.splash = splash;
    }

    pub fn splash(&self) -> crate::ui::splash::SplashState {
        self.splash
    }

    /// Build egui UI for this frame. Returns `true` when egui needs another
    /// immediate pass (menus, popups, settle frames). With `ControlFlow::Wait`
    /// the host must schedule a redraw or those only appear after the next
    /// pointer move.
    pub fn render(
        &mut self,
        window: &winit::window::Window,
        graph: &SceneGraph,
        engine: &Engine,
        ui_ctx: &EditorUiContext,
    ) -> bool {
        let raw_input = self.winit_state.take_egui_input(window);
        if self.applied_theme != Some(ui_ctx.ui_theme) {
            crate::ui::theme::apply_theme(&self.egui_ctx, ui_ctx.ui_theme, !self.fonts_ready);
            self.fonts_ready = true;
            self.applied_theme = Some(ui_ctx.ui_theme);
        }
        let ctx_menu = &mut self.context_menu_pos;
        let chrome = &mut self.chrome;
        let splash = &mut self.splash;
        let compositor = &mut self.compositor;
        let mut full_output = self.egui_ctx.run_ui(raw_input, |ui| {
            draw::build_ui(
                ui,
                graph,
                ui_ctx,
                ctx_menu,
                chrome,
                splash,
                compositor,
                &mut self.markdown_cache,
                &mut self.command_queue,
            );
        });
        chrome.pointer_over_egui = self.egui_ctx.is_pointer_over_egui();
        // egui-winit does not request redraw from platform_output; the host must
        // honor ViewportOutput::repaint_delay (menus open mid-click need +1 frame).
        // Only ZERO: non-zero delays are tooltips etc. and need a timer, not a spin.
        let wants_immediate_repaint = full_output
            .viewport_output
            .values()
            .any(|v| v.repaint_delay == std::time::Duration::ZERO);
        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        let device = engine.wgpu_device();
        let queue = engine.wgpu_queue();

        for (id, image_deltas) in &full_output.textures_delta.set {
            for image_delta in image_deltas {
                self.egui_renderer
                    .update_texture(device, queue, *id, image_delta);
            }
        }
        self.textures_to_free
            .extend(full_output.textures_delta.free.iter().copied());
        full_output.textures_delta.clear();

        self.pixels_per_point = full_output.pixels_per_point;
        self.clipped_primitives = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        wants_immediate_repaint
    }

    pub fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        for id in self.textures_to_free.drain(..) {
            self.egui_renderer.free_texture(&id);
        }

        let screen = ScreenDescriptor {
            size_in_pixels: self.screen_descriptor.size_in_pixels,
            pixels_per_point: self.pixels_per_point,
        };

        let user_cmd_bufs = self.egui_renderer.update_buffers(
            device,
            queue,
            encoder,
            &self.clipped_primitives,
            &screen,
        );
        if !user_cmd_bufs.is_empty() {
            queue.submit(user_cmd_bufs);
        }

        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("editor_egui_overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.egui_renderer.render(
            &mut pass.forget_lifetime(),
            &self.clipped_primitives,
            &screen,
        );
    }

    pub fn scene_pixel_rect(&self) -> Option<crate::ui::types::PixelRect> {
        points_to_pixel(self.chrome.scene_rect_points, self.pixels_per_point)
    }

    pub fn document_open(&self) -> bool {
        self.chrome.document_tab_open()
    }

    /// True when the pointer (window pixels) is on the nav cube, or the cube is being dragged.
    pub fn nav_cube_blocks_scene_pointer(&self, px: f32, py: f32) -> bool {
        if self.chrome.nav_cube_dragging {
            return true;
        }
        points_to_pixel(self.chrome.nav_cube_rect_points, self.pixels_per_point)
            .is_some_and(|r| r.contains(px, py))
    }

    pub fn nav_cube_hover_slot(&self) -> Option<u32> {
        self.chrome.nav_cube_hover_slot
    }

    /// True when the pointer is over chrome widgets (menus, docks, tool strip, flyouts),
    /// not the bare 3D viewport. Floating layers over the film must block Full 3D redraw.
    pub fn egui_blocks_scene_pointer(&self) -> bool {
        // Open menus: block even while the pointer crosses the film toward a submenu.
        if egui::Popup::is_any_open(&self.egui_ctx) || self.context_menu_pos.is_some() {
            return true;
        }
        let Some(pos) = self.egui_ctx.pointer_latest_pos() else {
            return self.chrome.pointer_over_egui;
        };
        if rect_contains_points(self.chrome.tool_strip_rect_points, pos) {
            return true;
        }
        if let Some(layer) = self.egui_ctx.layer_id_at(pos) {
            if layer.order > egui::Order::Middle {
                return true;
            }
        }
        if !self.chrome.pointer_over_egui {
            return false;
        }
        !rect_contains_points(self.chrome.scene_rect_points, pos)
    }

    pub fn compositor_open(&self) -> bool {
        self.chrome.compositor_tab_open()
    }

    pub fn compositor_graph(&self) -> &rc3d_render::CompositorGraph {
        &self.compositor.graph
    }
}

fn rect_contains_points(rect: Option<[f32; 4]>, pos: egui::Pos2) -> bool {
    let Some([x, y, w, h]) = rect else {
        return false;
    };
    pos.x >= x && pos.y >= y && pos.x < x + w && pos.y < y + h
}

fn points_to_pixel(rect: Option<[f32; 4]>, ppp: f32) -> Option<crate::ui::types::PixelRect> {
    let [x, y, w, h] = rect?;
    if w < 2.0 || h < 2.0 {
        return None;
    }
    Some(crate::ui::types::PixelRect {
        x: (x * ppp).round().max(0.0) as u32,
        y: (y * ppp).round().max(0.0) as u32,
        width: (w * ppp).round().max(1.0) as u32,
        height: (h * ppp).round().max(1.0) as u32,
    })
}
