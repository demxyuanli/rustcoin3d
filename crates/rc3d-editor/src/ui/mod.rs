//! In-viewport editor UI (egui) composited after the main scene pass.

pub mod draw;
pub mod panel;
pub mod types;

use std::collections::VecDeque;

use egui_wgpu::ScreenDescriptor;
use rc3d_engine_api::Engine;
use rc3d_scene::SceneGraph;
use wgpu;

pub use types::{EditorDisplayMode, EditorUiContext, NodeDataType, RenderFeatureFlags};

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
    /// Console log ring buffer.
    console_entries: Vec<String>,
    show_console: bool,
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
        let egui_renderer =
            egui_wgpu::Renderer::new(engine.wgpu_device(), format, None, 1, false);
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
            console_entries: Vec::new(),
            show_console: false,
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

    pub fn take_commands(&mut self) -> std::collections::vec_deque::IntoIter<crate::commands::EditorCommand> {
        std::mem::take(&mut self.command_queue).into_iter()
    }

    /// Push a log entry to the console ring buffer (max 200 entries).
    pub fn push_log(&mut self, msg: &str) {
        self.console_entries.push(msg.to_string());
        if self.console_entries.len() > 200 {
            self.console_entries.remove(0);
        }
    }

    /// Toggle console panel visibility.
    pub fn toggle_console(&mut self) {
        self.show_console = !self.show_console;
    }

    pub fn handle_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        self.on_window_event(window, event)
    }

    pub fn render(
        &mut self,
        window: &winit::window::Window,
        graph: &SceneGraph,
        engine: &Engine,
        ui_ctx: &EditorUiContext,
    ) {
        let raw_input = self.winit_state.take_egui_input(window);
        let ctx_menu = &mut self.context_menu_pos;
        let show_console = &mut self.show_console;
        let console_ref = &mut self.console_entries;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            draw::build_ui(
                ctx,
                graph,
                ui_ctx,
                ctx_menu,
                show_console,
                console_ref,
                &mut self.command_queue,
            );
        });
        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        let device = engine.wgpu_device();
        let queue = engine.wgpu_queue();

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.textures_to_free
            .extend(full_output.textures_delta.free.iter().copied());

        self.pixels_per_point = full_output.pixels_per_point;
        self.clipped_primitives = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
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
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.egui_renderer.render(
            &mut pass.forget_lifetime(),
            &self.clipped_primitives,
            &screen,
        );
    }
}
