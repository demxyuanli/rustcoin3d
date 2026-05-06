use std::time::Instant;

use rc3d_render::Renderer;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};

mod cli;
mod egui_paint;
mod engine;
mod panels;
mod render;

use engine::state::EngineState;
use panels::{CliPanel, DiagnosticsPanel, PropertiesPanel, ViewportPanel};
use render::RenderContext;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d_cli_editor=info"),
    )
    .init();
    log::info!("rc3d CLI Editor starting...");

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = CliEditorApp::new();
    event_loop.run_app(&mut app).expect("event loop error");
}

struct CliEditorApp {
    window: Option<Window>,
    render_ctx: Option<RenderContext>,
    engine_state: EngineState,
    cli_panel: CliPanel,
    viewport_panel: ViewportPanel,
    properties_panel: PropertiesPanel,
    diagnostics_panel: DiagnosticsPanel,
    last_frame: Instant,
    fps_smoother: f64,
}

impl CliEditorApp {
    fn new() -> Self {
        Self {
            window: None,
            render_ctx: None,
            engine_state: EngineState::new(),
            cli_panel: CliPanel::new(),
            viewport_panel: ViewportPanel::new(),
            properties_panel: PropertiesPanel::new(),
            diagnostics_panel: DiagnosticsPanel::new(),
            last_frame: Instant::now(),
            fps_smoother: 60.0,
        }
    }
}

impl ApplicationHandler for CliEditorApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = WindowAttributes::default()
            .with_title("rc3d CLI Editor")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 800u32));

        let window = event_loop
            .create_window(window_attrs)
            .expect("failed to create window");

        let renderer = pollster::block_on(Renderer::new(&window));
        let render_ctx = RenderContext::new(&window, renderer);

        self.window = Some(window);
        self.render_ctx = Some(render_ctx);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else { return };
        let Some(render_ctx) = self.render_ctx.as_mut() else { return };

        let _consumed = render_ctx.on_window_event(window, &event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f64();
                self.last_frame = now;
                if dt > 0.0 {
                    self.fps_smoother = self.fps_smoother * 0.9 + (1.0 / dt) * 0.1;
                }

                // Build egui UI
                render_ctx
                    .egui_ctx
                    .begin_pass(render_ctx.egui_winit.take_egui_input(window));

                self.diagnostics_panel.ui(
                    &render_ctx.egui_ctx,
                    &self.engine_state,
                    self.fps_smoother,
                    dt * 1000.0,
                );
                self.cli_panel
                    .ui(&render_ctx.egui_ctx, &mut self.engine_state);
                self.properties_panel
                    .ui(&render_ctx.egui_ctx, &self.engine_state);
                self.viewport_panel
                    .ui(&render_ctx.egui_ctx, &self.engine_state);

                let egui_output = render_ctx.egui_ctx.end_pass();
                let paint_jobs = render_ctx.egui_ctx.tessellate(
                    egui_output.shapes,
                    render_ctx.pixels_per_point,
                );

                for (id, delta) in egui_output.textures_delta.set {
                    render_ctx.egui_painter.update_texture(
                        &render_ctx.renderer.device,
                        &render_ctx.renderer.queue,
                        id,
                        &delta,
                    );
                }
                for id in egui_output.textures_delta.free {
                    render_ctx.egui_painter.free_texture(&id);
                }

                // Acquire surface and render egui
                let surface_tex = match render_ctx.renderer.acquire_surface_texture() {
                    Ok(t) => t,
                    Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                        let size = window.inner_size();
                        render_ctx.resize(size.width, size.height, window.scale_factor() as f32);
                        return;
                    }
                    Err(e) => {
                        log::error!("Surface error: {:?}", e);
                        return;
                    }
                };

                let view = surface_tex
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = render_ctx.renderer.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("egui frame"),
                    },
                );

                {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui render pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.05,
                                    g: 0.05,
                                    b: 0.05,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    let screen_size = window.inner_size();
                    render_ctx.egui_painter.paint(
                        &render_ctx.renderer.device,
                        &render_ctx.renderer.queue,
                        &mut rp,
                        &paint_jobs,
                        render_ctx.pixels_per_point,
                        screen_size.width,
                        screen_size.height,
                    );
                }

                render_ctx
                    .renderer
                    .queue
                    .submit(std::iter::once(encoder.finish()));
                surface_tex.present();

                window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                render_ctx.resize(size.width, size.height, window.scale_factor() as f32);
            }
            _ => {}
        }
    }
}
