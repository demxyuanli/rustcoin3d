use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rc3d_render::{render_action::apply_world_camera, Renderer};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};

mod cli;
mod egui_paint;
mod engine;
mod panels;
mod render;
mod session;
mod viewport_gpu;
mod tui;

use session::EditorSession;
use viewport_gpu::CliViewportRt;
use panels::{DiagnosticsPanel, PropertiesPanel, ViewportPanel};
use render::RenderContext;
use tui::AppEvent;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d_cli_editor=info"),
    )
    .init();
    log::info!("rc3d CLI Editor starting...");

    let event_loop =
        EventLoop::<AppEvent>::with_user_event().build().expect("failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let gui_alive = Arc::new(AtomicBool::new(true));
    let proxy = event_loop.create_proxy();

    let session = Arc::new(EditorSession::new());
    session.load_initial_demo();

    let session_tui = session.clone();
    let proxy_tui = proxy.clone();
    let alive_for_tui = gui_alive.clone();
    std::thread::spawn(move || {
        tui::run_tui_thread(session_tui, proxy_tui, alive_for_tui);
    });

    let mut app = CliEditorApp::new(session, gui_alive);

    event_loop.run_app(&mut app).expect("event loop error");
}

struct CliEditorApp {
    window: Option<Window>,
    render_ctx: Option<RenderContext>,
    session: Arc<EditorSession>,
    gui_alive: Arc<AtomicBool>,
    viewport_panel: ViewportPanel,
    properties_panel: PropertiesPanel,
    diagnostics_panel: DiagnosticsPanel,
    last_frame: Instant,
    fps_smoother: f64,
    viewport_rt: Option<CliViewportRt>,
}

impl CliEditorApp {
    fn new(session: Arc<EditorSession>, gui_alive: Arc<AtomicBool>) -> Self {
        Self {
            window: None,
            render_ctx: None,
            session,
            gui_alive,
            viewport_panel: ViewportPanel::new(),
            properties_panel: PropertiesPanel::new(),
            diagnostics_panel: DiagnosticsPanel::new(),
            last_frame: Instant::now(),
            fps_smoother: 60.0,
            viewport_rt: None,
        }
    }
}

impl ApplicationHandler<AppEvent> for CliEditorApp {
    fn user_event(&mut self, event_loop: &ActiveEventLoop, _event: AppEvent) {
        self.gui_alive.store(false, Ordering::Relaxed);
        event_loop.exit();
    }

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
                self.gui_alive.store(false, Ordering::Relaxed);
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let Ok(state) = self.session.state.read() else {
                    return;
                };

                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f64();
                self.last_frame = now;
                if dt > 0.0 {
                    self.fps_smoother = self.fps_smoother * 0.9 + (1.0 / dt) * 0.1;
                }

                self.viewport_panel.collect(&state);

                let [vx_req, vy_req] = self.viewport_panel.viewport_pixel_extent();
                let fmt = render_ctx.renderer.config.format;
                let recreated = viewport_gpu::ensure_viewport_rt(
                    &mut self.viewport_rt,
                    &render_ctx.renderer.device,
                    fmt,
                    vx_req,
                    vy_req,
                );
                if recreated {
                    if let Some(ref rt) = self.viewport_rt {
                        render_ctx.egui_painter.bind_user_texture_view(
                            &render_ctx.renderer.device,
                            self.viewport_panel.viewport_texture_id,
                            &rt.view,
                        );
                    }
                }

                if let Some(ref rt) = self.viewport_rt {
                    let vx = rt.extent[0];
                    let vy = rt.extent[1];
                    if !self.viewport_panel.draw_calls.is_empty() {
                        let aspect = vx as f32 / vy.max(1) as f32;
                        let proj = state.camera.projection_matrix(aspect);
                        let inv_proj = proj.inverse();
                        apply_world_camera(
                            &mut self.viewport_panel.draw_calls,
                            state.camera.view_matrix(),
                            proj,
                            state.camera.position(),
                        );
                        let stats = render_ctx.renderer.render_draw_calls_to_viewport_texture(
                            &self.viewport_panel.draw_calls,
                            &state.scene,
                            &rt.texture,
                            &rt.view,
                            vx,
                            vy,
                            proj,
                            inv_proj,
                        );
                        self.viewport_panel.frame_stats = Some(stats);
                    } else {
                        self.viewport_panel.frame_stats = None;
                    }
                }

                render_ctx
                    .egui_ctx
                    .begin_pass(render_ctx.egui_winit.take_egui_input(window));

                let ppp = render_ctx.pixels_per_point;
                self.diagnostics_panel.ui(
                    &render_ctx.egui_ctx,
                    &state,
                    self.fps_smoother,
                    dt * 1000.0,
                );
                self.properties_panel
                    .ui(&render_ctx.egui_ctx, &state);
                self.viewport_panel
                    .ui(&render_ctx.egui_ctx, &state, ppp);

                let egui_output = render_ctx.egui_ctx.end_pass();
                let paint_jobs = render_ctx.egui_ctx.tessellate(
                    egui_output.shapes,
                    render_ctx.pixels_per_point,
                );

                let viewport_tex_id = self.viewport_panel.viewport_texture_id;
                for (id, delta) in egui_output.textures_delta.set {
                    if id == viewport_tex_id {
                        continue;
                    }
                    render_ctx.egui_painter.update_texture(
                        &render_ctx.renderer.device,
                        &render_ctx.renderer.queue,
                        id,
                        &delta,
                    );
                }
                for id in egui_output.textures_delta.free {
                    if id == viewport_tex_id {
                        continue;
                    }
                    render_ctx.egui_painter.free_texture(&id);
                }

                let screen_size = window.inner_size();
                let batches = render_ctx.egui_painter.upload(
                    &render_ctx.renderer.device,
                    &render_ctx.renderer.queue,
                    &paint_jobs,
                    render_ctx.pixels_per_point,
                    screen_size.width,
                    screen_size.height,
                );

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
                let swap_view = surface_tex
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = render_ctx
                    .renderer
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("cli editor egui to swapchain"),
                    });
                {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("swapchain egui"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &swap_view,
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
                    render_ctx.egui_painter.draw_batches(&mut rp, &batches);
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


