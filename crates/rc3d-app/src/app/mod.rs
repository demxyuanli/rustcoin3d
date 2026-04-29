mod measurement;
mod streaming_lod;

use rc3d_core::{math::{Mat4, Vec3}, DisplayMode};
use rc3d_render::{DrawCall, FrameStats, Renderer};
use rc3d_scene::SceneGraph;
use std::sync::mpsc::TryRecvError;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::WindowAttributes,
};

use crate::camera_controller::CameraController;
use crate::world::World;
use streaming_lod::{FullResPatch, apply_decimated_preview, gaussian_triangle_budget, stream_step_ms};

type PickCallback = Box<dyn FnMut(&mut SceneGraph, rc3d_core::NodeId, Vec3)>;
const APPROX_VERTEX_BYTES: usize = 48;
const MAX_SAFE_VERTEX_BUFFER_BYTES: usize = 240 * 1024 * 1024;

struct FpsTracker {
    samples: std::collections::VecDeque<f32>,
    sum: f32,
    capacity: usize,
    last_log: Instant,
}

impl FpsTracker {
    fn new(capacity: usize) -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(capacity),
            sum: 0.0,
            capacity,
            last_log: Instant::now(),
        }
    }

    fn push(&mut self, frame_time_ms: f32) {
        if self.samples.len() == self.capacity {
            if let Some(old) = self.samples.pop_front() {
                self.sum -= old;
            }
        }
        self.samples.push_back(frame_time_ms);
        self.sum += frame_time_ms;
    }

    fn average_frame_ms(&self) -> f32 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.sum / self.samples.len() as f32
        }
    }

    fn fps(&self) -> f32 {
        let avg = self.average_frame_ms();
        if avg > 0.0 { 1000.0 / avg } else { 0.0 }
    }

    fn maybe_log(&mut self, stats: FrameStats, quality: &str) {
        if self.last_log.elapsed().as_secs() >= 1 {
            self.last_log = Instant::now();
            log::info!(
                "FPS: {:.1} | frame: {:.2}ms | tris: {} | draws: {} | culled: {} | quality: {}",
                self.fps(),
                self.average_frame_ms(),
                stats.visible_triangles,
                stats.visible_draw_calls,
                stats.culled_draw_calls,
                quality,
            );
        }
    }
}

pub struct App {
    pub world: World,
    pub renderer: Option<Renderer>,
    pub window: Option<winit::window::Window>,
    pub camera_controller: Option<CameraController>,
    pub on_pick: Option<PickCallback>,
    cursor_pos: (f64, f64),
    shift_pressed: bool,
    measurement_mode: bool,
    measurement_first_point: Option<Vec3>,
    measurements: Vec<(Vec3, Vec3, f32)>,
    perf_mode_last: bool,
    full_res_patches: Vec<FullResPatch>,
    preview_mode_active: bool,
    stream_next_tick: Option<Instant>,
    initial_display_mode: DisplayMode,
    enable_hdr_post_processing: bool,
    last_frame_time: Instant,
    fps_tracker: FpsTracker,
    pending_graph_rx: Option<std::sync::mpsc::Receiver<Result<SceneGraph, String>>>,
    graph_load_hook: Option<Box<dyn FnOnce(&mut App) + 'static>>,
}

impl App {
    pub fn new(graph: SceneGraph) -> Self {
        let mut graph = graph;
        let full_res_patches = apply_decimated_preview(&mut graph);
        let preview_mode_active = !full_res_patches.is_empty();
        if preview_mode_active {
            log::warn!(
                "Streaming mesh load: {} patch(es), Gaussian CDF sigma={}, duration={}s",
                full_res_patches.len(),
                0.22,
                2.2,
            );
        }
        Self {
            world: World::new(graph),
            renderer: None,
            window: None,
            camera_controller: None,
            on_pick: None,
            cursor_pos: (0.0, 0.0),
            shift_pressed: false,
            measurement_mode: false,
            measurement_first_point: None,
            measurements: Vec::new(),
            perf_mode_last: false,
            full_res_patches,
            preview_mode_active,
            stream_next_tick: None,
            initial_display_mode: DisplayMode::ShadedWithEdges,
            enable_hdr_post_processing: false,
            last_frame_time: Instant::now(),
            fps_tracker: FpsTracker::new(60),
            pending_graph_rx: None,
            graph_load_hook: None,
        }
    }

    pub fn set_pending_graph_receiver(&mut self, rx: std::sync::mpsc::Receiver<Result<SceneGraph, String>>) {
        self.pending_graph_rx = Some(rx);
    }

    pub fn set_graph_load_hook(&mut self, hook: impl FnOnce(&mut App) + 'static) {
        self.graph_load_hook = Some(Box::new(hook));
    }

    fn poll_pending_graph_load(&mut self) {
        let Some(rx) = self.pending_graph_rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(graph)) => {
                self.world.graph = graph;
                self.pending_graph_rx = None;
                if let Some(renderer) = &mut self.renderer {
                    self.world.invalidate_caches(renderer);
                } else {
                    self.world.collector.invalidate_mesh_cache();
                }
                log::info!("Async scene load applied");
                if let Some(hook) = self.graph_load_hook.take() {
                    hook(self);
                }
            }
            Ok(Err(e)) => {
                self.pending_graph_rx = None;
                log::error!("Async scene load failed: {}", e);
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.pending_graph_rx = None;
                log::warn!("Async scene load channel disconnected");
            }
        }
    }

    pub fn with_camera_controller(mut self, controller: CameraController) -> Self {
        self.camera_controller = Some(controller);
        self
    }

    pub fn with_engines(mut self, engines: rc3d_engine::EngineRegistry) -> Self {
        self.world.engines = Some(engines);
        self
    }

    pub fn on_pick(
        mut self,
        f: impl FnMut(&mut SceneGraph, rc3d_core::NodeId, Vec3) + 'static,
    ) -> Self {
        self.on_pick = Some(Box::new(f));
        self
    }

    pub fn scene_graph(&self) -> &SceneGraph {
        &self.world.graph
    }

    pub fn scene_graph_mut(&mut self) -> &mut SceneGraph {
        &mut self.world.graph
    }

    pub fn with_initial_display_mode(mut self, mode: DisplayMode) -> Self {
        self.initial_display_mode = mode;
        self
    }

    pub fn with_hdr_post_processing(mut self, enabled: bool) -> Self {
        self.enable_hdr_post_processing = enabled;
        self
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("rustcoin3d")
                        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
                        .with_resizable(true)
                        .with_maximized(true)
                        .with_visible(false),
                )
                .expect("failed to create window");
            let renderer = pollster::block_on(Renderer::new(&window));
            self.window = Some(window);
            self.renderer = Some(renderer);
            if let Some(renderer) = &mut self.renderer {
                renderer.set_display_mode(self.initial_display_mode);
                if self.enable_hdr_post_processing {
                    renderer.set_hdr_post_processing(true);
                }
            }
            if let Some(window) = &self.window {
                window.set_visible(true);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match &event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                let size = *physical_size;
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x, position.y);
            }
            WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    state: winit::event::ElementState::Pressed,
                    physical_key: key,
                    ..
                },
                ..
            } => {
                match key {
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyW) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.set_display_mode(DisplayMode::Wireframe);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyS) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.set_display_mode(DisplayMode::Shaded);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyE) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.set_display_mode(DisplayMode::ShadedWithEdges);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyH) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.set_display_mode(DisplayMode::HiddenLine);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyX) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.toggle_clip_plane(0);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyY) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.toggle_clip_plane(1);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyZ) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.toggle_clip_plane(2);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
                        self.world.graph.clear_selection();
                        self.measurements.clear();
                        self.measurement_first_point = None;
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyM) => {
                        self.measurement_mode = !self.measurement_mode;
                        self.measurement_first_point = None;
                        log::info!("Measurement mode: {}", self.measurement_mode);
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyI) => {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.cycle_ibl_preset();
                        }
                    }
                    _ => {}
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.shift_pressed = mods.state().shift_key();
            }
            WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                if self.measurement_mode {
                    self.do_measure_pick();
                } else if self.camera_controller.is_none() || self.shift_pressed {
                    self.do_pick();
                }
            }
            WindowEvent::RedrawRequested => {
                self.poll_pending_graph_load();
                if self.preview_mode_active && self.renderer.is_some() && self.stream_next_tick.is_none() {
                    self.stream_next_tick = Some(Instant::now() + Duration::from_millis(stream_step_ms()));
                }
                self.tick_mesh_stream();
                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    self.world.evaluate_engines();

                    if let Some(ref ctrl) = self.camera_controller {
                        ctrl.update_camera(&mut self.world.graph, aspect);
                    }

                    renderer.materials = self.world.materials.clone();
                    self.world.collector.draw_calls.clear();
                    self.world.collector.state = rc3d_actions::State::new();
                    self.world.collector.camera_pos = Vec3::new(0.0, 0.0, 5.0);
                    self.world.collector.view_matrix = Mat4::IDENTITY;
                    self.world.collector.projection_matrix = Mat4::IDENTITY;
                    self.world.collector.projection_orthographic = false;
                    self.world.collector.global_display_mode = renderer.display_mode();
                    self.world.collector.material_library = Some(self.world.materials.clone());
                    for &root in self.world.graph.roots() {
                        self.world.collector.traverse(&self.world.graph, root);
                    }

                    if !self.measurements.is_empty() {
                        let vp = self.world.collector.projection_matrix * self.world.collector.view_matrix;
                        for &(p1, p2, _dist) in &self.measurements {
                            self.world.collector.draw_calls.push(DrawCall {
                                vertices: Arc::new(Vec::new()),
                                indices: None,
                                edge_positions: Arc::new(vec![p1.to_array(), p2.to_array()]),
                                mvp: vp,
                                model_matrix: Mat4::IDENTITY,
                                camera_pos: self.world.collector.camera_pos,
                                light_dirs: [[0.0; 4]; 16],
                                light_colors: [[0.0; 4]; 16],
                                light_types: [[0.0; 4]; 16],
                                light_positions: [[0.0; 4]; 16],
                                spot_params: [[0.0; 4]; 16],
                                light_count: 0,
                                diffuse_color: Vec3::ZERO,
                                ambient_color: Vec3::ZERO,
                                specular_color: Vec3::ZERO,
                                shininess: 1.0,
                                base_color: Vec3::ZERO,
                                metallic: 0.0,
                                roughness: 0.5,
                                albedo_path: None,
                                aabb: None,
                                display_mode: DisplayMode::ShadedWithEdges,
                                selected: false,
                                overlay_color: Some([1.0, 1.0, 0.0, 1.0]),
                                mesh_hash: None,
                                meshlet_data: None,
                                projection_orthographic: self.world.collector.projection_orthographic,
                                depth_reversed_z: rc3d_core::depth_reversed_z_from_projection(
                                    self.world.collector.projection_matrix,
                                ),
                            });
                        }
                    }

                    if !self.world.collector.draw_calls.is_empty() {
                        let stats = renderer.render_draw_calls(&self.world.collector.draw_calls, &self.world.graph);
                        let now = Instant::now();
                        let frame_time_ms =
                            now.duration_since(self.last_frame_time).as_secs_f32() * 1000.0;
                        self.last_frame_time = now;
                        self.fps_tracker.push(frame_time_ms);
                        renderer.report_frame_time_ms(frame_time_ms);
                        let mode_name = format!("{:?} | IBL:{}", renderer.display_mode(), renderer.ibl_preset_name());
                        renderer.update_hud(
                            self.fps_tracker.fps(),
                            frame_time_ms,
                            stats,
                            &mode_name,
                        );
                        let quality = renderer.adaptive_quality_name().to_string();
                        self.fps_tracker.maybe_log(stats, &quality);
                        let perf_active = renderer.performance_mode_active();
                        if perf_active != self.perf_mode_last {
                            self.perf_mode_last = perf_active;
                            if perf_active {
                                window.set_title("rustcoin3d [Performance Mode]");
                                log::warn!("Performance mode is enabled");
                            } else {
                                window.set_title("rustcoin3d");
                                log::info!("Performance mode is disabled");
                            }
                        }
                        if self.preview_mode_active {
                            window.set_title("rustcoin3d [Stream mesh]");
                        }
                    } else {
                        drop(renderer.surface.get_current_texture());
                    }
                    if self.world.engines.is_some() {
                        window.request_redraw();
                    }
                }
                if self.preview_mode_active {
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }
            _ => {}
        }

        if let Some(ref mut ctrl) = self.camera_controller {
            if !self.measurement_mode {
                ctrl.handle_event(&event);
                if let Some(window) = &self.window {
                    match event {
                        WindowEvent::CursorMoved { .. }
                        | WindowEvent::MouseInput { .. }
                        | WindowEvent::MouseWheel { .. }
                        | WindowEvent::KeyboardInput { .. } => window.request_redraw(),
                        _ => {}
                    }
                }
            }
        }
    }
}

impl App {
    fn tick_mesh_stream(&mut self) {
        if self.full_res_patches.is_empty() {
            return;
        }
        let Some(fire_at) = self.stream_next_tick else {
            return;
        };
        let now = Instant::now();
        if now < fire_at {
            return;
        }
        self.stream_next_tick = Some(now + Duration::from_millis(stream_step_ms()));

        let mut remaining = Vec::new();
        let mut any_change = false;
        let mut finished_patches: Vec<FullResPatch> = Vec::new();

        for mut patch in self.full_res_patches.drain(..) {
            let start = *patch.stream_start.get_or_insert(now);
            let t_s = start.elapsed().as_secs_f64();

            let budget = gaussian_triangle_budget(t_s, patch.total_full_tris);
            let estimated_vertex_bytes = patch.full_points.len().saturating_mul(APPROX_VERTEX_BYTES);
            let skip_full_restore = estimated_vertex_bytes > MAX_SAFE_VERTEX_BUFFER_BYTES;

            if budget >= patch.total_full_tris || t_s >= 2.2 {
                if skip_full_restore {
                    // Do not restore full-res mesh for very large models, but still
                    // advance to the highest available streamed stage before finishing.
                    if let Some(last_stage) = patch.stream_stages.len().checked_sub(1) {
                        if last_stage > patch.current_stage {
                            patch.apply_stage_to_graph(&mut self.world.graph, last_stage);
                            let tris = patch.stage_tri_counts[last_stage];
                            log::info!(
                                "Mesh stream [Gaussian]: ~{}K tris at t={:.2}s (max staged LOD)",
                                tris / 1000,
                                t_s,
                            );
                            any_change = true;
                        }
                    }
                    log::warn!(
                        "Skip full-resolution restore (estimated_vertex_bytes={} > limit={}), keep max staged LOD",
                        estimated_vertex_bytes,
                        MAX_SAFE_VERTEX_BUFFER_BYTES,
                    );
                    continue;
                }
                finished_patches.push(patch);
                continue;
            }

            if let Some(target_stage) = patch.stage_for_budget(budget) {
                if target_stage > patch.current_stage {
                    patch.apply_stage_to_graph(&mut self.world.graph, target_stage);
                    let tris = patch.stage_tri_counts[target_stage];
                    log::info!(
                        "Mesh stream [Gaussian]: ~{}K tris at t={:.2}s (budget={})",
                        tris / 1000,
                        t_s,
                        budget,
                    );
                    patch.current_stage = target_stage;
                    any_change = true;
                }
            }
            remaining.push(patch);
        }

        let n_finished = finished_patches.len();
        for patch in finished_patches {
            let total_points = patch.full_points.len();
            let total_indices = patch.full_coord_index.len();
            let total_tris = total_indices / 4;
            let estimated_vertex_bytes = patch.full_points.len().saturating_mul(APPROX_VERTEX_BYTES);
            if estimated_vertex_bytes > MAX_SAFE_VERTEX_BUFFER_BYTES {
                log::warn!(
                    "Skip full-resolution restore (estimated_vertex_bytes={} > limit={}), keep streamed LOD",
                    estimated_vertex_bytes,
                    MAX_SAFE_VERTEX_BUFFER_BYTES,
                );
            } else {
                patch.apply_full_to_graph(&mut self.world.graph);
                log::warn!(
                    "Full-resolution mesh restored: points={}, indices={}, ~{}K triangles",
                    total_points,
                    total_indices,
                    total_tris / 1000,
                );
            }
        }

        self.full_res_patches = remaining;

        if any_change || n_finished > 0 {
            if let Some(renderer) = &mut self.renderer {
                renderer.invalidate_mesh_cache();
            }
            self.world.collector.invalidate_mesh_cache();
        }

        if self.full_res_patches.is_empty() {
            self.preview_mode_active = false;
            self.stream_next_tick = None;
            if let Some(window) = &self.window {
                window.set_title("rustcoin3d");
                window.request_redraw();
            }
        }
    }

    fn do_pick(&mut self) {
        let Some(window) = self.window.as_ref() else { return };
        let size = window.inner_size();
        let Some(ray) = measurement::build_pick_ray(&self.world.graph, self.cursor_pos, (size.width, size.height)) else {
            return;
        };

        let mut picker = rc3d_actions::RayPickAction::new(ray);
        rc3d_actions::apply_to_all_roots(&mut picker, &self.world.graph);

        if let Some(hit) = picker.hits.first() {
            self.world.graph.toggle_selection(hit.node);
            log::info!(
                "Pick hit: node={:?}, point={:?}, selected={}",
                hit.node, hit.point, self.world.graph.is_selected(hit.node)
            );
            if let Some(cb) = &mut self.on_pick {
                cb(&mut self.world.graph, hit.node, hit.point);
            }
        } else {
            log::info!("Pick miss (no hit)");
            if !self.shift_pressed {
                self.world.graph.clear_selection();
            }
        }
    }

    fn do_measure_pick(&mut self) {
        let Some(window) = self.window.as_ref() else { return };
        let size = window.inner_size();
        let Some(ray) = measurement::build_pick_ray(&self.world.graph, self.cursor_pos, (size.width, size.height)) else {
            return;
        };

        let mut picker = rc3d_actions::RayPickAction::new(ray);
        rc3d_actions::apply_to_all_roots(&mut picker, &self.world.graph);

        if let Some(hit) = picker.hits.first() {
            let point = hit.point;
            match self.measurement_first_point {
                None => {
                    self.measurement_first_point = Some(point);
                    log::info!("Measurement point A: {:?}", point);
                }
                Some(first) => {
                    let dist = (point - first).length();
                    self.measurements.push((first, point, dist));
                    self.measurement_first_point = None;
                    log::info!("Measurement: A={:?} B={:?} distance={:.4}", first, point, dist);
                }
            }
        }
    }
}
