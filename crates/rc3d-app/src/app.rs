use rc3d_actions::Ray;
use rc3d_core::{math::{Mat4, Vec3}, DisplayMode};
use rc3d_engine::EngineRegistry;
use rc3d_render::{DrawCall, FrameStats, RenderCollector, Renderer};
use rc3d_scene::{NodeData, SceneGraph};
use std::collections::HashMap;
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

type PickCallback = Box<dyn FnMut(&mut SceneGraph, rc3d_core::NodeId, Vec3)>;
type MeshLodStage = (Vec<Vec3>, Option<Vec<[f32; 2]>>, Vec<i32>);

const PREVIEW_TRIANGLE_THRESHOLD: usize = 1_000_000;

/// Gaussian-CDF streaming: approx erf function (Abramowitz & Stegun 7.1.26, max error 1.5e-7).
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

/// Standard-normal CDF using the erf approximation.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Streaming LOD configuration: Gaussian CDF mapped over `[0..STREAM_DURATION_S]`.
/// - `sigma` controls spread: smaller = more triangles early; larger = slower ramp.
/// - `STREAM_DURATION_S` is total wall-clock time from first GPU frame to full mesh.
const STREAM_SIGMA: f64 = 0.22;
const STREAM_DURATION_S: f64 = 2.2;

/// Triangle budgets sampled from the Gaussian CDF at evenly spaced time points.
/// Using 10 steps gives smooth visual progression over ~2 s.
const STREAM_LOD_TARGETS: &[usize] =
    &[12_000, 30_000, 60_000, 110_000, 180_000, 280_000, 430_000, 660_000, 1_000_000, 1_600_000];
/// Minimum wall time between applying consecutive LOD stages (avoid GPU stalls).
const STREAM_STEP_MS: u64 = 180;

struct FullResPatch {
    coord_node: rc3d_core::NodeId,
    ifs_node: rc3d_core::NodeId,
    tex_node: Option<rc3d_core::NodeId>,
    full_points: Vec<Vec3>,
    full_tex: Option<Vec<[f32; 2]>>,
    full_coord_index: Vec<i32>,
    total_full_tris: usize,
    /// Pre-computed LOD stages, triangle counts monotonically increasing.
    stream_stages: Vec<MeshLodStage>,
    /// Triangle count per stage (cached to avoid re-scanning).
    stage_tri_counts: Vec<usize>,
    /// Index of the last stage that was written into the graph.
    current_stage: usize,
    /// Instant when streaming started (first GPU frame with renderer).
    stream_start: Option<Instant>,
}

fn apply_lod_stage_to_graph(
    graph: &mut SceneGraph,
    coord_id: rc3d_core::NodeId,
    ifs_id: rc3d_core::NodeId,
    tex_node: Option<rc3d_core::NodeId>,
    stage: &MeshLodStage,
) {
    let (points, tex, coord_index) = stage;
    if let Some(coord_mut) = graph.get_mut(coord_id) {
        if let NodeData::Coordinate3(c) = &mut coord_mut.data {
            c.point = points.clone();
        }
    }
    if let (Some(tid), Some(tex_pts)) = (tex_node, tex.as_ref()) {
        if let Some(tex_mut) = graph.get_mut(tid) {
            if let NodeData::TextureCoordinate2(t) = &mut tex_mut.data {
                t.point = tex_pts.clone();
            }
        }
    }
    if let Some(ifs_mut) = graph.get_mut(ifs_id) {
        if let NodeData::IndexedFaceSet(ifs) = &mut ifs_mut.data {
            ifs.coord_index = coord_index.clone();
        }
    }
}

impl FullResPatch {
    fn apply_stage_to_graph(&self, graph: &mut SceneGraph, stage_i: usize) {
        apply_lod_stage_to_graph(
            graph,
            self.coord_node,
            self.ifs_node,
            self.tex_node,
            &self.stream_stages[stage_i],
        );
    }

    fn apply_full_to_graph(self, graph: &mut SceneGraph) {
        if let Some(coord_mut) = graph.get_mut(self.coord_node) {
            if let NodeData::Coordinate3(c) = &mut coord_mut.data {
                c.point = self.full_points;
            }
        }
        if let (Some(tex_id), Some(full_tex)) = (self.tex_node, self.full_tex) {
            if let Some(tex_mut) = graph.get_mut(tex_id) {
                if let NodeData::TextureCoordinate2(t) = &mut tex_mut.data {
                    t.point = full_tex;
                }
            }
        }
        if let Some(ifs_mut) = graph.get_mut(self.ifs_node) {
            if let NodeData::IndexedFaceSet(ifs) = &mut ifs_mut.data {
                ifs.coord_index = self.full_coord_index;
            }
        }
    }

    /// Find the best matching stage index for a target triangle budget (binary search).
    fn stage_for_budget(&self, budget: usize) -> Option<usize> {
        let n = self.stage_tri_counts.len();
        if n == 0 || budget == 0 {
            return None;
        }
        // Find last stage with tri count <= budget
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.stage_tri_counts[mid] <= budget {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            None
        } else {
            Some(lo - 1)
        }
    }
}

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
    pub graph: SceneGraph,
    pub renderer: Option<Renderer>,
    pub window: Option<winit::window::Window>,
    pub camera_controller: Option<CameraController>,
    pub engines: Option<EngineRegistry>,
    pub on_pick: Option<PickCallback>,
    cursor_pos: (f64, f64),
    shift_pressed: bool,
    start: std::time::Instant,
    measurement_mode: bool,
    measurement_first_point: Option<Vec3>,
    measurements: Vec<(Vec3, Vec3, f32)>,
    collector: RenderCollector,
    perf_mode_last: bool,
    full_res_patches: Vec<FullResPatch>,
    preview_mode_active: bool,
    /// When to apply the next streaming LOD step (armed on first GPU frame while streaming).
    stream_next_tick: Option<Instant>,
    initial_display_mode: DisplayMode,
    enable_hdr_post_processing: bool,
    last_frame_time: Instant,
    fps_tracker: FpsTracker,
    /// When set, `RedrawRequested` polls for a loaded scene from a background thread.
    pending_graph_rx: Option<std::sync::mpsc::Receiver<Result<SceneGraph, String>>>,
    /// Called once on the main thread after a successful async graph replace (camera fit, etc.).
    graph_load_hook: Option<Box<dyn FnOnce(&mut App) + 'static>>,
}

impl App {
    pub fn new(graph: SceneGraph) -> Self {
        let mut graph = graph;
        let full_res_patches = Self::apply_decimated_preview(&mut graph);
        let preview_mode_active = !full_res_patches.is_empty();
        if preview_mode_active {
            log::warn!(
                "Streaming mesh load: {} patch(es), Gaussian CDF sigma={}, duration={}s",
                full_res_patches.len(),
                STREAM_SIGMA,
                STREAM_DURATION_S,
            );
        }
        Self {
            graph,
            renderer: None,
            window: None,
            camera_controller: None,
            engines: None,
            on_pick: None,
            cursor_pos: (0.0, 0.0),
            shift_pressed: false,
            start: std::time::Instant::now(),
            measurement_mode: false,
            measurement_first_point: None,
            measurements: Vec::new(),
            collector: RenderCollector::new(),
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
                self.graph = graph;
                self.pending_graph_rx = None;
                if let Some(renderer) = &mut self.renderer {
                    renderer.invalidate_mesh_cache();
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

    pub fn with_engines(mut self, engines: EngineRegistry) -> Self {
        self.engines = Some(engines);
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
        &self.graph
    }

    pub fn scene_graph_mut(&mut self) -> &mut SceneGraph {
        &mut self.graph
    }

    pub fn with_initial_display_mode(mut self, mode: DisplayMode) -> Self {
        self.initial_display_mode = mode;
        self
    }

    /// Enable HDR scene target, tonemap/FXAA to `post_ldr`, then blit to swapchain (see `Renderer::set_hdr_post_processing`).
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
                        self.graph.clear_selection();
                        self.measurements.clear();
                        self.measurement_first_point = None;
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyM) => {
                        self.measurement_mode = !self.measurement_mode;
                        self.measurement_first_point = None;
                        log::info!("Measurement mode: {}", self.measurement_mode);
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
                    self.stream_next_tick = Some(Instant::now() + Duration::from_millis(STREAM_STEP_MS));
                }
                self.tick_mesh_stream();
                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    // Evaluate engines (animation)
                    if let Some(ref mut engines) = self.engines {
                        let time = self.start.elapsed().as_secs_f64();
                        engines.evaluate_all(&mut self.graph, time);
                    }

                    if let Some(ref ctrl) = self.camera_controller {
                        ctrl.update_camera(&mut self.graph, aspect);
                    }

                    self.collector.draw_calls.clear();
                    self.collector.state = rc3d_actions::State::new();
                    self.collector.camera_pos = Vec3::new(0.0, 0.0, 5.0);
                    self.collector.view_matrix = Mat4::IDENTITY;
                    self.collector.projection_matrix = Mat4::IDENTITY;
                    self.collector.projection_orthographic = false;
                    self.collector.global_display_mode = renderer.display_mode();
                    for &root in self.graph.roots() {
                        self.collector.traverse(&self.graph, root);
                    }

                    // Add measurement line draw calls
                    if !self.measurements.is_empty() {
                        let vp = self.collector.projection_matrix * self.collector.view_matrix;
                        for &(p1, p2, _dist) in &self.measurements {
                            self.collector.draw_calls.push(DrawCall {
                                vertices: Arc::new(Vec::new()),
                                indices: None,
                                edge_positions: Arc::new(vec![p1.to_array(), p2.to_array()]),
                                mvp: vp,
                                model_matrix: Mat4::IDENTITY,
                                camera_pos: self.collector.camera_pos,
                                light_dirs: [[0.0; 4]; 4],
                                light_colors: [[0.0; 4]; 4],
                                light_types: [[0.0; 4]; 4],
                                light_positions: [[0.0; 4]; 4],
                                spot_params: [[0.0; 4]; 4],
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
                                projection_orthographic: self.collector.projection_orthographic,
                                depth_reversed_z: rc3d_core::depth_reversed_z_from_projection(
                                    self.collector.projection_matrix,
                                ),
                            });
                        }
                    }

                    if !self.collector.draw_calls.is_empty() {
                        let stats = renderer.render_draw_calls(&self.collector.draw_calls, &self.graph);
                        let now = Instant::now();
                        let frame_time_ms =
                            now.duration_since(self.last_frame_time).as_secs_f32() * 1000.0;
                        self.last_frame_time = now;
                        self.fps_tracker.push(frame_time_ms);
                        renderer.report_frame_time_ms(frame_time_ms);
                        let mode_name = format!("{:?}", renderer.display_mode());
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
                    if self.engines.is_some() {
                        window.request_redraw();
                    }
                }
                // Without animation engines, redraws only happen on input; promotion must run on the main thread.
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
    fn build_stream_stages(
        full_points: &[Vec3],
        tex: Option<&[[f32; 2]]>,
        full_coord_index: &[i32],
        tri_count: usize,
    ) -> Vec<MeshLodStage> {
        let mut out = Vec::new();
        let n = STREAM_LOD_TARGETS.len();
        let mut last_tris: usize = 0;
        for i in 0..n {
            let t = STREAM_LOD_TARGETS[i];
            if t >= tri_count {
                break;
            }
            let stage = Self::decimate_indexed_face_set(full_points, tex, full_coord_index, t);
            let tris = stage.2.iter().filter(|&&v| v == -1).count();
            if tris > last_tris {
                last_tris = tris;
                out.push(stage);
            }
        }
        if out.is_empty() {
            let fallback = STREAM_LOD_TARGETS
                .iter()
                .copied()
                .find(|&x| x < tri_count)
                .unwrap_or(250_000);
            out.push(Self::decimate_indexed_face_set(
                full_points,
                tex,
                full_coord_index,
                fallback,
            ));
        }
        out
    }

    /// Gaussian CDF-based triangle budget for elapsed time `t_s` in `[0..STREAM_DURATION_S]`.
    fn gaussian_triangle_budget(t_s: f64, total_tris: usize) -> usize {
        if t_s <= 0.0 {
            return 0;
        }
        if t_s >= STREAM_DURATION_S {
            return total_tris;
        }
        let x = (t_s / STREAM_DURATION_S - 0.5) / STREAM_SIGMA;
        let fraction = normal_cdf(x).clamp(0.005, 0.999);
        ((total_tris as f64) * fraction) as usize
    }

    fn apply_decimated_preview(graph: &mut SceneGraph) -> Vec<FullResPatch> {
        let mut patches = Vec::new();
        for &root in graph.roots().to_vec().iter() {
            Self::collect_preview_patches(graph, root, &mut patches);
        }
        patches
    }

    fn collect_preview_patches(graph: &mut SceneGraph, node: rc3d_core::NodeId, patches: &mut Vec<FullResPatch>) {
        let children: Vec<rc3d_core::NodeId> = graph.children(node).to_vec();
        if children.len() >= 3 {
            for trip in children.windows(3) {
                let coord_id = trip[0];
                let tex_id = trip[1];
                let ifs_id = trip[2];
                let (Some(coord_entry), Some(tex_entry), Some(ifs_entry)) =
                    (graph.get(coord_id), graph.get(tex_id), graph.get(ifs_id))
                else {
                    continue;
                };
                let (NodeData::Coordinate3(coord), NodeData::TextureCoordinate2(tex), NodeData::IndexedFaceSet(ifs)) =
                    (&coord_entry.data, &tex_entry.data, &ifs_entry.data)
                else {
                    continue;
                };
                if tex.point.len() != coord.point.len() {
                    continue;
                }
                let tri_count = ifs.coord_index.iter().filter(|&&v| v == -1).count();
                if tri_count <= PREVIEW_TRIANGLE_THRESHOLD {
                    continue;
                }
                let full_points = coord.point.clone();
                let full_tex = tex.point.clone();
                let full_coord_index = ifs.coord_index.clone();
                let stream_stages =
                    Self::build_stream_stages(&full_points, Some(&full_tex), &full_coord_index, tri_count);
                let stage_tri_counts: Vec<usize> = stream_stages
                    .iter()
                    .map(|s| s.2.iter().filter(|&&v| v == -1).count())
                    .collect();
                apply_lod_stage_to_graph(graph, coord_id, ifs_id, Some(tex_id), &stream_stages[0]);
                patches.push(FullResPatch {
                    coord_node: coord_id,
                    ifs_node: ifs_id,
                    tex_node: Some(tex_id),
                    full_points,
                    full_tex: Some(full_tex),
                    full_coord_index,
                    total_full_tris: tri_count,
                    stream_stages,
                    stage_tri_counts,
                    current_stage: 0,
                    stream_start: None,
                });
            }
        }
        if children.len() >= 2 {
            for pair in children.windows(2) {
                let coord_id = pair[0];
                let ifs_id = pair[1];
                let (Some(coord_entry), Some(ifs_entry)) = (graph.get(coord_id), graph.get(ifs_id)) else {
                    continue;
                };
                let (NodeData::Coordinate3(coord), NodeData::IndexedFaceSet(ifs)) =
                    (&coord_entry.data, &ifs_entry.data)
                else {
                    continue;
                };
                let tri_count = ifs.coord_index.iter().filter(|&&v| v == -1).count();
                if tri_count <= PREVIEW_TRIANGLE_THRESHOLD {
                    continue;
                }
                let full_points = coord.point.clone();
                let full_coord_index = ifs.coord_index.clone();
                let stream_stages = Self::build_stream_stages(&full_points, None, &full_coord_index, tri_count);
                let stage_tri_counts: Vec<usize> = stream_stages
                    .iter()
                    .map(|s| s.2.iter().filter(|&&v| v == -1).count())
                    .collect();
                apply_lod_stage_to_graph(graph, coord_id, ifs_id, None, &stream_stages[0]);
                patches.push(FullResPatch {
                    coord_node: coord_id,
                    ifs_node: ifs_id,
                    tex_node: None,
                    full_points,
                    full_tex: None,
                    full_coord_index,
                    total_full_tris: tri_count,
                    stream_stages,
                    stage_tri_counts,
                    current_stage: 0,
                    stream_start: None,
                });
            }
        }
        for child in children {
            Self::collect_preview_patches(graph, child, patches);
        }
    }

    fn decimate_indexed_face_set(
        points: &[Vec3],
        tex: Option<&[[f32; 2]]>,
        coord_index: &[i32],
        target_triangles: usize,
    ) -> (Vec<Vec3>, Option<Vec<[f32; 2]>>, Vec<i32>) {
        if let Some(t) = tex {
            assert_eq!(t.len(), points.len(), "texture coords must match positions");
        }
        let tri_count = coord_index.iter().filter(|&&v| v == -1).count();
        if tri_count <= target_triangles || tri_count == 0 {
            return (
                points.to_vec(),
                tex.map(|t| t.to_vec()),
                coord_index.to_vec(),
            );
        }
        let stride = (tri_count / target_triangles).max(2);
        let mut new_points = Vec::new();
        let mut new_tex: Option<Vec<[f32; 2]>> = tex.map(|_| Vec::new());
        let mut new_index = Vec::new();
        let mut remap: HashMap<i32, i32> = HashMap::new();

        let mut face = Vec::with_capacity(4);
        let mut face_id = 0usize;
        for &idx in coord_index {
            if idx < 0 {
                if face.len() == 3 && face_id % stride == 0 {
                    for &src in &face {
                        let mapped = if let Some(&m) = remap.get(&src) {
                            m
                        } else {
                            let m = new_points.len() as i32;
                            remap.insert(src, m);
                            new_points.push(points[src as usize]);
                            if let (Some(t_in), Some(ref mut t_out)) = (tex, new_tex.as_mut()) {
                                t_out.push(t_in[src as usize]);
                            }
                            m
                        };
                        new_index.push(mapped);
                    }
                    new_index.push(-1);
                }
                face.clear();
                face_id += 1;
            } else {
                face.push(idx);
            }
        }
        if new_points.is_empty() || new_index.is_empty() {
            return (
                points.to_vec(),
                tex.map(|t| t.to_vec()),
                coord_index.to_vec(),
            );
        }
        (new_points, new_tex, new_index)
    }

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
        self.stream_next_tick = Some(now + Duration::from_millis(STREAM_STEP_MS));

        let mut remaining = Vec::new();
        let mut any_change = false;
        let mut finished_patches: Vec<FullResPatch> = Vec::new();

        for mut patch in self.full_res_patches.drain(..) {
            let start = *patch.stream_start.get_or_insert(now);
            let t_s = start.elapsed().as_secs_f64();

            let budget = Self::gaussian_triangle_budget(t_s, patch.total_full_tris);

            if budget >= patch.total_full_tris || t_s >= STREAM_DURATION_S {
                finished_patches.push(patch);
                continue;
            }

            if let Some(target_stage) = patch.stage_for_budget(budget) {
                if target_stage > patch.current_stage {
                    patch.apply_stage_to_graph(&mut self.graph, target_stage);
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
            patch.apply_full_to_graph(&mut self.graph);
            log::warn!(
                "Full-resolution mesh restored: points={}, indices={}, ~{}K triangles",
                total_points,
                total_indices,
                total_tris / 1000,
            );
        }

        self.full_res_patches = remaining;

        if any_change || n_finished > 0 {
            if let Some(renderer) = &mut self.renderer {
                renderer.invalidate_mesh_cache();
            }
            self.collector.invalidate_mesh_cache();
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

    fn collect_view_projection(&self) -> (Mat4, Mat4) {
        let mut collector = RenderCollector::new();
        for &root in self.graph.roots() {
            collector.traverse(&self.graph, root);
        }
        (collector.view_matrix, collector.projection_matrix)
    }

    fn build_pick_ray(&self) -> Option<Ray> {
        let window = self.window.as_ref()?;
        let size = window.inner_size();
        let (view, proj) = self.collect_view_projection();
        Some(Ray::from_screen_point(
            self.cursor_pos.0 as f32,
            self.cursor_pos.1 as f32,
            size.width as f32,
            size.height as f32,
            view,
            proj,
        ))
    }

    fn do_pick(&mut self) {
        let Some(ray) = self.build_pick_ray() else {
            return;
        };

        let mut picker = rc3d_actions::RayPickAction::new(ray);
        for &root in self.graph.roots() {
            picker.apply(&self.graph, root);
        }

        if let Some(hit) = picker.hits.first() {
            self.graph.toggle_selection(hit.node);
            log::info!(
                "Pick hit: node={:?}, point={:?}, selected={}",
                hit.node, hit.point, self.graph.is_selected(hit.node)
            );
            if let Some(cb) = &mut self.on_pick {
                cb(&mut self.graph, hit.node, hit.point);
            }
        } else {
            log::info!("Pick miss (no hit)");
            if !self.shift_pressed {
                self.graph.clear_selection();
            }
        }
    }

    fn do_measure_pick(&mut self) {
        let Some(ray) = self.build_pick_ray() else {
            return;
        };

        let mut picker = rc3d_actions::RayPickAction::new(ray);
        for &root in self.graph.roots() {
            picker.apply(&self.graph, root);
        }

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
