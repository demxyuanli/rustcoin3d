use rc3d_actions::Ray;
use rc3d_core::{math::{Mat4, Vec3}, DisplayMode};
use rc3d_engine::EngineRegistry;
use rc3d_render::{DrawCall, FrameStats, RenderCollector, Renderer};
use rc3d_scene::{NodeData, SceneGraph};
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver};
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
const PREVIEW_TRIANGLE_THRESHOLD: usize = 1_000_000;
const PREVIEW_TARGET_TRIANGLES: usize = 250_000;
const PREVIEW_MIN_MS: u128 = 1_500;
struct FullResPatch {
    coord_node: rc3d_core::NodeId,
    ifs_node: rc3d_core::NodeId,
    full_points: Vec<Vec3>,
    full_coord_index: Vec<i32>,
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
    full_res_ready_rx: Option<Receiver<()>>,
    preview_mode_active: bool,
    preview_started_at: Option<Instant>,
    initial_display_mode: DisplayMode,
    last_frame_time: Instant,
    fps_tracker: FpsTracker,
}

impl App {
    pub fn new(graph: SceneGraph) -> Self {
        let mut graph = graph;
        let full_res_patches = Self::apply_decimated_preview(&mut graph);
        let preview_mode_active = !full_res_patches.is_empty();
        let full_res_ready_rx = if preview_mode_active {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(800));
                let _ = tx.send(());
            });
            Some(rx)
        } else {
            None
        };
        if preview_mode_active {
            log::warn!(
                "Preview mode: {} patches decimated to {}K triangles, full-res restore after {}ms",
                full_res_patches.len(),
                PREVIEW_TARGET_TRIANGLES / 1000,
                PREVIEW_MIN_MS,
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
            full_res_ready_rx,
            preview_mode_active,
            preview_started_at: if preview_mode_active { Some(Instant::now()) } else { None },
            initial_display_mode: DisplayMode::ShadedWithEdges,
            last_frame_time: Instant::now(),
            fps_tracker: FpsTracker::new(60),
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
                self.try_promote_full_res();
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
                                aabb: None,
                                display_mode: DisplayMode::ShadedWithEdges,
                                selected: false,
                                overlay_color: Some([1.0, 1.0, 0.0, 1.0]),
                                mesh_hash: None,
                            });
                        }
                    }

                    if !self.collector.draw_calls.is_empty() {
                        let stats = renderer.render_draw_calls(&self.collector.draw_calls);
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
                            window.set_title("rustcoin3d [Preview]");
                        }
                    } else {
                        drop(renderer.surface.get_current_texture());
                    }
                    if self.engines.is_some() {
                        window.request_redraw();
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
    fn apply_decimated_preview(graph: &mut SceneGraph) -> Vec<FullResPatch> {
        let mut patches = Vec::new();
        for &root in graph.roots().to_vec().iter() {
            Self::collect_preview_patches(graph, root, &mut patches);
        }
        patches
    }

    fn collect_preview_patches(graph: &mut SceneGraph, node: rc3d_core::NodeId, patches: &mut Vec<FullResPatch>) {
        let children: Vec<rc3d_core::NodeId> = graph.children(node).to_vec();
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
                let (preview_points, preview_coord_index) =
                    Self::decimate_indexed_face_set(&full_points, &full_coord_index, PREVIEW_TARGET_TRIANGLES);
                if let Some(coord_mut) = graph.get_mut(coord_id) {
                    if let NodeData::Coordinate3(coord_mut_data) = &mut coord_mut.data {
                        coord_mut_data.point = preview_points;
                    }
                }
                if let Some(ifs_mut) = graph.get_mut(ifs_id) {
                    if let NodeData::IndexedFaceSet(ifs_mut_data) = &mut ifs_mut.data {
                        ifs_mut_data.coord_index = preview_coord_index;
                    }
                }
                patches.push(FullResPatch {
                    coord_node: coord_id,
                    ifs_node: ifs_id,
                    full_points,
                    full_coord_index,
                });
            }
        }
        for child in children {
            Self::collect_preview_patches(graph, child, patches);
        }
    }

    fn decimate_indexed_face_set(
        points: &[Vec3],
        coord_index: &[i32],
        target_triangles: usize,
    ) -> (Vec<Vec3>, Vec<i32>) {
        let tri_count = coord_index.iter().filter(|&&v| v == -1).count();
        if tri_count <= target_triangles || tri_count == 0 {
            return (points.to_vec(), coord_index.to_vec());
        }
        let stride = (tri_count / target_triangles).max(2);
        let mut new_points = Vec::new();
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
            return (points.to_vec(), coord_index.to_vec());
        }
        (new_points, new_index)
    }

    fn try_promote_full_res(&mut self) {
        let Some(rx) = &self.full_res_ready_rx else {
            return;
        };
        if self.full_res_patches.is_empty() {
            return;
        }
        if let Some(started) = self.preview_started_at {
            let elapsed = started.elapsed().as_millis();
            if elapsed < PREVIEW_MIN_MS {
                log::debug!("Full-res promotion waiting: {elapsed}ms < {PREVIEW_MIN_MS}ms");
                return;
            }
        }
        if rx.try_recv().is_err() {
            log::debug!("Full-res channel not ready yet");
            return;
        }
        let patch_count = self.full_res_patches.len();
        let mut total_points = 0usize;
        let mut total_indices = 0usize;
        for patch in self.full_res_patches.drain(..) {
            total_points += patch.full_points.len();
            total_indices += patch.full_coord_index.len();
            if let Some(coord_mut) = self.graph.get_mut(patch.coord_node) {
                if let NodeData::Coordinate3(coord_data) = &mut coord_mut.data {
                    coord_data.point = patch.full_points;
                }
            }
            if let Some(ifs_mut) = self.graph.get_mut(patch.ifs_node) {
                if let NodeData::IndexedFaceSet(ifs_data) = &mut ifs_mut.data {
                    ifs_data.coord_index = patch.full_coord_index;
                }
            }
        }
        let total_tris = total_indices / 4;
        self.full_res_ready_rx = None;
        self.preview_mode_active = false;
        self.preview_started_at = None;
        if let Some(renderer) = &mut self.renderer {
            renderer.invalidate_mesh_cache();
        }
        self.collector.invalidate_mesh_cache();
        log::warn!(
            "Full-resolution mesh restored: patches={}, points={}, indices={}, ~{}K triangles",
            patch_count, total_points, total_indices, total_tris / 1000,
        );
        if let Some(window) = &self.window {
            window.set_title("rustcoin3d");
            window.request_redraw();
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
