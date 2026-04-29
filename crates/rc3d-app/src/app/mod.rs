mod measurement;
mod streaming_lod;
mod gizmo_support;
mod box_select;

use rc3d_core::{math::{Mat4, Quat, Vec3}, Aabb, DisplayMode};
use rc3d_gizmo::GizmoMode;
use rc3d_actions::{
    Action, CommandHistory, Ray,
    SectionPlaneAction, SetRotationCommand, SetScaleCommand, SetTranslationCommand,
    update_all_lod_nodes,
};
use rc3d_render::{AdaptiveControl, DrawCall, FrameStats, Renderer};
use winit::event::{MouseButton, MouseScrollDelta, WindowEvent};
use rc3d_scene::SceneGraph;
use std::sync::mpsc::TryRecvError;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    event_loop::ActiveEventLoop,
    window::WindowAttributes,
};

use crate::camera_controller::CameraController;
use crate::viewport_camera::ViewportCameraSet;
use crate::world::World;
use rc3d_core::NodeId;
use streaming_lod::{FullResPatch, apply_decimated_preview, gaussian_triangle_budget, stream_step_ms};

type PickCallback = Box<dyn FnMut(&mut SceneGraph, rc3d_core::NodeId, Vec3)>;
const APPROX_VERTEX_BYTES: usize = 48;
const MAX_SAFE_VERTEX_BUFFER_BYTES: usize = 240 * 1024 * 1024;

struct FpsTracker {
    samples: std::collections::VecDeque<f32>,
    sum: f32,
    capacity: usize,
    ema_fps: f32,
    last_log: Instant,
}

impl FpsTracker {
    fn new(capacity: usize) -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(capacity),
            sum: 0.0,
            capacity,
            ema_fps: 0.0,
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
        let inst_fps = if frame_time_ms > 0.0 { 1000.0 / frame_time_ms } else { 0.0 };
        if self.ema_fps <= 0.0 {
            self.ema_fps = inst_fps;
        } else {
            let alpha = 0.08_f32;
            self.ema_fps += (inst_fps - self.ema_fps) * alpha;
        }
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

    fn smoothed_fps(&self) -> f32 {
        if self.ema_fps > 0.0 { self.ema_fps } else { self.fps() }
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
    pub camera_controller: Option<CameraController>,  // legacy; prefer viewport_cameras
    pub viewport_cameras: ViewportCameraSet,
    pub on_pick: Option<PickCallback>,
    cursor_pos: (f64, f64),
    shift_pressed: bool,
    ctrl_pressed: bool,
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
    panel_overlay_text_hook: Option<Box<dyn Fn() -> String>>,
    panel_overlay_key_hook: Option<Box<dyn FnMut(winit::keyboard::KeyCode)>>,
    panel_overlay_mouse_hook: Option<Box<dyn FnMut(f32, f32, u32, u32) -> bool>>,
    adaptive_quality_mode: AdaptiveQualityMode,
    adaptive_last_interaction: Instant,
    continuous_redraw: bool,
    gizmo: rc3d_gizmo::Gizmo,
    gizmo_dragging: bool,
    gizmo_pending_transform: Option<(NodeId, Mat4)>,
    command_history: CommandHistory,
    /// Axis-aligned clip toggles (merged with `SectionPlane` from the graph each frame).
    axis_clip: [bool; 3],
    box_select_drag: bool,
    box_select_anchor: (f32, f32),
    section_edit_mode: bool,
    view_split_drag: bool,
    last_camera_eye: Vec3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveQualityMode {
    Off,
    On,
    AutoIdleLock,
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
            viewport_cameras: ViewportCameraSet::new(),
            on_pick: None,
            cursor_pos: (0.0, 0.0),
            shift_pressed: false,
            ctrl_pressed: false,
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
            fps_tracker: FpsTracker::new(120),
            pending_graph_rx: None,
            graph_load_hook: None,
            panel_overlay_text_hook: None,
            panel_overlay_key_hook: None,
            panel_overlay_mouse_hook: None,
            adaptive_quality_mode: AdaptiveQualityMode::On,
            adaptive_last_interaction: Instant::now(),
            continuous_redraw: true,
            gizmo: rc3d_gizmo::Gizmo::new(),
            gizmo_dragging: false,
            gizmo_pending_transform: None,
            command_history: CommandHistory::new(128),
            axis_clip: [false, false, false],
            box_select_drag: false,
            box_select_anchor: (0.0, 0.0),
            section_edit_mode: false,
            view_split_drag: false,
            last_camera_eye: Vec3::new(0.0, 0.0, 5.0),
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
                self.viewport_cameras.cameras.clear();
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

    pub fn with_panel_overlay_text_hook(
        mut self,
        hook: impl Fn() -> String + 'static,
    ) -> Self {
        self.panel_overlay_text_hook = Some(Box::new(hook));
        self
    }

    pub fn with_panel_overlay_key_hook(
        mut self,
        hook: impl FnMut(winit::keyboard::KeyCode) + 'static,
    ) -> Self {
        self.panel_overlay_key_hook = Some(Box::new(hook));
        self
    }

    pub fn with_panel_overlay_mouse_hook(
        mut self,
        hook: impl FnMut(f32, f32, u32, u32) -> bool + 'static,
    ) -> Self {
        self.panel_overlay_mouse_hook = Some(Box::new(hook));
        self
    }

    pub fn with_adaptive_quality_mode(mut self, mode: AdaptiveQualityMode) -> Self {
        self.adaptive_quality_mode = mode;
        self
    }

    pub fn with_continuous_redraw(mut self, enabled: bool) -> Self {
        self.continuous_redraw = enabled;
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
                self.adaptive_last_interaction = Instant::now();
                let size = *physical_size;
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.adaptive_last_interaction = Instant::now();
                self.cursor_pos = (position.x, position.y);
                if !self.measurement_mode {
                    self.editor_on_cursor_moved();
                }
            }
            WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    state: winit::event::ElementState::Pressed,
                    physical_key: key,
                    ..
                },
                ..
            } => {
                if let winit::keyboard::PhysicalKey::Code(code) = key {
                    self.adaptive_last_interaction = Instant::now();
                    if let Some(hook) = &mut self.panel_overlay_key_hook {
                        hook(*code);
                    }
                }
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
                        self.axis_clip[0] = !self.axis_clip[0];
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyY) => {
                        if self.ctrl_pressed {
                            let _ = self.command_history.redo(&mut self.world.graph);
                        } else {
                            self.axis_clip[1] = !self.axis_clip[1];
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyZ) => {
                        if self.ctrl_pressed {
                            let _ = self.command_history.undo(&mut self.world.graph);
                        } else {
                            self.axis_clip[2] = !self.axis_clip[2];
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyF) => {
                        self.fit_selection_to_view();
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyT) => {
                        self.gizmo.mode = GizmoMode::Translate;
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyR) => {
                        self.gizmo.mode = GizmoMode::Rotate;
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyG) => {
                        self.gizmo.mode = GizmoMode::Scale;
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyP) => {
                        self.section_edit_mode = !self.section_edit_mode;
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::BracketLeft) => {
                        if self.section_edit_mode {
                            self.nudge_section_planes(-0.04);
                        }
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::BracketRight) => {
                        if self.section_edit_mode {
                            self.nudge_section_planes(0.04);
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
                // Dispatch non-pointer events to scene-graph EventCallback nodes
                if let WindowEvent::KeyboardInput { event, .. } = &event {
                    if event.state == winit::event::ElementState::Pressed {
                        let evt = rc3d_actions::Event::KeyPress {
                            key: format!("{:?}", event.logical_key),
                        };
                        if let (Some(_r), Some(active)) = (&self.renderer, self.viewport_cameras.active()) {
                            let view = active.controller.view_matrix();
                            let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 1000.0);
                            let ctx = rc3d_actions::EventContext::new(evt, view, proj);
                            let mut action = rc3d_actions::HandleEventAction::new(ctx);
                            action.apply_non_pointer_only(&self.world.graph, self.world.graph.roots().first().copied().unwrap_or(NodeId::default()));
                            // Event callback nodes found; application can inspect action.event_callback_nodes
                        }
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.adaptive_last_interaction = Instant::now();
                self.shift_pressed = mods.state().shift_key();
                self.ctrl_pressed = mods.state().control_key();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.adaptive_last_interaction = Instant::now();
                    if *state == winit::event::ElementState::Pressed {
                        if let Some(hook) = &mut self.panel_overlay_mouse_hook {
                            if let Some(window) = self.window.as_ref() {
                                let size = window.inner_size();
                                if hook(
                                    self.cursor_pos.0 as f32,
                                    self.cursor_pos.1 as f32,
                                    size.width,
                                    size.height,
                                ) {
                                    if let Some(window) = &self.window {
                                        window.request_redraw();
                                    }
                                    return;
                                }
                            }
                        }
                        if self.measurement_mode {
                            self.do_measure_pick();
                        } else {
                            self.editor_on_left_down();
                        }
                    } else if *state == winit::event::ElementState::Released {
                        self.editor_on_left_up();
                    }
                }
            }
            WindowEvent::MouseWheel { .. } => {
                self.adaptive_last_interaction = Instant::now();
            }
            WindowEvent::RedrawRequested => {
                self.poll_pending_graph_load();
                if self.preview_mode_active && self.renderer.is_some() && self.stream_next_tick.is_none() {
                    self.stream_next_tick = Some(Instant::now() + Duration::from_millis(stream_step_ms()));
                }
                self.tick_mesh_stream();

                // Snapshot viewport layout for camera updates (before renderer is mutably borrowed)
                if !self.viewport_cameras.cameras.is_empty() {
                    let viewports: Vec<_> = self.renderer.as_ref().map(|r| {
                        r.viewport_layout.viewports.iter().map(|v| {
                            (v.id, v.rect.aspect())
                        }).collect::<Vec<_>>()
                    }).unwrap_or_default();
                    for vc in &self.viewport_cameras.cameras {
                        if let Some(&(_, aspect)) = viewports.iter().find(|&&(id, _)| id == vc.viewport_id) {
                            vc.controller.update_camera_node(&mut self.world.graph, vc.camera_node, aspect);
                        }
                    }
                }

                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    let roots_lod: Vec<NodeId> = self.world.graph.roots().to_vec();
                    for &r in &roots_lod {
                        update_all_lod_nodes(&mut self.world.graph, r, self.last_camera_eye);
                    }
                    {
                        let mut section = SectionPlaneAction::new();
                        for &r in &roots_lod {
                            section.apply(&self.world.graph, r);
                        }
                        let mut merged = section.planes;
                        if self.axis_clip[0] {
                            merged.push([1.0, 0.0, 0.0, 0.0]);
                        }
                        if self.axis_clip[1] {
                            merged.push([0.0, 1.0, 0.0, 0.0]);
                        }
                        if self.axis_clip[2] {
                            merged.push([0.0, 0.0, 1.0, 0.0]);
                        }
                        renderer.set_clip_planes(merged);
                    }
                    self.world.evaluate_engines();

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
                    self.last_camera_eye = self.world.collector.camera_pos;
                    gizmo_support::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);

                    if !self.measurements.is_empty() {
                        let vp = self.world.collector.projection_matrix * self.world.collector.view_matrix;
                        let depth_reversed_z = rc3d_core::depth_reversed_z_from_projection(
                            self.world.collector.projection_matrix,
                        );
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
                                opacity: 1.0,
                                albedo_path: None,
                                aabb: None,
                                display_mode: DisplayMode::ShadedWithEdges,
                                selected: false,
                                overlay_color: Some([1.0, 1.0, 0.0, 1.0]),
                                mesh_hash: None,
                                meshlet_data: None,
                                projection_orthographic: self.world.collector.projection_orthographic,
                                depth_reversed_z,
                            });
                        }
                    }
                    if self.gizmo.visible {
                        let vp = self.world.collector.projection_matrix * self.world.collector.view_matrix;
                        let depth_reversed_z = rc3d_core::depth_reversed_z_from_projection(
                            self.world.collector.projection_matrix,
                        );
                        for (line_verts, color) in self.gizmo.generate_lines() {
                            let mut ep: Vec<[f32; 3]> = Vec::new();
                            for c in line_verts.chunks_exact(2) {
                                ep.push(c[0].position);
                                ep.push(c[1].position);
                            }
                            if !ep.is_empty() {
                                self.world.collector.draw_calls.push(DrawCall {
                                    vertices: Arc::new(Vec::new()),
                                    indices: None,
                                    edge_positions: Arc::new(ep),
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
                                    opacity: 1.0,
                                    albedo_path: None,
                                    aabb: None,
                                    display_mode: DisplayMode::ShadedWithEdges,
                                    selected: false,
                                    overlay_color: Some(color),
                                    mesh_hash: None,
                                    meshlet_data: None,
                                    projection_orthographic: self.world.collector.projection_orthographic,
                                    depth_reversed_z,
                                });
                            }
                        }
                    }

                    if !self.world.collector.draw_calls.is_empty() {
                        let stats = renderer.render_draw_calls(&self.world.collector.draw_calls, &self.world.graph);
                        let now = Instant::now();
                        let frame_time_ms =
                            now.duration_since(self.last_frame_time).as_secs_f32() * 1000.0;
                        self.last_frame_time = now;
                        self.fps_tracker.push(frame_time_ms);
                        let idle_for_secs = self.adaptive_last_interaction.elapsed().as_secs_f32();
                        let has_dynamic_scene = self.world.engines.is_some();
                        let allow_downgrade = idle_for_secs < 0.35 || has_dynamic_scene;
                        let lock_idle = idle_for_secs >= 2.0 && !has_dynamic_scene;
                        let adaptive_control = match self.adaptive_quality_mode {
                            AdaptiveQualityMode::Off => AdaptiveControl::Disabled,
                            AdaptiveQualityMode::On => AdaptiveControl::Dynamic { allow_downgrade },
                            AdaptiveQualityMode::AutoIdleLock => {
                                if lock_idle {
                                    AdaptiveControl::Locked
                                } else {
                                    AdaptiveControl::Dynamic { allow_downgrade }
                                }
                            }
                        };
                        renderer.report_frame_time_ms(frame_time_ms, adaptive_control);
                        let mut mode_name = format!("{:?} | IBL:{}", renderer.display_mode(), renderer.ibl_preset_name());
                        if let Some(text_hook) = &self.panel_overlay_text_hook {
                            let overlay = text_hook();
                            if !overlay.is_empty() {
                                mode_name.push('\n');
                                mode_name.push_str(&overlay);
                            }
                        }
                        renderer.update_hud(
                            self.fps_tracker.smoothed_fps(),
                            self.fps_tracker.average_frame_ms(),
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
                        match renderer.surface.get_current_texture() {
                            Ok(tex) => drop(tex),
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                // Surface needs reconfiguration — handled on next resize event
                            }
                            Err(e) => log::warn!("Surface error during no-op frame: {:?}", e),
                        }
                    }
                    if self.continuous_redraw && self.world.engines.is_some() {
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

        // Camera control (legacy single-camera path)
        let block_orbit = self.gizmo_dragging || self.box_select_drag || self.view_split_drag;
        if let Some(ref mut ctrl) = self.camera_controller {
            if !self.measurement_mode && !block_orbit {
                Self::dispatch_camera_event(ctrl, &event, &self.cursor_pos);
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

        // Viewport camera set (new multi-viewport path)
        if !self.viewport_cameras.cameras.is_empty() && !self.measurement_mode && !block_orbit {
            if let Some(vc) = self.viewport_cameras.active_mut() {
                Self::dispatch_camera_event(&mut vc.controller, &event, &self.cursor_pos);
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

    fn dispatch_camera_event(
        ctrl: &mut CameraController,
        event: &WindowEvent,
        cursor_pos: &(f64, f64),
    ) {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                match (button, state) {
                    (MouseButton::Left, winit::event::ElementState::Pressed) => {
                        ctrl.orbiting = true;
                    }
                    (MouseButton::Left, winit::event::ElementState::Released) => {
                        ctrl.orbiting = false;
                    }
                    (MouseButton::Right, winit::event::ElementState::Pressed) => {
                        ctrl.panning = true;
                    }
                    (MouseButton::Right, winit::event::ElementState::Released) => {
                        ctrl.panning = false;
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let dx = (position.x - cursor_pos.0) as f32 * 0.005;
                let dy = (position.y - cursor_pos.1) as f32 * 0.005;
                if ctrl.orbiting {
                    ctrl.orbit(dx, dy);
                }
                if ctrl.panning {
                    // Mouse delta in pixels, scaled for pan
                    let dx = (position.x - cursor_pos.0) as f32;
                    let dy = (position.y - cursor_pos.1) as f32;
                    ctrl.pan(dx, dy);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };
                ctrl.zoom(scroll);
            }
            _ => {}
        }
    }

    fn do_pick(&mut self) {
        let Some(ray) = self.build_pick_ray() else { return };
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
        let Some(ray) = self.build_pick_ray() else { return };
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

    fn build_pick_ray(&self) -> Option<Ray> {
        let w = self.window.as_ref()?;
        let s = w.inner_size();
        if !self.viewport_cameras.cameras.is_empty() {
            if let (Some(r), Some(vc)) = (self.renderer.as_ref(), self.viewport_cameras.active()) {
                if let Some(avp) = r
                    .viewport_layout
                    .viewports
                    .iter()
                    .find(|v| v.id == vc.viewport_id)
                {
                    let (v, p) = gizmo_support::pick_view_proj(&self.world.graph, vc, avp);
                    return Some(gizmo_support::pick_ray_in_viewport(self.cursor_pos, avp, v, p));
                }
            }
        }
        if let Some(c) = &self.camera_controller {
            return Some(gizmo_support::pick_ray_full_window(
                self.cursor_pos,
                (s.width, s.height),
                c,
            ));
        }
        let (v, p) = self.active_camera_matrices(s.width as f32, s.height as f32);
        Some(Ray::from_screen_point(
            self.cursor_pos.0 as f32,
            self.cursor_pos.1 as f32,
            s.width as f32,
            s.height as f32,
            v,
            p,
        ))
    }

    fn editor_on_left_down(&mut self) {
        if let (Some(r), _w) = (&mut self.renderer, &self.window) {
            if r.viewport_layout.viewports.len() > 1 {
                if let Some(vp) = r
                    .viewport_layout
                    .viewport_at(self.cursor_pos.0 as f32, self.cursor_pos.1 as f32)
                {
                    self.viewport_cameras
                        .set_active(vp.id, &mut r.viewport_layout);
                }
            }
        }
        if self.ctrl_pressed {
            self.box_select_drag = true;
            self.box_select_anchor = (self.cursor_pos.0 as f32, self.cursor_pos.1 as f32);
            return;
        }
        if let Some(ray) = self.build_pick_ray() {
            gizmo_support::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);
            if self.gizmo.visible {
                if let Some((h, _)) = self.gizmo.hit_test(&ray) {
                    if let Some(n) = self.gizmo.target_node {
                        if let Some(e) = self.world.graph.get(n) {
                            if let rc3d_scene::NodeData::Transform(t) = &e.data {
                                let old = Mat4::from_scale_rotation_translation(
                                    t.scale,
                                    Quat::from_mat4(&t.rotation),
                                    t.translation,
                                );
                                self.gizmo_pending_transform = Some((n, old));
                            }
                        }
                    }
                    self.gizmo.start_drag(&ray, h);
                    self.gizmo_dragging = true;
                    return;
                }
            }
        }
        let can_pick = self.camera_controller.is_none() && self.viewport_cameras.cameras.is_empty()
            || self.shift_pressed;
        if can_pick {
            self.do_pick();
            gizmo_support::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);
        }
    }

    fn editor_on_left_up(&mut self) {
        if self.view_split_drag {
            self.view_split_drag = false;
        }
        if self.box_select_drag {
            self.box_select_drag = false;
            if let (Some(r), Some(w)) = (self.renderer.as_ref(), self.window.as_ref()) {
                let s = w.inner_size();
                let roots = self.world.graph.roots().to_vec();
                if !self.viewport_cameras.cameras.is_empty() {
                    if let (Some(vc), Some(_)) = (self.viewport_cameras.active(), self.build_pick_ray()) {
                        if let Some(avp) = r
                            .viewport_layout
                            .viewports
                            .iter()
                            .find(|v| v.id == vc.viewport_id)
                        {
                            let (v, p) = gizmo_support::pick_view_proj(&self.world.graph, vc, avp);
                            for &root in &roots {
                                box_select::select_nodes_in_viewport_box(
                                    &mut self.world.graph,
                                    root,
                                    self.box_select_anchor,
                                    (self.cursor_pos.0 as f32, self.cursor_pos.1 as f32),
                                    avp,
                                    v,
                                    p,
                                );
                            }
                        }
                    }
                } else {
                    let (v, p) = self.active_camera_matrices(s.width as f32, s.height as f32);
                    for &root in &roots {
                        box_select::select_nodes_in_screen_box(
                            &mut self.world.graph,
                            root,
                            self.box_select_anchor,
                            (self.cursor_pos.0 as f32, self.cursor_pos.1 as f32),
                            s.width as f32,
                            s.height as f32,
                            v,
                            p,
                        );
                    }
                }
                gizmo_support::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);
            }
            return;
        }
        if self.gizmo_dragging {
            self.gizmo.end_drag();
            if let Some((n, old_mat)) = self.gizmo_pending_transform.take() {
                if let Some(e) = self.world.graph.get(n) {
                    if let rc3d_scene::NodeData::Transform(t) = &e.data {
                        let new_mat = Mat4::from_scale_rotation_translation(
                            t.scale,
                            Quat::from_mat4(&t.rotation),
                            t.translation,
                        );
                        let (old_scale, _old_rot, old_trans) = old_mat.to_scale_rotation_translation();
                        let (new_scale, _new_rot, new_trans) = new_mat.to_scale_rotation_translation();
                        if (new_trans - old_trans).length_squared() > 1e-8 {
                            self.command_history.execute(
                                Box::new(SetTranslationCommand {
                                    node: n,
                                    old_value: old_trans,
                                    new_value: new_trans,
                                }),
                                &mut self.world.graph,
                            );
                        }
                        if (new_scale - old_scale).length_squared() > 1e-8 {
                            self.command_history.execute(
                                Box::new(SetScaleCommand {
                                    node: n,
                                    old_value: old_scale,
                                    new_value: new_scale,
                                }),
                                &mut self.world.graph,
                            );
                        }
                        let old_rot_cols = Mat4::from_quat(_old_rot).to_cols_array_2d();
                        let new_rot_cols = Mat4::from_quat(_new_rot).to_cols_array_2d();
                        if old_rot_cols != new_rot_cols {
                            self.command_history.execute(
                                Box::new(SetRotationCommand {
                                    node: n,
                                    old_value: Mat4::from_quat(_old_rot),
                                    new_value: Mat4::from_quat(_new_rot),
                                }),
                                &mut self.world.graph,
                            );
                        }
                    }
                }
            }
            self.gizmo_dragging = false;
        }
    }

    fn fit_selection_to_view(&mut self) {
        if self.world.graph.selected_nodes().is_empty() {
            return;
        }
        use rc3d_actions::GetBoundingBoxAction;
        let mut aabb: Option<Aabb> = None;
        for &id in self.world.graph.selected_nodes() {
            let mut a = GetBoundingBoxAction::new();
            a.apply(&self.world.graph, id);
            if a.bounding_box.min.x <= a.bounding_box.max.x {
                aabb = Some(match aabb {
                    Some(p) => p.union(&a.bounding_box),
                    None => a.bounding_box,
                });
            }
        }
        if let Some(b) = aabb {
            if let Some(cc) = &mut self.camera_controller {
                cc.fit_bounds(&b, 60.0f32.to_radians());
            } else if let Some(vc) = self.viewport_cameras.active_mut() {
                vc.controller.fit_bounds(&b, 60.0f32.to_radians());
            }
        }
    }

    fn nudge_section_planes(&mut self, d: f32) {
        for &root in &self.world.graph.roots().to_vec() {
            nudge_sp_rec(&mut self.world.graph, root, d);
        }
    }

    fn editor_on_cursor_moved(&mut self) {
        if self.gizmo_dragging {
            if let Some(ray) = self.build_pick_ray() {
                if let Some((n, old_mat)) = &self.gizmo_pending_transform {
                    if let Some(d) = self.gizmo.drag_delta(
                        &ray,
                        Mat4::IDENTITY,
                        Mat4::IDENTITY,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ) {
                        let new_mat = d * *old_mat;
                        let (scale, rot, trans) = new_mat.to_scale_rotation_translation();
                        if let Some(e) = self.world.graph.get_mut(*n) {
                            if let rc3d_scene::NodeData::Transform(t) = &mut e.data {
                                t.translation = trans;
                                t.rotation = Mat4::from_quat(rot);
                                t.scale = scale;
                            }
                        }
                    }
                }
            }
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            return;
        }
        if !self.gizmo.visible {
            return;
        }
        if self.renderer.is_some() {
            if let (Some(ray), Some(_r)) = (self.build_pick_ray(), self.renderer.as_ref()) {
                gizmo_support::sync_gizmo_from_selection(&mut self.gizmo, &self.world.graph);
                self.gizmo.hovered = self.gizmo.hit_test(&ray).map(|(h, _)| h);
            }
        }
    }

    fn active_camera_matrices(&self, width: f32, height: f32) -> (Mat4, Mat4) {
        let aspect = width / height.max(1.0);
        let proj = Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0);
        if let Some(vc) = self.viewport_cameras.active() {
            (vc.controller.view_matrix(), proj)
        } else if let Some(ctrl) = &self.camera_controller {
            (ctrl.view_matrix(), proj)
        } else {
            (Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y), proj)
        }
    }
}

fn nudge_sp_rec(graph: &mut rc3d_scene::SceneGraph, id: NodeId, d: f32) {
    use rc3d_scene::node_data::NodeData;
    let children: Vec<_> = graph
        .get(id)
        .map(|e| e.children.to_vec())
        .unwrap_or_default();
    if let Some(e) = graph.get_mut(id) {
        if let NodeData::SectionPlane(sp) = &mut e.data {
            sp.plane[3] += d;
        }
    }
    for c in children {
        nudge_sp_rec(graph, c, d);
    }
}
