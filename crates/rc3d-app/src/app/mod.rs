mod app_state;
mod box_select;
mod editor_commands;
mod editor_interaction;
mod editor_session;
mod event_handler;
mod fps_tracker;
mod gizmo_support;
mod input_state;
mod lod_state;
mod measurement;
mod streaming_lod;

pub(crate) use app_state::AppState;
pub(crate) use editor_session::EditorSession;
pub(crate) use input_state::InputState;
pub(crate) use lod_state::LODState;

use rc3d_actions::{
    Action, CommandHistory, Event, EventContext, HandleEventAction, MarkupAction, Ray,
};
use rc3d_core::{
    math::{Mat4, Vec3},
    DisplayMode,
};
use rc3d_render::FrameStats;
use rc3d_scene::SceneGraph;
use std::sync::mpsc::TryRecvError;
use std::time::{Duration, Instant};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::{application::ApplicationHandler, event_loop::ActiveEventLoop};

use crate::adaptive_quality::AdaptiveQualityMode;
use crate::camera_controller::CameraController;
use crate::editor_ui::{EditorUi, EditorUiContext, RenderFeatureFlags};

use crate::viewport_camera::{ViewportCamera, ViewportCameraSet};
use crate::world::World;
use fps_tracker::FpsTracker;
use rc3d_core::NodeId;
use streaming_lod::{gaussian_triangle_budget, stream_step_ms, FullResPatch};

type PickCallback = Box<dyn FnMut(&mut SceneGraph, rc3d_core::NodeId, Vec3)>;
const APPROX_VERTEX_BYTES: usize = 48;
const MAX_SAFE_VERTEX_BUFFER_BYTES: usize = 128 * 1024 * 1024;
const MAX_SAFE_TRIANGLES: usize = 1_000_000;

fn patch_exceeds_full_restore_limits(patch: &FullResPatch) -> bool {
    let estimated_vertex_bytes = patch.full_points.len().saturating_mul(APPROX_VERTEX_BYTES);
    estimated_vertex_bytes > MAX_SAFE_VERTEX_BUFFER_BYTES
        || patch.total_full_tris > MAX_SAFE_TRIANGLES
}

pub struct App {
    pub state: AppState,
    pub editor: EditorSession,
    pub input: InputState,
    pub lod: LODState,
    // Cross-cutting hooks
    pub on_pick: Option<PickCallback>,
    pub pending_graph_rx: Option<std::sync::mpsc::Receiver<rc3d_core::EngineResult<SceneGraph>>>,
    pub graph_load_hook: Option<Box<dyn FnOnce(&mut App) + 'static>>,
    pub panel_overlay_text_hook: Option<Box<dyn Fn() -> String>>,
    pub panel_overlay_key_hook: Option<Box<dyn FnMut(winit::keyboard::KeyCode) -> bool>>,
    pub panel_overlay_mouse_hook: Option<Box<dyn FnMut(f32, f32, u32, u32) -> bool>>,
    pub pre_render_hook: Option<Box<dyn FnMut(&mut rc3d_render::Renderer)>>,
}

impl App {
    pub fn new(graph: SceneGraph) -> Self {
        let full_res_patches = Vec::new();
        let preview_mode_active = false;
        let app = Self {
            state: AppState {
                world: World::new(graph),
                renderer: None,
                window: None,
                camera_controller: None,
                viewport_cameras: ViewportCameraSet::new(),
                initial_display_mode: DisplayMode::Shaded,
                initial_post_effect_params: None,
                enable_hdr_post_processing: false,
                adaptive_quality_mode: AdaptiveQualityMode::On,
                adaptive_last_interaction: Instant::now(),
                // Event-driven by default so camera interaction uses inline render_interaction_frame()
                // (see request_redraw_from_camera). Opt in with with_continuous_redraw(true) for
                // idle animation / always-on redraw loops.
                continuous_redraw: false,
                last_frame_time: Instant::now(),
                last_frame_time_ms: 0.0,
                fps_tracker: FpsTracker::new(120),
                last_render_stats: FrameStats::default(),
                window_title: "rustcoin3d".into(),
                editor_ui: None,
                editor_ui_enabled: false,
                editor_commands: std::collections::VecDeque::new(),
                hidden_nodes: std::collections::HashSet::new(),
                last_camera_eye: Vec3::new(0.0, 0.0, 5.0),
                perf_mode_last: false,
                bg_settings: None,
                pending_effect_graph: None,
                dynamic_surfaces: Vec::new(),
                interaction_frame_rendered: false,
                last_inline_render_time: std::time::Instant::now(),
                interaction_render_count: 0,
            },
            editor: EditorSession {
                gizmo: rc3d_gizmo::Gizmo::new(),
                gizmo_dragging: false,
                gizmo_pending_transform: None,
                command_history: CommandHistory::new(128),
                markup_action: MarkupAction::new(),
                measurement_mode: false,
                measurement_type: None,
                measurement_first_point: None,
                measurements: Vec::new(),
                axis_clip: [false, false, false],
                grid_enabled: false,
                box_select_drag: false,
                box_select_anchor: (0.0, 0.0),
                section_edit_mode: false,
                view_split_drag: None,
                orbit_drag_viewport_id: None,
                left_orbit_drag_viewport_id: None,
                pan_drag_viewport_id: None,
                left_pick_arm_pos: None,
                left_drag_suppresses_pick: false,
            },
            input: InputState {
                cursor_pos: (0.0, 0.0),
                shift_pressed: false,
                ctrl_pressed: false,
            },
            lod: LODState {
                full_res_patches,
                preview_mode_active,
                stream_next_tick: None,
            },
            on_pick: None,
            pending_graph_rx: None,
            graph_load_hook: None,
            panel_overlay_text_hook: None,
            panel_overlay_key_hook: None,
            panel_overlay_mouse_hook: None,
            pre_render_hook: None,
        };
        log_scene_stats(&app.state.world.graph);
        app
    }

    pub fn with_window_title(mut self, title: impl Into<String>) -> Self {
        self.state.window_title = title.into();
        self
    }

    pub fn with_editor_ui(mut self, enabled: bool) -> Self {
        self.state.editor_ui_enabled = enabled;
        self
    }

    pub fn with_background(mut self, settings: rc3d_render::background::BgSettings) -> Self {
        self.state.bg_settings = Some(settings);
        self
    }

    pub fn set_pending_graph_receiver(
        &mut self,
        rx: std::sync::mpsc::Receiver<rc3d_core::EngineResult<SceneGraph>>,
    ) {
        self.pending_graph_rx = Some(rx);
    }

    pub fn set_graph_load_hook(&mut self, hook: impl FnOnce(&mut App) + 'static) {
        self.graph_load_hook = Some(Box::new(hook));
    }

    fn poll_pending_graph_load(&mut self) -> bool {
        let Some(rx) = self.pending_graph_rx.as_ref() else {
            return false;
        };
        match rx.try_recv() {
            Ok(Ok(graph)) => {
                self.state.world.graph = graph;
                self.state.viewport_cameras.cameras.clear();
                self.editor.orbit_drag_viewport_id = None;
                self.editor.left_orbit_drag_viewport_id = None;
                self.editor.pan_drag_viewport_id = None;
                self.editor.left_pick_arm_pos = None;
                self.editor.left_drag_suppresses_pick = false;
                self.pending_graph_rx = None;
                if let Some(renderer) = &mut self.state.renderer {
                    self.state.world.invalidate_caches(renderer);
                } else {
                    self.state.world.collector.invalidate_mesh_cache();
                }
                log::info!("Async scene load applied");
                if let Some(hook) = self.graph_load_hook.take() {
                    hook(self);
                }
                true
            }
            Ok(Err(e)) => {
                self.pending_graph_rx = None;
                log::error!("Async scene load failed: {}", e);
                false
            }
            Err(TryRecvError::Empty) => false,
            Err(TryRecvError::Disconnected) => {
                self.pending_graph_rx = None;
                log::warn!("Async scene load channel disconnected");
                false
            }
        }
    }

    pub fn with_camera_controller(mut self, controller: CameraController) -> Self {
        self.state.camera_controller = Some(controller);
        self
    }

    pub fn with_engines(mut self, engines: rc3d_engine::EngineRegistry) -> Self {
        // Engines advance only inside RedrawRequested; event-driven redraw would stall.
        // Callers can override with .with_continuous_redraw(false) after this (e.g. idle/power).
        if !engines.engines.is_empty() {
            self.state.continuous_redraw = true;
        }
        self.state.world.engines = Some(engines);
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
        &self.state.world.graph
    }

    pub fn scene_graph_mut(&mut self) -> &mut SceneGraph {
        &mut self.state.world.graph
    }

    pub fn with_initial_display_mode(mut self, mode: DisplayMode) -> Self {
        self.state.initial_display_mode = mode;
        self
    }

    pub fn with_hdr_post_processing(mut self, enabled: bool) -> Self {
        self.state.enable_hdr_post_processing = enabled;
        self
    }

    pub fn with_initial_post_effect_params(
        mut self,
        vignette: f32,
        chromatic_aberration: f32,
        bloom_strength: f32,
        grain: f32,
    ) -> Self {
        self.state.initial_post_effect_params =
            Some((vignette, chromatic_aberration, bloom_strength, grain));
        self
    }

    pub fn with_panel_overlay_text_hook(mut self, hook: impl Fn() -> String + 'static) -> Self {
        self.panel_overlay_text_hook = Some(Box::new(hook));
        self
    }

    pub fn with_panel_overlay_key_hook(
        mut self,
        hook: impl FnMut(winit::keyboard::KeyCode) -> bool + 'static,
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

    pub fn with_pre_render_hook(
        mut self,
        hook: impl FnMut(&mut rc3d_render::Renderer) + 'static,
    ) -> Self {
        self.pre_render_hook = Some(Box::new(hook));
        self
    }

    pub fn with_grid_enabled(mut self, enabled: bool) -> Self {
        self.editor.grid_enabled = enabled;
        self
    }

    pub fn with_adaptive_quality_mode(mut self, mode: AdaptiveQualityMode) -> Self {
        self.state.adaptive_quality_mode = mode;
        self
    }

    pub fn with_continuous_redraw(mut self, enabled: bool) -> Self {
        self.state.continuous_redraw = enabled;
        self
    }
}

/// Log scene node statistics at startup for performance baselining.
fn log_scene_stats(graph: &SceneGraph) {
    let mut total = 0u32;
    let mut type_counts: std::collections::HashMap<&str, u32> = Default::default();
    for &root in graph.roots() {
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if let Some(entry) = graph.get(id) {
                total += 1;
                *type_counts.entry(entry.data.type_name()).or_default() += 1;
                stack.extend(entry.children.iter().copied());
            }
        }
    }
    log::info!("Scene: {} nodes, {} roots", total, graph.roots().len());
    let mut types: Vec<(&str, u32)> = type_counts.into_iter().collect();
    rc3d_core::utils::sort::sort_by_count_desc(&mut types);
    for (name, count) in types.iter().take(10) {
        log::info!("  {}: {}", name, count);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_handler::resumed(self, event_loop);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let applied = self.poll_pending_graph_load();
        if let Some(w) = &self.state.window {
            if applied || self.pending_graph_rx.is_some() {
                w.request_redraw();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        event_handler::window_event(self, event_loop, _window_id, event);
    }
}

impl App {
    fn tick_mesh_stream(&mut self) {
        if self.lod.full_res_patches.is_empty() {
            return;
        }
        let Some(fire_at) = self.lod.stream_next_tick else {
            return;
        };
        let now = Instant::now();
        if now < fire_at {
            return;
        }
        self.lod.stream_next_tick = Some(now + Duration::from_millis(stream_step_ms()));

        let mut remaining = Vec::new();
        let mut any_change = false;
        let mut finished_patches: Vec<FullResPatch> = Vec::new();

        for mut patch in self.lod.full_res_patches.drain(..) {
            let start = *patch.stream_start.get_or_insert(now);
            let t_s = start.elapsed().as_secs_f64();

            let budget = gaussian_triangle_budget(t_s, patch.total_full_tris);
            let skip_full_restore = patch_exceeds_full_restore_limits(&patch);

            if budget >= patch.total_full_tris || t_s >= 2.2 {
                if skip_full_restore {
                    // Do not restore full-res mesh for very large models, but still
                    // advance to the highest available streamed stage before finishing.
                    if let Some(last_stage) = patch.final_stage_index() {
                        if last_stage > patch.current_stage {
                            patch.apply_stage_to_graph(&mut self.state.world.graph, last_stage);
                            let tris = patch.stage_tri_counts[last_stage];
                            log::info!(
                                "Mesh stream [Gaussian]: ~{}K tris at t={:.2}s (max staged LOD)",
                                tris / 1000,
                                t_s,
                            );
                            any_change = true;
                        }
                    }
                    let estimated_vertex_bytes =
                        patch.full_points.len().saturating_mul(APPROX_VERTEX_BYTES);
                    log::warn!(
                        "Skip full-resolution restore (estimated_vertex_bytes={}, triangles={}, limits: {}MB / {}M tris), keep max staged LOD",
                        estimated_vertex_bytes,
                        patch.total_full_tris,
                        MAX_SAFE_VERTEX_BUFFER_BYTES / (1024 * 1024),
                        MAX_SAFE_TRIANGLES / 1_000_000,
                    );
                    continue;
                }
                finished_patches.push(patch);
                continue;
            }

            if let Some(target_stage) = patch.stage_for_budget(budget) {
                if target_stage > patch.current_stage {
                    patch.apply_stage_to_graph(&mut self.state.world.graph, target_stage);
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
            let estimated_vertex_bytes =
                patch.full_points.len().saturating_mul(APPROX_VERTEX_BYTES);
            if patch_exceeds_full_restore_limits(&patch) {
                if let Some(last_stage) = patch.final_stage_index() {
                    if last_stage > patch.current_stage {
                        patch.apply_stage_to_graph(&mut self.state.world.graph, last_stage);
                        any_change = true;
                    }
                }
                log::warn!(
                    "Skip full-resolution restore (points={}, ~{}K tris, vertex_bytes={}, limits: {}MB / {}M tris)",
                    total_points,
                    total_tris / 1000,
                    estimated_vertex_bytes,
                    MAX_SAFE_VERTEX_BUFFER_BYTES / (1024 * 1024),
                    MAX_SAFE_TRIANGLES / 1_000_000,
                );
            } else {
                patch.apply_full_to_graph(&mut self.state.world.graph);
                log::warn!(
                    "Full-resolution mesh restored: points={}, indices={}, ~{}K triangles",
                    total_points,
                    total_indices,
                    total_tris / 1000,
                );
            }
        }

        self.lod.full_res_patches = remaining;

        if any_change || n_finished > 0 {
            if let Some(renderer) = &mut self.state.renderer {
                renderer.invalidate_mesh_cache();
            }
            self.state.world.collector.invalidate_mesh_cache();
        }

        if self.lod.full_res_patches.is_empty() {
            self.lod.preview_mode_active = false;
            self.lod.stream_next_tick = None;
            if let Some(window) = &self.state.window {
                window.set_title("rustcoin3d");
                window.request_redraw();
            }
        }
    }

    fn update_camera_recursive(
        ctrl: &CameraController,
        graph: &mut SceneGraph,
        node: NodeId,
        aspect: f32,
    ) {
        let Some(entry) = graph.get(node) else { return };
        if matches!(
            entry.data,
            rc3d_scene::NodeData::PerspectiveCamera(_)
                | rc3d_scene::NodeData::OrthographicCamera(_)
        ) {
            ctrl.update_camera_node(graph, node, aspect);
            return;
        }
        let children: Vec<NodeId> = entry.children.clone();
        for child in children {
            Self::update_camera_recursive(ctrl, graph, child, aspect);
        }
    }

    fn find_camera_projection(graph: &SceneGraph, node: NodeId) -> Option<Mat4> {
        let entry = graph.get(node)?;
        match &entry.data {
            rc3d_scene::NodeData::PerspectiveCamera(cam) => {
                return Some(cam.projection_matrix());
            }
            rc3d_scene::NodeData::OrthographicCamera(cam) => {
                return Some(cam.projection_matrix());
            }
            _ => {}
        }
        for &child in &entry.children {
            if let Some(p) = Self::find_camera_projection(graph, child) {
                return Some(p);
            }
        }
        None
    }

    /// Viewport id under the cursor (winit physical pixels), using the same viewport resolution rules as scene picking.
    pub(super) fn viewport_id_under_cursor(&self) -> Option<rc3d_render::viewport::ViewportId> {
        let r = self.state.renderer.as_ref()?;
        let cx = self.input.cursor_pos.0 as f32;
        let cy = self.input.cursor_pos.1 as f32;
        let vp = r
            .viewport_layout()
            .viewport_at(cx, cy)
            .or_else(|| r.viewport_layout().active())
            .or_else(|| r.viewport_layout().viewports.first())?;
        Some(vp.id)
    }

    /// winit physical pixels: viewport-local `(x, y)` + `(width, height)` and view/projection
    /// for the viewport under the cursor (or full surface when using legacy camera).
    fn pointer_pick_frame(
        &self,
        cx: f32,
        cy: f32,
        surface_w: u32,
        surface_h: u32,
    ) -> Option<(f32, f32, f32, f32, Mat4, Mat4)> {
        if !self.state.viewport_cameras.cameras.is_empty() {
            let r = self.state.renderer.as_ref()?;
            let vp_ref = r
                .viewport_layout()
                .viewport_at(cx, cy)
                .or_else(|| r.viewport_layout().active())
                .or_else(|| r.viewport_layout().viewports.first())?;
            if let Some(vc) = self.state.viewport_cameras.find(vp_ref.id) {
                let (v, p) = gizmo_support::pick_view_proj(&self.state.world.graph, vc, vp_ref);
                let lx = cx - vp_ref.rect.x as f32;
                let ly = cy - vp_ref.rect.y as f32;
                let vw = vp_ref.rect.width.max(1) as f32;
                let vh = vp_ref.rect.height.max(1) as f32;
                return Some((lx, ly, vw, vh, v, p));
            }
        }
        if let Some(ctrl) = &self.state.camera_controller {
            let vw = surface_w.max(1) as f32;
            let vh = surface_h.max(1) as f32;
            let aspect = vw / vh;
            let v = ctrl.view_matrix();
            let p = Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0);
            Some((cx, cy, vw, vh, v, p))
        } else {
            let vw = surface_w.max(1) as f32;
            let vh = surface_h.max(1) as f32;
            let (v, p) = self.active_camera_matrices(vw, vh);
            Some((cx, cy, vw, vh, v, p))
        }
    }

    pub(super) fn build_pointer_handle_event_context(&self, event: Event) -> Option<EventContext> {
        let w = self.state.window.as_ref()?;
        let s = w.inner_size();
        let cx = self.input.cursor_pos.0 as f32;
        let cy = self.input.cursor_pos.1 as f32;
        let (local_event, view, proj, pick_vp) = match &event {
            Event::MouseMove { dx, dy, .. } => {
                let (lx, ly, vw, vh, v, p) = self.pointer_pick_frame(cx, cy, s.width, s.height)?;
                (
                    Event::MouseMove {
                        x: lx,
                        y: ly,
                        dx: *dx,
                        dy: *dy,
                    },
                    v,
                    p,
                    Some((vw, vh)),
                )
            }
            Event::ButtonPress { button, .. } => {
                let (lx, ly, vw, vh, v, p) = self.pointer_pick_frame(cx, cy, s.width, s.height)?;
                (
                    Event::ButtonPress {
                        button: *button,
                        x: lx,
                        y: ly,
                    },
                    v,
                    p,
                    Some((vw, vh)),
                )
            }
            Event::ButtonRelease { button, .. } => {
                let (lx, ly, vw, vh, v, p) = self.pointer_pick_frame(cx, cy, s.width, s.height)?;
                (
                    Event::ButtonRelease {
                        button: *button,
                        x: lx,
                        y: ly,
                    },
                    v,
                    p,
                    Some((vw, vh)),
                )
            }
            _ => return None,
        };
        let mut ctx = EventContext::new(local_event, view, proj);
        ctx.pointer_pick_viewport = pick_vp;
        Some(ctx)
    }

    pub(super) fn dispatch_handle_event_for_pointer(&self, event: Event) {
        let Some(ctx) = self.build_pointer_handle_event_context(event) else {
            return;
        };
        let mut action = HandleEventAction::new(ctx);
        let root = self
            .state
            .world
            .graph
            .roots()
            .first()
            .copied()
            .unwrap_or(NodeId::default());
        action.apply(&self.state.world.graph, root);
        if let Some(n) = action.hit_node {
            log::trace!("HandleEventAction pointer: pick hit {:?}", n);
        }
    }

    pub(super) fn dispatch_handle_event_for_scroll(&self, dx: f32, dy: f32) {
        let Some(w) = self.state.window.as_ref() else {
            return;
        };
        let s = w.inner_size();
        let cx = self.input.cursor_pos.0 as f32;
        let cy = self.input.cursor_pos.1 as f32;
        let Some((_, _, _, _, view, proj)) = self.pointer_pick_frame(cx, cy, s.width, s.height)
        else {
            return;
        };
        let ctx = EventContext::new(Event::Scroll { dx, dy }, view, proj);
        let mut action = HandleEventAction::new(ctx);
        let root = self
            .state
            .world
            .graph
            .roots()
            .first()
            .copied()
            .unwrap_or(NodeId::default());
        action.apply(&self.state.world.graph, root);
        if !action.event_callback_nodes.is_empty() {
            log::trace!(
                "HandleEventAction scroll: {} EventCallback node(s) collected",
                action.event_callback_nodes.len()
            );
        }
    }

    pub(super) fn should_block_handle_event_pointer_dispatch(&self, for_cursor_move: bool) -> bool {
        if self.editor.view_split_drag.is_some() {
            return true;
        }
        if for_cursor_move && (self.editor.gizmo_dragging || self.editor.box_select_drag) {
            return true;
        }
        if for_cursor_move {
            let cam_active = self.state.camera_controller.as_ref().map_or(false, |c| {
                c.middle_orbit_held || c.left_orbit_held || c.panning
            }) || self.state.viewport_cameras.cameras.iter().any(|vc| {
                vc.controller.middle_orbit_held
                    || vc.controller.left_orbit_held
                    || vc.controller.panning
            });
            if cam_active {
                return true;
            }
        }
        false
    }

    fn viewport_camera_by_id_mut(
        &mut self,
        id: rc3d_render::viewport::ViewportId,
    ) -> Option<&mut ViewportCamera> {
        let i = self
            .state
            .viewport_cameras
            .cameras
            .iter()
            .position(|vc| vc.viewport_id == id)?;
        self.state.viewport_cameras.cameras.get_mut(i)
    }

    /// Multi-viewport camera: wheel uses cursor viewport; middle/left/right drag locks to press viewport until release.
    pub(super) fn dispatch_multi_viewport_camera(
        &mut self,
        event: &WindowEvent,
        cursor_pos: &(f64, f64),
        left_orbit_enabled: bool,
    ) {
        let active_id = self.state.viewport_cameras.active_viewport;
        let vid_cursor_resolved = || {
            self.viewport_id_under_cursor()
                .filter(|id| self.state.viewport_cameras.find(*id).is_some())
                .unwrap_or(active_id)
        };

        match event {
            WindowEvent::MouseInput { state, button, .. } => match (*button, *state) {
                (MouseButton::Middle, ElementState::Pressed) => {
                    let vid = vid_cursor_resolved();
                    self.editor.orbit_drag_viewport_id = Some(vid);
                    if let Some(vc) = self.viewport_camera_by_id_mut(vid) {
                        Self::dispatch_camera_event(
                            &mut vc.controller,
                            event,
                            cursor_pos,
                            left_orbit_enabled,
                        );
                    }
                }
                (MouseButton::Middle, ElementState::Released) => {
                    let vid = self.editor.orbit_drag_viewport_id.unwrap_or(active_id);
                    if let Some(vc) = self.viewport_camera_by_id_mut(vid) {
                        Self::dispatch_camera_event(
                            &mut vc.controller,
                            event,
                            cursor_pos,
                            left_orbit_enabled,
                        );
                    }
                    self.editor.orbit_drag_viewport_id = None;
                }
                (MouseButton::Left, ElementState::Pressed) if left_orbit_enabled => {
                    let vid = vid_cursor_resolved();
                    self.editor.left_orbit_drag_viewport_id = Some(vid);
                    if let Some(vc) = self.viewport_camera_by_id_mut(vid) {
                        Self::dispatch_camera_event(
                            &mut vc.controller,
                            event,
                            cursor_pos,
                            left_orbit_enabled,
                        );
                    }
                }
                (MouseButton::Left, ElementState::Released) if left_orbit_enabled => {
                    let vid = self.editor.left_orbit_drag_viewport_id.unwrap_or(active_id);
                    if let Some(vc) = self.viewport_camera_by_id_mut(vid) {
                        Self::dispatch_camera_event(
                            &mut vc.controller,
                            event,
                            cursor_pos,
                            left_orbit_enabled,
                        );
                    }
                    self.editor.left_orbit_drag_viewport_id = None;
                }
                (MouseButton::Right, ElementState::Pressed) => {
                    let vid = vid_cursor_resolved();
                    self.editor.pan_drag_viewport_id = Some(vid);
                    if let Some(vc) = self.viewport_camera_by_id_mut(vid) {
                        Self::dispatch_camera_event(
                            &mut vc.controller,
                            event,
                            cursor_pos,
                            left_orbit_enabled,
                        );
                    }
                }
                (MouseButton::Right, ElementState::Released) => {
                    let vid = self.editor.pan_drag_viewport_id.unwrap_or(active_id);
                    if let Some(vc) = self.viewport_camera_by_id_mut(vid) {
                        Self::dispatch_camera_event(
                            &mut vc.controller,
                            event,
                            cursor_pos,
                            left_orbit_enabled,
                        );
                    }
                    self.editor.pan_drag_viewport_id = None;
                }
                _ => {}
            },
            WindowEvent::CursorMoved { .. } => {
                let mut targ: Vec<rc3d_render::viewport::ViewportId> = Vec::new();
                for vid in [
                    self.editor.orbit_drag_viewport_id,
                    self.editor.left_orbit_drag_viewport_id,
                    self.editor.pan_drag_viewport_id,
                ]
                .into_iter()
                .flatten()
                {
                    if !targ.contains(&vid) {
                        targ.push(vid);
                    }
                }
                for id in targ {
                    if let Some(vc) = self.viewport_camera_by_id_mut(id) {
                        Self::dispatch_camera_event(
                            &mut vc.controller,
                            event,
                            cursor_pos,
                            left_orbit_enabled,
                        );
                    }
                }
            }
            WindowEvent::MouseWheel { .. } => {
                let vid = vid_cursor_resolved();
                if let Some(vc) = self.viewport_camera_by_id_mut(vid) {
                    Self::dispatch_camera_event(
                        &mut vc.controller,
                        event,
                        cursor_pos,
                        left_orbit_enabled,
                    );
                }
            }
            _ => {}
        }
    }

    /// Left-drag orbit when not using the full editor UI, measurement picks, or an `on_pick` demo hook.
    pub(super) fn camera_left_orbit_enabled(&self) -> bool {
        !self.state.editor_ui_enabled && !self.editor.measurement_mode && self.on_pick.is_none()
    }

    fn dispatch_camera_event(
        ctrl: &mut CameraController,
        event: &WindowEvent,
        cursor_pos: &(f64, f64),
        left_orbit_enabled: bool,
    ) {
        match event {
            WindowEvent::MouseInput { state, button, .. } => match (button, state) {
                (MouseButton::Middle, winit::event::ElementState::Pressed) => {
                    ctrl.middle_orbit_held = true;
                }
                (MouseButton::Middle, winit::event::ElementState::Released) => {
                    ctrl.middle_orbit_held = false;
                }
                (MouseButton::Left, winit::event::ElementState::Pressed) if left_orbit_enabled => {
                    ctrl.left_orbit_held = true;
                }
                (MouseButton::Left, winit::event::ElementState::Released) if left_orbit_enabled => {
                    ctrl.left_orbit_held = false;
                }
                (MouseButton::Right, winit::event::ElementState::Pressed) => {
                    ctrl.panning = true;
                }
                (MouseButton::Right, winit::event::ElementState::Released) => {
                    ctrl.panning = false;
                }
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                let dx = (position.x - cursor_pos.0) as f32 * 0.005;
                let dy = (position.y - cursor_pos.1) as f32 * 0.005;
                if ctrl.middle_orbit_held || ctrl.left_orbit_held {
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
        let Some(ray) = self.build_pick_ray() else {
            return;
        };
        let mut picker = rc3d_actions::RayPickAction::new(ray);
        rc3d_actions::apply_to_all_roots(&mut picker, &self.state.world.graph);

        if let Some(hit) = picker.hits.first() {
            self.state.world.graph.toggle_selection(hit.node);
            log::info!(
                "Pick hit: node={:?}, point={:?}, selected={}",
                hit.node,
                hit.point,
                self.state.world.graph.is_selected(hit.node)
            );
            if let Some(cb) = &mut self.on_pick {
                cb(&mut self.state.world.graph, hit.node, hit.point);
            }
        } else {
            log::info!("Pick miss (no hit)");
            if !self.input.shift_pressed {
                self.state.world.graph.clear_selection();
            }
        }
    }

    fn do_measure_pick(&mut self) {
        let Some(ray) = self.build_pick_ray() else {
            return;
        };
        let mut picker = rc3d_actions::RayPickAction::new(ray);
        rc3d_actions::apply_to_all_roots(&mut picker, &self.state.world.graph);

        if let Some(hit) = picker.hits.first() {
            let point = hit.point;
            match self.editor.measurement_first_point {
                None => {
                    self.editor.measurement_first_point = Some(point);
                    log::info!("Measurement point A: {:?}", point);
                }
                Some(first) => {
                    let dist = (point - first).length();
                    self.editor.measurements.push((first, point, dist));
                    self.editor.measurement_first_point = None;
                    log::info!(
                        "Measurement: A={:?} B={:?} distance={:.4}",
                        first,
                        point,
                        dist
                    );
                }
            }
        }
    }

    fn build_pick_ray(&self) -> Option<Ray> {
        let w = self.state.window.as_ref()?;
        let s = w.inner_size();
        let (lx, ly, vw, vh, v, p) = self.pointer_pick_frame(
            self.input.cursor_pos.0 as f32,
            self.input.cursor_pos.1 as f32,
            s.width,
            s.height,
        )?;
        Some(Ray::from_screen_point(lx, ly, vw, vh, v, p))
    }

    fn nudge_section_planes(&mut self, d: f32) {
        for &root in &self.state.world.graph.roots().to_vec() {
            nudge_sp_rec(&mut self.state.world.graph, root, d);
        }
    }

    fn active_camera_controller(&self) -> Option<&CameraController> {
        if let Some(vc) = self.state.viewport_cameras.active() {
            Some(&vc.controller)
        } else if let Some(ref ctrl) = self.state.camera_controller {
            Some(ctrl)
        } else {
            None
        }
    }

    fn active_camera_controller_mut(&mut self) -> Option<&mut CameraController> {
        if let Some(vc) = self.state.viewport_cameras.active_mut() {
            Some(&mut vc.controller)
        } else if let Some(ref mut ctrl) = self.state.camera_controller {
            Some(ctrl)
        } else {
            None
        }
    }

    fn is_camera_interacting(&self) -> bool {
        self.state.camera_controller.as_ref().map_or(false, |c| {
            c.middle_orbit_held || c.left_orbit_held || c.panning
        }) || self.state.viewport_cameras.cameras.iter().any(|vc| {
            vc.controller.middle_orbit_held
                || vc.controller.left_orbit_held
                || vc.controller.panning
        })
    }

    pub(super) fn prepare_editor_ui_frame(&mut self) {
        if !self.state.editor_ui_enabled {
            return;
        }
        let selected_set = self.state.world.graph.selected_nodes().clone();
        let selected_count = selected_set.len();
        let bookmarks: [(bool, &'static str); 9] = {
            let mut bm = [(false, ""); 9];
            if let Some(ctrl) = self.active_camera_controller_mut() {
                for (i, slot) in ctrl.bookmarks.iter().enumerate() {
                    if slot.is_some() {
                        bm[i] = (true, "saved");
                    }
                }
            }
            bm
        };
        if let (Some(ui), Some(window), Some(renderer)) = (
            &mut self.state.editor_ui,
            &self.state.window,
            &self.state.renderer,
        ) {
            let ui_ctx = EditorUiContext {
                selected: selected_set,
                display_mode_label: format!("{:?}", renderer.display_mode()),
                ibl_label: renderer.ibl_preset_name().to_string(),
                ibl_preset: renderer.ibl_preset,
                gizmo_mode: self.editor.gizmo.mode,
                layout_mode: renderer.viewport_layout().layout_mode,
                layout_mode_label: format!("{:?}", renderer.viewport_layout().layout_mode),
                active_viewport_label: format!("{:?}", renderer.viewport_layout().active_id),
                smoothed_fps: self.state.fps_tracker.smoothed_fps(),
                frame_time_ms: self.state.last_frame_time_ms,
                diagnostics: self.state.last_render_stats.diagnostics.clone(),
                hidden_nodes: self.state.hidden_nodes.clone(),
                render_features: RenderFeatureFlags {
                    taa: renderer.enable_taa,
                    motion_blur: renderer.enable_motion_blur,
                    ssr: renderer.enable_ssr,
                    color_grading: renderer.enable_color_grading,
                    dof: renderer.enable_dof,
                    volumetric_fog: renderer.enable_volumetric_fog,
                    cluster_lights: renderer.enable_cluster_lights,
                    omni_shadows: renderer.enable_omni_shadows,
                    xray: renderer.xray_mode,
                },
                hdr_enabled: renderer.hdr_post_processing,
                vsync_enabled: matches!(
                    renderer.config.present_mode,
                    wgpu::PresentMode::AutoVsync
                ),
                grid_enabled: self.editor.grid_enabled,
                hud_enabled: renderer.hud_enabled,
                outline_width: renderer.outline_width,
                outline_color: renderer.outline_color,
                xray_mode: renderer.xray_mode,
                adaptive_quality_mode: self.state.adaptive_quality_mode,
                adaptive_quality_name: renderer.adaptive_quality_name().to_string(),
                cad_display_tier: renderer.requested_display_tier(),
                bookmarks,
                selected_count,
            };
            ui.run(window, &self.state.world.graph, renderer, &ui_ctx);
            for cmd in ui.take_commands() {
                self.state.editor_commands.push_back(cmd);
            }
        }
        self.apply_editor_commands();
    }

    pub(super) fn request_redraw_from_camera(&mut self) {
        if self.is_camera_interacting() {
            self.render_interaction_frame();
            return;
        }
        if let Some(window) = &self.state.window {
            window.request_redraw();
        }
    }

    fn render_interaction_frame(&mut self) {
        // Rate-limit: CursorMoved can fire 100s/sec on Windows. Rendering every event
        // would stall the event loop. Skip if we rendered recently.
        const MIN_INTERACTION_FRAME_MS: u64 = 8;
        let since_last = self.state.last_inline_render_time.elapsed().as_millis() as u64;
        if since_last < MIN_INTERACTION_FRAME_MS {
            return;
        }
        let t0 = std::time::Instant::now();
        if self.state.world.cached_draw_calls.is_empty() {
            // No draw list yet (first frames or cache invalidated). Inline path cannot render;
            // still schedule a full RedrawRequested so camera motion is not dropped.
            if let Some(window) = &self.state.window {
                window.request_redraw();
            }
            return;
        }
        self.prepare_editor_ui_frame();
        let roots = self.state.world.graph.roots().to_vec();
        let aspect = self.state.renderer.as_ref()
            .map(|r| r.config.width as f32 / r.config.height.max(1) as f32).unwrap_or(1.0);
        let vp_proj = roots.first().and_then(|&r| Self::find_camera_projection(&self.state.world.graph, r))
            .unwrap_or_else(|| Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0));
        let (vp_view, cam_pos) = if let Some(vc) = self.state.viewport_cameras.active() {
            (vc.controller.view_matrix(), vc.controller.eye_position())
        } else if let Some(ref ctrl) = self.state.camera_controller {
            (ctrl.view_matrix(), ctrl.eye_position())
        } else { return; };
        let sync_interaction_active = self.is_camera_interacting();
        self.state.world.collector.draw_calls = self.state.world.cached_draw_calls.clone();
        rc3d_render::render_action::apply_world_camera(
            &mut self.state.world.collector.draw_calls, vp_view, vp_proj, cam_pos);
        let Some(renderer) = self.state.renderer.as_mut() else { return };
        // Interaction quality: use Flat mode + no HDR for faster rendering during drag.
        let saved_display = renderer.display_mode();
        let saved_hdr = renderer.hdr_post_processing;
        let tris: u64 = self.state.world.collector.draw_calls.iter()
            .map(|dc| dc.indices.as_ref().map_or(dc.vertices.len() as u64 / 3, |i| i.len() as u64 / 3))
            .sum();
        let saved_interaction_scale = renderer.interaction_render_scale;
        let interaction_scale = if tris > 1_000_000 {
            0.5
        } else if tris > 500_000 {
            0.67
        } else {
            1.0
        };
        renderer.set_interaction_render_scale(interaction_scale);
        if tris > 500_000 {
            renderer.set_display_mode(DisplayMode::Flat);
        }
        renderer.hdr_post_processing = false;
        renderer.interaction_active = true;
        let use_editor_overlay =
            self.state.editor_ui_enabled && self.state.editor_ui.is_some();
        if use_editor_overlay {
            let ui_ptr = self.state.editor_ui.as_mut().unwrap() as *mut EditorUi;
            let device = renderer.device.clone();
            let queue = renderer.queue.clone();
            renderer.render_draw_calls_with_overlay(
                &self.state.world.collector.draw_calls,
                &self.state.world.graph,
                Some(&mut |encoder, view| {
                    let ui = unsafe { &mut *ui_ptr };
                    ui.paint(&device, &queue, encoder, view);
                }),
            );
        } else {
            renderer.render_draw_calls(
                &self.state.world.collector.draw_calls,
                &self.state.world.graph,
            );
        }
        renderer.set_interaction_render_scale(saved_interaction_scale);
        renderer.set_display_mode(saved_display);
        renderer.hdr_post_processing = saved_hdr;
        // Inline render forced interaction_active for tier/degrade paths; sync back so release
        // is not stuck in interaction quality until a full RedrawRequested runs.
        renderer.interaction_active = sync_interaction_active;
        // Prevent the next RedrawRequested from double-rendering.
        self.state.interaction_frame_rendered = true;
        let now = std::time::Instant::now();
        let dt = now.duration_since(t0);
        self.state.last_inline_render_time = now;
        self.state.interaction_render_count += 1;
        if self.state.interaction_render_count % 30 == 0 {
            log::info!(
                "[INTERACTION] frame {:.1}ms tris={} scale={:.2}",
                dt.as_secs_f64() * 1000.0, tris, interaction_scale,
            );
        }
    }

    fn active_camera_matrices(&self, width: f32, height: f32) -> (Mat4, Mat4) {
        let aspect = width / height.max(1.0);
        let proj = Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0);
        if let Some(vc) = self.state.viewport_cameras.active() {
            (vc.controller.view_matrix(), proj)
        } else if let Some(ctrl) = &self.state.camera_controller {
            (ctrl.view_matrix(), proj)
        } else {
            (
                Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y),
                proj,
            )
        }
    }

    /// Call after any `ViewportLayout::rebuild` so [`ViewportCamera::viewport_id`] matches new slots.
    pub fn sync_viewport_camera_ids(&mut self) {
        if self.state.viewport_cameras.cameras.is_empty() {
            return;
        }
        let Some(renderer) = &self.state.renderer else {
            return;
        };
        self.state
            .viewport_cameras
            .remap_viewport_ids_from_layout(renderer.viewport_layout());
        self.editor.orbit_drag_viewport_id = None;
        self.editor.left_orbit_drag_viewport_id = None;
        self.editor.pan_drag_viewport_id = None;
    }

    fn apply_editor_commands(&mut self) {
        editor_commands::apply_editor_commands(self);
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

fn bookmark_slot_from_key(code: winit::keyboard::KeyCode) -> usize {
    use winit::keyboard::KeyCode;
    match code {
        KeyCode::Digit1 => 0,
        KeyCode::Digit2 => 1,
        KeyCode::Digit3 => 2,
        KeyCode::Digit4 => 3,
        KeyCode::Digit5 => 4,
        KeyCode::Digit6 => 5,
        KeyCode::Digit7 => 6,
        KeyCode::Digit8 => 7,
        KeyCode::Digit9 => 8,
        _ => 0,
    }
}
