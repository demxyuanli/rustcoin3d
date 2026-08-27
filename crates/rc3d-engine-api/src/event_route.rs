//! Shared winit → scene-graph event path (`HandleEventAction` + camera + pick).

use rc3d_actions::{apply_to_all_roots, Event, EventContext, HandleEventAction};
use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_render::viewport::{ProjectionType, Viewport};
use rc3d_scene::node_data::NodeData;
use rc3d_scene::SceneGraph;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::PhysicalKey;

use crate::engine::Engine;
use crate::gizmo_bind::scene_pick_matrices;
use crate::viewport::ViewportCamera;

/// Options for [`Engine::handle_window_event`].
#[derive(Clone, Copy, Debug)]
pub struct EventRouteOpts {
    /// Left-button drag orbits the camera (examples). Editor uses middle-button only.
    pub left_orbit: bool,
    /// Click (no drag) invokes [`Engine::pick_at`] when pick callbacks are set.
    pub pick_on_click: bool,
    /// Forward the event to [`CameraController::dispatch_window_event`].
    pub camera: bool,
}

impl Default for EventRouteOpts {
    fn default() -> Self {
        Self {
            left_orbit: true,
            pick_on_click: true,
            camera: true,
        }
    }
}

/// Result of routing a window event through the engine.
#[derive(Clone, Debug, Default)]
pub struct EventRouteResult {
    pub camera_changed: bool,
    pub redraw: bool,
    pub overlay_consumed: bool,
    pub hit_node: Option<NodeId>,
    pub event_callback_nodes: Vec<NodeId>,
}

impl Engine {
    /// Viewport-local pick frame: `(local_x, local_y, vw, vh, view, proj)`.
    pub fn pointer_pick_frame(
        &self,
        cx: f32,
        cy: f32,
        surface_w: u32,
        surface_h: u32,
    ) -> Option<(f32, f32, f32, f32, Mat4, Mat4)> {
        if !self.viewport_cameras.cameras.is_empty() {
            let r = self.renderer.as_ref()?;
            let layout = r.viewport_layout();
            let vp_ref = layout
                .viewport_at(cx, cy)
                .or_else(|| layout.active())
                .or_else(|| layout.viewports.first())?;
            if let Some(vc) = self.viewport_cameras.find(vp_ref.id) {
                let (v, p) = viewport_pick_matrices(&self.world.graph, vc, vp_ref);
                let lx = cx - vp_ref.rect.x as f32;
                let ly = cy - vp_ref.rect.y as f32;
                let vw = vp_ref.rect.width.max(1) as f32;
                let vh = vp_ref.rect.height.max(1) as f32;
                return Some((lx, ly, vw, vh, v, p));
            }
        }
        let vw = surface_w.max(1) as f32;
        let vh = surface_h.max(1) as f32;
        let (v, p) = scene_pick_matrices(self);
        Some((cx, cy, vw, vh, v, p))
    }

    /// Update [`Engine::input`] from a winit event (cursor, modifiers, click-vs-drag).
    pub fn feed_input(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((px, py)) = self.input.pending_pick {
                    let dx = position.x as f32 - px;
                    let dy = position.y as f32 - py;
                    if dx * dx + dy * dy > 25.0 {
                        self.input.left_dragged = true;
                    }
                }
                self.input.cursor_prev = self.input.cursor_pos;
                self.input.cursor_pos = (position.x, position.y);
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.input.shift_pressed = mods.state().shift_key();
                self.input.ctrl_pressed = mods.state().control_key();
            }
            WindowEvent::Resized(size) => {
                self.input.window_size = (size.width, size.height);
            }
            _ => {}
        }
    }

    /// Convert winit input into [`HandleEventAction`], camera, and optional click-pick.
    ///
    /// Does not handle redraw / close; the event loop keeps those.
    pub fn handle_window_event(
        &mut self,
        event: &WindowEvent,
        opts: EventRouteOpts,
    ) -> EventRouteResult {
        self.feed_input(event);
        self.dispatch_routed_event(event, opts)
    }

    /// Scene routing after [`Self::feed_input`] (editor can insert gizmo handling in between).
    pub fn dispatch_routed_event(
        &mut self,
        event: &WindowEvent,
        opts: EventRouteOpts,
    ) -> EventRouteResult {
        let mut result = EventRouteResult::default();
        let (cx, cy) = (
            self.input.cursor_pos.0 as f32,
            self.input.cursor_pos.1 as f32,
        );
        let (ww, wh) = self.input.window_size;

        if let WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button: MouseButton::Left,
            ..
        } = event
        {
            if let Some(ref hook) = self.panel_overlay_mouse_hook {
                result.overlay_consumed = hook(cx, cy, ww, wh);
                if result.overlay_consumed {
                    result.redraw = true;
                }
            }
        }

        if let WindowEvent::MouseInput {
            state,
            button: MouseButton::Left,
            ..
        } = event
        {
            match state {
                ElementState::Pressed if !result.overlay_consumed => {
                    self.input.pending_pick = Some((cx, cy));
                    self.input.left_dragged = false;
                }
                ElementState::Released => {
                    if opts.pick_on_click
                        && !self.input.left_dragged
                        && (self.on_pick.is_some() || self.on_pick_hit.is_some())
                    {
                        if let Some((x, y)) = self.input.pending_pick {
                            self.pick_at(x, y, ww, wh);
                            result.redraw = true;
                        }
                    }
                    self.input.pending_pick = None;
                    self.input.left_dragged = false;
                }
                _ => {}
            }
        }

        let (lx, ly, vw, vh, view, proj) = self
            .pointer_pick_frame(cx, cy, ww, wh)
            .unwrap_or_else(|| {
                let (v, p) = scene_pick_matrices(self);
                (
                    cx,
                    cy,
                    ww.max(1) as f32,
                    wh.max(1) as f32,
                    v,
                    p,
                )
            });
        if let Some(scene_event) =
            window_event_to_scene(event, lx, ly, self.input.cursor_prev)
        {
            let mut ctx = EventContext::new(scene_event, view, proj);
            ctx.pointer_pick_viewport = Some((vw, vh));
            let mut action = HandleEventAction::new(ctx);
            apply_to_all_roots(&mut action, &self.world.graph);
            result.hit_node = action.hit_node;
            result.event_callback_nodes = action.event_callback_nodes;
        }

        if let WindowEvent::KeyboardInput { event: key, .. } = event {
            if key.state == ElementState::Pressed {
                if let Some(ref mut hook) = self.panel_overlay_key_hook {
                    if hook(key.physical_key) {
                        result.redraw = true;
                    }
                }
            }
        }

        let left_orbit = opts.left_orbit && !result.overlay_consumed;
        if opts.camera {
            result.camera_changed = self.controller.dispatch_window_event(
                event,
                self.input.cursor_prev,
                left_orbit,
            );
            if result.camera_changed {
                result.redraw = true;
            }
        }

        result
    }
}

fn viewport_pick_matrices(graph: &SceneGraph, vc: &ViewportCamera, vport: &Viewport) -> (Mat4, Mat4) {
    if let Some(e) = graph.get(vc.camera_node) {
        match &e.data {
            NodeData::PerspectiveCamera(c) => return (c.view_matrix(), c.projection_matrix()),
            NodeData::OrthographicCamera(c) => return (c.view_matrix(), c.projection_matrix()),
            _ => {}
        }
    }
    let aspect = vport.rect.aspect();
    let v = vc.controller.view_matrix();
    let p = match vport.projection_type {
        ProjectionType::Perspective => {
            Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 1000.0)
        }
        ProjectionType::Orthographic => {
            let height = vc.controller.distance * 1.2;
            let w = height * aspect;
            rc3d_render::shadow_map::orthographic_wgpu_rh(
                -w * 0.5,
                w * 0.5,
                -height * 0.5,
                height * 0.5,
                0.1,
                1000.0,
            )
        }
    };
    (v, p)
}

fn window_event_to_scene(
    event: &WindowEvent,
    cx: f32,
    cy: f32,
    cursor_prev: (f64, f64),
) -> Option<Event> {
    match event {
        WindowEvent::CursorMoved { position, .. } => Some(Event::MouseMove {
            x: cx,
            y: cy,
            dx: (position.x - cursor_prev.0) as f32,
            dy: (position.y - cursor_prev.1) as f32,
        }),
        WindowEvent::MouseInput { state, button, .. } => {
            let button = match button {
                MouseButton::Left => 0,
                MouseButton::Middle => 1,
                MouseButton::Right => 2,
                _ => 0,
            };
            match state {
                ElementState::Pressed => Some(Event::ButtonPress { button, x: cx, y: cy }),
                ElementState::Released => Some(Event::ButtonRelease { button, x: cx, y: cy }),
            }
        }
        WindowEvent::MouseWheel { delta, .. } => {
            let (dx, dy) = match delta {
                MouseScrollDelta::LineDelta(x, y) => (*x, *y),
                MouseScrollDelta::PixelDelta(p) => (p.x as f32 / 50.0, p.y as f32 / 50.0),
            };
            Some(Event::Scroll { dx, dy })
        }
        WindowEvent::KeyboardInput { event, .. } => Some(key_event_to_scene(event)),
        _ => None,
    }
}

fn key_event_to_scene(event: &KeyEvent) -> Event {
    let key = match event.physical_key {
        PhysicalKey::Code(code) => format!("{code:?}"),
        PhysicalKey::Unidentified(_) => "Unidentified".into(),
    };
    if event.state == ElementState::Pressed {
        Event::KeyPress { key }
    } else {
        Event::KeyRelease { key }
    }
}
