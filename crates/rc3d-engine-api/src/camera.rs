use rc3d_core::aabb::Aabb;
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};
use std::cell::Cell;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

/// Saved camera state for a bookmark slot.
#[derive(Clone, Copy, Debug)]
pub struct CameraBookmark {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub name: &'static str,
}

/// State for smooth fly-to animation.
#[derive(Clone, Debug)]
struct FlyToState {
    start_target: Vec3,
    start_distance: f32,
    start_yaw: f32,
    start_pitch: f32,
    end_target: Vec3,
    end_distance: f32,
    end_yaw: f32,
    end_pitch: f32,
    elapsed: f32,
    duration: f32,
}

/// View orientation presets for industrial visualization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewPreset {
    Top,
    Bottom,
    Front,
    Back,
    Right,
    Left,
    Iso,
}

impl ViewPreset {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Top" => Some(Self::Top),
            "Bottom" => Some(Self::Bottom),
            "Front" => Some(Self::Front),
            "Back" => Some(Self::Back),
            "Right" => Some(Self::Right),
            "Left" => Some(Self::Left),
            "Iso" => Some(Self::Iso),
            _ => None,
        }
    }

    pub fn yaw_pitch(&self) -> (f32, f32) {
        match self {
            Self::Top => (0.0, std::f32::consts::FRAC_PI_2),
            Self::Bottom => (0.0, -std::f32::consts::FRAC_PI_2),
            Self::Front => (0.0, 0.0),
            Self::Back => (std::f32::consts::PI, 0.0),
            Self::Right => (std::f32::consts::FRAC_PI_2, 0.0),
            Self::Left => (-std::f32::consts::FRAC_PI_2, 0.0),
            Self::Iso => (std::f32::consts::FRAC_PI_4, 0.615),
        }
    }

    /// World up for `look_at` (Top/Bottom cannot use +Y — eye is on the Y axis).
    pub fn up_vector(&self) -> Vec3 {
        match self {
            Self::Top | Self::Bottom => -Vec3::Z,
            _ => Vec3::Y,
        }
    }
}

/// Orbit camera controller decoupled from windowing events.
/// Core math: orbit around target via yaw/pitch, pan, zoom.
pub struct CameraController {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub up: Vec3,
    /// Middle mouse held: apply orbit deltas on drag.
    pub middle_orbit_held: bool,
    /// Left mouse held: orbit (click-without-drag can still pick in the viewer).
    pub left_orbit_held: bool,
    /// Whether the controller is currently panning (for drag state tracking).
    pub panning: bool,
    /// When true, use first-person walk navigation instead of orbit.
    pub walk_mode: bool,
    /// Saved camera bookmarks (slots 0-8, keys 1-9).
    pub bookmarks: [Option<CameraBookmark>; 9],
    /// Active fly-to animation state.
    fly_to: Option<FlyToState>,
    /// Set when camera position actually changed (user input or animation).
    pub position_changed: Cell<bool>,
}

impl CameraController {
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            yaw: 0.0,
            pitch: 0.4,
            up: Vec3::Y,
            middle_orbit_held: false,
            left_orbit_held: false,
            panning: false,
            walk_mode: false,
            bookmarks: [None; 9],
            fly_to: None,
            position_changed: Cell::new(true), // first frame needs traversal
        }
    }

    /// Orbit by mouse delta (dx, dy in radians-scale).
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx;
        self.pitch -= dy;
        self.pitch = self.pitch.clamp(-1.5, 1.5);
        self.position_changed.set(true);
    }

    /// Pan the target perpendicular to the view direction.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let right = self.right_vector();
        let up = self.up_vector();
        let speed = self.distance * 0.002;
        self.target -= right * dx * speed;
        self.target += up * dy * speed;
        self.position_changed.set(true);
    }

    /// Zoom by a multiplicative factor (scroll delta).
    pub fn zoom(&mut self, delta: f32) {
        self.distance *= 1.0 - delta * 0.1;
        self.distance = self.distance.max(0.01);
        self.position_changed.set(true);
    }

    /// Walk forward/right relative to the current view direction.
    /// `forward` and `right` are normalized directional inputs.
    pub fn walk(&mut self, forward: f32, right: f32, up_down: f32, speed: f32) {
        let fwd = self.forward_vector();
        let rgt = self.right_vector();
        let delta = fwd * forward + rgt * right + self.up * up_down;
        self.target += delta * speed;
        self.position_changed.set(true);
    }

    /// Turn the view via mouse delta (first-person look).
    pub fn turn(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch -= dy * 0.005;
        self.pitch = self.pitch.clamp(-1.5, 1.5);
        self.position_changed.set(true);
    }

    /// Toggle walk mode on/off.
    pub fn toggle_walk_mode(&mut self) {
        self.walk_mode = !self.walk_mode;
    }

    pub fn eye_position(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let offset = Vec3::new(cp * sy, sp, cp * cy) * self.distance;
        self.target + offset
    }

    /// Look-at view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye_position(), self.target, self.up)
    }

    /// Fit the camera to view a bounding box.
    pub fn fit_bounds(&mut self, aabb: &Aabb, fov_y: f32) {
        self.target = aabb.center();
        let size = aabb.size().length();
        self.distance = size / (fov_y * 0.5).sin();
        self.distance = self.distance.max(0.1);
    }

    /// Point the camera at a view preset.
    pub fn set_view_preset(&mut self, preset: ViewPreset) {
        let (yaw, pitch) = preset.yaw_pitch();
        self.yaw = yaw;
        self.pitch = pitch;
        self.up = preset.up_vector();
        self.position_changed.set(true);
    }

    /// Update a camera node in the scene graph from current state.
    pub fn update_camera_node(&self, graph: &mut SceneGraph, camera_node: NodeId, aspect: f32) {
        let eye = self.eye_position();
        let mut changed = false;
        if let Some(entry) = graph.get_mut(camera_node) {
            match &mut entry.data {
                NodeData::PerspectiveCamera(cam) => {
                    changed = cam.aspect != aspect;
                    cam.position = eye;
                    cam.orientation = Mat4::look_at_rh(eye, self.target, self.up);
                    cam.aspect = aspect;
                    cam.near = (self.distance * 0.001).max(0.01);
                    cam.far = (self.distance * 20.0).max(100.0);
                }
                NodeData::OrthographicCamera(cam) => {
                    changed = cam.aspect != aspect;
                    cam.position = eye;
                    cam.orientation = Mat4::look_at_rh(eye, self.target, self.up);
                    let height = self.distance * 1.2;
                    cam.height = height;
                    cam.aspect = aspect;
                    cam.near = (self.distance * 0.001).max(0.01);
                    cam.far = (self.distance * 20.0).max(100.0);
                }
                _ => {}
            }
        }
        if self.position_changed.get() || changed {
            rc3d_render::dirty_flags::mark_node_dirty(
                graph,
                camera_node,
                rc3d_scene::node_entry::dirty_flags::TRANSFORM,
            );
            self.position_changed.set(false);
        }
    }

    /// Recursively update all PerspectiveCamera/OrthographicCamera nodes
    /// in the subtree rooted at `node` with this controller's current state.
    pub fn update_camera_recursive(
        &self,
        graph: &mut SceneGraph,
        node: NodeId,
        aspect: f32,
    ) {
        let Some(entry) = graph.get(node) else { return };
        if matches!(
            entry.data,
            NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_)
        ) {
            self.update_camera_node(graph, node, aspect);
            return;
        }
        let children: Vec<NodeId> = entry.children.clone();
        for child in children {
            self.update_camera_recursive(graph, child, aspect);
        }
    }

    /// Dispatch a winit window event to the camera controller.
    ///
    /// Handles mouse orbit (middle/left drag), pan (right drag), and zoom
    /// (scroll wheel). Returns `true` if the camera state changed.
    ///
    /// `cursor_pos` should be the cursor position BEFORE this event
    /// (used to compute deltas for `CursorMoved`).
    /// `left_orbit_enabled` enables left-button orbit on press. Left release
    /// always clears orbit so a later HUD click cannot leave the button stuck.
    pub fn dispatch_window_event(
        &mut self,
        event: &WindowEvent,
        cursor_pos: (f64, f64),
        left_orbit_enabled: bool,
    ) -> bool {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                match (button, state) {
                    (MouseButton::Middle, ElementState::Pressed) => {
                        self.middle_orbit_held = true;
                    }
                    (MouseButton::Middle, ElementState::Released) => {
                        self.middle_orbit_held = false;
                    }
                    (MouseButton::Left, ElementState::Pressed) if left_orbit_enabled => {
                        self.left_orbit_held = true;
                    }
                    (MouseButton::Left, ElementState::Released) => {
                        self.left_orbit_held = false;
                    }
                    (MouseButton::Right, ElementState::Pressed) => {
                        self.panning = true;
                    }
                    (MouseButton::Right, ElementState::Released) => {
                        self.panning = false;
                    }
                    _ => return false,
                }
                self.position_changed.set(true);
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let dx = (position.x - cursor_pos.0) as f32 * 0.005;
                let dy = (position.y - cursor_pos.1) as f32 * 0.005;
                if self.middle_orbit_held || self.left_orbit_held {
                    self.orbit(dx, dy);
                    return true;
                }
                if self.panning {
                    let dx = (position.x - cursor_pos.0) as f32;
                    let dy = (position.y - cursor_pos.1) as f32;
                    self.pan(dx, dy);
                    return true;
                }
                false
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };
                self.zoom(scroll);
                true
            }
            _ => false,
        }
    }

    /// Save current camera state to bookmark slot 0-8.
    pub fn save_bookmark(&mut self, slot: usize, name: &'static str) {
        if slot < 9 {
            self.bookmarks[slot] = Some(CameraBookmark {
                target: self.target,
                distance: self.distance,
                yaw: self.yaw,
                pitch: self.pitch,
                name,
            });
        }
    }

    /// Start fly-to animation to bookmark slot.
    pub fn recall_bookmark(&mut self, slot: usize) {
        if slot >= 9 {
            return;
        }
        let Some(bm) = self.bookmarks[slot] else {
            return;
        };
        self.fly_to = Some(FlyToState {
            start_target: self.target,
            start_distance: self.distance,
            start_yaw: self.yaw,
            start_pitch: self.pitch,
            end_target: bm.target,
            end_distance: bm.distance,
            end_yaw: bm.yaw,
            end_pitch: bm.pitch,
            elapsed: 0.0,
            duration: 0.5,
        });
    }

    /// Advance fly-to animation by `dt` seconds. Returns true while animation is active.
    pub fn tick_fly(&mut self, dt: f32) -> bool {
        let Some(ref fly) = self.fly_to else {
            return false;
        };
        let t = (fly.elapsed / fly.duration).clamp(0.0, 1.0);
        // Smooth ease-in-out
        let t_eased = t * t * (3.0 - 2.0 * t);
        self.target = fly.start_target + (fly.end_target - fly.start_target) * t_eased;
        self.distance = fly.start_distance + (fly.end_distance - fly.start_distance) * t_eased;
        self.yaw = fly.start_yaw + (fly.end_yaw - fly.start_yaw) * t_eased;
        self.pitch = fly.start_pitch + (fly.end_pitch - fly.start_pitch) * t_eased;
        if fly.elapsed >= fly.duration {
            self.fly_to = None;
            return false;
        }
        // Update elapsed in the struct (needs mutable access)
        if let Some(ref mut f) = self.fly_to {
            f.elapsed += dt;
        }
        true
    }

    pub fn is_flying(&self) -> bool {
        self.fly_to.is_some()
    }

    fn forward_vector(&self) -> Vec3 {
        (self.target - self.eye_position()).normalize()
    }

    fn right_vector(&self) -> Vec3 {
        let forward = self.forward_vector();
        forward.cross(self.up).normalize()
    }

    fn up_vector(&self) -> Vec3 {
        let right = self.right_vector();
        self.up.cross(right).normalize()
    }
}
