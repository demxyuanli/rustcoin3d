use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};
use winit::event::{MouseButton, MouseScrollDelta, WindowEvent};

/// Orbit camera controller: left-drag to orbit, right-drag to pan, scroll to zoom.
pub struct CameraController {
    pub camera_node: NodeId,
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub up: Vec3,
    pub orbiting: bool,
    pub panning: bool,
    pub last_mouse: (f64, f64),
}

impl CameraController {
    pub fn new(camera_node: NodeId, target: Vec3, distance: f32) -> Self {
        Self {
            camera_node,
            target,
            distance,
            yaw: 0.0,
            pitch: 0.4,
            up: Vec3::Y,
            orbiting: false,
            panning: false,
            last_mouse: (0.0, 0.0),
        }
    }

    pub fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                match (button, state) {
                    (MouseButton::Left, winit::event::ElementState::Pressed) => {
                        self.orbiting = true;
                    }
                    (MouseButton::Left, winit::event::ElementState::Released) => {
                        self.orbiting = false;
                    }
                    (MouseButton::Right, winit::event::ElementState::Pressed) => {
                        self.panning = true;
                    }
                    (MouseButton::Right, winit::event::ElementState::Released) => {
                        self.panning = false;
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x - self.last_mouse.0;
                let dy = position.y - self.last_mouse.1;
                if self.orbiting {
                    self.yaw -= dx as f32 * 0.005;
                    self.pitch -= dy as f32 * 0.005;
                    self.pitch = self.pitch.clamp(-1.5, 1.5);
                }
                if self.panning {
                    let right = self.right_vector();
                    let up = self.up_vector();
                    let pan_speed = self.distance * 0.002;
                    self.target -= right * dx as f32 * pan_speed;
                    self.target += up * dy as f32 * pan_speed;
                }
                self.last_mouse = (position.x, position.y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };
                self.distance *= 1.0 - scroll * 0.1;
                self.distance = self.distance.max(0.1);
            }
            _ => {}
        }
    }

    /// Update the camera node in the scene graph from current orbit state.
    pub fn update_camera(&self, graph: &mut SceneGraph, aspect: f32) {
        let eye = self.eye_position();
        if let Some(entry) = graph.get_mut(self.camera_node) {
            if let NodeData::PerspectiveCamera(cam) = &mut entry.data {
                cam.position = eye;
                cam.orientation = Mat4::look_at_rh(eye, self.target, self.up);
                cam.aspect = aspect;
            }
        }
    }

    pub fn eye_position(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let offset = Vec3::new(cp * sy, sp, cp * cy) * self.distance;
        self.target + offset
    }

    fn right_vector(&self) -> Vec3 {
        let forward = (self.target - self.eye_position()).normalize();
        forward.cross(self.up).normalize()
    }

    fn up_vector(&self) -> Vec3 {
        let right = self.right_vector();
        self.up.cross(right).normalize()
    }
}
