//! Camera types.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_scene::node_data::{NodeData, OrthographicCameraNode, PerspectiveCameraNode};

/// Perspective camera with position, orientation, and projection parameters.
#[derive(Clone, Debug)]
pub struct PerspectiveCamera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    pub reverse_depth: bool,
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            eye: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 100.0,
            aspect: 16.0 / 9.0,
            reverse_depth: false,
        }
    }
}

impl PerspectiveCamera {
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3, fov: f32, aspect: f32) -> Self {
        Self { eye, target, up, fov, aspect, ..Default::default() }
    }

    pub fn fov(mut self, v: f32) -> Self { self.fov = v; self }
    pub fn near(mut self, v: f32) -> Self { self.near = v; self }
    pub fn far(mut self, v: f32) -> Self { self.far = v; self }
    pub fn aspect(mut self, v: f32) -> Self { self.aspect = v; self }
    pub fn reverse_depth(mut self, v: bool) -> Self { self.reverse_depth = v; self }

    pub(crate) fn to_node(&self) -> NodeData {
        NodeData::PerspectiveCamera(PerspectiveCameraNode {
            position: self.eye,
            orientation: Mat4::look_at_rh(self.eye, self.target, self.up),
            fov: self.fov,
            near: self.near,
            far: self.far,
            aspect: self.aspect,
            reverse_depth: self.reverse_depth,
        })
    }
}

/// Orthographic camera.
#[derive(Clone, Debug)]
pub struct OrthographicCamera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub height: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    pub reverse_depth: bool,
}

impl Default for OrthographicCamera {
    fn default() -> Self {
        Self {
            eye: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            height: 10.0,
            near: 0.1,
            far: 100.0,
            aspect: 16.0 / 9.0,
            reverse_depth: false,
        }
    }
}

impl OrthographicCamera {
    pub fn new(eye: Vec3, target: Vec3, up: Vec3, height: f32, aspect: f32) -> Self {
        Self { eye, target, up, height, aspect, ..Default::default() }
    }

    pub fn height(mut self, v: f32) -> Self { self.height = v; self }
    pub fn near(mut self, v: f32) -> Self { self.near = v; self }
    pub fn far(mut self, v: f32) -> Self { self.far = v; self }
    pub fn aspect(mut self, v: f32) -> Self { self.aspect = v; self }

    pub(crate) fn to_node(&self) -> NodeData {
        NodeData::OrthographicCamera(OrthographicCameraNode {
            position: self.eye,
            orientation: Mat4::look_at_rh(self.eye, self.target, self.up),
            height: self.height,
            near: self.near,
            far: self.far,
            aspect: self.aspect,
            reverse_depth: self.reverse_depth,
        })
    }
}
