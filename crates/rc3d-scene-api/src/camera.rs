//! Camera types.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_scene::node_data::{
    CubeCameraNode, NodeData, OrthographicCameraNode, PerspectiveCameraNode, StereoCameraNode,
};
pub use rc3d_scene::node_data::StereoMode;

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

/// Local reflection probe: captures six 90-degree faces from `position` into a cubemap
/// that replaces the global IBL environment (three.js CubeCamera analog).
#[derive(Clone, Debug)]
pub struct CubeCamera {
    pub position: Vec3,
    pub near: f32,
    pub far: f32,
    pub resolution: u32,
    pub update_period: u32,
    pub enabled: bool,
}

impl Default for CubeCamera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            near: 0.1,
            far: 100.0,
            resolution: 128,
            update_period: 0,
            enabled: true,
        }
    }
}

impl CubeCamera {
    pub fn at(position: Vec3) -> Self {
        Self { position, ..Default::default() }
    }

    pub fn near(mut self, v: f32) -> Self { self.near = v; self }
    pub fn far(mut self, v: f32) -> Self { self.far = v; self }
    pub fn resolution(mut self, v: u32) -> Self { self.resolution = v; self }
    pub fn update_period(mut self, v: u32) -> Self { self.update_period = v; self }

    pub(crate) fn to_node(&self) -> NodeData {
        NodeData::CubeCamera(CubeCameraNode {
            position: self.position,
            near: self.near,
            far: self.far,
            resolution: self.resolution,
            update_period: self.update_period,
            enabled: self.enabled,
        })
    }
}

/// Stereo helper (three.js StereoCamera / StereoEffect analog).
/// Renders left/right eyes from the scene's perspective camera.
#[derive(Clone, Debug)]
pub struct StereoCamera {
    pub interocular_distance: f32,
    pub convergence_distance: f32,
    pub mode: StereoMode,
}

impl Default for StereoCamera {
    fn default() -> Self {
        Self {
            interocular_distance: 0.065,
            convergence_distance: 2.0,
            mode: StereoMode::SideBySide,
        }
    }
}

impl StereoCamera {
    pub fn side_by_side() -> Self {
        Self::default()
    }

    pub fn interocular(mut self, meters: f32) -> Self {
        self.interocular_distance = meters;
        self
    }

    pub fn convergence(mut self, meters: f32) -> Self {
        self.convergence_distance = meters;
        self
    }

    pub fn mode(mut self, mode: StereoMode) -> Self {
        self.mode = mode;
        self
    }

    pub(crate) fn to_node(&self, base_camera: rc3d_core::NodeId) -> NodeData {
        NodeData::StereoCamera(StereoCameraNode {
            base_camera,
            interocular_distance: self.interocular_distance,
            convergence_distance: self.convergence_distance,
            mode: self.mode,
        })
    }
}
