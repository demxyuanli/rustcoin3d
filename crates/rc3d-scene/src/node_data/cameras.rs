//! Camera nodes.
use rc3d_core::math::{Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};

/// Camera with perspective projection.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PerspectiveCameraNode {
    pub position: Vec3,
    pub orientation: Mat4,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    /// When `true`, builds reverse-Z clip mapping (near plane -> ndc_z ~ 1, far -> ~0). Match WebGPU
    /// depth test `Greater` and clear `0.0`.
    pub reverse_depth: bool,
}

impl Default for PerspectiveCameraNode {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            orientation: Mat4::IDENTITY,
            fov: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 100.0,
            aspect: 1.0,
            reverse_depth: false,
        }
    }
}

impl PerspectiveCameraNode {
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3, fov: f32, aspect: f32) -> Self {
        Self {
            position: eye,
            orientation: Mat4::look_at_rh(eye, target, up),
            fov,
            near: 0.1,
            far: 100.0,
            aspect,
            reverse_depth: false,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        self.orientation
    }

    pub fn projection_matrix(&self) -> Mat4 {
        // WebGPU clip Z in [0, w]; xy unchanged vs OpenGL NDC.
        let fov_clamped = self.fov.clamp(f32::EPSILON, std::f32::consts::PI - f32::EPSILON);
        let f = 1.0 / (fov_clamped * 0.5).tan();
        let d = (self.far - self.near).max(f32::EPSILON);
        let aspect = self.aspect.max(f32::EPSILON);
        if self.reverse_depth {
            let a = self.near / d;
            let b = self.near * self.far / d;
            Mat4::from_cols(
                Vec4::new(f / aspect, 0.0, 0.0, 0.0),
                Vec4::new(0.0, f, 0.0, 0.0),
                Vec4::new(0.0, 0.0, a, -1.0),
                Vec4::new(0.0, 0.0, b, 0.0),
            )
        } else {
            let nf = 1.0 / (self.near - self.far);
            Mat4::from_cols(
                Vec4::new(f / aspect, 0.0, 0.0, 0.0),
                Vec4::new(0.0, f, 0.0, 0.0),
                Vec4::new(0.0, 0.0, self.far * nf, -1.0),
                Vec4::new(0.0, 0.0, self.near * self.far * nf, 0.0),
            )
        }
    }
}

/// Camera with orthographic projection.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OrthographicCameraNode {
    pub position: Vec3,
    pub orientation: Mat4,
    pub height: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    /// When `true`, ndc_z increases toward the near plane (reverse-Z); pair with `Greater` + clear `0`.
    pub reverse_depth: bool,
}

impl Default for OrthographicCameraNode {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            orientation: Mat4::IDENTITY,
            height: 2.0,
            near: 0.1,
            far: 100.0,
            aspect: 1.0,
            reverse_depth: false,
        }
    }
}

/// Stereo camera for VR/AR side-by-side rendering.
/// Encapsulates left/right eye transforms derived from a base camera.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoMode { SideBySide, TopBottom, Anaglyph }
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StereoCameraNode {
    pub base_camera: rc3d_core::NodeId,
    pub interocular_distance: f32,
    pub convergence_distance: f32,
    pub mode: StereoMode,
}
impl Default for StereoCameraNode {
    fn default() -> Self {
        Self { base_camera: rc3d_core::NodeId::default(), interocular_distance: 0.065, convergence_distance: 2.0, mode: StereoMode::SideBySide }
    }
}

impl OrthographicCameraNode {
    pub fn view_matrix(&self) -> Mat4 {
        self.orientation
    }

    pub fn projection_matrix(&self) -> Mat4 {
        let half_h = self.height.max(f32::EPSILON) / 2.0;
        let half_w = half_h * self.aspect;
        let rml = half_w * 2.0;
        let tmb = half_h * 2.0;
        let fmn = self.far - self.near;
        if self.reverse_depth {
            Mat4::from_cols(
                Vec4::new(2.0 / rml, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 2.0 / tmb, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0 / fmn, 0.0),
                Vec4::new(0.0, 0.0, self.far / fmn, 1.0),
            )
        } else {
            Mat4::from_cols(
                Vec4::new(2.0 / rml, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 2.0 / tmb, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0 / (self.near - self.far), 0.0),
                Vec4::new(0.0, 0.0, -self.near / fmn, 1.0),
            )
        }
    }
}
