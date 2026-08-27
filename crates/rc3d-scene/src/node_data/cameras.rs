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
pub enum StereoMode {
    SideBySide,
    TopBottom,
    Anaglyph,
}

impl StereoMode {
    /// Per-eye aspect for the given surface size (half-width / half-height / full).
    pub fn eye_aspect(self, surface_width: u32, surface_height: u32) -> f32 {
        let w = surface_width.max(1) as f32;
        let h = surface_height.max(1) as f32;
        match self {
            StereoMode::SideBySide => (w * 0.5) / h,
            StereoMode::TopBottom => w / (h * 0.5),
            StereoMode::Anaglyph => w / h,
        }
    }
}

/// One stereo eye: view, projection, and world-space position.
#[derive(Clone, Copy, Debug)]
pub struct StereoEyePose {
    pub view: Mat4,
    pub projection: Mat4,
    pub position: Vec3,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StereoCameraNode {
    pub base_camera: rc3d_core::NodeId,
    pub interocular_distance: f32,
    pub convergence_distance: f32,
    pub mode: StereoMode,
}
impl Default for StereoCameraNode {
    fn default() -> Self {
        Self {
            base_camera: rc3d_core::NodeId::default(),
            interocular_distance: 0.065,
            convergence_distance: 2.0,
            mode: StereoMode::SideBySide,
        }
    }
}

impl StereoCameraNode {
    /// Parallel cameras + off-axis frustum (no toe-in). `None` when IPD is ~0
    /// (monoscopic fallback) or the basis is degenerate.
    pub fn eyes_from_perspective(
        &self,
        cam: &PerspectiveCameraNode,
        eye_aspect: f32,
    ) -> Option<[StereoEyePose; 2]> {
        let half_iod = self.interocular_distance.abs() * 0.5;
        if half_iod < 1e-6 {
            return None;
        }
        let inv = cam.view_matrix().inverse();
        let right = inv.x_axis.truncate().normalize_or_zero();
        let up = inv.y_axis.truncate().normalize_or_zero();
        let forward = (-inv.z_axis.truncate()).normalize_or_zero();
        if right.length_squared() < 1e-8 || up.length_squared() < 1e-8 || forward.length_squared() < 1e-8
        {
            return None;
        }
        let near = cam.near.max(f32::EPSILON);
        let far = cam.far.max(near + f32::EPSILON);
        let conv = self.convergence_distance.max(near + f32::EPSILON);
        let fov = cam.fov.clamp(f32::EPSILON, std::f32::consts::PI - f32::EPSILON);
        let aspect = eye_aspect.max(f32::EPSILON);
        let wd2 = near * (fov * 0.5).tan();
        let shift = half_iod * (near / conv);
        let a_wd2 = aspect * wd2;

        let pos_l = cam.position - right * half_iod;
        let pos_r = cam.position + right * half_iod;
        let view_l = Mat4::look_at_rh(pos_l, pos_l + forward, up);
        let view_r = Mat4::look_at_rh(pos_r, pos_r + forward, up);
        let proj_l = frustum_rh(
            -a_wd2 - shift,
            a_wd2 - shift,
            -wd2,
            wd2,
            near,
            far,
            cam.reverse_depth,
        );
        let proj_r = frustum_rh(
            -a_wd2 + shift,
            a_wd2 + shift,
            -wd2,
            wd2,
            near,
            far,
            cam.reverse_depth,
        );
        Some([
            StereoEyePose {
                view: view_l,
                projection: proj_l,
                position: pos_l,
            },
            StereoEyePose {
                view: view_r,
                projection: proj_r,
                position: pos_r,
            },
        ])
    }
}

/// WebGPU clip-Z [0, 1] right-handed asymmetric frustum (matches `PerspectiveCameraNode`).
fn frustum_rh(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    z_near: f32,
    z_far: f32,
    reverse_depth: bool,
) -> Mat4 {
    let rml = (right - left).max(f32::EPSILON);
    let tmb = (top - bottom).max(f32::EPSILON);
    let n = z_near.max(f32::EPSILON);
    let f = z_far.max(n + f32::EPSILON);
    if reverse_depth {
        let d = (f - n).max(f32::EPSILON);
        Mat4::from_cols(
            Vec4::new(2.0 * n / rml, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 * n / tmb, 0.0, 0.0),
            Vec4::new((right + left) / rml, (top + bottom) / tmb, n / d, -1.0),
            Vec4::new(0.0, 0.0, n * f / d, 0.0),
        )
    } else {
        let nf = 1.0 / (n - f);
        Mat4::from_cols(
            Vec4::new(2.0 * n / rml, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 * n / tmb, 0.0, 0.0),
            Vec4::new((right + left) / rml, (top + bottom) / tmb, f * nf, -1.0),
            Vec4::new(0.0, 0.0, n * f * nf, 0.0),
        )
    }
}

/// Dynamic cube-map probe (three.js CubeCamera). Renders six 90-degree faces
/// from `position` into a cubemap used as a local IBL environment.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct CubeCameraNode {
    pub position: Vec3,
    pub near: f32,
    pub far: f32,
    /// Face resolution (square). Clamped to 32..512 at capture time.
    pub resolution: u32,
    /// 0 = capture once, 1 = every frame, N = every N frames.
    pub update_period: u32,
    pub enabled: bool,
}

impl Default for CubeCameraNode {
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

impl OrthographicCameraNode {
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3, height: f32, aspect: f32) -> Self {
        Self {
            position: eye,
            orientation: Mat4::look_at_rh(eye, target, up),
            height,
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
