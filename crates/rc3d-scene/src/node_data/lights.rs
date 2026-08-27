//! Light nodes.
use rc3d_core::math::Vec3;
use serde::{Deserialize, Serialize};

/// Directional (infinite) light.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DirectionalLightNode {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub light_group: Option<String>,
}

impl Default for DirectionalLightNode {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }
    }
}

/// Point light.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PointLightNode {
    pub location: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub light_group: Option<String>,
}

impl Default for PointLightNode {
    fn default() -> Self {
        Self {
            location: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }
    }
}

/// Spot light.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SpotLightNode {
    pub location: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub cut_off_angle: f32,
    pub drop_off_rate: f32,
    pub light_group: Option<String>,
}

impl Default for SpotLightNode {
    fn default() -> Self {
        Self {
            location: Vec3::ZERO,
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::ONE,
            intensity: 1.0,
            cut_off_angle: 0.785,
            drop_off_rate: 0.0,
            light_group: None,
        }
    }
}

/// Area light shape.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum AreaLightShape {
    Rectangle,
    Disc,
}

/// Area light: rectangle or disc emitter with soft shadows.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AreaLightNode {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    /// Width of the light (rectangle width, or disc diameter).
    pub width: f32,
    /// Height of the light (rectangle height, ignored for disc).
    pub height: f32,
    pub shape: AreaLightShape,
    pub light_group: Option<String>,
}

impl Default for AreaLightNode {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::ONE,
            intensity: 1.0,
            width: 1.0,
            height: 1.0,
            shape: AreaLightShape::Rectangle,
            light_group: None,
        }
    }
}

/// Hemisphere (sky/ground) ambient light. Indirect diffuse only; no shadows.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HemisphereLightNode {
    pub sky_color: Vec3,
    pub ground_color: Vec3,
    pub intensity: f32,
    /// Sky / up direction (world space).
    pub direction: Vec3,
}

    impl Default for HemisphereLightNode {
    fn default() -> Self {
        Self {
            sky_color: Vec3::new(0.4, 0.6, 1.0),
            ground_color: Vec3::new(0.4, 0.25, 0.1),
            intensity: 1.0,
            direction: Vec3::Y,
        }
    }
}

/// L2 spherical-harmonic irradiance probe (three.js `LightProbe`).
///
/// Band order matches three.js `SphericalHarmonics3` / `shGetIrradianceAt`:
/// `[L00, L1-1 (y), L10 (z), L11 (x), L2-2 (xy), L2-1 (yz), L20, L21 (xz), L22]`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LightProbeNode {
    pub sh: [[f32; 3]; 9],
    pub intensity: f32,
}

impl Default for LightProbeNode {
    fn default() -> Self {
        Self {
            sh: [[0.0; 3]; 9],
            intensity: 1.0,
        }
    }
}

impl LightProbeNode {
    pub fn from_sh(sh: [[f32; 3]; 9], intensity: f32) -> Self {
        Self { sh, intensity }
    }

    /// Constant ambient: L0 = `color * 2 * sqrt(PI)` so irradiance recovers `PI * color`.
    pub fn from_ambient(color: Vec3, intensity: f32) -> Self {
        let s = 2.0 * std::f32::consts::PI.sqrt();
        let mut sh = [[0.0f32; 3]; 9];
        sh[0] = (color * s).to_array();
        Self { sh, intensity }
    }

    /// Y-up sky/ground projected onto L0 + L1 (three.js band order).
    pub fn from_hemisphere(sky: Vec3, ground: Vec3, intensity: f32) -> Self {
        let pi = std::f32::consts::PI;
        let l0 = (sky + ground) * pi.sqrt();
        let l1y = (sky - ground) * (0.488603 * pi);
        let mut sh = [[0.0f32; 3]; 9];
        sh[0] = l0.to_array();
        sh[1] = l1y.to_array();
        Self { sh, intensity }
    }

    pub fn packed_sh_l2(&self) -> [[f32; 4]; 9] {
        let mut out = [[0.0f32; 4]; 9];
        for i in 0..9 {
            out[i] = [self.sh[i][0], self.sh[i][1], self.sh[i][2], 0.0];
        }
        out
    }
}
