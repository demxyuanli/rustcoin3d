//! Scene effect nodes (explode, reflection, decal).
use rc3d_core::math::Vec3;
use serde::{Deserialize, Serialize};

/// Exploded view: offsets children along direction proportionally (HOOPS explode).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExplodedViewNode {
    pub direction: Vec3,
    pub factor: f32,
    pub center: Vec3,
}
impl Default for ExplodedViewNode {
    fn default() -> Self { Self { direction: Vec3::Y, factor: 1.0, center: Vec3::ZERO } }
}

/// Planar reflection plane (HOOPS reflection equivalent).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReflectionPlaneNode {
    pub normal: Vec3,
    pub origin: Vec3,
    pub enabled: bool,
}
impl Default for ReflectionPlaneNode {
    fn default() -> Self { Self { normal: Vec3::Y, origin: Vec3::ZERO, enabled: true } }
}

/// Screen-space projected texture decal.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DecalNode {
    pub position: Vec3,
    pub direction: Vec3,
    pub size: [f32; 2],
    pub texture_path: String,
    pub color: [f32; 4],
    pub opacity: f32,
}

impl Default for DecalNode {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::new(0.0, -1.0, 0.0),
            size: [1.0, 1.0],
            texture_path: String::new(),
            color: [1.0, 1.0, 1.0, 1.0],
            opacity: 1.0,
        }
    }
}
