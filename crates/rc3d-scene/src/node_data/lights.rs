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
