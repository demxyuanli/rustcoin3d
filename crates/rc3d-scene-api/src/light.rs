//! Light types.

use rc3d_core::math::Vec3;
use rc3d_scene::node_data::{DirectionalLightNode, NodeData, PointLightNode};

/// Directional (sun) light.
#[derive(Clone, Debug)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub light_group: Option<String>,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }
    }
}

impl DirectionalLight {
    /// Sun-like light from a direction.
    pub fn sun(direction: Vec3, intensity: f32) -> Self {
        Self {
            direction: direction.normalize(),
            color: Vec3::ONE,
            intensity,
            light_group: None,
        }
    }

    pub fn direction(mut self, x: f32, y: f32, z: f32) -> Self {
        self.direction = Vec3::new(x, y, z).normalize();
        self
    }

    pub fn color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.color = Vec3::new(r, g, b);
        self
    }

    pub fn intensity(mut self, v: f32) -> Self {
        self.intensity = v;
        self
    }

    pub(crate) fn to_node(&self) -> NodeData {
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: self.direction,
            color: self.color,
            intensity: self.intensity,
            light_group: self.light_group.clone(),
        })
    }
}

/// Point (omnidirectional) light.
#[derive(Clone, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub light_group: Option<String>,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }
    }
}

impl PointLight {
    pub fn new(position: Vec3, color: Vec3, intensity: f32) -> Self {
        Self { position, color, intensity, light_group: None }
    }

    pub fn at(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = Vec3::new(x, y, z);
        self
    }

    pub fn color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.color = Vec3::new(r, g, b);
        self
    }

    pub fn intensity(mut self, v: f32) -> Self {
        self.intensity = v;
        self
    }

    pub(crate) fn to_node(&self) -> NodeData {
        NodeData::PointLight(PointLightNode {
            location: self.position,
            color: self.color,
            intensity: self.intensity,
            light_group: self.light_group.clone(),
        })
    }
}
