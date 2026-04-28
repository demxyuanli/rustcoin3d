use rc3d_core::math::{Mat4, Vec3};
use std::any::Any;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementId(pub u16);

pub trait Element: Any + std::fmt::Debug {
    fn element_id(&self) -> ElementId;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_box(&self) -> Box<dyn Element>;
}

impl Clone for Box<dyn Element> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// --- Concrete Elements ---

#[derive(Clone, Debug)]
pub struct ModelMatrixElement {
    pub matrix: Mat4,
}

impl Default for ModelMatrixElement {
    fn default() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
        }
    }
}

impl Element for ModelMatrixElement {
    fn element_id(&self) -> ElementId {
        ElementId(0)
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug)]
pub struct ViewMatrixElement {
    pub matrix: Mat4,
}

impl Default for ViewMatrixElement {
    fn default() -> Self {
        Self { matrix: Mat4::IDENTITY }
    }
}

impl Element for ViewMatrixElement {
    fn element_id(&self) -> ElementId { ElementId(1) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug)]
pub struct ProjectionMatrixElement {
    pub matrix: Mat4,
}

impl Default for ProjectionMatrixElement {
    fn default() -> Self {
        Self { matrix: Mat4::IDENTITY }
    }
}

impl Element for ProjectionMatrixElement {
    fn element_id(&self) -> ElementId { ElementId(2) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug, Default)]
pub struct CoordinateElement {
    pub points: Vec<Vec3>,
}

impl Element for CoordinateElement {
    fn element_id(&self) -> ElementId { ElementId(3) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug, Default)]
pub struct NormalElement {
    pub vectors: Vec<Vec3>,
}

impl Element for NormalElement {
    fn element_id(&self) -> ElementId { ElementId(4) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug)]
pub struct MaterialElement {
    pub diffuse: Vec3,
    pub ambient: Vec3,
    pub specular: Vec3,
    pub shininess: f32,
}

impl Default for MaterialElement {
    fn default() -> Self {
        Self {
            diffuse: Vec3::new(0.8, 0.8, 0.8),
            ambient: Vec3::new(0.2, 0.2, 0.2),
            specular: Vec3::new(0.0, 0.0, 0.0),
            shininess: 0.0,
        }
    }
}

impl Element for MaterialElement {
    fn element_id(&self) -> ElementId { ElementId(5) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

#[derive(Clone, Debug, Default)]
pub struct LightElement {
    pub lights: Vec<LightData>,
}

#[derive(Clone, Debug)]
pub struct LightData {
    pub light_type: LightType,
    pub direction: Vec3,
    pub location: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub cut_off_angle: f32,
    pub drop_off_rate: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightType {
    Directional,
    Point,
    Spot,
}

impl Element for LightElement {
    fn element_id(&self) -> ElementId { ElementId(6) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn Element> { Box::new(self.clone()) }
}

pub const NUM_ELEMENT_TYPES: usize = 7;
