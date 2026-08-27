//! Compose/decompose vectors, matrices, and rotations.

use rc3d_core::math::{Mat4, Quat, Vec2, Vec3, Vec4};
use rc3d_fields::FieldValue;
use rc3d_scene::{field_as_bool, field_as_f32, SceneGraph};

use crate::engine::Engine;

/// Split a Vec3 into x/y/z (Coin3D `SoDecomposeVec3f`).
#[derive(Debug, Clone)]
pub struct DecomposeVec3fEngine {
    vector: Vec3,
}

impl Default for DecomposeVec3fEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DecomposeVec3fEngine {
    pub fn new() -> Self {
        Self { vector: Vec3::ZERO }
    }
}

impl Engine for DecomposeVec3fEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}
    fn set_input(&mut self, port: &str, value: FieldValue) {
        if port == "vector" {
            if let FieldValue::Vec3f(v) = value {
                self.vector = v;
            }
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "vector" => Some(FieldValue::Vec3f(self.vector)),
            "x" => Some(FieldValue::Float(self.vector.x)),
            "y" => Some(FieldValue::Float(self.vector.y)),
            "z" => Some(FieldValue::Float(self.vector.z)),
            _ => None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Split a Vec2 into x/y (Coin3D `SoDecomposeVec2f`).
#[derive(Debug, Clone)]
pub struct DecomposeVec2fEngine {
    vector: Vec2,
}

impl Default for DecomposeVec2fEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DecomposeVec2fEngine {
    pub fn new() -> Self {
        Self { vector: Vec2::ZERO }
    }
}

impl Engine for DecomposeVec2fEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}
    fn set_input(&mut self, port: &str, value: FieldValue) {
        if port == "vector" {
            if let FieldValue::Vec2f(v) = value {
                self.vector = v;
            }
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "vector" => Some(FieldValue::Vec2f(self.vector)),
            "x" => Some(FieldValue::Float(self.vector.x)),
            "y" => Some(FieldValue::Float(self.vector.y)),
            _ => None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Split a Vec4 into x/y/z/w (Coin3D `SoDecomposeVec4f`).
#[derive(Debug, Clone)]
pub struct DecomposeVec4fEngine {
    vector: Vec4,
}

impl Default for DecomposeVec4fEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DecomposeVec4fEngine {
    pub fn new() -> Self {
        Self { vector: Vec4::ZERO }
    }
}

impl Engine for DecomposeVec4fEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}
    fn set_input(&mut self, port: &str, value: FieldValue) {
        if port == "vector" {
            if let FieldValue::Vec4f(v) = value {
                self.vector = v;
            }
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "vector" => Some(FieldValue::Vec4f(self.vector)),
            "x" => Some(FieldValue::Float(self.vector.x)),
            "y" => Some(FieldValue::Float(self.vector.y)),
            "z" => Some(FieldValue::Float(self.vector.z)),
            "w" => Some(FieldValue::Float(self.vector.w)),
            _ => None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// x,y → Vec2 (Coin3D `SoComposeVec2f`).
#[derive(Debug, Clone)]
pub struct ComposeVec2fEngine {
    x: f32,
    y: f32,
}

impl Default for ComposeVec2fEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ComposeVec2fEngine {
    pub fn new() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

impl Engine for ComposeVec2fEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}
    fn set_input(&mut self, port: &str, value: FieldValue) {
        if let Some(f) = field_as_f32(&value) {
            match port {
                "x" => self.x = f,
                "y" => self.y = f,
                _ => {}
            }
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "vector" => Some(FieldValue::Vec2f(Vec2::new(self.x, self.y))),
            "x" => Some(FieldValue::Float(self.x)),
            "y" => Some(FieldValue::Float(self.y)),
            _ => None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// x,y,z,w → Vec4 (Coin3D `SoComposeVec4f`).
#[derive(Debug, Clone)]
pub struct ComposeVec4fEngine {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Default for ComposeVec4fEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ComposeVec4fEngine {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }
}

impl Engine for ComposeVec4fEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}
    fn set_input(&mut self, port: &str, value: FieldValue) {
        if let Some(f) = field_as_f32(&value) {
            match port {
                "x" => self.x = f,
                "y" => self.y = f,
                "z" => self.z = f,
                "w" => self.w = f,
                _ => {}
            }
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "vector" => Some(FieldValue::Vec4f(Vec4::new(self.x, self.y, self.z, self.w))),
            "x" => Some(FieldValue::Float(self.x)),
            "y" => Some(FieldValue::Float(self.y)),
            "z" => Some(FieldValue::Float(self.z)),
            "w" => Some(FieldValue::Float(self.w)),
            _ => None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
/// TRS split of a matrix (Coin3D `SoDecomposeMatrix`).
#[derive(Debug, Clone)]
pub struct DecomposeMatrixEngine {
    matrix: Mat4,
    translation: Vec3,
    scale: Vec3,
    rotation: Mat4,
}

impl Default for DecomposeMatrixEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DecomposeMatrixEngine {
    pub fn new() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
            translation: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Mat4::IDENTITY,
        }
    }
}

impl Engine for DecomposeMatrixEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {
        let (scale, rot, trans) = self.matrix.to_scale_rotation_translation();
        self.scale = scale;
        self.translation = trans;
        self.rotation = Mat4::from_quat(rot);
    }
    fn set_input(&mut self, port: &str, value: FieldValue) {
        if port == "matrix" {
            if let FieldValue::Mat4f(m) = value {
                self.matrix = m;
            }
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "matrix" => Some(FieldValue::Mat4f(self.matrix)),
            "translation" => Some(FieldValue::Vec3f(self.translation)),
            "scale" => Some(FieldValue::Vec3f(self.scale)),
            "rotation" => Some(FieldValue::Mat4f(self.rotation)),
            _ => None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Multiply a vector by a matrix (Coin3D `SoTransformVec3f`).
#[derive(Debug, Clone)]
pub struct TransformVec3fEngine {
    vector: Vec3,
    matrix: Mat4,
    pub as_point: bool,
    output: Vec3,
}

impl Default for TransformVec3fEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformVec3fEngine {
    pub fn new() -> Self {
        Self {
            vector: Vec3::ZERO,
            matrix: Mat4::IDENTITY,
            as_point: true,
            output: Vec3::ZERO,
        }
    }
}

impl Engine for TransformVec3fEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {
        self.output = if self.as_point {
            self.matrix.transform_point3(self.vector)
        } else {
            self.matrix.transform_vector3(self.vector)
        };
    }
    fn set_input(&mut self, port: &str, value: FieldValue) {
        match port {
            "vector" => {
                if let FieldValue::Vec3f(v) = value {
                    self.vector = v;
                }
            }
            "matrix" => {
                if let FieldValue::Mat4f(m) = value {
                    self.matrix = m;
                }
            }
            "asPoint" | "as_point" => {
                if let Some(b) = field_as_bool(&value) {
                    self.as_point = b;
                }
            }
            _ => {}
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "vector" | "output" => Some(FieldValue::Vec3f(self.output)),
            _ => None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Axis-angle → rotation matrix (Coin3D `SoComposeRotation`).
#[derive(Debug, Clone)]
pub struct ComposeRotationEngine {
    axis: Vec3,
    angle: f32,
}

impl Default for ComposeRotationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ComposeRotationEngine {
    pub fn new() -> Self {
        Self {
            axis: Vec3::Y,
            angle: 0.0,
        }
    }
}

impl Engine for ComposeRotationEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}
    fn set_input(&mut self, port: &str, value: FieldValue) {
        match port {
            "axis" => {
                if let FieldValue::Vec3f(v) = value {
                    self.axis = v;
                }
            }
            "angle" => {
                if let Some(a) = field_as_f32(&value) {
                    self.angle = a;
                }
            }
            _ => {}
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "rotation" | "output" => {
                let axis = if self.axis.length_squared() < 1e-8 {
                    Vec3::Y
                } else {
                    self.axis.normalize()
                };
                Some(FieldValue::Mat4f(Mat4::from_axis_angle(axis, self.angle)))
            }
            _ => None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Rotation matrix → axis-angle (Coin3D `SoDecomposeRotation`).
#[derive(Debug, Clone)]
pub struct DecomposeRotationEngine {
    matrix: Mat4,
    axis: Vec3,
    angle: f32,
}

impl Default for DecomposeRotationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DecomposeRotationEngine {
    pub fn new() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
            axis: Vec3::Y,
            angle: 0.0,
        }
    }
}

impl Engine for DecomposeRotationEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {
        let q = Quat::from_mat4(&self.matrix);
        let (axis, angle) = q.to_axis_angle();
        self.axis = axis;
        self.angle = angle;
    }
    fn set_input(&mut self, port: &str, value: FieldValue) {
        if port == "rotation" || port == "matrix" {
            if let FieldValue::Mat4f(m) = value {
                self.matrix = m;
            }
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "axis" => Some(FieldValue::Vec3f(self.axis)),
            "angle" => Some(FieldValue::Float(self.angle)),
            _ => None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
