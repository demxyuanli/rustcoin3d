use std::any::Any;

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::FieldId;
use rc3d_scene::SceneGraph;

/// An engine computes output values from input values (lazy evaluation).
pub trait Engine: Any + std::fmt::Debug {
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Rotates a Transform node continuously using the elapsed time.
#[derive(Debug)]
pub struct ElapsedTimeEngine {
    pub transform_node: rc3d_core::NodeId,
    pub speed: f32,
    pub axis: Vec3,
}

impl ElapsedTimeEngine {
    pub fn new(transform_node: rc3d_core::NodeId, speed: f32, axis: Vec3) -> Self {
        Self {
            transform_node,
            speed,
            axis,
        }
    }
}

impl Engine for ElapsedTimeEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64) {
        let elapsed = time as f32;
        let angle = elapsed * self.speed;
        let rotation = Mat4::from_axis_angle(self.axis, angle);
        if let Some(entry) = graph.get_mut(self.transform_node) {
            if let rc3d_scene::node_data::NodeData::Transform(t) = &mut entry.data {
                t.rotation = rotation;
            }
        }
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Oscillates a Transform field using a sine wave driven by elapsed time.
#[derive(Debug)]
pub struct SineOscillatorEngine {
    pub transform_node: rc3d_core::NodeId,
    pub frequency: f32,
    pub amplitude: f32,
    pub field: SineField,
}

#[derive(Clone, Copy, Debug)]
pub enum SineField {
    ScaleX,
    ScaleY,
    ScaleZ,
    TranslationY,
}

impl SineOscillatorEngine {
    pub fn new(transform_node: rc3d_core::NodeId, frequency: f32, amplitude: f32, field: SineField) -> Self {
        Self {
            transform_node,
            frequency,
            amplitude,
            field,
        }
    }
}

impl Engine for SineOscillatorEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64) {
        let elapsed = time as f32;
        let value = (elapsed * self.frequency * std::f32::consts::TAU).sin() * self.amplitude;
        if let Some(entry) = graph.get_mut(self.transform_node) {
            if let rc3d_scene::node_data::NodeData::Transform(t) = &mut entry.data {
                match self.field {
                    SineField::ScaleX => t.scale.x = value.abs().max(0.1),
                    SineField::ScaleY => t.scale.y = value.abs().max(0.1),
                    SineField::ScaleZ => t.scale.z = value.abs().max(0.1),
                    SineField::TranslationY => t.translation.y = value,
                }
            }
        }
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Expression-based calculator (Coin3D SoCalculator pattern).
/// Evaluates expressions like "oA = sin(iA) * 3.0" on field values.
pub struct CalculatorEngine {
    pub expressions: Vec<String>,
    pub output_field_ids: Vec<FieldId>,
}

impl CalculatorEngine {
    pub fn new(expressions: Vec<String>, outputs: Vec<FieldId>) -> Self {
        Self { expressions, output_field_ids: outputs }
    }
}

impl Engine for CalculatorEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl std::fmt::Debug for CalculatorEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalculatorEngine").field("expressions", &self.expressions).finish()
    }
}

/// Compose TRS into Mat4 (Coin3D SoComposeMatrix pattern).
pub struct ComposeMatrixEngine {
    pub translation_field: Option<FieldId>,
    pub rotation_field: Option<FieldId>,
    pub scale_field: Option<FieldId>,
    pub output_field: FieldId,
}

impl ComposeMatrixEngine {
    pub fn new(output: FieldId) -> Self {
        Self { translation_field: None, rotation_field: None, scale_field: None, output_field: output }
    }
}

impl Engine for ComposeMatrixEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl std::fmt::Debug for ComposeMatrixEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComposeMatrixEngine").finish()
    }
}

/// Registry of all active engines, evaluated each frame.
pub struct EngineRegistry {
    pub engines: Vec<Box<dyn Engine>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self {
            engines: Vec::new(),
        }
    }

    pub fn add(&mut self, engine: impl Engine + 'static) {
        self.engines.push(Box::new(engine));
    }

    pub fn evaluate_all(&mut self, graph: &mut SceneGraph, time: f64) {
        for engine in &mut self.engines {
            engine.evaluate(graph, time);
        }
    }
}

impl Default for EngineRegistry {
    fn default() -> Self {
        Self::new()
    }
}
