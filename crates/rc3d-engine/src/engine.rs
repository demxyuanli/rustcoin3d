use std::any::Any;
use std::time::Instant;

use rc3d_core::math::{Mat4, Vec3};
use rc3d_scene::SceneGraph;

/// An engine computes output values from input values (lazy evaluation).
pub trait Engine: Any + std::fmt::Debug {
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Engine that outputs elapsed time since creation.
/// Drives animation by updating a Transform node's rotation.
#[derive(Debug)]
pub struct ElapsedTimeEngine {
    pub transform_node: rc3d_core::NodeId,
    pub start: Instant,
    pub speed: f32,
    pub axis: Vec3,
}

impl ElapsedTimeEngine {
    pub fn new(transform_node: rc3d_core::NodeId, speed: f32, axis: Vec3) -> Self {
        Self {
            transform_node,
            start: Instant::now(),
            speed,
            axis,
        }
    }
}

impl Engine for ElapsedTimeEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, _time: f64) {
        let elapsed = self.start.elapsed().as_secs_f64() as f32;
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

/// Engine that oscillates a value using sine wave.
#[derive(Debug)]
pub struct SineOscillatorEngine {
    pub transform_node: rc3d_core::NodeId,
    pub start: Instant,
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
            start: Instant::now(),
            frequency,
            amplitude,
            field,
        }
    }
}

impl Engine for SineOscillatorEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, _time: f64) {
        let elapsed = self.start.elapsed().as_secs_f64() as f32;
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
