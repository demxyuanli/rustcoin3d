use std::any::Any;

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::FieldId;
use rc3d_scene::node_data::NodeData;
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
    last_time: Option<f64>,
}

impl ElapsedTimeEngine {
    pub fn new(transform_node: rc3d_core::NodeId, speed: f32, axis: Vec3) -> Self {
        let axis = axis.normalize();
        let axis = if axis.length_squared() < 1e-8 {
            Vec3::Y
        } else {
            axis
        };
        Self {
            transform_node,
            speed,
            axis,
            last_time: None,
        }
    }
}

impl Engine for ElapsedTimeEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64) {
        let dt = match self.last_time {
            Some(prev) => ((time - prev) as f32).min(0.1), // cap at 100ms to avoid jump
            None => 0.0,
        };
        self.last_time = Some(time);
        if dt <= 0.0 {
            return;
        }
        let angle = dt * self.speed;
        let rotation = Mat4::from_axis_angle(self.axis, angle);
        if let Some(entry) = graph.get_mut(self.transform_node) {
            if let rc3d_scene::node_data::NodeData::Transform(t) = &mut entry.data {
                t.rotation = rotation * t.rotation; // compound
            }
            entry.dirty_flags |= rc3d_scene::node_entry::dirty_flags::GEOMETRY;
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
    TranslationX,
    TranslationY,
    TranslationZ,
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
                    SineField::TranslationX => t.translation.x = value,
                    SineField::TranslationY => t.translation.y = value,
                    SineField::TranslationZ => t.translation.z = value,
                }
            }
            entry.dirty_flags |= rc3d_scene::node_entry::dirty_flags::TRANSFORM;
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
    pub node_id: rc3d_core::NodeId,
}

impl CalculatorEngine {
    pub fn new(expressions: Vec<String>, outputs: Vec<FieldId>, node: rc3d_core::NodeId) -> Self {
        Self { expressions, output_field_ids: outputs, node_id: node }
    }
}

impl Engine for CalculatorEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, _time: f64) {
        use rc3d_fields::FieldValue;
        let Some(entry) = graph.get_mut(self.node_id) else { return };
        for (i, expr) in self.expressions.iter().enumerate() {
            if i < self.output_field_ids.len() {
                let value = Self::eval_simple(expr);
                if let Some(val) = value {
                    entry.fields.set(self.output_field_ids[i], FieldValue::Float(val));
                }
            }
        }
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl CalculatorEngine {
    fn eval_simple(expr: &str) -> Option<f32> {
        let parts: Vec<&str> = expr.split('=').collect();
        if parts.len() < 2 { return None; }
        let rhs = parts[1].trim();
        if let Ok(v) = rhs.parse::<f32>() { return Some(v); }
        if rhs.starts_with("sin(") {
            let inner = rhs.trim_start_matches("sin(").trim_end_matches(')');
            if let Ok(x) = inner.parse::<f32>() { return Some(x.sin()); }
        }
        if rhs.starts_with("cos(") {
            let inner = rhs.trim_start_matches("cos(").trim_end_matches(')');
            if let Ok(x) = inner.parse::<f32>() { return Some(x.cos()); }
        }
        None
    }
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
    pub node_id: rc3d_core::NodeId,
}

impl ComposeMatrixEngine {
    pub fn new(output: FieldId, node: rc3d_core::NodeId) -> Self {
        Self { translation_field: None, rotation_field: None, scale_field: None, output_field: output, node_id: node }
    }
}

impl Engine for ComposeMatrixEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, _time: f64) {
        use rc3d_core::math::{Mat4, Quat, Vec3};
        use rc3d_fields::FieldValue;
        let Some(entry) = graph.get_mut(self.node_id) else { return };
        let t = self.translation_field
            .and_then(|fid| entry.fields.get(fid))
            .and_then(|v| match v { FieldValue::Vec3f(v) => Some(*v), _ => None })
            .unwrap_or(Vec3::ZERO);
        let r = self.rotation_field
            .and_then(|fid| entry.fields.get(fid))
            .and_then(|v| match v { FieldValue::Vec3f(v) => {
                let q = Quat::from_xyzw(v.x, v.y, v.z, 1.0);
                Some(Mat4::from_quat(q))
            }, _ => None })
            .unwrap_or(Mat4::IDENTITY);
        let s = self.scale_field
            .and_then(|fid| entry.fields.get(fid))
            .and_then(|v| match v { FieldValue::Vec3f(v) => Some(*v), _ => None })
            .unwrap_or(Vec3::ONE);
        let m = Mat4::from_scale_rotation_translation(s, Quat::from_mat4(&r), t);
        entry.fields.set(self.output_field, FieldValue::Mat4f(m));
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

impl std::fmt::Debug for ComposeMatrixEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComposeMatrixEngine").finish()
    }
}

/// One-shot trigger engine (Coin3D SoOneShot).
/// Fires once for a specified duration when triggered, then stops.
#[derive(Debug, Clone)]
pub struct OneShotEngine {
    pub duration: f64,
    pub triggered: bool,
    pub elapsed: f64,
    pub active: bool,
    last_time: f64,
}

impl OneShotEngine {
    pub fn new(duration: f64) -> Self {
        Self { duration, triggered: false, elapsed: 0.0, active: false, last_time: 0.0 }
    }
    pub fn trigger(&mut self) { self.triggered = true; self.active = true; self.elapsed = 0.0; }
    pub fn is_active(&self) -> bool { self.active }
}

impl Engine for OneShotEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, time: f64) {
        if !self.active { return; }
        if self.elapsed == 0.0 { self.last_time = time; }
        let dt = (time - self.last_time).clamp(0.0, 0.1);
        self.last_time = time;
        self.elapsed += dt;
        if self.elapsed >= self.duration { self.active = false; }
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Integer counter engine (Coin3D SoCounter).
/// Increments or decrements on trigger, wraps at min/max.
#[derive(Debug, Clone)]
pub struct CounterEngine {
    pub value: i32,
    pub min: i32,
    pub max: i32,
    pub step: i32,
    pub triggered: bool,
}

impl CounterEngine {
    pub fn new(min: i32, max: i32, step: i32) -> Self {
        Self { value: min, min, max, step: step.max(1), triggered: false }
    }
    pub fn trigger(&mut self) { self.triggered = true; }
    pub fn value(&self) -> i32 { self.value }
}

impl Engine for CounterEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {
        if !self.triggered { return; }
        self.triggered = false;
        self.value += self.step;
        if self.value > self.max { self.value = self.min; }
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Interpolates a [`TransformNode`](rc3d_scene::node_data::TransformNode) translation (Coin3D SoInterpolate-style).
#[derive(Debug)]
pub struct InterpolateVec3Engine {
    pub transform_node: rc3d_core::NodeId,
    pub from: Vec3,
    pub to: Vec3,
    pub period_secs: f64,
}

impl Engine for InterpolateVec3Engine {
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64) {
        if self.period_secs <= 1e-6 {
            return;
        }
        let u = (time % self.period_secs) / self.period_secs;
        let p = self.from.lerp(self.to, u as f32);
        if let Some(e) = graph.get_mut(self.transform_node) {
            if let NodeData::Transform(t) = &mut e.data {
                t.translation = p;
            }
            e.dirty_flags |= rc3d_scene::node_entry::dirty_flags::TRANSFORM;
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
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

/// Float interpolation engine (Coin3D SoInterpolateFloat).
#[derive(Debug, Clone)]
pub struct InterpolateFloatEngine {
    pub node_id: rc3d_core::NodeId,
    pub field_id: rc3d_core::FieldId,
    pub from: f32, pub to: f32,
    pub duration: f64, pub elapsed: f64,
    last_time: f64,
}
impl InterpolateFloatEngine {
    pub fn new(node: rc3d_core::NodeId, field: rc3d_core::FieldId, from: f32, to: f32, duration: f64) -> Self {
        Self { node_id: node, field_id: field, from, to, duration: duration.max(0.001), elapsed: 0.0, last_time: 0.0 }
    }
}
impl Engine for InterpolateFloatEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64) {
        if self.elapsed == 0.0 { self.last_time = time; }
        let dt = (time - self.last_time).clamp(0.0, 0.1);
        self.last_time = time;
        self.elapsed += dt;
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        let v = self.from + (self.to - self.from) * t as f32;
        if let Some(entry) = graph.get_mut(self.node_id) {
            entry.fields.set(self.field_id, rc3d_fields::FieldValue::Float(v));
        }
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Rotation interpolation engine (Coin3D SoInterpolateRotation).
#[derive(Debug, Clone)]
pub struct InterpolateRotationEngine {
    pub node_id: rc3d_core::NodeId,
    pub from: rc3d_core::math::Quat, pub to: rc3d_core::math::Quat,
    pub duration: f64, pub elapsed: f64,
    last_time: f64,
}
impl InterpolateRotationEngine {
    pub fn new(node: rc3d_core::NodeId, from: rc3d_core::math::Quat, to: rc3d_core::math::Quat, duration: f64) -> Self {
        Self { node_id: node, from, to, duration: duration.max(0.001), elapsed: 0.0, last_time: 0.0 }
    }
}
impl Engine for InterpolateRotationEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, time: f64) {
        if self.elapsed == 0.0 { self.last_time = time; }
        let dt = (time - self.last_time).clamp(0.0, 0.1);
        self.last_time = time;
        self.elapsed += dt;
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        let q = self.from.slerp(self.to, t as f32);
        if let Some(entry) = graph.get_mut(self.node_id) {
            if let rc3d_scene::NodeData::Transform(tf) = &mut entry.data {
                tf.rotation = rc3d_core::math::Mat4::from_quat(q);
            }
            entry.dirty_flags |= rc3d_scene::node_entry::dirty_flags::TRANSFORM;
        }
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Compose a 3D vector from three float fields (Coin3D SoComposeVec3f).
#[derive(Debug, Clone)]
pub struct ComposeVec3fEngine {
    pub node_id: rc3d_core::NodeId,
    pub x_field: rc3d_core::FieldId, pub y_field: rc3d_core::FieldId, pub z_field: rc3d_core::FieldId,
    pub output_field: rc3d_core::FieldId,
}
impl ComposeVec3fEngine {
    pub fn new(node: rc3d_core::NodeId, x: rc3d_core::FieldId, y: rc3d_core::FieldId, z: rc3d_core::FieldId, out: rc3d_core::FieldId) -> Self {
        Self { node_id: node, x_field: x, y_field: y, z_field: z, output_field: out }
    }
}
impl Engine for ComposeVec3fEngine {
    fn evaluate(&mut self, graph: &mut SceneGraph, _time: f64) {
        let Some(entry) = graph.get_mut(self.node_id) else { return };
        let x = entry.fields.get(self.x_field).and_then(|v| match v { rc3d_fields::FieldValue::Float(v) => Some(*v), _ => None }).unwrap_or(0.0);
        let y = entry.fields.get(self.y_field).and_then(|v| match v { rc3d_fields::FieldValue::Float(v) => Some(*v), _ => None }).unwrap_or(0.0);
        let z = entry.fields.get(self.z_field).and_then(|v| match v { rc3d_fields::FieldValue::Float(v) => Some(*v), _ => None }).unwrap_or(0.0);
        entry.fields.set(self.output_field, rc3d_fields::FieldValue::Vec3f(rc3d_core::math::Vec3::new(x, y, z)));
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Toggle engine (Coin3D SoOnOff).
#[derive(Debug, Clone)]
pub struct OnOffEngine { pub state: bool, pub triggered: bool }
impl Default for OnOffEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OnOffEngine {
    pub fn new() -> Self { Self { state: false, triggered: false } }
    pub fn trigger(&mut self) { self.triggered = true; }
}
impl Engine for OnOffEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {
        if self.triggered { self.state = !self.state; self.triggered = false; }
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Trigger on any input change (Coin3D SoTriggerAny).
#[derive(Debug, Clone)]
pub struct TriggerAnyEngine { pub fired: bool, pub triggered: bool }
impl Default for TriggerAnyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TriggerAnyEngine {
    pub fn new() -> Self { Self { fired: false, triggered: false } }
    pub fn trigger(&mut self) { self.triggered = true; }
    pub fn has_fired(&self) -> bool { self.fired }
}
impl Engine for TriggerAnyEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {
        if self.triggered { self.fired = true; self.triggered = false; }
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
