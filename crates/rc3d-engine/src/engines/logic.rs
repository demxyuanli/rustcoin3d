//! Gate, concatenate, select, bool, time counter.


use rc3d_fields::FieldValue;
use rc3d_scene::{field_as_bool, field_as_f32, field_as_i32, SceneGraph};

use crate::connection;
use crate::engine::Engine;

/// Pass-through when `enable` is true, or once on `trigger` (Coin3D `SoGate`).
#[derive(Debug, Clone)]
pub struct GateEngine {
    pub enable: bool,
    input: Option<FieldValue>,
    output: Option<FieldValue>,
    triggered: bool,
}

impl Default for GateEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl GateEngine {
    pub fn new() -> Self {
        Self {
            enable: true,
            input: None,
            output: None,
            triggered: false,
        }
    }
}

impl Engine for GateEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {
        let pass = self.enable || self.triggered;
        self.triggered = false;
        if pass {
            if let Some(v) = self.input.clone() {
                self.output = Some(v);
            }
        }
    }
    fn set_input(&mut self, port: &str, value: FieldValue) {
        match port {
            "enable" => {
                if let Some(b) = field_as_bool(&value) {
                    self.enable = b;
                }
            }
            "input" => self.input = Some(value),
            "trigger" if connection::is_trigger(&value) => self.triggered = true,
            _ => {}
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "output" => self.output.clone(),
            "enable" => Some(FieldValue::Bool(self.enable)),
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

fn input_slot(port: &str) -> Option<usize> {
    if port == "input" {
        return Some(0);
    }
    port.strip_prefix("input")?.parse().ok()
}

/// Concatenate up to 8 scalar/vector inputs into an array (Coin3D `SoConcatenate`).
#[derive(Debug, Clone)]
pub struct ConcatenateEngine {
    inputs: [Option<FieldValue>; 8],
}

impl Default for ConcatenateEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ConcatenateEngine {
    pub fn new() -> Self {
        Self {
            inputs: [None, None, None, None, None, None, None, None],
        }
    }
}

impl Engine for ConcatenateEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}
    fn set_input(&mut self, port: &str, value: FieldValue) {
        if let Some(i) = input_slot(port) {
            if i < 8 {
                self.inputs[i] = Some(value);
            }
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        if port != "output" {
            return None;
        }
        let filled: Vec<&FieldValue> = self.inputs.iter().flatten().collect();
        if filled.is_empty() {
            return Some(FieldValue::FloatArray(Vec::new()));
        }
        if filled.iter().all(|v| matches!(v, FieldValue::Vec3f(_))) {
            let arr = filled
                .iter()
                .filter_map(|v| match v {
                    FieldValue::Vec3f(p) => Some(*p),
                    _ => None,
                })
                .collect();
            return Some(FieldValue::Vec3fArray(arr));
        }
        if filled.iter().all(|v| matches!(v, FieldValue::Int32(_))) {
            let arr = filled
                .iter()
                .filter_map(|v| match v {
                    FieldValue::Int32(p) => Some(*p),
                    _ => None,
                })
                .collect();
            return Some(FieldValue::Int32Array(arr));
        }
        let arr: Vec<f32> = filled.iter().filter_map(|v| field_as_f32(v)).collect();
        Some(FieldValue::FloatArray(arr))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Pick one element from an array by index (Coin3D `SoSelectOne`).
#[derive(Debug, Clone)]
pub struct SelectOneEngine {
    index: i32,
    input: Option<FieldValue>,
}

impl Default for SelectOneEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectOneEngine {
    pub fn new() -> Self {
        Self {
            index: 0,
            input: None,
        }
    }
}

impl Engine for SelectOneEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}
    fn set_input(&mut self, port: &str, value: FieldValue) {
        match port {
            "index" => {
                if let Some(i) = field_as_i32(&value) {
                    self.index = i;
                }
            }
            "input" => self.input = Some(value),
            _ => {}
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        if port != "output" {
            return None;
        }
        let Some(input) = self.input.as_ref() else {
            return None;
        };
        let i = self.index.max(0) as usize;
        match input {
            FieldValue::FloatArray(a) => a.get(i).copied().map(FieldValue::Float),
            FieldValue::Vec3fArray(a) => a.get(i).copied().map(FieldValue::Vec3f),
            FieldValue::Int32Array(a) => a.get(i).copied().map(FieldValue::Int32),
            other => Some(other.clone()),
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Boolean combination (Coin3D `SoBoolOperation`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoolOp {
    #[default]
    And,
    Or,
    Xor,
    Not,
    Nand,
    Nor,
}

#[derive(Debug, Clone)]
pub struct BoolOperationEngine {
    pub op: BoolOp,
    a: bool,
    b: bool,
}

impl BoolOperationEngine {
    pub fn new(op: BoolOp) -> Self {
        Self {
            op,
            a: false,
            b: false,
        }
    }
}

impl Engine for BoolOperationEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, _time: f64) {}
    fn set_input(&mut self, port: &str, value: FieldValue) {
        match port {
            "a" => {
                if let Some(v) = field_as_bool(&value) {
                    self.a = v;
                }
            }
            "b" => {
                if let Some(v) = field_as_bool(&value) {
                    self.b = v;
                }
            }
            _ => {}
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        if port != "output" && port != "result" {
            return None;
        }
        let r = match self.op {
            BoolOp::And => self.a && self.b,
            BoolOp::Or => self.a || self.b,
            BoolOp::Xor => self.a ^ self.b,
            BoolOp::Not => !self.a,
            BoolOp::Nand => !(self.a && self.b),
            BoolOp::Nor => !(self.a || self.b),
        };
        Some(FieldValue::Bool(r))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
/// Frequency-driven integer cycle (Coin3D `SoTimeCounter`).
#[derive(Debug, Clone)]
pub struct TimeCounterEngine {
    pub min: i32,
    pub max: i32,
    pub step: i32,
    pub frequency: f32,
    value: i32,
    last_time: Option<f64>,
    accum: f32,
}

impl TimeCounterEngine {
    pub fn new(min: i32, max: i32, frequency: f32) -> Self {
        Self {
            min,
            max: max.max(min),
            step: 1,
            frequency: frequency.max(0.0),
            value: min,
            last_time: None,
            accum: 0.0,
        }
    }
}

impl Engine for TimeCounterEngine {
    fn evaluate(&mut self, _graph: &mut SceneGraph, time: f64) {
        let dt = match self.last_time {
            Some(prev) => ((time - prev) as f32).min(0.1),
            None => 0.0,
        };
        self.last_time = Some(time);
        if self.frequency <= 0.0 || dt <= 0.0 {
            return;
        }
        self.accum += dt * self.frequency;
        while self.accum >= 1.0 {
            self.accum -= 1.0;
            self.value += self.step;
            if self.value > self.max {
                self.value = self.min;
            }
        }
    }
    fn output(&self, port: &str) -> Option<FieldValue> {
        match port {
            "output" => Some(FieldValue::Int32(self.value)),
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
