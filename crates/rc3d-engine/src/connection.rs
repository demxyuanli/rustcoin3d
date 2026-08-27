//! Engine-to-engine and engine-to-node connections (Coin3D `SoEngineOutput`).
//!
//! When [`EngineRegistry::connections`] is empty, engines still evaluate in
//! insertion order. Otherwise engine-to-engine edges are topologically sorted
//! (`rc3d_core::utils::graph::toposort_linear`); a cycle falls back to
//! insertion order.

use rc3d_core::NodeId;
use rc3d_fields::FieldValue;
use rc3d_scene::{read_node_field, write_node_field, SceneGraph};

use crate::engine::EngineRegistry;

pub(crate) use rc3d_scene::field_as_f32;

/// Index of an engine in [`EngineRegistry::engines`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EngineId(pub usize);

/// One end of an [`EngineConnection`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EngineEndpoint {
    Engine { id: EngineId, port: String },
    Node { node: NodeId, field_index: u16 },
}

/// Directed value flow from `from` to `to`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EngineConnection {
    pub from: EngineEndpoint,
    pub to: EngineEndpoint,
}

impl EngineRegistry {
    pub fn from_engines(engines: Vec<Box<dyn crate::engine::Engine>>) -> Self {
        Self {
            engines,
            connections: Vec::new(),
        }
    }

    pub fn connect(&mut self, from: EngineEndpoint, to: EngineEndpoint) {
        self.connections.push(EngineConnection { from, to });
    }

    pub fn connect_engines(
        &mut self,
        from: EngineId,
        from_port: &str,
        to: EngineId,
        to_port: &str,
    ) {
        self.connect(
            EngineEndpoint::Engine {
                id: from,
                port: from_port.to_string(),
            },
            EngineEndpoint::Engine {
                id: to,
                port: to_port.to_string(),
            },
        );
    }

    pub fn connect_to_node(
        &mut self,
        from: EngineId,
        from_port: &str,
        node: NodeId,
        field_index: u16,
    ) {
        self.connect(
            EngineEndpoint::Engine {
                id: from,
                port: from_port.to_string(),
            },
            EngineEndpoint::Node { node, field_index },
        );
    }

    pub fn connect_from_node(
        &mut self,
        node: NodeId,
        field_index: u16,
        to: EngineId,
        to_port: &str,
    ) {
        self.connect(
            EngineEndpoint::Node { node, field_index },
            EngineEndpoint::Engine {
                id: to,
                port: to_port.to_string(),
            },
        );
    }
}

pub(crate) fn evaluate_registry(reg: &mut EngineRegistry, graph: &mut SceneGraph, time: f64) {
    if reg.connections.is_empty() {
        for engine in &mut reg.engines {
            engine.evaluate(graph, time);
        }
        return;
    }

    let n = reg.engines.len();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for c in &reg.connections {
        if let (
            EngineEndpoint::Engine { id: from, .. },
            EngineEndpoint::Engine { id: to, .. },
        ) = (&c.from, &c.to)
        {
            if from.0 < n && to.0 < n {
                edges.push((from.0, to.0));
            }
        }
    }
    let order = rc3d_core::utils::graph::toposort_linear(&edges, n)
        .unwrap_or_else(|_| (0..n).collect());

    for &i in &order {
        let mut pulled: Vec<(String, FieldValue)> = Vec::new();
        for c in &reg.connections {
            match (&c.from, &c.to) {
                (
                    EngineEndpoint::Engine {
                        id: src,
                        port: src_port,
                    },
                    EngineEndpoint::Engine {
                        id: dst,
                        port: dst_port,
                    },
                ) if dst.0 == i && src.0 < n => {
                    if let Some(v) = reg.engines[src.0].output(src_port) {
                        pulled.push((dst_port.clone(), v));
                    }
                }
                (
                    EngineEndpoint::Node { node, field_index },
                    EngineEndpoint::Engine { id: dst, port },
                ) if dst.0 == i => {
                    if let Some(v) = read_node_field(graph, *node, *field_index) {
                        pulled.push((port.clone(), v));
                    }
                }
                _ => {}
            }
        }
        for (port, value) in pulled {
            reg.engines[i].set_input(&port, value);
        }
        reg.engines[i].evaluate(graph, time);

        let mut node_writes: Vec<(NodeId, u16, FieldValue)> = Vec::new();
        for c in &reg.connections {
            if let (
                EngineEndpoint::Engine { id: src, port },
                EngineEndpoint::Node { node, field_index },
            ) = (&c.from, &c.to)
            {
                if src.0 == i {
                    if let Some(v) = reg.engines[i].output(port) {
                        node_writes.push((*node, *field_index, v));
                    }
                }
            }
        }
        for (node, field_index, value) in node_writes {
            write_node_field(graph, node, field_index, &value);
        }
    }

    let mut node_to_node: Vec<(NodeId, u16, NodeId, u16)> = Vec::new();
    for c in &reg.connections {
        if let (
            EngineEndpoint::Node {
                node: from,
                field_index: fi,
            },
            EngineEndpoint::Node {
                node: to,
                field_index: ti,
            },
        ) = (&c.from, &c.to)
        {
            node_to_node.push((*from, *fi, *to, *ti));
        }
    }
    for (from, fi, to, ti) in node_to_node {
        if let Some(v) = read_node_field(graph, from, fi) {
            write_node_field(graph, to, ti, &v);
        }
    }
}

pub(crate) fn is_trigger(v: &FieldValue) -> bool {
    match v {
        FieldValue::Bool(b) => *b,
        FieldValue::Int32(i) => *i != 0,
        FieldValue::Float(f) => *f != 0.0,
        FieldValue::Float64(f) => *f != 0.0,
        _ => true,
    }
}

pub(crate) fn calc_port_index(port: &str) -> Option<usize> {
    match port {
        "iA" | "oA" => Some(0),
        "iB" | "oB" => Some(1),
        "iC" | "oC" => Some(2),
        "iD" | "oD" => Some(3),
        "iE" | "oE" => Some(4),
        "iF" | "oF" => Some(5),
        "iG" | "oG" => Some(6),
        "iH" | "oH" => Some(7),
        _ => None,
    }
}

const CALC_INPUT_NAMES: [&str; 8] = ["iA", "iB", "iC", "iD", "iE", "iF", "iG", "iH"];

pub(crate) fn eval_calculator_expr(expr: &str, inputs: &[f32; 8]) -> Option<f32> {
    let parts: Vec<&str> = expr.split('=').collect();
    if parts.len() < 2 {
        return None;
    }
    let mut rhs = parts[1].trim().to_string();
    for (i, name) in CALC_INPUT_NAMES.iter().enumerate() {
        rhs = rhs.replace(name, &inputs[i].to_string());
    }
    eval_rhs(&rhs)
}

fn eval_rhs(s: &str) -> Option<f32> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(i) = find_op(s, b'+') {
        return Some(eval_rhs(&s[..i])? + eval_rhs(&s[i + 1..])?);
    }
    if let Some(i) = find_op(s, b'-') {
        return Some(eval_rhs(&s[..i])? - eval_rhs(&s[i + 1..])?);
    }
    if let Some(i) = find_op(s, b'*') {
        return Some(eval_rhs(&s[..i])? * eval_rhs(&s[i + 1..])?);
    }
    if let Some(i) = find_op(s, b'/') {
        let b = eval_rhs(&s[i + 1..])?;
        if b.abs() < 1e-12 {
            return None;
        }
        return Some(eval_rhs(&s[..i])? / b);
    }
    if s.starts_with("sin(") && s.ends_with(')') {
        return eval_rhs(&s[4..s.len() - 1]).map(f32::sin);
    }
    if s.starts_with("cos(") && s.ends_with(')') {
        return eval_rhs(&s[4..s.len() - 1]).map(f32::cos);
    }
    s.parse().ok()
}

fn find_op(s: &str, op: u8) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    for i in (0..bytes.len()).rev() {
        match bytes[i] {
            b')' => depth += 1,
            b'(' => depth -= 1,
            c if depth == 0 && c == op && i > 0 => return Some(i),
            _ => {}
        }
    }
    None
}
