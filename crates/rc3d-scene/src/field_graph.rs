//! Cross-node field connection graph (Coin3D `SoField::connectFrom`).
//!
//! Directed value flow with reverse lookup (`field_sources` / `field_targets`).
//! Values do not ping-pong; cycles fall back to a single pass over listed edges.

use serde::{Deserialize, Serialize};

use rc3d_core::NodeId;
use rc3d_core::utils::graph::toposort_linear;

use crate::field_access::{read_node_field, write_node_field};
use crate::SceneGraph;

/// Identity of a typed node field (`NodeData::field_descriptors` index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldRef {
    pub node: NodeId,
    pub field_index: u16,
}

impl FieldRef {
    pub fn new(node: NodeId, field_index: u16) -> Self {
        Self { node, field_index }
    }
}

/// Directed edges between [`FieldRef`]s. Empty graph is a no-op on propagate.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FieldGraph {
    pub edges: Vec<(FieldRef, FieldRef)>,
}

impl SceneGraph {
    pub fn field_graph(&self) -> &FieldGraph {
        &self.field_graph
    }

    /// Connect `from` → `to`. Returns false if the edge already exists or is a self-loop.
    pub fn connect_fields(&mut self, from: FieldRef, to: FieldRef) -> bool {
        if from == to {
            return false;
        }
        if self.field_graph.edges.iter().any(|&e| e == (from, to)) {
            return false;
        }
        self.field_graph.edges.push((from, to));
        true
    }

    pub fn disconnect_fields(&mut self, from: FieldRef, to: FieldRef) -> bool {
        let before = self.field_graph.edges.len();
        self.field_graph.edges.retain(|&e| e != (from, to));
        self.field_graph.edges.len() != before
    }

    pub fn field_targets(&self, from: FieldRef) -> Vec<FieldRef> {
        self.field_graph
            .edges
            .iter()
            .filter_map(|&(f, t)| if f == from { Some(t) } else { None })
            .collect()
    }

    pub fn field_sources(&self, to: FieldRef) -> Vec<FieldRef> {
        self.field_graph
            .edges
            .iter()
            .filter_map(|&(f, t)| if t == to { Some(f) } else { None })
            .collect()
    }

    /// Copy values along edges. Engines should run first so sources are up to date.
    pub fn propagate_fields(&mut self) {
        if self.field_graph.edges.is_empty() {
            return;
        }

        let mut index_of: Vec<FieldRef> = Vec::new();
        for &(from, to) in &self.field_graph.edges {
            if !index_of.contains(&from) {
                index_of.push(from);
            }
            if !index_of.contains(&to) {
                index_of.push(to);
            }
        }
        let n = index_of.len();
        let idx = |r: FieldRef| index_of.iter().position(|&x| x == r);

        let mut edges: Vec<(usize, usize)> = Vec::new();
        for &(from, to) in &self.field_graph.edges {
            if let (Some(a), Some(b)) = (idx(from), idx(to)) {
                edges.push((a, b));
            }
        }

        let listed = self.field_graph.edges.clone();
        match toposort_linear(&edges, n) {
            Ok(order) => {
                for &i in &order {
                    let src = index_of[i];
                    let Some(value) = read_node_field(self, src.node, src.field_index) else {
                        continue;
                    };
                    for &(from, to) in &listed {
                        if from == src {
                            write_node_field(self, to.node, to.field_index, &value);
                        }
                    }
                }
            }
            Err(()) => {
                for (from, to) in listed {
                    if let Some(value) = read_node_field(self, from.node, from.field_index) {
                        write_node_field(self, to.node, to.field_index, &value);
                    }
                }
            }
        }
    }
}
