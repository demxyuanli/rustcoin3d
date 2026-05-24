use std::collections::HashMap;
use super::vertex::VertexId;

/// Edge sense relative to its defining curve direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeSense {
    Forward,
    Reversed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32);

#[derive(Debug, Clone)]
pub struct TopoEdge {
    pub id: EdgeId,
    pub start: VertexId,
    pub end: VertexId,
    /// Entity ID of the STEP curve geometry (0 = synthetic/line segment)
    pub curve_entity_id: u64,
    pub sense: EdgeSense,
    /// Curve tolerance (from GLOBAL_UNCERTAINTY or EDGE_CURVE)
    pub tolerance: f32,
}

#[derive(Debug, Default)]
pub struct EdgeRegistry {
    edges: Vec<TopoEdge>,
    /// (start.0, end.0) → edge index
    index: HashMap<(u32, u32), u32>,
}

impl EdgeRegistry {
    pub fn new() -> Self { Self::default() }

    /// Insert an edge. Deduplicates by (start, end) vertex pair.
    pub fn insert(&mut self, start: VertexId, end: VertexId,
                  curve_entity_id: u64, tolerance: f32) -> EdgeId {
        let key = (start.0, end.0);
        if let Some(&idx) = self.index.get(&key) {
            return EdgeId(idx);
        }
        let id = EdgeId(self.edges.len() as u32);
        self.edges.push(TopoEdge {
            id, start, end, curve_entity_id,
            sense: EdgeSense::Forward, tolerance,
        });
        self.index.insert(key, id.0);
        id
    }

    pub fn get(&self, id: EdgeId) -> Option<&TopoEdge> {
        self.edges.get(id.0 as usize)
    }

    pub fn len(&self) -> usize { self.edges.len() }
}
