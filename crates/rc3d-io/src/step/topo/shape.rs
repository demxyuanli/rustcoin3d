use super::edge::EdgeId;

/// Unique ID for topological shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeId {
    Face(u32),
    Shell(u32),
}

/// A face in the shared-topology B-rep.
#[derive(Debug, Clone)]
pub struct TopoFace {
    pub id: ShapeId,
    /// One outer loop, zero or more inner loops (holes)
    pub outer_loop: TopoLoop,
    pub inner_loops: Vec<TopoLoop>,
    /// STEP surface entity ID
    pub surface_entity_id: Option<u64>,
    pub same_sense: bool,
}

#[derive(Debug, Clone)]
pub struct TopoLoop {
    /// Ordered edge IDs with per-loop sense.
    /// Each entry: (edge_id, reversed_in_this_loop)
    pub edges: Vec<(EdgeId, bool)>,
}

#[derive(Debug, Clone)]
pub struct TopoShell {
    pub id: ShapeId,
    pub faces: Vec<TopoFace>,
    pub is_closed: bool,
}
