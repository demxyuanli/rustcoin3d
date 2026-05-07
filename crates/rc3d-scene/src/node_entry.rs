use std::collections::HashMap;

use rc3d_core::{DisplayMode, NodeId};
use rc3d_fields::FieldMap;
use serde::{Deserialize, Serialize};

use crate::node_data::NodeData;

/// Dirty flag bits for incremental render cache updates.
pub mod dirty_flags {
    pub const TRANSFORM: u8 = 1 << 0;
    pub const MATERIAL: u8  = 1 << 1;
    pub const GEOMETRY: u8  = 1 << 2;
    pub const CHILDREN: u8  = 1 << 3;  // child added/removed/reordered
    pub const REMOVED: u8   = 1 << 4;
    pub const FROZEN: u8    = 1 << 7;  // static subtree, skip traversal
}

#[derive(Serialize, Deserialize)]
pub struct NodeEntry {
    pub data: NodeData,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    /// Bitflag tracking which aspects of this node have changed since the last
    /// render cache update. See `dirty_flags` module for bit definitions.
    pub dirty_flags: u8,
    pub name: Option<String>,
    pub display_mode: Option<DisplayMode>,
    pub fields: FieldMap,
    /// User-defined key-value attributes (e.g. part_number, material_grade).
    pub attributes: HashMap<String, String>,
}
