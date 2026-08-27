use std::collections::HashMap;

use rc3d_core::{Appearance, DisplayMode, EdgeStyle, FillStyle, NodeId};
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
    /// Overrides fill from `display_mode` / parent when `Some`.
    #[serde(default)]
    pub fill_style: Option<FillStyle>,
    /// Overrides edges from `display_mode` / parent when `Some`.
    #[serde(default)]
    pub edge_style: Option<EdgeStyle>,
    pub fields: FieldMap,
    /// User-defined key-value attributes (e.g. part_number, material_grade).
    pub attributes: HashMap<String, String>,
}

impl NodeEntry {
    pub fn appearance(&self, inherited: Appearance) -> Appearance {
        Appearance::resolve(inherited, self.display_mode, self.fill_style, self.edge_style)
    }
}
