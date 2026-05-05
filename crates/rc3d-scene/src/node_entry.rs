use std::collections::HashMap;

use rc3d_core::{DisplayMode, NodeId};
use rc3d_fields::FieldMap;

use crate::node_data::NodeData;

pub struct NodeEntry {
    pub data: NodeData,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub name: Option<String>,
    pub display_mode: Option<DisplayMode>,
    pub fields: FieldMap,
    /// User-defined key-value attributes (e.g. part_number, material_grade).
    pub attributes: HashMap<String, String>,
}
