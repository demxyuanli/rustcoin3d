use rc3d_core::NodeId;

use crate::node_data::NodeData;

pub struct NodeEntry {
    pub data: NodeData,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub name: Option<String>,
}
