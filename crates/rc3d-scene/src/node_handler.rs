use std::fmt::Debug;

use rc3d_core::NodeId;

use crate::SceneGraph;

/// Custom traversal for `NodeData::HandlerNode` without growing the core `NodeData` enum.
pub trait NodeHandler: Debug + Send + Sync {
    fn handler_name(&self) -> &'static str;

    fn traverse(&self, _graph: &SceneGraph, _node: NodeId, children: &[NodeId], recur: &mut dyn FnMut(NodeId)) {
        for &c in children {
            recur(c);
        }
    }
}

/// Placeholder handler produced by deserialization; triggers default child traversal.
#[derive(Debug)]
pub struct DummyHandler;

impl NodeHandler for DummyHandler {
    fn handler_name(&self) -> &'static str {
        "DummyHandler"
    }
}
