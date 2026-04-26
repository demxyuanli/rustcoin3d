use rc3d_core::NodeId;
use slotmap::SlotMap;

use crate::node_data::NodeData;
use crate::node_entry::NodeEntry;

pub struct SceneGraph {
    nodes: SlotMap<NodeId, NodeEntry>,
    roots: Vec<NodeId>,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            roots: Vec::new(),
        }
    }

    pub fn add_root(&mut self, data: NodeData) -> NodeId {
        let id = self.nodes.insert(NodeEntry {
            data,
            parent: None,
            children: Vec::new(),
            name: None,
        });
        self.roots.push(id);
        id
    }

    pub fn add_child(&mut self, parent: NodeId, data: NodeData) -> NodeId {
        let id = self.nodes.insert(NodeEntry {
            data,
            parent: Some(parent),
            children: Vec::new(),
            name: None,
        });
        if let Some(entry) = self.nodes.get_mut(parent) {
            entry.children.push(id);
        }
        id
    }

    pub fn remove(&mut self, id: NodeId) {
        // Remove from parent's children list
        if let Some(entry) = self.nodes.remove(id) {
            if let Some(parent_id) = entry.parent {
                if let Some(parent) = self.nodes.get_mut(parent_id) {
                    parent.children.retain(|&c| c != id);
                }
            }
        }
        // Remove from roots if present
        self.roots.retain(|&r| r != id);
    }

    pub fn get(&self, id: NodeId) -> Option<&NodeEntry> {
        self.nodes.get(id)
    }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut NodeEntry> {
        self.nodes.get_mut(id)
    }

    pub fn children(&self, id: NodeId) -> &[NodeId] {
        self.nodes
            .get(id)
            .map(|e| e.children.as_slice())
            .unwrap_or(&[])
    }

    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}
