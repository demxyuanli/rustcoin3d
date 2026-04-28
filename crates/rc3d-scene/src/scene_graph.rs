use std::collections::HashSet;

use rc3d_core::NodeId;
use slotmap::SlotMap;

use crate::node_data::NodeData;
use crate::node_entry::NodeEntry;

pub struct SceneGraph {
    nodes: SlotMap<NodeId, NodeEntry>,
    roots: Vec<NodeId>,
    selected: HashSet<NodeId>,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            roots: Vec::new(),
            selected: HashSet::new(),
        }
    }

    pub fn add_root(&mut self, data: NodeData) -> NodeId {
        let id = self.nodes.insert(NodeEntry {
            data,
            parent: None,
            children: Vec::new(),
            name: None,
            display_mode: None,
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
            display_mode: None,
        });
        if let Some(entry) = self.nodes.get_mut(parent) {
            entry.children.push(id);
        }
        id
    }

    pub fn insert_child(&mut self, parent: NodeId, index: usize, data: NodeData) -> NodeId {
        let id = self.nodes.insert(NodeEntry {
            data,
            parent: Some(parent),
            children: Vec::new(),
            name: None,
            display_mode: None,
        });
        if let Some(entry) = self.nodes.get_mut(parent) {
            let idx = index.min(entry.children.len());
            entry.children.insert(idx, id);
        }
        id
    }

    pub fn remove(&mut self, id: NodeId) {
        if let Some(entry) = self.nodes.remove(id) {
            if let Some(parent_id) = entry.parent {
                if let Some(parent) = self.nodes.get_mut(parent_id) {
                    parent.children.retain(|&c| c != id);
                }
            }
            // Recursively remove orphaned children
            for child in entry.children {
                self.remove(child);
            }
        }
        self.roots.retain(|&r| r != id);
        self.selected.remove(&id);
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

    pub fn select(&mut self, id: NodeId) {
        if self.nodes.contains_key(id) {
            self.selected.insert(id);
        }
    }

    pub fn deselect(&mut self, id: NodeId) {
        self.selected.remove(&id);
    }

    pub fn toggle_selection(&mut self, id: NodeId) {
        if self.selected.contains(&id) {
            self.selected.remove(&id);
        } else if self.nodes.contains_key(id) {
            self.selected.insert(id);
        }
    }

    pub fn clear_selection(&mut self) {
        self.selected.clear();
    }

    pub fn is_selected(&self, id: NodeId) -> bool {
        self.selected.contains(&id)
    }

    pub fn selected_nodes(&self) -> &HashSet<NodeId> {
        &self.selected
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}
