use rc3d_core::NodeId;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Selection {
    pub nodes: HashSet<NodeId>,
    pub primary: Option<NodeId>,
}

impl Selection {
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            primary: None,
        }
    }
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.primary = None;
    }
    pub fn set_single(&mut self, node: NodeId) {
        self.clear();
        self.nodes.insert(node);
        self.primary = Some(node);
    }
    pub fn toggle(&mut self, node: NodeId) {
        if self.nodes.contains(&node) {
            self.nodes.remove(&node);
            if self.primary == Some(node) {
                self.primary = self.nodes.iter().next().copied();
            }
        } else {
            self.nodes.insert(node);
            self.primary = Some(node);
        }
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for Selection {
    fn default() -> Self {
        Self::new()
    }
}
