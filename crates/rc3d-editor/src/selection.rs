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

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64) -> NodeId {
        NodeId::from(slotmap::KeyData::from_ffi(id))
    }

    #[test]
    fn selection_new_is_empty() {
        let s = Selection::new();
        assert!(s.is_empty());
        assert!(s.nodes.is_empty());
        assert!(s.primary.is_none());
    }

    #[test]
    fn selection_set_single_sets_primary() {
        let mut s = Selection::new();
        let n = node(1);
        s.set_single(n);
        assert_eq!(s.primary, Some(n));
        assert!(s.nodes.contains(&n));
        assert_eq!(s.nodes.len(), 1);
    }

    #[test]
    fn selection_toggle_adds_and_removes() {
        let mut s = Selection::new();
        let n = node(1);

        // First toggle: add
        s.toggle(n);
        assert!(s.nodes.contains(&n));
        assert_eq!(s.primary, Some(n));

        // Second toggle: remove
        s.toggle(n);
        assert!(!s.nodes.contains(&n));
        assert!(s.is_empty());
        assert!(s.primary.is_none());
    }

    #[test]
    fn selection_clear_removes_all() {
        let mut s = Selection::new();
        s.set_single(node(1));
        s.toggle(node(2)); // now has {1,2}, primary=2
        assert_eq!(s.nodes.len(), 2);

        s.clear();
        assert!(s.is_empty());
        assert!(s.primary.is_none());
        assert!(s.nodes.is_empty());
    }

    #[test]
    fn selection_toggle_primary_falls_back_to_another() {
        let mut s = Selection::new();
        let a = node(1);
        let b = node(2);
        s.set_single(a);
        s.toggle(b); // now nodes = {a, b}, primary = b
        s.toggle(b); // remove b, primary should fall back to a
        assert!(s.nodes.contains(&a));
        assert!(!s.nodes.contains(&b));
        assert_eq!(s.primary, Some(a));
    }
}
