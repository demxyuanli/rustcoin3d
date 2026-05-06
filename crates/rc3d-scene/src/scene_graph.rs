use std::collections::{HashMap, HashSet};

use rc3d_core::NodeId;
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;

use crate::node_data::NodeData;
use crate::node_entry::NodeEntry;

/// Core scene container: a hierarchical directed graph of typed nodes.
///
/// Nodes are stored in a `SlotMap` for O(1) access and stable IDs.
/// Supports root nodes, parent-child relationships, selection, and
/// subtree operations. Serializable via serde (skips transient selection state).
#[derive(Serialize, Deserialize)]
pub struct SceneGraph {
    nodes: SlotMap<NodeId, NodeEntry>,
    /// Top-level nodes with no parent.
    roots: Vec<NodeId>,
    #[serde(default, skip_serializing)]
    selected: HashSet<NodeId>,
    #[serde(default, skip_serializing)]
    selection_sets: HashMap<String, HashSet<NodeId>>,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            roots: Vec::new(),
            selected: HashSet::new(),
            selection_sets: HashMap::new(),
        }
    }

    pub fn add_root(&mut self, data: NodeData) -> NodeId {
        let id = self.nodes.insert(NodeEntry {
            data,
            parent: None,
            children: Vec::new(),
            name: None,
            display_mode: None,
            fields: rc3d_fields::FieldMap::new(),
            attributes: std::collections::HashMap::new(),
        });
        self.roots.push(id);
        id
    }

    pub fn add_child(&mut self, parent: NodeId, data: NodeData) -> NodeId {
        if !self.nodes.contains_key(parent) {
            return NodeId::from(slotmap::KeyData::default());
        }
        let id = self.nodes.insert(NodeEntry {
            data,
            parent: Some(parent),
            children: Vec::new(),
            name: None,
            display_mode: None,
            fields: rc3d_fields::FieldMap::new(),
            attributes: std::collections::HashMap::new(),
        });
        if let Some(entry) = self.nodes.get_mut(parent) {
            entry.children.push(id);
        }
        id
    }

    pub fn insert_child(&mut self, parent: NodeId, index: usize, data: NodeData) -> NodeId {
        if !self.nodes.contains_key(parent) {
            return NodeId::from(slotmap::KeyData::default());
        }
        let id = self.nodes.insert(NodeEntry {
            data,
            parent: Some(parent),
            children: Vec::new(),
            name: None,
            display_mode: None,
            fields: rc3d_fields::FieldMap::new(),
            attributes: std::collections::HashMap::new(),
        });
        if let Some(entry) = self.nodes.get_mut(parent) {
            let idx = index.min(entry.children.len());
            entry.children.insert(idx, id);
        }
        id
    }

    pub fn remove(&mut self, id: NodeId) {
        // Extract needed data before any mutation (avoid borrow conflict)
        let (parent_id, children) = match self.nodes.get(id) {
            Some(entry) => (entry.parent, entry.children.clone()),
            None => return,
        };

        // Unlink from parent
        if let Some(pid) = parent_id {
            if let Some(parent) = self.nodes.get_mut(pid) {
                parent.children.retain(|&c| c != id);
            }
            self.roots.retain(|&r| r != id);
        } else {
            self.roots.retain(|&r| r != id);
        }

        // Iterative removal of subtree (avoid stack overflow on deep chains)
        self.nodes.remove(id);
        self.selected.remove(&id);

        let mut stack: Vec<NodeId> = children;
        while let Some(node_id) = stack.pop() {
            if let Some(child) = self.nodes.get(node_id) {
                stack.extend(child.children.clone());
                self.selected.remove(&node_id);
            }
            self.nodes.remove(node_id);
        }
    }

    pub fn get(&self, id: NodeId) -> Option<&NodeEntry> {
        self.nodes.get(id)
    }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut NodeEntry> {
        self.nodes.get_mut(id)
    }

    pub fn children(&self, id: NodeId) -> Option<&[NodeId]> {
        self.nodes.get(id).map(|e| e.children.as_slice())
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

    /// Add all existing nodes from `ids` to the selection.
    pub fn select_many(&mut self, ids: impl IntoIterator<Item = NodeId>) {
        for id in ids {
            if self.nodes.contains_key(id) {
                self.selected.insert(id);
            }
        }
    }

    pub fn is_selected(&self, id: NodeId) -> bool {
        self.selected.contains(&id)
    }

    pub fn selected_nodes(&self) -> &HashSet<NodeId> {
        &self.selected
    }

    /// Named selection sets for group operations.
    pub fn selection_set(&self, name: &str) -> Option<&HashSet<NodeId>> {
        self.selection_sets.get(name)
    }

    pub fn selection_set_mut(&mut self, name: &str) -> &mut HashSet<NodeId> {
        self.selection_sets.entry(name.to_string()).or_default()
    }

    pub fn selection_set_names(&self) -> impl Iterator<Item = &String> {
        self.selection_sets.keys()
    }

    /// Add nodes to a named selection set (creating it if needed).
    pub fn selection_set_add(&mut self, name: &str, ids: &[NodeId]) {
        let valid: Vec<NodeId> = ids.iter().filter(|&&id| self.nodes.contains_key(id)).copied().collect();
        let set = self.selection_set_mut(name);
        for id in valid {
            set.insert(id);
        }
    }

    /// Remove nodes from a named selection set.
    pub fn selection_set_remove(&mut self, name: &str, ids: &[NodeId]) {
        if let Some(set) = self.selection_sets.get_mut(name) {
            for &id in ids {
                set.remove(&id);
            }
        }
    }

    /// Apply a named selection set to the active selection.
    pub fn selection_set_select(&mut self, name: &str) {
        if let Some(set) = self.selection_sets.get(name) {
            self.selected = set.clone();
        }
    }

    /// Delete a named selection set.
    pub fn selection_set_delete(&mut self, name: &str) -> bool {
        self.selection_sets.remove(name).is_some()
    }

    /// Marks every field on each node in the subtree as dirty (e.g. after a structural edit).
    pub fn mark_fields_dirty_subtree(&mut self, root: NodeId) {
        let ids = self.subtree_preorder_ids(root);
        for id in ids {
            if let Some(e) = self.get_mut(id) {
                e.fields.mark_all_dirty();
            }
        }
    }

    fn subtree_preorder_ids(&self, root: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            out.push(id);
            if let Some(e) = self.get(id) {
                for &c in e.children.iter().rev() {
                    stack.push(c);
                }
            }
        }
        out
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_data::{MaterialNode, TransformNode, GroupNode};

    fn make_material() -> NodeData {
        NodeData::Material(MaterialNode::default())
    }

    fn make_group() -> NodeData {
        NodeData::Group(GroupNode)
    }

    // ── Root ops ──

    #[test]
    fn test_add_root_has_no_parent() {
        let mut g = SceneGraph::new();
        let id = g.add_root(make_group());
        assert_eq!(g.roots(), &[id]);
        assert!(g.get(id).unwrap().parent.is_none());
    }

    #[test]
    fn test_multiple_roots() {
        let mut g = SceneGraph::new();
        let a = g.add_root(make_group());
        let b = g.add_root(make_material());
        assert_eq!(g.roots().len(), 2);
        assert!(g.roots().contains(&a));
        assert!(g.roots().contains(&b));
    }

    // ── Child ops ──

    #[test]
    fn test_add_child_to_nonexistent_parent_returns_default() {
        let mut g = SceneGraph::new();
        let bad_parent = NodeId::default();
        let id = g.add_child(bad_parent, make_group());
        assert_eq!(id, NodeId::default());
    }

    #[test]
    fn test_insert_child_at_index() {
        let mut g = SceneGraph::new();
        let root = g.add_root(make_group());
        let a = g.add_child(root, make_material());
        let b = g.insert_child(root, 0, make_material());
        let children = g.children(root).unwrap();
        assert_eq!(children[0], b); // inserted at 0, before a
        assert_eq!(children[1], a);
    }

    #[test]
    fn test_insert_child_index_clamped() {
        let mut g = SceneGraph::new();
        let root = g.add_root(make_group());
        let a = g.add_child(root, make_material());
        // index 999 should clamp to children.len()
        let b = g.insert_child(root, 999, make_material());
        let children = g.children(root).unwrap();
        assert_eq!(children[children.len() - 1], b);
        assert!(children.contains(&a));
    }

    // ── Attributes ──

    #[test]
    fn test_node_attributes_empty_by_default() {
        let mut g = SceneGraph::new();
        let id = g.add_root(make_group());
        assert!(g.get(id).unwrap().attributes.is_empty());
    }

    #[test]
    fn test_node_attributes_set_and_read() {
        let mut g = SceneGraph::new();
        let id = g.add_root(make_material());
        if let Some(e) = g.get_mut(id) {
            e.attributes.insert("part_number".into(), "PN-001".into());
        }
        assert_eq!(
            g.get(id).unwrap().attributes.get("part_number"),
            Some(&"PN-001".to_string())
        );
    }

    #[test]
    fn test_attributes_survive_node_operations() {
        let mut g = SceneGraph::new();
        let root = g.add_root(make_group());
        if let Some(e) = g.get_mut(root) {
            e.attributes.insert("key".into(), "val".into());
        }
        let child = g.add_child(root, make_material());
        // Child has independent (empty) attributes
        assert!(g.get(child).unwrap().attributes.is_empty());
        // Root's attributes unchanged
        assert_eq!(
            g.get(root).unwrap().attributes.get("key"),
            Some(&"val".to_string())
        );
    }

    // ── Hierarchical containment ──

    #[test]
    fn test_root_is_not_child() {
        let mut g = SceneGraph::new();
        let root = g.add_root(make_group());
        let child = g.add_child(root, make_material());
        assert_eq!(g.roots(), &[root]);
        assert!(g.children(root).unwrap().contains(&child));
    }

    #[test]
    fn test_subtree_mark_dirty_reaches_all_nodes() {
        let mut g = SceneGraph::new();
        let root = g.add_root(make_group());
        let a = g.add_child(root, make_material());
        let b = g.add_child(root, make_material());
        // Add a field to each node so any_dirty() can detect dirtiness
        for &id in &[root, a, b] {
            if let Some(e) = g.get_mut(id) {
                e.fields.insert(id, 0, rc3d_fields::FieldValue::Bool(false));
            }
        }

        g.mark_fields_dirty_subtree(root);
        assert!(g.get(root).unwrap().fields.any_dirty());
        assert!(g.get(a).unwrap().fields.any_dirty());
        assert!(g.get(b).unwrap().fields.any_dirty());
    }
}

#[cfg(test)]
mod serde_tests {
    use super::*;
    use crate::node_data::*;

    fn make_test_scene() -> SceneGraph {
        let mut g = SceneGraph::new();
        let root = g.add_root(NodeData::Separator(SeparatorNode));
        let mat_id = g.add_child(
            root,
            NodeData::Material(MaterialNode {
                base_color: rc3d_core::math::Vec3::new(0.8, 0.2, 0.2),
                metallic: 0.5,
                roughness: 0.3,
                ..Default::default()
            }),
        );
        let cube = g.add_child(root, NodeData::Cube(CubeNode::default()));
        let light = g.add_child(
            root,
            NodeData::DirectionalLight(DirectionalLightNode::default()),
        );
        let camera = g.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::default()),
        );
        let markup = g.add_child(
            root,
            NodeData::Markup(MarkupNode {
                elements: vec![MarkupElement::Line {
                    start: [0.0, 0.0],
                    end: [100.0, 100.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                    width: 2.0,
                }],
                layer_name: "layer1".into(),
                visible: true,
            }),
        );
        g
    }

    #[test]
    fn test_scene_graph_json_roundtrip() {
        let g = make_test_scene();
        let json = serde_json::to_string_pretty(&g).expect("serialize");
        let g2: SceneGraph = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(g2.roots().len(), g.roots().len());
        // Root node exists
        let root = g2.roots()[0];
        assert!(g2.get(root).is_some());
    }

    #[test]
    fn test_node_data_roundtrip_via_json() {
        let nd = NodeData::Material(MaterialNode {
            base_color: rc3d_core::math::Vec3::new(1.0, 0.0, 0.0),
            metallic: 1.0,
            roughness: 0.2,
            ..Default::default()
        });
        let json = serde_json::to_string(&nd).expect("serialize");
        let nd2: NodeData = serde_json::from_str(&json).expect("deserialize");
        match nd2 {
            NodeData::Material(m) => {
                assert_eq!(m.base_color, rc3d_core::math::Vec3::new(1.0, 0.0, 0.0));
                assert_eq!(m.metallic, 1.0);
                assert_eq!(m.roughness, 0.2);
            }
            _ => panic!("expected Material"),
        }
    }

    #[test]
    fn test_handler_node_roundtrip() {
        let nd = NodeData::HandlerNode(std::sync::Arc::new(
            crate::node_handler::DummyHandler,
        ));
        let json = serde_json::to_string(&nd).expect("serialize");
        let nd2: NodeData = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(nd2, NodeData::HandlerNode(_)));
    }

    #[test]
    fn test_field_value_serde() {
        use rc3d_fields::FieldValue;
        let cases = vec![
            FieldValue::Bool(true),
            FieldValue::Float(3.14),
            FieldValue::String("hello".into()),
            FieldValue::Vec3f(rc3d_core::math::Vec3::new(1.0, 2.0, 3.0)),
        ];
        for fv in cases {
            let json = serde_json::to_string(&fv).expect("serialize");
            let fv2: FieldValue = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(fv, fv2);
        }
    }

    #[test]
    fn test_empty_scene_roundtrip() {
        let g = SceneGraph::new();
        let json = serde_json::to_string(&g).expect("serialize");
        let g2: SceneGraph = serde_json::from_str(&json).expect("deserialize");
        assert!(g2.roots().is_empty());
    }
}
