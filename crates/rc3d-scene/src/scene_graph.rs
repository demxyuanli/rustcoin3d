use std::collections::{HashMap, HashSet};

use rc3d_core::{DisplayMode, EdgeStyle, FillStyle, NodeId, VisualStyle};
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;

use crate::node_data::{EdgeTint, FaceTint, NodeData};
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
    /// HOOPS-style sub-entity face colors (runtime; not serialized).
    #[serde(default, skip_serializing)]
    face_tints: HashMap<NodeId, Vec<FaceTint>>,
    /// HOOPS-style sub-entity edge colors (runtime; not serialized).
    #[serde(default, skip_serializing)]
    edge_tints: HashMap<NodeId, Vec<EdgeTint>>,
    /// Cross-node field connections (`SoField::connectFrom`).
    #[serde(default)]
    pub(crate) field_graph: crate::field_graph::FieldGraph,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            roots: Vec::new(),
            selected: HashSet::new(),
            selection_sets: HashMap::new(),
            face_tints: HashMap::new(),
            edge_tints: HashMap::new(),
            field_graph: crate::field_graph::FieldGraph::default(),
        }
    }

    pub fn add_root(&mut self, data: NodeData) -> NodeId {
        let id = self.nodes.insert(NodeEntry {
            data,
            parent: None,
            children: Vec::new(),
            name: None,
            display_mode: None,
            fill_style: None,
            edge_style: None,
            fields: rc3d_fields::FieldMap::new(),
            attributes: std::collections::HashMap::new(),
            dirty_flags: 0,
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
            fill_style: None,
            edge_style: None,
            fields: rc3d_fields::FieldMap::new(),
            attributes: std::collections::HashMap::new(),
            dirty_flags: 0,
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
            fill_style: None,
            edge_style: None,
            fields: rc3d_fields::FieldMap::new(),
            attributes: std::collections::HashMap::new(),
            dirty_flags: 0,
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

    /// Coin3D `SoDrawStyle` on this node: Separator isolates the style to its subtree.
    /// Clears independent fill/edge overrides so the preset is the full appearance.
    pub fn set_display_mode(&mut self, id: NodeId, mode: DisplayMode) {
        if let Some(entry) = self.nodes.get_mut(id) {
            entry.display_mode = Some(mode);
            entry.fill_style = None;
            entry.edge_style = None;
        }
    }

    /// Clear a per-node display-mode override (inherit parent / global again).
    pub fn clear_display_mode(&mut self, id: NodeId) {
        if let Some(entry) = self.nodes.get_mut(id) {
            entry.display_mode = None;
            entry.fill_style = None;
            entry.edge_style = None;
        }
    }

    pub fn set_fill_style(&mut self, id: NodeId, fill: FillStyle) {
        if let Some(entry) = self.nodes.get_mut(id) {
            entry.fill_style = Some(fill);
        }
    }

    pub fn set_edge_style(&mut self, id: NodeId, edges: EdgeStyle) {
        if let Some(entry) = self.nodes.get_mut(id) {
            entry.edge_style = Some(edges);
        }
    }

    pub fn clear_fill_style(&mut self, id: NodeId) {
        if let Some(entry) = self.nodes.get_mut(id) {
            entry.fill_style = None;
        }
    }

    pub fn clear_edge_style(&mut self, id: NodeId) {
        if let Some(entry) = self.nodes.get_mut(id) {
            entry.edge_style = None;
        }
    }

    pub fn set_name(&mut self, id: NodeId, name: impl Into<String>) {
        if let Some(entry) = self.nodes.get_mut(id) {
            entry.name = Some(name.into());
        }
    }

    pub fn find_named(&self, name: &str) -> Option<NodeId> {
        self.nodes.iter().find_map(|(id, e)| {
            e.name.as_deref().filter(|n| *n == name).map(|_| id)
        })
    }

    pub fn all_node_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().collect()
    }

    /// Resolve PMI names on every `AnnotationSet` and stamp unbound points.
    pub fn bind_pmi(&mut self) -> usize {
        crate::annotation::bind_scene_pmi(self)
    }

    /// Apply a named HOOPS-style visual style to this node (typically a Separator).
    pub fn apply_visual_style(&mut self, id: NodeId, style: &VisualStyle) {
        if let Some(entry) = self.nodes.get_mut(id) {
            entry.display_mode = Some(style.appearance.to_display_mode());
            entry.fill_style = Some(style.appearance.fill);
            entry.edge_style = Some(style.appearance.edges);
        }
    }

    pub fn children(&self, id: NodeId) -> Option<&[NodeId]> {
        self.nodes.get(id).map(|e| e.children.as_slice())
    }

    /// Set `LodNode.range_scale` on `id`, or the first Lod descendant.
    pub fn set_lod_range_scale(&mut self, id: NodeId, scale: f32) -> bool {
        let scale = scale.max(1e-4);
        if self.apply_lod_range_scale(id, scale) {
            return true;
        }
        let mut stack = match self.children(id) {
            Some(c) => c.to_vec(),
            None => return false,
        };
        while let Some(n) = stack.pop() {
            if self.apply_lod_range_scale(n, scale) {
                return true;
            }
            if let Some(c) = self.children(n) {
                stack.extend(c.iter().copied());
            }
        }
        false
    }

    fn apply_lod_range_scale(&mut self, id: NodeId, scale: f32) -> bool {
        if let Some(entry) = self.get_mut(id) {
            if let NodeData::Lod(lod) = &mut entry.data {
                lod.range_scale = scale;
                return true;
            }
        }
        false
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

    /// Map a picked triangle index to a CAD face id (cube: `tri / 2`; IFS: `face_ids`).
    pub fn face_id_from_triangle(&self, node: NodeId, triangle: u32) -> u32 {
        let Some(entry) = self.get(node) else {
            return triangle;
        };
        match &entry.data {
            NodeData::IndexedFaceSet(ifs) => ifs.face_id(triangle),
            NodeData::Cube(_) => triangle / 2,
            _ => triangle,
        }
    }

    pub fn set_face_tint(&mut self, node: NodeId, id: u32, color: [f32; 4]) {
        if !self.nodes.contains_key(node) {
            return;
        }
        let tints = self.face_tints.entry(node).or_default();
        if let Some(existing) = tints.iter_mut().find(|t| t.id == id) {
            existing.color = color;
        } else {
            tints.push(FaceTint { id, color });
        }
    }

    pub fn clear_face_tints(&mut self, node: NodeId) {
        self.face_tints.remove(&node);
    }

    pub fn face_tints(&self, node: NodeId) -> &[FaceTint] {
        self.face_tints
            .get(&node)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn set_edge_tint(&mut self, node: NodeId, triangle: u32, edge: u8, color: [f32; 4]) {
        if !self.nodes.contains_key(node) {
            return;
        }
        let edge = edge.min(2);
        let tints = self.edge_tints.entry(node).or_default();
        if let Some(existing) = tints
            .iter_mut()
            .find(|t| t.triangle == triangle && t.edge == edge)
        {
            existing.color = color;
        } else {
            tints.push(EdgeTint {
                triangle,
                edge,
                color,
            });
        }
    }

    pub fn clear_edge_tints(&mut self, node: NodeId) {
        self.edge_tints.remove(&node);
    }

    pub fn edge_tints(&self, node: NodeId) -> &[EdgeTint] {
        self.edge_tints
            .get(&node)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
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

    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.nodes.get(id).and_then(|e| e.parent)
    }

    /// True if this node or any ancestor is selected (HOOPS isolate covers a subtree).
    pub fn is_in_selection(&self, id: NodeId) -> bool {
        let mut cur = Some(id);
        while let Some(nid) = cur {
            if self.selected.contains(&nid) {
                return true;
            }
            cur = self.parent(nid);
        }
        false
    }

    pub fn selected_nodes(&self) -> &HashSet<NodeId> {
        &self.selected
    }

    /// Clear dirty flags on all nodes by iterating the SlotMap directly (zero allocation).
    pub fn clear_all_dirty_flags(&mut self) {
        for (_id, entry) in self.nodes.iter_mut() {
            entry.dirty_flags = 0;
        }
    }

    /// Check if any node has dirty flags set (early-out on first hit).
    pub fn has_any_dirty(&self) -> bool {
        self.nodes.values().any(|e| e.dirty_flags != 0)
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
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

    /// Update all `LodNode` current_level fields based on camera distance.
    /// Walk from each root accumulating Transform matrices, then call
    /// `LodNode::select_level(distance)` for each LodNode found.
    pub fn update_lod_levels(&mut self, camera_pos: rc3d_core::math::Vec3) {
        let roots = self.roots().to_vec();
        for &root in &roots {
            self.update_lod_recursive(root, camera_pos, rc3d_core::math::Mat4::IDENTITY);
        }
    }

    fn update_lod_recursive(
        &mut self,
        node: NodeId,
        camera_pos: rc3d_core::math::Vec3,
        model: rc3d_core::math::Mat4,
    ) -> rc3d_core::math::Mat4 {
        // Read transform and children under immutable borrow
        let (local_model, is_lod, children) = {
            let entry = match self.get(node) {
                Some(e) => e,
                None => return model,
            };
            let local_model = match &entry.data {
                NodeData::Transform(t) => {
                    let xform = rc3d_core::math::Mat4::from_translation(-t.center)
                        * rc3d_core::math::Mat4::from_scale(t.scale)
                        * t.rotation
                        * rc3d_core::math::Mat4::from_translation(t.center + t.translation);
                    model * xform
                }
                NodeData::Rotation(r) => model * r.to_matrix(),
                NodeData::RotationXYZ(r) => model * r.to_matrix(),
                _ => model,
            };
            let is_lod = matches!(&entry.data, NodeData::Lod(_));
            let children = entry.children.to_vec();
            (local_model, is_lod, children)
        };
        // Update LOD level under mutable borrow
        if is_lod {
            if let Some(entry) = self.get_mut(node) {
                if let NodeData::Lod(lod) = &mut entry.data {
                    let (_, _, trans) = local_model.to_scale_rotation_translation();
                    let distance = (trans - camera_pos).length();
                    let scaled = distance / lod.range_scale.max(1e-4);
                    lod.select_level(scaled);
                }
            }
        }
        for child in children {
            self.update_lod_recursive(child, camera_pos, local_model);
        }
        local_model
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
    use crate::node_data::{MaterialNode, GroupNode};

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
        let _mat_id = g.add_child(
            root,
            NodeData::Material(MaterialNode {
                base_color: rc3d_core::math::Vec3::new(0.8, 0.2, 0.2),
                metallic: 0.5,
                roughness: 0.3,
                ..Default::default()
            }),
        );
        let _cube = g.add_child(root, NodeData::Cube(CubeNode::default()));
        let _light = g.add_child(
            root,
            NodeData::DirectionalLight(DirectionalLightNode::default()),
        );
        let _camera = g.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::default()),
        );
        let _markup = g.add_child(
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
