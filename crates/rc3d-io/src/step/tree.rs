//! Assembly tree: preserves product hierarchy from STEP assemblies.

use std::collections::HashMap;
use rc3d_core::math::Mat4;

/// A node in the assembly tree.
#[derive(Debug, Clone)]
pub struct AssemblyNode {
    /// Product name (from PRODUCT entity)
    pub name: String,
    /// Product description
    pub description: String,
    /// Accumulated transform from root
    pub transform: Mat4,
    /// Child indices
    pub children: Vec<usize>,
    /// Shell entity IDs owned by this node
    pub shells: Vec<u64>,
    /// Product ID for reference
    pub product_id: u64,
}

/// Full assembly tree: flat node array with root index.
#[derive(Debug, Clone)]
pub struct AssemblyTree {
    pub nodes: Vec<AssemblyNode>,
    pub root_index: usize,
}

impl AssemblyTree {
    /// Walk the tree depth-first, applying a function to each node.
    pub fn walk<F>(&self, visitor: &mut F)
    where F: FnMut(&AssemblyNode, &Mat4, usize) // (node, world_transform, depth)
    {
        if !self.nodes.is_empty() {
            self.walk_node(self.root_index, &Mat4::IDENTITY, 0, visitor);
        }
    }

    fn walk_node<F>(&self, idx: usize, parent_xform: &Mat4, depth: usize, visitor: &mut F)
    where F: FnMut(&AssemblyNode, &Mat4, usize)
    {
        if idx >= self.nodes.len() { return; }
        let node = &self.nodes[idx];
        let world = *parent_xform * node.transform;
        visitor(node, &world, depth);
        for &child in &node.children {
            self.walk_node(child, &world, depth + 1, visitor);
        }
    }

    /// Get flat list of (shell_id, world_transform) pairs for rendering.
    pub fn flatten_shells(&self) -> Vec<(u64, Mat4)> {
        let mut result = Vec::new();
        self.walk(&mut |node, world, _depth| {
            for &shell_id in &node.shells {
                result.push((shell_id, *world));
            }
        });
        result
    }

    /// Resolve inherited styles: for each node's shells, returns the effective
    /// style (own style if present, otherwise inherited from nearest styled ancestor).
    ///
    /// `shell_styles` is the map from shell_step_id → StyleInfo extracted by
    /// `assembly::extract_shell_styles`. Returns a map enriched with inherited values.
    pub fn resolve_inherited_styles(
        &self,
        shell_styles: &HashMap<u64, super::assembly::StyleInfo>,
    ) -> HashMap<u64, super::assembly::StyleInfo> {
        let mut inherited = shell_styles.clone();
        let mut stack: Vec<(usize, Option<super::assembly::StyleInfo>)> = Vec::new();
        self.walk_inherit(self.root_index, None, &mut inherited, &mut stack);
        inherited
    }

    fn walk_inherit(
        &self,
        idx: usize,
        parent_style: Option<super::assembly::StyleInfo>,
        inherited: &mut HashMap<u64, super::assembly::StyleInfo>,
        _stack: &mut Vec<(usize, Option<super::assembly::StyleInfo>)>,
    ) {
        if idx >= self.nodes.len() {
            return;
        }
        let node = &self.nodes[idx];

        // Determine this node's effective style
        let node_style = node.shells.iter().find_map(|sid| {
            inherited.get(sid).cloned()
        });

        let effective = node_style.or(parent_style);

        // If parent had a style and a shell doesn't have its own, inherit
        if let Some(ref style) = effective {
            for &sid in &node.shells {
                if !inherited.contains_key(&sid) {
                    inherited.insert(sid, style.clone());
                }
            }
        }

        for &child in &node.children {
            self.walk_inherit(child, effective.clone(), inherited, _stack);
        }
    }
}

/// Product-level metadata extracted from STEP entities.
#[derive(Debug, Clone, Default)]
pub struct ProductMetadata {
    pub name: String,
    pub description: String,
    pub formation_id: String,
    pub shape_name: String,
}

use crate::step::parser::EntityIndex;

/// Extract metadata for all products in the entity index.
pub fn extract_all_metadata(entities: &EntityIndex) -> HashMap<u64, ProductMetadata> {
    let mut metadata = HashMap::new();

    for (&eid, record) in entities.iter() {
        if record.name == "PRODUCT" {
            let name = record.params.nth_param(1)
                .and_then(|v| match v {
                    crate::step::value::StepValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();

            let description = record.params.nth_param(2)
                .and_then(|v| match v {
                    crate::step::value::StepValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();

            metadata.insert(eid, ProductMetadata {
                name,
                description,
                ..Default::default()
            });
        }
    }

    metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_walk_visits_all_nodes() {
        let nodes = vec![
            AssemblyNode {
                name: "root".into(), description: "".into(),
                transform: Mat4::IDENTITY, children: vec![1, 2],
                shells: vec![], product_id: 1,
            },
            AssemblyNode {
                name: "child1".into(), description: "".into(),
                transform: Mat4::IDENTITY, children: vec![],
                shells: vec![10], product_id: 2,
            },
            AssemblyNode {
                name: "child2".into(), description: "".into(),
                transform: Mat4::IDENTITY, children: vec![],
                shells: vec![20], product_id: 3,
            },
        ];
        let tree = AssemblyTree { nodes, root_index: 0 };
        let mut visited = Vec::new();
        tree.walk(&mut |node, _, _| visited.push(node.name.clone()));
        assert_eq!(visited, vec!["root", "child1", "child2"]);
    }

    #[test]
    fn test_flatten_shells() {
        let nodes = vec![
            AssemblyNode {
                name: "root".into(), description: "".into(),
                transform: Mat4::IDENTITY, children: vec![1],
                shells: vec![100], product_id: 1,
            },
            AssemblyNode {
                name: "child".into(), description: "".into(),
                transform: Mat4::IDENTITY, children: vec![],
                shells: vec![200], product_id: 2,
            },
        ];
        let tree = AssemblyTree { nodes, root_index: 0 };
        let shells = tree.flatten_shells();
        assert_eq!(shells.len(), 2);
        assert!(shells.iter().any(|(id, _)| *id == 100));
        assert!(shells.iter().any(|(id, _)| *id == 200));
    }
}
