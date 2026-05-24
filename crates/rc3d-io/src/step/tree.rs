//! Assembly tree: preserves product hierarchy from STEP assemblies.

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
