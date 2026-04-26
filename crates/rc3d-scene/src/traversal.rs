use rc3d_core::NodeId;

use crate::scene_graph::SceneGraph;

/// Depth-first pre-order traversal yielding node IDs.
pub struct DfsPreOrder<'a> {
    graph: &'a SceneGraph,
    stack: Vec<NodeId>,
}

impl<'a> DfsPreOrder<'a> {
    pub fn new(graph: &'a SceneGraph, roots: &[NodeId]) -> Self {
        let mut stack = roots.to_vec();
        stack.reverse();
        Self { graph, stack }
    }
}

impl Iterator for DfsPreOrder<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.stack.pop()?;
        if let Some(entry) = self.graph.get(id) {
            // Push children in reverse so leftmost child is visited first
            for child in entry.children.iter().rev() {
                self.stack.push(*child);
            }
        }
        Some(id)
    }
}

impl SceneGraph {
    pub fn traverse_dfs(&self, root: NodeId) -> DfsPreOrder<'_> {
        DfsPreOrder::new(self, &[root])
    }

    pub fn traverse_all(&self) -> DfsPreOrder<'_> {
        DfsPreOrder::new(self, self.roots())
    }
}
