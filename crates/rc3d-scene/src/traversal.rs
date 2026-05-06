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

    /// Parallel traversal: process each root subtree in its own rayon thread.
    ///
    /// `action_factory` is called per root to create a thread-local action.
    /// The produced action must be `Send` so it can be moved into the worker thread.
    pub fn traverse_parallel<F, T>(graph: &SceneGraph, roots: &[NodeId], action_factory: F)
    where
        F: Fn() -> T + Sync,
        T: FnMut(&SceneGraph, NodeId) + Send + Sync,
    {
        let roots = roots.to_vec();
        rayon::scope(|s| {
            for &root in &roots {
                let mut action = action_factory();
                s.spawn(move |_| {
                    action(graph, root);
                });
            }
        });
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

    /// Parallel traversal over all roots of this scene graph.
    pub fn traverse_parallel_all<F, T>(&self, action_factory: F)
    where
        F: Fn() -> T + Sync,
        T: FnMut(&SceneGraph, NodeId) + Send + Sync,
    {
        DfsPreOrder::traverse_parallel(self, self.roots(), action_factory);
    }
}
