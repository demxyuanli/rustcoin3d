//! Scene graph query utilities for type-safe traversal.

use rc3d_core::Aabb;
use rc3d_core::NodeId;
use rc3d_scene::node_data::NodeData;
use rc3d_scene::SceneGraph;

/// A type-safe query over the scene graph.
pub struct Query<'a> {
    graph: &'a SceneGraph,
}

impl<'a> Query<'a> {
    pub(crate) fn new(graph: &'a SceneGraph) -> Self {
        Self { graph }
    }

    /// Check if the graph contains at least one node whose data matches the predicate.
    pub fn any<P>(&self, predicate: P) -> bool
    where
        P: Fn(&NodeData) -> bool,
    {
        self.graph
            .traverse_all()
            .filter_map(|id| self.graph.get(id))
            .any(|entry| predicate(&entry.data))
    }

    /// Collect all node IDs whose data matches the predicate.
    pub fn find_all<P>(&self, predicate: P) -> Vec<NodeId>
    where
        P: Fn(&NodeData) -> bool,
    {
        self.graph
            .traverse_all()
            .filter(|&id| {
                self.graph
                    .get(id)
                    .is_some_and(|entry| predicate(&entry.data))
            })
            .collect()
    }

    /// Compute the bounding box of a subtree (in local space).
    pub fn bounds_of(&self, root: NodeId) -> Option<Aabb> {
        let mut aabb: Option<Aabb> = None;
        for id in self.graph.traverse_dfs(root) {
            if let Some(entry) = self.graph.get(id) {
                match &entry.data {
                    NodeData::Cube(c) => {
                        let hw = c.width / 2.0;
                        let hh = c.height / 2.0;
                        let hd = c.depth / 2.0;
                        let ab = Aabb {
                            min: (-hw, -hh, -hd).into(),
                            max: (hw, hh, hd).into(),
                        };
                        aabb = Some(aabb.map_or(ab.clone(), |a| a.union(&ab)));
                    }
                    NodeData::Sphere(s) => {
                        let r = s.radius;
                        let ab = Aabb {
                            min: (-r, -r, -r).into(),
                            max: (r, r, r).into(),
                        };
                        aabb = Some(aabb.map_or(ab.clone(), |a| a.union(&ab)));
                    }
                    _ => {}
                }
            }
        }
        aabb
    }

    /// Count all nodes matching a predicate.
    pub fn count<P>(&self, predicate: P) -> usize
    where
        P: Fn(&NodeData) -> bool,
    {
        self.graph
            .traverse_all()
            .filter(|&id| {
                self.graph
                    .get(id)
                    .is_some_and(|e| predicate(&e.data))
            })
            .count()
    }
}

/// Convenience methods on SceneGraph for type-based queries.
pub trait SceneQueryExt {
    fn has_any<P: Fn(&NodeData) -> bool>(&self, predicate: P) -> bool;
    fn find_all_ids<P: Fn(&NodeData) -> bool>(&self, predicate: P) -> Vec<NodeId>;
}

impl SceneQueryExt for SceneGraph {
    fn has_any<P: Fn(&NodeData) -> bool>(&self, predicate: P) -> bool {
        self.traverse_all()
            .filter_map(|id| self.get(id))
            .any(|e| predicate(&e.data))
    }

    fn find_all_ids<P: Fn(&NodeData) -> bool>(&self, predicate: P) -> Vec<NodeId> {
        self.traverse_all()
            .filter(|&id| self.get(id).is_some_and(|e| predicate(&e.data)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_scene::SceneGraph;
    use rc3d_scene::node_data::{
        CubeNode, NodeData, PerspectiveCameraNode, SeparatorNode,
    };

    fn build_test_graph() -> SceneGraph {
        let mut g = SceneGraph::new();
        let root = g.add_root(NodeData::Separator(SeparatorNode));
        let cam = PerspectiveCameraNode::look_at(
            [0.0, 0.0, 5.0].into(),
            [0.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            1.0,
            1.0,
        );
        g.add_child(root, NodeData::PerspectiveCamera(cam));
        let shape_sep = g.add_child(root, NodeData::Separator(SeparatorNode));
        g.add_child(shape_sep, NodeData::Cube(CubeNode::default()));
        g
    }

    #[test]
    fn test_any_finds_camera() {
        let g = build_test_graph();
        let q = Query::new(&g);
        assert!(q.any(|d| matches!(d, NodeData::PerspectiveCamera(_))));
    }

    #[test]
    fn test_any_no_light() {
        let g = build_test_graph();
        let q = Query::new(&g);
        assert!(!q.any(|d| matches!(d, NodeData::DirectionalLight(_))));
    }

    #[test]
    fn test_find_all_cubes() {
        let g = build_test_graph();
        let q = Query::new(&g);
        let ids = q.find_all(|d| matches!(d, NodeData::Cube(_)));
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_count_nodes() {
        let g = build_test_graph();
        let q = Query::new(&g);
        let n = q.count(|d| matches!(d, NodeData::Separator(_)));
        assert_eq!(n, 2);
    }
}
