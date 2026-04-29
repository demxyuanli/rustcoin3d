//! Scene graph path: chain from root to a target node.
//! Follows Coin3D SoPath pattern.

use rc3d_core::math::Mat4;
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

/// A path through the scene graph: chain of (node_id, child_index).
#[derive(Clone, Debug)]
pub struct ScenePath {
    pub nodes: Vec<NodeId>,
    pub child_indices: Vec<usize>,
}

impl ScenePath {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), child_indices: Vec::new() }
    }

    pub fn head(&self) -> Option<NodeId> {
        self.nodes.first().copied()
    }

    pub fn tail(&self) -> Option<NodeId> {
        self.nodes.last().copied()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Accumulated world transform along this path.
    pub fn get_matrix(&self, graph: &SceneGraph) -> Mat4 {
        let mut m = Mat4::IDENTITY;
        for &node_id in &self.nodes {
            if let Some(entry) = graph.get(node_id) {
                if let NodeData::Transform(t) = &entry.data {
                    m = m * t.to_matrix();
                }
            }
        }
        m
    }

    pub fn node_slice(&self) -> &[NodeId] {
        &self.nodes
    }
}

impl Default for ScenePath {
    fn default() -> Self {
        Self::new()
    }
}

/// Search action: find nodes by name or type.
pub struct SearchAction {
    pub name_filter: Option<String>,
    pub type_filter: Option<String>,
    pub results: Vec<ScenePath>,
    current_path: ScenePath,
}

impl SearchAction {
    pub fn new() -> Self {
        Self {
            name_filter: None,
            type_filter: None,
            results: Vec::new(),
            current_path: ScenePath::new(),
        }
    }

    pub fn by_name(name: &str) -> Self {
        let mut s = Self::new();
        s.name_filter = Some(name.to_string());
        s
    }

    pub fn by_type(type_name: &str) -> Self {
        let mut s = Self::new();
        s.type_filter = Some(type_name.to_string());
        s
    }
}

/// Compute world-space matrix for the tail of a scene path.
pub struct GetMatrixAction {
    pub path: ScenePath,
    pub matrix: Mat4,
}

impl GetMatrixAction {
    pub fn new(path: ScenePath) -> Self {
        Self {
            path,
            matrix: Mat4::IDENTITY,
        }
    }

    pub fn apply(&mut self, graph: &SceneGraph) {
        self.matrix = self.path.get_matrix(graph);
    }
}
