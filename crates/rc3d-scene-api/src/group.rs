//! Group node for explicit state isolation.

use rc3d_core::NodeId;
use rc3d_scene::node_data::{NodeData, SeparatorNode, TransformNode};
use rc3d_scene::SceneGraph;

use crate::scene::NodeHandle;
use crate::shape::Shape;

/// A group of shapes with explicit state isolation.
///
/// Each `Group` maps to a `Separator` in the scene graph, ensuring transforms
/// and materials set inside don't leak to siblings.
#[derive(Clone, Debug, Default)]
pub struct Group {
    name: String,
    children: Vec<ChildData>,
}

#[derive(Clone, Debug)]
struct ChildData {
    translation: Option<rc3d_core::math::Vec3>,
    rotation: Option<rc3d_core::math::Mat4>,
    scale: Option<rc3d_core::math::Vec3>,
    material: Option<crate::material::Material>,
    /// Geometry nodes to add (Coordinate3, Normal, IFS, etc.) — supports multi-node shapes.
    geometry_nodes: Vec<NodeData>,
}

impl Group {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), children: Vec::new() }
    }

    /// Add a shape to this group.
    pub fn add(mut self, shape: impl Shape + Clone + 'static) -> Self {
        let mut geometry_nodes = Vec::new();
        // Build geometry nodes into a temporary graph to capture all siblings
        let mut tmp_graph = SceneGraph::new();
        let tmp_root = tmp_graph.add_root(NodeData::Separator(SeparatorNode));
        let tmp_sep = tmp_graph.add_child(tmp_root, NodeData::Separator(SeparatorNode));
        shape.add_geometry_nodes(&mut tmp_graph, tmp_sep);
        // Collect all children added under the separator
        if let Some(entry) = tmp_graph.get(tmp_sep) {
            for &child_id in &entry.children {
                if let Some(child_entry) = tmp_graph.get(child_id) {
                    geometry_nodes.push(child_entry.data.clone());
                }
            }
        }

        self.children.push(ChildData {
            translation: shape.shape_translation().copied(),
            rotation: shape.shape_rotation().copied(),
            scale: shape.shape_scale().copied(),
            material: shape.shape_material().cloned(),
            geometry_nodes,
        });
        self
    }

    /// Name of this group.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of direct children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Add this group to a scene graph, returning the separator node id.
    pub(crate) fn add_to_graph(&self, graph: &mut SceneGraph, parent: NodeId) -> NodeHandle {
        let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));
        for child in &self.children {
            let sub = graph.add_child(sep, NodeData::Separator(SeparatorNode));
            if child.translation.is_some() || child.rotation.is_some() || child.scale.is_some() {
                graph.add_child(
                    sub,
                    NodeData::Transform(TransformNode {
                        translation: child.translation.unwrap_or(rc3d_core::math::Vec3::ZERO),
                        rotation: child.rotation.unwrap_or(rc3d_core::math::Mat4::IDENTITY),
                        scale: child.scale.unwrap_or(rc3d_core::math::Vec3::ONE),
                        center: rc3d_core::math::Vec3::ZERO,
                    }),
                );
            }
            if let Some(ref mat) = child.material {
                graph.add_child(sub, NodeData::Material(mat.to_node()));
            }
            // Add all geometry nodes (Coordinate3, Normal, IFS, etc.)
            for node_data in &child.geometry_nodes {
                graph.add_child(sub, node_data.clone());
            }
        }
        NodeHandle::from_id(sep)
    }
}
