use rc3d_core::math::Vec3;
use rc3d_core::NodeId;
use rc3d_scene::{NodeData, SceneGraph};

/// Scene light queries shared by rendering (shadows, shading).
#[derive(Clone, Copy, Debug)]
pub struct LightSubsystem;

impl LightSubsystem {
    pub fn primary_directional_dir(graph: &SceneGraph) -> Option<Vec3> {
        for &root in graph.roots() {
            if let Some(d) = Self::walk_dir_light(graph, root) {
                return Some(d);
            }
        }
        None
    }

    fn walk_dir_light(graph: &SceneGraph, node: NodeId) -> Option<Vec3> {
        let entry = graph.get(node)?;
        match &entry.data {
            NodeData::DirectionalLight(l) => Some(l.direction.normalize()),
            NodeData::HandlerNode(_) => {
                for &c in &entry.children {
                    if let Some(d) = Self::walk_dir_light(graph, c) {
                        return Some(d);
                    }
                }
                None
            }
            NodeData::Separator(_) => {
                for &c in &entry.children {
                    if let Some(d) = Self::walk_dir_light(graph, c) {
                        return Some(d);
                    }
                }
                None
            }
            NodeData::Group(_) | NodeData::Billboard(_) | NodeData::Transform(_) => {
                for &c in &entry.children {
                    if let Some(d) = Self::walk_dir_light(graph, c) {
                        return Some(d);
                    }
                }
                None
            }
            _ => None,
        }
    }
}
