//! Node kits — Coin3D SoBaseKit pattern: pre-built subgraphs for common node combinations.

use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::NodeId;
use rc3d_scene::node_data::{
    Coordinate3Node, IndexedFaceSetNode, MaterialNode, NodeData, NormalNode, SeparatorNode,
    TextureCoordinate2Node,
};
use rc3d_scene::SceneGraph;
use rc3d_mesh::TriangleMesh;

/// Build a ShapeKit: Separator + Material + Coordinate3 + Normal + TextureCoordinate2 + IndexedFaceSet.
/// Returns the Separator's NodeId (the root of the kit).
pub fn shape_kit(graph: &mut SceneGraph, parent: NodeId, mesh: &TriangleMesh, material: MaterialNode) -> NodeId {
    let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));
    graph.add_child(sep, NodeData::Material(material));
    graph.add_child(sep, NodeData::Coordinate3(Coordinate3Node::from_points(mesh.positions.clone())));
    graph.add_child(sep, NodeData::Normal(NormalNode { vector: mesh.normals.clone() }));
    if !mesh.texcoords.is_empty() {
        graph.add_child(sep, NodeData::TextureCoordinate2(TextureCoordinate2Node { point: mesh.texcoords.clone() }));
    }
    let coord_index: Vec<i32> = mesh.tri_indices.iter().map(|&i| i as i32).collect();
    graph.add_child(sep, NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index }));
    sep
}

/// Build a TransformKit: Transform + child geometry. Returns the Transform's NodeId.
pub fn transform_kit(graph: &mut SceneGraph, parent: NodeId, translation: Vec3, rotation: Option<Mat4>, scale: Option<Vec3>) -> NodeId {
    use rc3d_scene::node_data::TransformNode;
    graph.add_child(parent, NodeData::Transform(TransformNode {
        translation,
        rotation: rotation.unwrap_or(Mat4::IDENTITY),
        scale: scale.unwrap_or(Vec3::ONE),
        center: Vec3::ZERO,
    }))
}

/// Build a LightKit: Separator + directional light. Returns the Separator's NodeId.
pub fn directional_light_kit(graph: &mut SceneGraph, parent: NodeId, direction: Vec3, color: Vec3, intensity: f32) -> NodeId {
    use rc3d_scene::node_data::DirectionalLightNode;
    let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));
    graph.add_child(sep, NodeData::DirectionalLight(DirectionalLightNode {
        direction: direction.normalize(),
        color,
        intensity,
        light_group: None,
    }));
    sep
}

/// Build a CameraKit: Separator + perspective camera. Returns the Separator's NodeId.
pub fn perspective_camera_kit(graph: &mut SceneGraph, parent: NodeId, position: Vec3, target: Vec3, fov: f32, aspect: f32) -> NodeId {
    use rc3d_scene::node_data::PerspectiveCameraNode;
    let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));
    let cam = PerspectiveCameraNode::look_at(position, target, Vec3::Y, fov, aspect);
    graph.add_child(sep, NodeData::PerspectiveCamera(cam));
    sep
}

/// Build a SwitchKit: Separator + Switch with N child groups. Returns (separator, switch, child_nodes).
pub fn switch_kit(graph: &mut SceneGraph, parent: NodeId, child_count: usize) -> (NodeId, NodeId, Vec<NodeId>) {
    use rc3d_scene::node_data::SwitchNode;
    let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));
    let sw = graph.add_child(sep, NodeData::Switch(SwitchNode { which_child: -1, children: Vec::new() }));
    let mut children = Vec::with_capacity(child_count);
    for _ in 0..child_count {
        let c = graph.add_child(sw, NodeData::Separator(SeparatorNode));
        children.push(c);
    }
    // Update SwitchNode's child list to match the scene-graph children under it
    if let Some(entry) = graph.get_mut(sw) {
        if let NodeData::Switch(ref mut s) = entry.data {
            s.children.clone_from(&children);
        }
    }
    (sep, sw, children)
}
