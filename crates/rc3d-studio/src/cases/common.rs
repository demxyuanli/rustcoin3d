use rc3d_core::math::Vec3;
use rc3d_core::NodeId;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

pub fn add_camera(graph: &mut SceneGraph, root: NodeId, eye: Vec3, target: Vec3) {
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            eye,
            target,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            1.0,
        )),
    );
}

pub fn add_key_light(graph: &mut SceneGraph, root: NodeId, intensity: f32) {
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity,
            light_group: None,
        }),
    );
}

pub fn add_floor(graph: &mut SceneGraph, root: NodeId, size: f32) {
    let floor = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        floor,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.35, 0.36, 0.4),
            base_color: Vec3::new(0.35, 0.36, 0.4),
            roughness: 0.9,
            ..Default::default()
        }),
    );
    graph.add_child(
        floor,
        NodeData::Cube(CubeNode {
            width: size,
            height: 0.15,
            depth: size,
        }),
    );
}

pub fn add_pbr_sphere(
    graph: &mut SceneGraph,
    root: NodeId,
    pos: Vec3,
    color: Vec3,
    metallic: f32,
    roughness: f32,
    radius: f32,
) -> NodeId {
    let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(sep, NodeData::Transform(TransformNode::from_translation(pos)));
    let mat = graph.add_child(
        sep,
        NodeData::Material(MaterialNode {
            diffuse_color: color,
            base_color: color,
            metallic,
            roughness,
            ..Default::default()
        }),
    );
    graph.set_name(mat, "case_mat");
    graph.add_child(sep, NodeData::Sphere(SphereNode { radius }));
    mat
}

pub fn first_material_mut(graph: &mut SceneGraph) -> Option<&mut MaterialNode> {
    let mut named = None;
    let mut last = None;
    for id in graph.all_node_ids() {
        let Some(e) = graph.get(id) else {
            continue;
        };
        if !matches!(e.data, NodeData::Material(_)) {
            continue;
        }
        if e.name.as_deref() == Some("case_mat") {
            named = Some(id);
            break;
        }
        last = Some(id);
    }
    let id = named.or(last)?;
    match &mut graph.get_mut(id)?.data {
        NodeData::Material(m) => Some(m),
        _ => None,
    }
}

pub fn first_dir_light_mut(graph: &mut SceneGraph) -> Option<&mut DirectionalLightNode> {
    let id = graph.all_node_ids().into_iter().find(|&id| {
        matches!(
            graph.get(id).map(|e| &e.data),
            Some(NodeData::DirectionalLight(_))
        )
    })?;
    match &mut graph.get_mut(id)?.data {
        NodeData::DirectionalLight(l) => Some(l),
        _ => None,
    }
}

pub fn first_section_mut(graph: &mut SceneGraph) -> Option<&mut SectionPlaneNode> {
    let id = graph.all_node_ids().into_iter().find(|&id| {
        matches!(
            graph.get(id).map(|e| &e.data),
            Some(NodeData::SectionPlane(_))
        )
    })?;
    match &mut graph.get_mut(id)?.data {
        NodeData::SectionPlane(s) => Some(s),
        _ => None,
    }
}
