use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

pub fn build_demo_scene() -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(5.0, 4.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            1.0,
        )),
    );
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.2,
            light_group: None,
        }),
    );

    let floor = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        floor,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.4, 0.4, 0.45),
            base_color: Vec3::new(0.4, 0.4, 0.45),
            roughness: 0.9,
            ..Default::default()
        }),
    );
    graph.add_child(
        floor,
        NodeData::Cube(CubeNode {
            width: 10.0,
            height: 0.2,
            depth: 10.0,
        }),
    );

    let body = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        body,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.0, 0.0))),
    );
    graph.add_child(
        body,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.2, 0.55, 0.85),
            base_color: Vec3::new(0.2, 0.55, 0.85),
            metallic: 0.15,
            roughness: 0.35,
            ..Default::default()
        }),
    );
    graph.add_child(body, NodeData::Sphere(SphereNode::default()));

    graph.add_child(
        root,
        NodeData::SectionPlane(SectionPlaneNode {
            plane: [0.0, 1.0, 0.0, -0.5],
            enabled: false,
            ..Default::default()
        }),
    );

    graph
}
