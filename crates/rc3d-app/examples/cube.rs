use rc3d_app::App;
use rc3d_scene::node_data::*;
use rc3d_core::math::Vec3;

fn main() {
    env_logger::init();

    let mut graph = rc3d_scene::SceneGraph::new();

    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera looking at origin from above-right-front
    let camera = PerspectiveCameraNode::look_at(
        Vec3::new(2.0, 2.0, 4.0),
        Vec3::ZERO,
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        800.0 / 600.0,
    );
    graph.add_child(root, NodeData::PerspectiveCamera(camera));

    // Light
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
        }),
    );

    // Material
    graph.add_child(
        root,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.5, 0.8))),
    );

    // Cube
    graph.add_child(root, NodeData::Cube(CubeNode::default()));

    let mut app = App::new(graph);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}
