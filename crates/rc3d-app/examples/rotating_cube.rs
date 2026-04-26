use rc3d_app::{App, CameraController};
use rc3d_core::math::Vec3;
use rc3d_engine::{ElapsedTimeEngine, EngineRegistry};
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();

    let mut graph = rc3d_scene::SceneGraph::new();

    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera
    let camera_id = graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(3.0, 3.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Light
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
        }),
    );

    // Animated cube: Separator { Transform(animated), Material, Cube }
    let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    let transform_id = graph.add_child(sep, NodeData::Transform(TransformNode::default()));
    graph.add_child(
        sep,
        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.5, 0.8))),
    );
    graph.add_child(sep, NodeData::Cube(CubeNode::default()));

    // Engine: rotate the transform continuously
    let mut engines = EngineRegistry::new();
    engines.add(ElapsedTimeEngine::new(transform_id, 1.0, Vec3::Y));

    let ctrl = CameraController::new(camera_id, Vec3::ZERO, 7.0);

    let mut app = App::new(graph)
        .with_camera_controller(ctrl)
        .with_engines(engines);

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}
