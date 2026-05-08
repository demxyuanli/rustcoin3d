//! Engine demo: ElapsedTime + SineOscillator + Calculator + OneShot + Counter.
use rc3d_app::{App, CameraController};
use rc3d_core::math::Vec3;
use rc3d_engine::engine::{
    CounterEngine, ElapsedTimeEngine, EngineRegistry, OneShotEngine, SineField,
    SineOscillatorEngine,
};
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    log::info!("=== {} demo ===", "engines_demo");
    println!("=== {} ===", "EnginesDemo");
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.7, 0.3, 0.3), roughness: 0.2, ..Default::default() }));
    // Rotating cube using ElapsedTime engine
    let cube_node = g.add_child(root, NodeData::Transform(TransformNode::default()));
    g.add_child(cube_node, NodeData::Cube(CubeNode::default()));
    // Register engines
    let mut registry = EngineRegistry::new();
    registry.add(ElapsedTimeEngine::new(cube_node, 0.5, Vec3::Y));
    registry.add(SineOscillatorEngine::new(cube_node, 2.0, 0.2, SineField::ScaleX));
    registry.add(OneShotEngine::new(3.0));
    registry.add(CounterEngine::new(0, 5, 1));
    let mut app = App::new(g)
        .with_camera_controller(CameraController::new(Vec3::new(2.0, 0.0, 0.0), 11.0))
        .with_engines(registry);
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut app).expect("event loop");
}
