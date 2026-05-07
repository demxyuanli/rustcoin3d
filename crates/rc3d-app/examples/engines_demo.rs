//! Engine demo: ElapsedTime + SineOscillator + Calculator + OneShot + Counter.
use rc3d_app::App; use rc3d_core::math::Vec3;
use rc3d_engine::engine::{ElapsedTimeEngine, SineOscillatorEngine, SineField, CalculatorEngine,
    OneShotEngine, CounterEngine, EngineRegistry};
use rc3d_core::FieldId; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.7, 0.3, 0.3), roughness: 0.2, ..Default::default() }));
    // Rotating cube using ElapsedTime engine
    let cube_node = g.add_child(root, NodeData::Transform(TransformNode::default()));
    g.add_child(cube_node, NodeData::Cube(CubeNode::default()));
    // Register engines
    let mut registry = EngineRegistry::new();
    registry.push(Box::new(ElapsedTimeEngine::new(cube_node, 0.5f64.to_radians(), Vec3::Y))); // rotation speed
    registry.push(Box::new(SineOscillatorEngine { node_id: cube_node, field: SineField::Scale, amplitude: 0.2, frequency: 2.0, phase: 0.0 }));
    registry.push(Box::new(OneShotEngine::new(3.0)));
    registry.push(Box::new(CounterEngine::new(0, 5, 1)));
    let mut app = App::new(g).with_engines(registry);
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut app).expect("event loop");
}
