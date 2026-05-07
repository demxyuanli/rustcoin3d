//! Annotation demo — children rendered as overlay without depth test.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    log::info!("=== {} demo ===", "annotation");
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(0.0, 1.0, 4.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    // Feature label
    g.add_child(root, NodeData::Text2(Text2Node { string: "Annotation".into(), position: [10.0, 10.0], size: 18.0, color: [1.0, 0.9, 0.3, 1.0] }));
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.4, 0.4, 0.8), roughness: 0.5, ..Default::default() }));
    g.add_child(root, NodeData::Cube(CubeNode::default()));
    // Annotation — children render on top regardless of depth
    let ann = g.add_child(root, NodeData::Annotation(AnnotationNode));
    g.add_child(ann, NodeData::Text2(Text2Node { string: "OVERLAY TEXT".into(), position: [200.0, 100.0], size: 32.0, color: [1.0, 0.8, 0.0, 1.0] }));
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
