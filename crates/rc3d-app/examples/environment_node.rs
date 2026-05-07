//! Environment demo — ambient lighting + fog settings.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    log::info!("=== {} demo ===", "environment_node");
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(2.0, 1.0, 4.0), Vec3::new(0.0, 0.5, 0.0), Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    // Environment: red ambient + blue fog
    // Feature label
    g.add_child(root, NodeData::Text2(Text2Node { string: "Environment Node".into(), position: [10.0, 10.0], size: 18.0, color: [1.0, 0.9, 0.3, 1.0] }));
    g.add_child(root, NodeData::Environment(EnvironmentNode { ambient_intensity: 0.4, ambient_color: Vec3::new(0.8, 0.2, 0.2), attenuation: Vec3::new(0.0, 0.0, 1.0), fog_color: Vec3::new(0.3, 0.5, 0.9), fog_visibility: 8.0 }));
    // Spheres at different distances to show fog
    for i in 0..5i32 {
        let c = MaterialNode { base_color: Vec3::new(0.2 + i as f32 * 0.15, 0.6, 0.3), roughness: 0.3, ..Default::default() };
        g.add_child(root, NodeData::Material(c));
        g.add_child(root, NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 0.5, -i as f32 * 2.0))));
        g.add_child(root, NodeData::Sphere(SphereNode { radius: 0.5 }));
    }
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
