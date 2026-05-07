//! Light linking demo — light_group include/exclude for selective illumination.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(3.0, 2.0, 6.0), Vec3::new(1.5, 0.0, 0.0), Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    // Red light: only affects group "A"
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::new(1.0, 0.3, 0.3), intensity: 2.0, light_group: Some("A".into()) }));
    // Blue light: only affects group "B"
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(1.0,-1.0,-1.0).normalize(), color: Vec3::new(0.3, 0.3, 1.0), intensity: 2.0, light_group: Some("B".into()) }));
    // Sphere in group "A" — illuminated red
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.9, 0.9, 0.9), roughness: 0.3, light_group: Some("A".into()), ..Default::default() }));
    g.add_child(root, NodeData::Transform(TransformNode::from_translation(Vec3::new(1.0, 0.0, 0.0))));
    g.add_child(root, NodeData::Sphere(SphereNode { radius: 0.8 }));
    // Sphere in group "B" — illuminated blue
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.9, 0.9, 0.9), roughness: 0.3, light_group: Some("B".into()), ..Default::default() }));
    g.add_child(root, NodeData::Transform(TransformNode::from_translation(Vec3::new(2.5, 0.0, 0.0))));
    g.add_child(root, NodeData::Sphere(SphereNode { radius: 0.8 }));
    println!("Light linking: left sphere=group A (red light), right=group B (blue light)");
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
