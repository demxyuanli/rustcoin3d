//! ReflectionPlane demo — mirror reflection of geometry.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(3.0, 1.5, 5.0), Vec3::new(0.0, 0.5, 0.0), Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    // Red sphere above Y=0
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.9, 0.2, 0.2), roughness: 0.2, ..Default::default() }));
    g.add_child(root, NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.0, 0.0))));
    g.add_child(root, NodeData::Sphere(SphereNode { radius: 0.5 }));
    // Reflection plane at Y=0 — renders mirrored sphere below
    g.add_child(root, NodeData::ReflectionPlane(ReflectionPlaneNode { normal: Vec3::Y, origin: Vec3::ZERO, enabled: true }));
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
