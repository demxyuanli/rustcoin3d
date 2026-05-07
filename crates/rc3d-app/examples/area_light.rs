//! AreaLight demo — rectangle and disc area lights with soft falloff.
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();
    log::info!("=== {} demo ===", "area_light");
    let mut g = rc3d_scene::SceneGraph::new();
    let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
        Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0 / 600.0,
    )));
    // Feature label
    g.add_child(root, NodeData::Text2(Text2Node { string: "Area Light".into(), position: [10.0, 10.0], size: 18.0, color: [1.0, 0.9, 0.3, 1.0] }));
    // Rectangle area light above the scene
    g.add_child(root, NodeData::AreaLight(AreaLightNode {
        position: Vec3::new(0.0, 3.0, 0.0), direction: Vec3::new(0.0, -1.0, 0.0),
        color: Vec3::new(1.0, 0.9, 0.7), intensity: 2.0, width: 2.0, height: 1.0,
        shape: AreaLightShape::Rectangle, light_group: None,
    }));
    // Ground plane
    g.add_child(root, NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.5, 0.5, 0.5))));
    let ground = g.add_child(root, NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, -1.2, 0.0))));
    g.add_child(ground, NodeData::Cube(CubeNode { width: 5.0, height: 0.1, depth: 5.0 }));
    // Sphere receiving area light
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.8, 0.3, 0.3), roughness: 0.3, ..Default::default() }));
    g.add_child(root, NodeData::Sphere(SphereNode { radius: 1.0 }));
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
