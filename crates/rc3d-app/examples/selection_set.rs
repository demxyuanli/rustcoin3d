//! SelectionSet demo — named selection groups for batch operations.
use rc3d_app::App; use rc3d_core::{NodeId, math::Vec3}; use rc3d_scene::{SceneGraph, node_data::*};

fn main() {
    env_logger::init(); let mut g = SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    // Create 5 spheres
    let mut ids = Vec::new();
    for i in 0..5i32 {
        g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.3 + i as f32 * 0.1, 0.2, 0.7), roughness: 0.3, ..Default::default() }));
        g.add_child(root, NodeData::Transform(TransformNode::from_translation(Vec3::new(i as f32 * 1.5 - 3.0, 0.0, 0.0))));
        ids.push(g.add_child(root, NodeData::Sphere(SphereNode { radius: 0.4 })));
    }
    // Add to named selection set
    g.selection_set_add("spheres", &ids);
    // Select from named set
    g.selection_set_select("spheres");
    println!("Selected {} spheres via named set", g.selected_nodes().len());
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
