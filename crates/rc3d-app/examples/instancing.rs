//! MultipleCopy + Switch + Lod demo — scenegraph traversal variants.
use rc3d_app::App; use rc3d_core::math::{Mat4, Vec3}; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(4.0, 2.0, 8.0), Vec3::new(2.0, 0.0, 0.0), Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    // MultipleCopy: 6 instances along X
    let copies: Vec<Mat4> = (0..6).map(|i| Mat4::from_translation(Vec3::new(i as f32 * 1.5, 0.0, 0.0))).collect();
    let mc = g.add_child(root, NodeData::MultipleCopy(MultipleCopyNode { copies, children: vec![] }));
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.3, 0.6, 0.9), roughness: 0.3, ..Default::default() }));
    let sphere = g.add_child(mc, NodeData::Sphere(SphereNode { radius: 0.4 }));
    // Switch: toggle between sphere (index 0) and cube (index 1)
    let sw = g.add_child(root, NodeData::Switch(SwitchNode { which_child: 0,
        children: vec![sphere] }));
    // Lod: simple two-level LOD
    let lod_levels = vec![LodLevel { children: vec![sw], max_distance: 10.0 }];
    g.add_child(root, NodeData::Lod(LodNode { levels: lod_levels, current_level: 0 }));
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
