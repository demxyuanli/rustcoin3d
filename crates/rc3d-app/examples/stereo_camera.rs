//! StereoCamera demo — side-by-side stereo rendering with interocular distance.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    let cam_id = g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.5, 0.3, 0.8), roughness: 0.2, ..Default::default() }));
    g.add_child(root, NodeData::Sphere(SphereNode { radius: 1.0 }));
    // StereoCamera wrapping the base camera
    g.add_child(root, NodeData::StereoCamera(StereoCameraNode { base_camera: cam_id, interocular_distance: 0.065, convergence_distance: 2.0, mode: StereoMode::SideBySide }));
    println!("Stereo camera configured: SideBySide mode, interocular=0.065m, convergence=2.0m");
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
