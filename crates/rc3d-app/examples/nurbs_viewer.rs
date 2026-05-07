//! Curve viewer — demonstrates IndexedLineSet for parametric curve visualization.
//! For full NURBS integration, see crates/rc3d-nurbs/ (curve, surface, tessellator).
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(0.0, 0.0, 8.0), Vec3::new(2.0, 1.0, 0.0), Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    g.add_child(root, NodeData::DirectionalLight(DirectionalLightNode { direction: Vec3::new(-1.0,-1.0,-1.0).normalize(), color: Vec3::ONE, intensity: 1.0, light_group: None }));
    // Parametric curve: sine wave visualized as line segments
    let n = 64; let mut points = Vec::new();
    for i in 0..=n { let t = i as f32 / n as f32; points.push(Vec3::new(t * 4.0, (t * std::f32::consts::TAU).sin() * 1.5, 0.0)); }
    let mut indices: Vec<i32> = Vec::new();
    for i in 0..n { indices.push(i); indices.push(i + 1); }
    g.add_child(root, NodeData::Coordinate3(Coordinate3Node::from_points(points)));
    g.add_child(root, NodeData::IndexedLineSet(IndexedLineSetNode { coord_index: indices, line_width: 3.0 }));
    println!("Curve: {} points → {} line segments", n+1, n);
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
