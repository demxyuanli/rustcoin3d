//! IndexedLineSet demo — wireframe edges and grid lines.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    log::info!("=== {} demo ===", "indexed_line_set");
    println!("=== {} ===", "IndexedLineSet");
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(0.0, 2.0, 4.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    // Grid of lines in XZ plane
    let mut coord = Vec::new(); let mut indices = Vec::new(); let n = 10i32;
    for i in -n..=n { coord.push(Vec3::new(i as f32, 0.0, -n as f32)); coord.push(Vec3::new(i as f32, 0.0, n as f32)); let b = ((i+n)*2) as i32; indices.push(b); indices.push(b+1); }
    for i in -n..=n { coord.push(Vec3::new(-n as f32, 0.0, i as f32)); coord.push(Vec3::new(n as f32, 0.0, i as f32)); let b = ((n*2+1)*2+(i+n)*2) as i32; indices.push(b); indices.push(b+1); }
    g.add_child(root, NodeData::Coordinate3(Coordinate3Node::from_points(coord)));
    g.add_child(root, NodeData::IndexedLineSet(IndexedLineSetNode { coord_index: indices, line_width: 1.0 }));
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
