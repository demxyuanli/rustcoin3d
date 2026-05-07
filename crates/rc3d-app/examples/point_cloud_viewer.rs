//! PointCloud demo — out-of-core point cloud rendering with octree spatial index.
use rc3d_app::App; use rc3d_core::math::Vec3; use rc3d_scene::node_data::*;

fn main() {
    env_logger::init(); let mut g = rc3d_scene::SceneGraph::new(); let root = g.add_root(NodeData::Separator(SeparatorNode));
    g.add_child(root, NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 800.0/600.0)));
    // PointCloud node referencing an external point cloud file
    g.add_child(root, NodeData::PointCloud(PointCloudNode { file_path: "points.bin".into(), max_visible_points: 50000, point_size: 2.0, color: [0.5, 0.5, 1.0, 1.0] }));
    // Reference sphere showing the point cloud bounds
    g.add_child(root, NodeData::Material(MaterialNode { base_color: Vec3::new(0.3, 0.3, 0.9), roughness: 0.5, opacity: 0.3, alpha_mode: AlphaMode::Blend, ..Default::default() }));
    g.add_child(root, NodeData::Sphere(SphereNode { radius: 5.0 }));
    println!("PointCloud OOC demo — file streamed via rc3d-pointcloud crate");
    winit::event_loop::EventLoop::new().unwrap().run_app(&mut App::new(g)).expect("event loop");
}
