//! PointCloud demo — out-of-core point cloud rendering with octree spatial index.
//!
//! Usage: cargo run -p rc3d-examples --example point_cloud_viewer

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Point Cloud", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(0.0, 0.0, 10.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        graph.add_child(root, NodeData::PointCloud(PointCloudNode {
            file_path: "points.bin".into(),
            max_visible_points: 50000,
            point_size: 2.0,
            color: [0.5, 0.5, 1.0, 1.0],
        }));

        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.3, 0.3, 0.9),
            roughness: 0.5,
            opacity: 0.3,
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        }));
        graph.add_child(root, NodeData::Sphere(SphereNode { radius: 5.0 }));
    });
}
