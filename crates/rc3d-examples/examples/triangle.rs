//! Triangle example — single indexed face set.
//!
//! Usage: cargo run -p rc3d-examples --example triangle

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Triangle", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(0.0, 0.0, 5.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        graph.add_child(
            root,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.6, 0.8))),
        );

        let verts = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
        ];
        graph.add_child(root, NodeData::Coordinate3(Coordinate3Node::from_points(verts)));
        graph.add_child(root, NodeData::IndexedFaceSet(IndexedFaceSetNode {
            coord_index: vec![0, 1, 2],
        }));
    });
}
