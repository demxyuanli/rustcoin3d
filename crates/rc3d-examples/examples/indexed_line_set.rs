//! IndexedLineSet demo — wireframe grid lines.
//!
//! Usage: cargo run -p rc3d-examples --example indexed_line_set

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("IndexedLineSet", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(0.0, 2.0, 4.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        let mut coord = Vec::new();
        let mut indices = Vec::new();
        let n = 10i32;
        for i in -n..=n {
            coord.push(Vec3::new(i as f32, 0.0, -n as f32));
            coord.push(Vec3::new(i as f32, 0.0, n as f32));
            let b = ((i + n) * 2) as i32;
            indices.push(b);
            indices.push(b + 1);
        }
        for i in -n..=n {
            coord.push(Vec3::new(-n as f32, 0.0, i as f32));
            coord.push(Vec3::new(n as f32, 0.0, i as f32));
            let b = ((n * 2 + 1) * 2 + (i + n) * 2) as i32;
            indices.push(b);
            indices.push(b + 1);
        }

        graph.add_child(root, NodeData::Coordinate3(Coordinate3Node::from_points(coord)));
        graph.add_child(root, NodeData::IndexedLineSet(IndexedLineSetNode {
            coord_index: indices,
            line_width: 1.0,
        }));
    });
}
