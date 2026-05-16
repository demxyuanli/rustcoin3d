//! Billboard example — children always face the camera.
//!
//! Usage: cargo run -p rc3d-examples --example billboard

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Billboard", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(3.0, 2.0, 5.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        graph.add_child(
            root,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
                color: Vec3::ONE,
                intensity: 1.0,
                light_group: None,
            }),
        );

        let billboard = graph.add_child(root, NodeData::Billboard(BillboardNode {
            axis_aligned: false,
        }));

        graph.add_child(billboard, NodeData::Text2(Text2Node {
            string: "Facing Camera".into(),
            position: [0.0, 0.0],
            size: 24.0,
            color: [1.0, 0.8, 0.0, 1.0],
        }));

        graph.add_child(root, NodeData::Cube(CubeNode {
            width: 0.5, height: 0.5, depth: 0.5,
        }));
    });
}
