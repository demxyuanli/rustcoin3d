//! Text3D demo — world-space label quads on a cube with camera orbit.
//!
//! Usage: cargo run -p rc3d-examples --example text3d

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Text3D — World Labels", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(3.0, 2.5, 4.0),
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

        // Reference cube at origin
        graph.add_child(
            root,
            NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.4, 0.5, 0.7),
                roughness: 0.4,
                ..Default::default()
            }),
        );
        graph.add_child(root, NodeData::Cube(CubeNode::default()));

        // Labels on cube faces
        let labels = [
            ("+X", Vec3::new(1.1, 0.0, 0.0), [1.0, 0.3, 0.3, 1.0]),
            ("-X", Vec3::new(-1.1, 0.0, 0.0), [1.0, 0.3, 0.3, 1.0]),
            ("+Y", Vec3::new(0.0, 1.1, 0.0), [0.3, 1.0, 0.3, 1.0]),
            ("-Y", Vec3::new(0.0, -1.1, 0.0), [0.3, 1.0, 0.3, 1.0]),
            ("+Z", Vec3::new(0.0, 0.0, 1.1), [0.4, 0.4, 1.0, 1.0]),
            ("-Z", Vec3::new(0.0, 0.0, -1.1), [0.4, 0.4, 1.0, 1.0]),
        ];

        for (text, pos, color) in &labels {
            graph.add_child(
                root,
                NodeData::Text3(Text3Node {
                    string: text.to_string(),
                    position: *pos,
                    size: 32.0,
                    color: *color,
                }),
            );
        }
    });
}
