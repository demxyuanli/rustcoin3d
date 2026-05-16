//! Annotation demo — overlay text rendered on top without depth test.
//!
//! Usage: cargo run -p rc3d-examples --example annotation

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Annotation", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(0.0, 1.0, 4.0),
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

        graph.add_child(
            root,
            NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.4, 0.4, 0.8),
                roughness: 0.5,
                ..Default::default()
            }),
        );
        graph.add_child(root, NodeData::Cube(CubeNode::default()));

        let ann = graph.add_child(root, NodeData::Annotation(AnnotationNode));
        graph.add_child(ann, NodeData::Text2(Text2Node {
            string: "OVERLAY TEXT".into(),
            position: [200.0, 100.0],
            size: 32.0,
            color: [1.0, 0.8, 0.0, 1.0],
        }));
    });
}
