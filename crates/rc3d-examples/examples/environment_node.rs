//! Environment demo — ambient lighting + fog settings.
//!
//! Usage: cargo run -p rc3d-examples --example environment_node

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Environment", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(2.0, 1.0, 4.0),
                Vec3::new(0.0, 0.5, 0.0),
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        graph.add_child(
            root,
            NodeData::Environment(EnvironmentNode {
                ambient_intensity: 0.4,
                ambient_color: Vec3::new(0.8, 0.2, 0.2),
                attenuation: Vec3::new(0.0, 0.0, 1.0),
                fog_color: Vec3::new(0.3, 0.5, 0.9),
                fog_visibility: 8.0,
            }),
        );

        for i in 0..5i32 {
            graph.add_child(root, NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.2 + i as f32 * 0.15, 0.6, 0.3),
                roughness: 0.3,
                ..Default::default()
            }));
            graph.add_child(
                root,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(
                    0.0, 0.5, -i as f32 * 2.0,
                ))),
            );
            graph.add_child(root, NodeData::Sphere(SphereNode { radius: 0.5 }));
        }
    });
}
