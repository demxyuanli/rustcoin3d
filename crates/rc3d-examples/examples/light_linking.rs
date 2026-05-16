//! Light linking demo — light_group include/exclude for selective illumination.
//!
//! Usage: cargo run -p rc3d-examples --example light_linking

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Light Linking", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(3.0, 2.0, 6.0),
                Vec3::new(1.5, 0.0, 0.0),
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        // Red light: only affects group "A"
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::new(1.0, 0.3, 0.3),
            intensity: 2.0,
            light_group: Some("A".into()),
        }));
        // Blue light: only affects group "B"
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(1.0, -1.0, -1.0).normalize(),
            color: Vec3::new(0.3, 0.3, 1.0),
            intensity: 2.0,
            light_group: Some("B".into()),
        }));

        // Sphere in group "A"
        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.9, 0.9, 0.9),
            roughness: 0.3,
            light_group: Some("A".into()),
            ..Default::default()
        }));
        graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(1.0, 0.0, 0.0))),
        );
        graph.add_child(root, NodeData::Sphere(SphereNode { radius: 0.8 }));

        // Sphere in group "B"
        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.9, 0.9, 0.9),
            roughness: 0.3,
            light_group: Some("B".into()),
            ..Default::default()
        }));
        graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(2.5, 0.0, 0.0))),
        );
        graph.add_child(root, NodeData::Sphere(SphereNode { radius: 0.8 }));
    });
}
