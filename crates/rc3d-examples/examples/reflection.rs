//! Reflection example — demonstrates the reflection node.
//!
//! Usage: cargo run -p rc3d-examples --example reflection

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Reflection", |engine| {
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

        // Reflective floor
        let floor = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            floor,
            NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.4, 0.4, 0.5),
                roughness: 0.3,
                metallic: 0.9,
                ..Default::default()
            }),
        );
        graph.add_child(floor, NodeData::Cube(CubeNode {
            width: 4.0,
            height: 0.1,
            depth: 4.0,
        }));

        graph.add_child(
            root,
            NodeData::ReflectionPlane(ReflectionPlaneNode {
                normal: Vec3::Y,
                origin: Vec3::new(0.0, -0.05, 0.0),
                enabled: true,
            }),
        );

        // Sphere above the floor
        let sphere_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            sphere_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.5, 0.0))),
        );
        graph.add_child(
            sphere_sep,
            NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.8, 0.3, 0.3),
                roughness: 0.2,
                metallic: 0.1,
                ..Default::default()
            }),
        );
        graph.add_child(sphere_sep, NodeData::Sphere(SphereNode { radius: 0.8 }));
    });
}
