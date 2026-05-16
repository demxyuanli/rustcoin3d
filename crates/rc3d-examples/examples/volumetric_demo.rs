//! Volumetric fog demo — scene with lights in foggy environment.
//!
//! Usage: cargo run -p rc3d-examples --example volumetric_demo

use rc3d_core::{math::Vec3, DisplayMode};
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Volumetric Demo", |engine| {
        engine.set_display_mode(DisplayMode::Shaded);

        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        // Camera
        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(5.0, 2.5, 3.0),
                Vec3::new(0.0, 1.0, -5.0),
                Vec3::Y,
                std::f32::consts::FRAC_PI_3,
                800.0 / 600.0,
            )),
        );

        // Strong directional light for god rays
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.4, -0.5, 0.6).normalize(),
            color: Vec3::new(1.0, 0.9, 0.7),
            intensity: 3.0,
            light_group: None,
        }));

        // Point lights for atmosphere
        let colors = [
            Vec3::new(1.0, 0.3, 0.2),
            Vec3::new(0.2, 0.5, 1.0),
            Vec3::new(0.3, 1.0, 0.3),
        ];
        for i in 0..3i32 {
            let s = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(s, NodeData::Transform(TransformNode::from_translation(
                Vec3::new((i as f32 - 1.0) * 6.0, 3.0, -3.0),
            )));
            graph.add_child(s, NodeData::PointLight(PointLightNode {
                location: Vec3::ZERO,
                color: colors[i as usize],
                intensity: 50.0,
                light_group: None,
            }));
        }

        // Ground
        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.5, 0.5, 0.5),
            roughness: 0.9,
            ..Default::default()
        }));
        graph.add_child(root, NodeData::Cube(CubeNode {
            width: 30.0, height: 0.2, depth: 30.0,
        }));

        // Environment with fog-like attenuation
        graph.add_child(root, NodeData::Environment(EnvironmentNode {
            ambient_intensity: 0.1,
            ambient_color: Vec3::new(0.2, 0.3, 0.4),
            attenuation: Vec3::new(0.01, 0.01, 0.01),
            fog_color: Vec3::new(0.5, 0.6, 0.7),
            fog_visibility: 10.0,
        }));

        // Spheres at various distances
        for i in 0..8i32 {
            let s = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(s, NodeData::Transform(TransformNode::from_translation(
                Vec3::new((i as f32 - 3.5) * 3.0, 1.0, i as f32 * -2.0),
            )));
            graph.add_child(s, NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.8, 0.3 + i as f32 * 0.08, 0.4),
                roughness: 0.4,
                ..Default::default()
            }));
            graph.add_child(s, NodeData::Sphere(SphereNode { radius: 0.8 }));
        }
    });
}
