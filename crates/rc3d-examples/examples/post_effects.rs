//! PostEffectParams demo — vignette, chromatic aberration, bloom, film grain.
//!
//! Usage: cargo run -p rc3d-examples --example post_effects

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Post Effects", |engine| {
        engine.set_post_effects(0.4, 0.01, 1.0, 0.02);

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

        graph.add_child(
            root,
            NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.6, 0.3, 0.3),
                roughness: 0.3,
                metallic: 0.5,
                ..Default::default()
            }),
        );

        graph.add_child(root, NodeData::Sphere(SphereNode { radius: 1.0 }));
    });
}
