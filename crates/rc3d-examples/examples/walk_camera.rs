//! Walk Camera example — first-person WASD movement.
//!
//! Usage: cargo run -p rc3d-examples --example walk_camera

use rc3d_core::math::Vec3;
use rc3d_engine_api::background::BackgroundSettings;
use rc3d_examples::common::run_example;
use rc3d_render::background::BgMode;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Walk Camera", |engine| {
        engine.camera_mut().walk_mode = true;

        engine.set_background(BackgroundSettings {
            mode: BgMode::VerticalGradient,
            clear_color: [0.03, 0.05, 0.12, 1.0],
            ..Default::default()
        });

        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(3.0, 1.5, 5.0),
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

        // Floor grid of cubes
        for x in -2..=2i32 {
            for z in -2..=2i32 {
                let tf = graph.add_child(
                    root,
                    NodeData::Transform(TransformNode::from_translation(Vec3::new(
                        x as f32 * 2.0, 0.0, z as f32 * 2.0,
                    ))),
                );
                graph.add_child(tf, NodeData::Material(MaterialNode::from_diffuse(Vec3::new(
                    0.2, 0.4 + x as f32 * 0.1, 0.3 + z as f32 * 0.1,
                ))));
                graph.add_child(tf, NodeData::Cube(CubeNode::default()));
            }
        }
    });
}
