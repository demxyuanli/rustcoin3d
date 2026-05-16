//! Decal demo — screen-space projected texture overlay on geometry.
//!
//! Usage: cargo run -p rc3d-examples --example decal_viewer

use rc3d_core::math::Vec3;
use rc3d_engine_api::background::BackgroundSettings;
use rc3d_examples::common::run_example;
use rc3d_render::background::BgMode;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Decal Viewer", |engine| {
        engine.set_background(BackgroundSettings {
            mode: BgMode::Image,
            image_path: Some("decal.png".into()),
            ..Default::default()
        });

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

        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }));

        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.5, 0.5, 0.5),
            roughness: 0.4,
            albedo_texture: Some("decal.png".into()),
            ..Default::default()
        }));

        let ground = graph.add_child(
            root,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, -1.0, 0.0))),
        );
        graph.add_child(ground, NodeData::Cube(CubeNode {
            width: 5.0, height: 0.1, depth: 5.0,
        }));

        graph.add_child(root, NodeData::Decal(DecalNode {
            position: Vec3::new(0.0, 0.5, 0.0),
            direction: Vec3::NEG_Y,
            size: [2.0, 2.0],
            texture_path: "decal.png".into(),
            color: [1.0, 0.5, 0.0, 0.8],
            opacity: 0.8,
        }));
    });
}
