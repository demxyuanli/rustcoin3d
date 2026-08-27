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

        graph.add_child(
            root,
            NodeData::Sprite(SpriteNode {
                color: [0.2, 0.85, 1.0, 1.0],
                size: 0.8,
                size_attenuation: true,
                ..Default::default()
            }),
        );
        let offset = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            offset,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(1.5, 0.5, 0.0))),
        );
        graph.add_child(
            offset,
            NodeData::Sprite(SpriteNode {
                color: [1.0, 0.4, 0.15, 0.9],
                size: 0.6,
                size_attenuation: false,
                ..Default::default()
            }),
        );

        let font_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            font_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.4, 0.0))),
        );
        graph.add_child(
            font_sep,
            NodeData::Font(FontNode {
                name: String::new(),
                size: 32.0,
                style: FontStyle::Serif,
            }),
        );
        graph.add_child(
            font_sep,
            NodeData::Text3(Text3Node {
                string: "SoFont SDF".into(),
                position: Vec3::ZERO,
                size: 24.0,
                color: [0.95, 0.92, 0.75, 1.0],
            }),
        );
    });
}
