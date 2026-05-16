//! MaterialBinding + ShapeHints + Texture2Transform demo.
//!
//! Usage: cargo run -p rc3d-examples --example material_variants

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Material Variants", |engine| {
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
            NodeData::ShapeHints(ShapeHintsNode {
                vertex_ordering: VertexOrdering::CounterClockwise,
                ..Default::default()
            }),
        );

        graph.add_child(
            root,
            NodeData::MaterialBinding(MaterialBindingNode {
                value: MaterialBinding::PerVertex,
            }),
        );

        for i in 0..3i32 {
            let c = [
                Vec3::new(0.9, 0.2, 0.2),
                Vec3::new(0.2, 0.9, 0.2),
                Vec3::new(0.2, 0.2, 0.9),
            ][i as usize];
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(sep, NodeData::Material(MaterialNode {
                base_color: c,
                roughness: 0.3,
                metallic: i as f32 * 0.2,
                ..Default::default()
            }));
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(
                    i as f32 * 2.0 - 2.0, 0.0, 0.0,
                ))),
            );
            graph.add_child(sep, NodeData::Cube(CubeNode::default()));
        }

        graph.add_child(
            root,
            NodeData::Texture2Transform(Texture2TransformNode {
                rotation: 0.5,
                ..Default::default()
            }),
        );
    });
}
