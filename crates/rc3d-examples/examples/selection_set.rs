//! SelectionSet demo — named selection groups for batch operations.
//!
//! Usage: cargo run -p rc3d-examples --example selection_set

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Selection Set", |engine| {
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

        let mut ids = Vec::new();
        for i in 0..5i32 {
            graph.add_child(root, NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.3 + i as f32 * 0.1, 0.2, 0.7),
                roughness: 0.3,
                ..Default::default()
            }));
            graph.add_child(
                root,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(
                    i as f32 * 1.5 - 3.0, 0.0, 0.0,
                ))),
            );
            ids.push(graph.add_child(root, NodeData::Sphere(SphereNode { radius: 0.4 })));
        }

        graph.selection_set_add("spheres", &ids);
        graph.selection_set_select("spheres");
    });
}
