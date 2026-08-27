//! MultipleCopy + InstancedMesh patterns, plus BatchedMesh (mixed cube/sphere).
//!
//! Usage: cargo run -p rc3d-examples --example instancing

use rc3d_core::math::{Mat4, Vec3};
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Instancing", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(4.0, 2.0, 8.0),
                Vec3::new(2.0, 0.0, 0.0),
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

        let copies: Vec<Mat4> = (0..6)
            .map(|i| Mat4::from_translation(Vec3::new(i as f32 * 1.5, 0.0, 0.0)))
            .collect();
        let mc = graph.add_child(root, NodeData::MultipleCopy(MultipleCopyNode {
            copies, children: vec![],
        }));

        graph.add_child(
            root,
            NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.3, 0.6, 0.9),
                roughness: 0.3,
                ..Default::default()
            }),
        );

        let sphere = graph.add_child(mc, NodeData::Sphere(SphereNode { radius: 0.4 }));

        let sw = graph.add_child(root, NodeData::Switch(SwitchNode {
            which_child: 0,
            children: vec![sphere],
        }));

        graph.add_child(root, NodeData::Lod(LodNode {
            levels: vec![LodLevel { children: vec![sw], max_distance: 10.0 }],
            current_level: 0,
            ..Default::default()
        }));

        // Mixed cube/sphere instances share one vertex buffer (three.js BatchedMesh).
        let mut batch = rc3d_scene_api::BatchedMesh::new();
        let cube_g = batch.add_cube(0.55, 0.55, 0.55);
        let sphere_g = batch.add_sphere(0.32);
        for i in 0..8 {
            let x = (i % 4) as f32 * 1.2;
            let z = -2.4 - (i / 4) as f32 * 1.2;
            let geo = if i % 2 == 0 { cube_g } else { sphere_g };
            let id = batch.add_instance(
                geo,
                Mat4::from_translation(Vec3::new(x, 0.0, z)),
            );
            if i % 2 == 0 {
                batch.set_instance_color(id, [0.9, 0.45, 0.2, 1.0]);
            } else {
                batch.set_instance_color(id, [0.35, 0.8, 0.5, 1.0]);
            }
        }
        graph.add_child(root, batch.into_node());
    });
}
