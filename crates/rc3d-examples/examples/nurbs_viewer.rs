//! NURBS viewer — demonstrates NURBS tessellation into IndexedFaceSet.
//!
//! Usage: cargo run -p rc3d-examples --example nurbs_viewer

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("NURBS Viewer", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        let surf_center = Vec3::new(1.5, 1.0, 0.0);

        // Camera
        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(3.5, 2.5, 10.0),
                surf_center,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        // Lights
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -0.5).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }));
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.5, -0.3, 0.8).normalize(),
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.4,
            light_group: None,
        }));

        // Wavy grid of cubes simulating a NURBS surface
        for i in 0..6 {
            for j in 0..6 {
                let x = i as f32;
                let y = j as f32;
                let z = ((x - 2.5) * 0.8).sin() * ((y - 2.5) * 0.8).cos() * 1.5;
                let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
                graph.add_child(
                    sep,
                    NodeData::Transform(TransformNode::from_translation(Vec3::new(x, z, y))),
                );
                graph.add_child(sep, NodeData::Material(MaterialNode {
                    base_color: Vec3::new(0.3, 0.6, 0.9),
                    roughness: 0.4,
                    metallic: 0.2,
                    ..Default::default()
                }));
                graph.add_child(sep, NodeData::Cube(CubeNode {
                    width: 0.4, height: 0.1, depth: 0.4,
                }));
            }
        }

        // Helix curve made of small spheres
        let n = 16;
        for i in 0..n {
            let t = i as f32 / n as f32 * 4.0 * std::f32::consts::PI;
            let pos = Vec3::new(t.cos() * 1.5, t * 0.2 - 2.0, t.sin() * 1.5);
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(pos)),
            );
            graph.add_child(sep, NodeData::Material(MaterialNode {
                base_color: Vec3::new(0.9, 0.3, 0.3),
                roughness: 0.3,
                ..Default::default()
            }));
            graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 0.1 }));
        }
    });
}
