//! Scene graph example — multiple node hierarchy with per-node material and transform.
//!
//! Usage: cargo run -p rc3d-examples --example scene_graph

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Scene Graph", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        // Key light
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -1.0, -0.5).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }));
        // Fill light
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(1.0, -0.6, 0.8).normalize(),
            color: Vec3::new(0.9, 0.92, 1.0),
            intensity: 0.5,
            light_group: None,
        }));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(0.0, 3.0, 8.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        // Red cube at (-2.5, 0, 0)
        {
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(-2.5, 0.0, 0.0))),
            );
            graph.add_child(
                sep,
                NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.9, 0.2, 0.2))),
            );
            graph.add_child(sep, NodeData::Cube(CubeNode::default()));
        }

        // Green sphere at (0, 0, 0)
        {
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                sep,
                NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.8, 0.2))),
            );
            graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 0.7 }));
        }

        // Blue cone at (2.5, 0, 0)
        {
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(2.5, 0.0, 0.0))),
            );
            graph.add_child(
                sep,
                NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.3, 0.9))),
            );
            graph.add_child(sep, NodeData::Cone(ConeNode { bottom_radius: 0.6, height: 1.5 }));
        }

        // Yellow cylinder at (-1.25, 0, -2.5)
        {
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(-1.25, 0.0, -2.5))),
            );
            graph.add_child(
                sep,
                NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.9, 0.8, 0.1))),
            );
            graph.add_child(sep, NodeData::Cylinder(CylinderNode { radius: 0.4, height: 1.2 }));
        }

        // Purple scaled cube at (1.25, 0, -2.5)
        {
            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(sep, NodeData::Transform(TransformNode {
                translation: Vec3::new(1.25, 0.0, -2.5),
                scale: Vec3::new(0.5, 1.5, 0.5),
                ..Default::default()
            }));
            graph.add_child(
                sep,
                NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.7, 0.2, 0.8))),
            );
            graph.add_child(sep, NodeData::Cube(CubeNode::default()));
        }

        graph.add_child(root, NodeData::EventCallback(EventCallbackNode::default()));
    });
}
