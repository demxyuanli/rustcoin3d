//! Markup Dimensions demo — 3D annotations on Cube/Sphere/Cylinder.
//!
//! Usage: cargo run -p rc3d-examples --example markup_dimensions

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Markup Dimensions", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(4.0, 3.0, 12.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );

        // Directional light
        graph.add_child(root, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }));

        // Cube (left, orange)
        let cube_x = -3.5;
        let cube_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            cube_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(cube_x, 0.0, 0.0))),
        );
        graph.add_child(
            cube_sep,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(1.0, 0.6, 0.2))),
        );
        graph.add_child(cube_sep, NodeData::Cube(CubeNode {
            width: 1.5, height: 1.5, depth: 1.5,
        }));

        // Sphere (center, green)
        let sphere_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            sphere_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 0.5, 0.0))),
        );
        graph.add_child(
            sphere_sep,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.8, 0.3))),
        );
        graph.add_child(sphere_sep, NodeData::Sphere(SphereNode { radius: 1.0 }));

        // Cylinder (right, blue)
        let cyl_x = 3.5;
        let cyl_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            cyl_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(cyl_x, 0.0, 0.0))),
        );
        graph.add_child(
            cyl_sep,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.3, 0.4, 0.9))),
        );
        graph.add_child(cyl_sep, NodeData::Cylinder(CylinderNode {
            radius: 0.8, height: 2.0,
        }));

        // Annotation overlay
        let ann = graph.add_child(root, NodeData::Annotation(AnnotationNode));
        graph.add_child(ann, NodeData::Text2(Text2Node {
            string: "Cube".into(),
            position: [100.0, 50.0],
            size: 18.0,
            color: [1.0, 0.6, 0.2, 1.0],
        }));
        graph.add_child(ann, NodeData::Text2(Text2Node {
            string: "Sphere".into(),
            position: [380.0, 50.0],
            size: 18.0,
            color: [0.2, 0.8, 0.3, 1.0],
        }));
        graph.add_child(ann, NodeData::Text2(Text2Node {
            string: "Cylinder".into(),
            position: [650.0, 50.0],
            size: 18.0,
            color: [0.3, 0.4, 0.9, 1.0],
        }));
    });
}
