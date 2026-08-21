//! Picking example — click a shape to outline it (three.js OutlinePass).
//!
//! Left-click selects; click empty space clears. Orbit with the right mouse button.
//!
//! Usage: cargo run -p rc3d-examples --example picking
//!   --screenshot --screenshot-exit  writes target/picking.png with the red cube outlined

use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Picking", |engine| {
        engine.set_display_mode(DisplayMode::Shaded);
        if let Some(renderer) = engine.renderer.as_mut() {
            renderer.set_outline_width(1.0);
            renderer.set_outline_color([1.0, 0.5, 0.0, 1.0]);
        }

        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

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

        graph.add_child(
            root,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-0.5, -1.0, -0.5).normalize(),
                color: Vec3::ONE,
                intensity: 1.0,
                light_group: None,
            }),
        );

        // Red cube
        let sep1 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            sep1,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(-2.5, 0.0, 0.0))),
        );
        graph.add_child(
            sep1,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.9, 0.2, 0.2))),
        );
        let red_cube = graph.add_child(sep1, NodeData::Cube(CubeNode::default()));

        // Green sphere
        let sep2 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            sep2,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.8, 0.2))),
        );
        graph.add_child(sep2, NodeData::Sphere(SphereNode { radius: 0.8 }));

        // Blue cone
        let sep3 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            sep3,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(2.5, 0.0, 0.0))),
        );
        graph.add_child(
            sep3,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.3, 0.9))),
        );
        graph.add_child(sep3, NodeData::Cone(ConeNode {
            bottom_radius: 0.7, height: 1.5,
        }));

        // Yellow cylinder
        let sep4 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            sep4,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(-1.25, 0.0, -2.5))),
        );
        graph.add_child(
            sep4,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.9, 0.8, 0.1))),
        );
        graph.add_child(sep4, NodeData::Cylinder(CylinderNode {
            radius: 0.5, height: 1.2,
        }));

        // Purple scaled cube
        let sep5 = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep5, NodeData::Transform(TransformNode {
            translation: Vec3::new(1.25, 0.0, -2.5),
            scale: Vec3::new(0.5, 1.5, 0.5),
            ..Default::default()
        }));
        graph.add_child(
            sep5,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.7, 0.2, 0.8))),
        );
        graph.add_child(sep5, NodeData::Cube(CubeNode::default()));

        graph.add_child(root, NodeData::EventCallback(EventCallbackNode::default()));

        graph.select(red_cube);

        engine.on_pick = Some(Box::new(|graph, node, _point| {
            graph.clear_selection();
            graph.select(node);
        }));
        engine.hud_text_hook = Some(Box::new(|| {
            "Left-click a shape to outline it".to_string()
        }));
    });
}
