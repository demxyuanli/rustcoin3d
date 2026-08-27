//! Section plane cap — sphere + cube clipped with ANSI/ISO hatch.
//!
//! Usage: cargo run -p rc3d-examples --example section_caps

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Section Caps", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        let eye = Vec3::new(1.4, 4.2, 4.6);
        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                eye,
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
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.3, 0.6, 0.9))),
        );

        graph.add_child(root, NodeData::Sphere(SphereNode { radius: 1.5 }));

        let box_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            box_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(-2.6, 0.0, 0.0))),
        );
        graph.add_child(
            box_sep,
            NodeData::Cube(CubeNode {
                width: 1.8,
                height: 1.8,
                depth: 1.8,
            }),
        );

        graph.add_child(
            root,
            NodeData::SectionPlane(SectionPlaneNode {
                // Keep y <= 0 so a high camera sees the hatched disk.
                plane: [0.0, -1.0, 0.0, 0.0],
                enabled: true,
                cap_color: [0.82, 0.36, 0.28, 1.0],
                cap_enabled: true,
                hatch_enabled: true,
                hatch_spacing: 0.16,
                hatch_angle_deg: 45.0,
                hatch_width: 0.20,
                hatch_color: [0.16, 0.07, 0.06, 1.0],
                hatch_cross: true,
            }),
        );

        let dist = eye.length();
        let ctrl = engine.camera_mut();
        ctrl.target = Vec3::new(-0.8, 0.0, 0.0);
        ctrl.distance = dist;
        ctrl.yaw = eye.x.atan2(eye.z);
        ctrl.pitch = (eye.y / dist).asin();
    });
}
