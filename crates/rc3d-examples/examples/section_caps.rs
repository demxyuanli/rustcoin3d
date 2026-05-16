//! Section plane cap — sphere clipped by a plane with filled cross-section.
//!
//! Usage: cargo run -p rc3d-examples --example section_caps

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Section Caps", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        let eye = Vec3::new(3.0, 2.5, 4.0);
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

        graph.add_child(
            root,
            NodeData::SectionPlane(SectionPlaneNode {
                plane: [0.0, 1.0, 0.0, 0.3],
                enabled: true,
                cap_color: [0.8, 0.3, 0.3, 1.0],
                cap_enabled: true,
            }),
        );

        let dist = eye.length();
        let ctrl = engine.camera_mut();
        ctrl.target = Vec3::ZERO;
        ctrl.distance = dist;
        ctrl.yaw = eye.x.atan2(eye.z);
        ctrl.pitch = (eye.y / dist).asin();
    });
}
