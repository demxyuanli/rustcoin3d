//! StereoCamera demo — side-by-side stereo rendering with interocular distance.
//!
//! Usage: cargo run -p rc3d-examples --example stereo_camera

use rc3d_core::math::Vec3;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Stereo Camera", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        let cam_id = graph.add_child(
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

        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.5, 0.3, 0.8),
            roughness: 0.2,
            ..Default::default()
        }));
        graph.add_child(root, NodeData::Sphere(SphereNode { radius: 1.0 }));

        graph.add_child(root, NodeData::StereoCamera(StereoCameraNode {
            base_camera: cam_id,
            interocular_distance: 0.065,
            convergence_distance: 2.0,
            mode: StereoMode::SideBySide,
        }));
    });
}
