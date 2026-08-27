//! BlendNode animation demo — clip blending with weight parameter.
//!
//! Usage: cargo run -p rc3d-examples --example blend_animation

use rc3d_core::math::{Mat4, Vec3};
use rc3d_examples::common::run_example;
use rc3d_scene::{
    animation::{AnimationClip, BlendNode, Joint, Skeleton},
    node_data::*,
};

fn main() {
    run_example("Blend Animation", |engine| {
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

        graph.add_child(root, NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.3, 0.7, 0.9),
            roughness: 0.4,
            ..Default::default()
        }));

        graph.add_child(
            root,
            NodeData::Transform(TransformNode {
                translation: Vec3::new(0.0, 1.5, 0.0),
                ..Default::default()
            }),
        );
        graph.add_child(root, NodeData::Sphere(SphereNode { radius: 0.6 }));

        // Build a simple blend tree
        let skeleton = Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: 1,
            bind_transform: Mat4::IDENTITY,
            inverse_bind_matrix: Mat4::IDENTITY,
        }]);
        let idle = AnimationClip { name: "idle".into(), duration: 1.0, ..Default::default() };
        let walk = AnimationClip { name: "walk".into(), duration: 0.5, ..Default::default() };
        let blend = BlendNode::Blend {
            left: Box::new(BlendNode::Clip { clip: idle, speed: 1.0, start_time: 0.0 }),
            right: Box::new(BlendNode::Clip { clip: walk, speed: 1.0, start_time: 0.0 }),
            weight: 0.5,
        };
        if blend.sample(0.0, &skeleton).is_some() {
            log::info!("Blend tree sampled successfully at time 0");
        }
    });
}
