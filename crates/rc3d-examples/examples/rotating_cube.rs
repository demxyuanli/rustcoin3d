//! Rotating cube — engine-driven rotation animation.
//!
//! Usage: cargo run -p rc3d-examples --example rotating_cube

use rc3d_core::math::Vec3;
use rc3d_engine::{ElapsedTimeEngine, EngineRegistry};
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Rotating Cube", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(3.0, 3.0, 5.0),
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

        let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let transform_id = graph.add_child(sep, NodeData::Transform(TransformNode::default()));
        graph.add_child(
            sep,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.2, 0.5, 0.8))),
        );
        graph.add_child(sep, NodeData::Cube(CubeNode::default()));

        let mut engines = EngineRegistry::new();
        engines.add(ElapsedTimeEngine::new(transform_id, 1.0, Vec3::Y));
        engine.world_mut().engines = Some(engines);
    });
}
