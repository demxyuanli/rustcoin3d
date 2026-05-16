//! Engine demo: ElapsedTime + SineOscillator + Calculator + OneShot + Counter.
//!
//! Usage: cargo run -p rc3d-examples --example engines_demo

use rc3d_core::math::Vec3;
use rc3d_engine::engine::{
    CounterEngine, ElapsedTimeEngine, EngineRegistry, OneShotEngine, SineField,
    SineOscillatorEngine,
};
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;

fn main() {
    run_example("Engines Demo", |engine| {
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
            base_color: Vec3::new(0.7, 0.3, 0.3),
            roughness: 0.2,
            ..Default::default()
        }));

        let cube_node = graph.add_child(root, NodeData::Transform(TransformNode::default()));
        graph.add_child(cube_node, NodeData::Cube(CubeNode::default()));

        let mut registry = EngineRegistry::new();
        registry.add(ElapsedTimeEngine::new(cube_node, 0.5, Vec3::Y));
        registry.add(SineOscillatorEngine::new(cube_node, 2.0, 0.2, SineField::ScaleX));
        registry.add(OneShotEngine::new(3.0));
        registry.add(CounterEngine::new(0, 5, 1));

        engine.world_mut().engines = Some(registry);
    });
}
