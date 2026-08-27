//! Rotating cube — engine family + cross-node field graph.
//!
//! Usage: cargo run -p rc3d-examples --example rotating_cube

use rc3d_core::math::Vec3;
use rc3d_engine::{
    CalculatorEngine, ComposeVec3fEngine, ConcatenateEngine, DecomposeVec3fEngine,
    ElapsedTimeEngine, EngineRegistry, GateEngine, SineOscillatorEngine,
};
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;
use rc3d_scene::FieldRef;

fn main() {
    run_example("Rotating Cube", |engine| {
        let graph = engine.scene_mut();
        let root = graph.add_root(NodeData::Separator(SeparatorNode));

        graph.add_child(
            root,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(5.0, 4.0, 8.0),
                Vec3::new(1.0, 0.0, 0.0),
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

        let follow_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let follow_tf = graph.add_child(
            follow_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(-2.0, 0.0, 0.0))),
        );
        graph.add_child(
            follow_sep,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.3, 0.7, 0.4))),
        );
        graph.add_child(follow_sep, NodeData::Cube(CubeNode::default()));
        graph.connect_fields(
            FieldRef::new(transform_id, 1),
            FieldRef::new(follow_tf, 1),
        );

        let bounce_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let bounce_tf = graph.add_child(
            bounce_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(2.0, 0.0, 0.0))),
        );
        graph.add_child(
            bounce_sep,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.8, 0.4, 0.2))),
        );
        graph.add_child(bounce_sep, NodeData::Cube(CubeNode::default()));

        let pulse_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        let pulse_tf = graph.add_child(
            pulse_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(4.0, 0.0, 0.0))),
        );
        graph.add_child(
            pulse_sep,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.7, 0.7, 0.2))),
        );
        graph.add_child(pulse_sep, NodeData::Sphere(SphereNode::default()));

        let mut engines = EngineRegistry::new();
        engines.add(ElapsedTimeEngine::new(transform_id, 1.0, Vec3::Y));

        let sine = engines.add(SineOscillatorEngine::unbound(1.2, 1.0));
        let gate = engines.add(GateEngine::new());
        let calc = engines.add(CalculatorEngine::from_expr("oA = iA * 0.5"));
        let compose = engines.add(ComposeVec3fEngine::unbound_xyz(2.0, 0.0, 0.0));
        let decomp = engines.add(DecomposeVec3fEngine::new());
        let concat = engines.add(ConcatenateEngine::new());
        let pulse = engines.add(ComposeVec3fEngine::unbound_xyz(4.0, 0.0, 0.0));

        engines.connect_engines(sine, "value", gate, "input");
        engines.connect_engines(gate, "output", calc, "iA");
        engines.connect_engines(calc, "oA", compose, "y");
        engines.connect_to_node(compose, "vector", bounce_tf, 0);
        engines.connect_engines(compose, "vector", decomp, "vector");
        engines.connect_engines(decomp, "x", concat, "input0");
        engines.connect_engines(decomp, "y", concat, "input1");
        engines.connect_engines(decomp, "y", pulse, "y");
        engines.connect_to_node(pulse, "vector", pulse_tf, 0);

        engine.world_mut().engines = Some(engines);
    });
}
