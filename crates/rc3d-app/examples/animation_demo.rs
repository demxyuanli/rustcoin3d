//! Animation demo — engine-driven transform animation.
//!
//! Shows:
//!   - Rotating cube (ElapsedTimeEngine)
//!   - Oscillating sphere (SineOscillatorEngine)
//!   - Orbiting point lights
//!   - Static ground plane for reference
//!
//! Controls: mouse drag to orbit, scroll to zoom, ESC to exit.

use rc3d_app::{App, CameraController};
use rc3d_core::math::Vec3;
use rc3d_engine::{ElapsedTimeEngine, EngineRegistry, InterpolateVec3Engine, SineField, SineOscillatorEngine};
use rc3d_render::background::{BgMode, BgSettings};
use rc3d_scene::node_data::*;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d=info"),
    )
    .init();

    let mut graph = rc3d_scene::SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(4.0, 3.0, 6.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Directional light
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 0.8,
            light_group: None,
        }),
    );

    // Two point lights orbiting the scene
    let light1_id = graph.add_child(
        root,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(3.0, 2.0, 0.0))),
    );
    graph.add_child(
        light1_id,
        NodeData::PointLight(PointLightNode {
            location: Vec3::ZERO,
            color: Vec3::new(1.0, 0.3, 0.2),
            intensity: 15.0,
            light_group: None,
        }),
    );
    let light2_id = graph.add_child(
        root,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(-3.0, 2.0, 0.0))),
    );
    graph.add_child(
        light2_id,
        NodeData::PointLight(PointLightNode {
            location: Vec3::ZERO,
            color: Vec3::new(0.2, 0.5, 1.0),
            intensity: 15.0,
            light_group: None,
        }),
    );

    // Ground plane
    graph.add_child(
        root,
        NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.3, 0.3, 0.35),
            roughness: 0.8,
            ..Default::default()
        }),
    );
    let ground = graph.add_child(
        root,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, -0.55, 0.0))),
    );
    graph.add_child(ground, NodeData::Cube(CubeNode { width: 8.0, height: 0.1, depth: 8.0 }));

    // ===== Animated objects =====

    // 1. Rotating cube (blue)
    let cube_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
    let cube_tf = graph.add_child(cube_sep, NodeData::Transform(TransformNode::from_translation(
        Vec3::new(-1.5, 1.0, 0.0),
    )));
    graph.add_child(
        cube_sep,
        NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.2, 0.4, 0.9),
            metallic: 0.3,
            roughness: 0.3,
            ..Default::default()
        }),
    );
    graph.add_child(cube_sep, NodeData::Cube(CubeNode { width: 0.8, height: 0.8, depth: 0.8 }));

    // 2. Oscillating sphere (orange)
    let sphere_tf = graph.add_child(root, NodeData::Transform(TransformNode::from_translation(
        Vec3::new(0.0, 1.5, 0.0),
    )));
    graph.add_child(
        root,
        NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.9, 0.5, 0.1),
            metallic: 0.1,
            roughness: 0.4,
            ..Default::default()
        }),
    );
    graph.add_child(sphere_tf, NodeData::Sphere(SphereNode { radius: 0.6 }));

    // 3. Sliding cube (green, moves back and forth via InterpolateVec3Engine)
    let slide_tf = graph.add_child(root, NodeData::Transform(TransformNode::from_translation(
        Vec3::new(1.5, 0.4, 0.0),
    )));
    graph.add_child(
        root,
        NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.1, 0.8, 0.3),
            metallic: 0.2,
            roughness: 0.5,
            ..Default::default()
        }),
    );
    graph.add_child(slide_tf, NodeData::Cube(CubeNode { width: 0.6, height: 0.6, depth: 0.6 }));

    // ==== Engines ====
    let mut engines = EngineRegistry::new();

    // Cube rotates around Y + wobble around Z
    engines.add(ElapsedTimeEngine::new(cube_tf, 1.2, Vec3::new(0.0, 1.0, 0.3).normalize()));

    // Sphere bounces up and down
    engines.add(SineOscillatorEngine::new(sphere_tf, 0.8, 0.8, SineField::TranslationY));

    // Green cube slides left-right
    engines.add(InterpolateVec3Engine {
        transform_node: slide_tf,
        from: Vec3::new(1.5, 0.4, -1.5),
        to: Vec3::new(1.5, 0.4, 1.5),
        period_secs: 3.0,
    });

    // Two point lights orbit
    engines.add(ElapsedTimeEngine::new(light1_id, 0.6, Vec3::new(0.0, 0.0, 1.0)));
    engines.add(ElapsedTimeEngine::new(light2_id, -0.6, Vec3::new(0.0, 0.0, 1.0)));

    let ctrl = CameraController::new(Vec3::new(0.0, 1.0, 0.0), 8.0);

    println!("Animation demo started:");
    println!("  Blue cube:   rotating");
    println!("  Orange sphere: bouncing");
    println!("  Green cube:  sliding left-right");
    println!("  Red+Blue lights: orbiting");

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut App::new(graph)
            .with_camera_controller(ctrl)
            .with_engines(engines)
            .with_background(BgSettings {
                mode: BgMode::Gradient,
                top_color: [0.05, 0.1, 0.2, 1.0],
                bot_color: [0.01, 0.02, 0.05, 1.0],
                ..Default::default()
            }))
        .expect("event loop");
}
