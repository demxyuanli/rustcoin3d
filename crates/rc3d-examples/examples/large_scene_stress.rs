//! Large scene stress test — tests instanced draw batching, traversal, GPU pipelines.
//!
//! Keys:
//!   1-5: Object count presets (1K/2K/5K/10K/20K)
//!   Mouse drag: orbit camera
//!   Escape: clear selection | close window to quit

use std::sync::{Arc, Mutex};

use rc3d_engine_api::CameraController;
use rc3d_core::math::Vec3;
use rc3d_render::AdaptiveControl;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use rc3d_examples::common::run_example_with_hooks;

#[derive(Clone, Default)]
struct StressState {
    drawable_objects: usize,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();

    let target = 10000;
    let (scene, object_count) = build_large_scene(target);

    println!("Scene built: {} drawable objects", object_count);

    let state = Arc::new(Mutex::new(StressState {
        drawable_objects: object_count,
    }));

    let state_clone = state.clone();

    let ctrl = CameraController::new(Vec3::new(0.0, 1.0, 0.0), 22.0);

    run_example_with_hooks("Large Scene Stress Test", move |engine| {
        engine.load_scene(scene);
        engine.controller = ctrl;
        engine.continuous_redraw = true;
        engine.set_adaptive_quality(AdaptiveControl::Disabled);
        engine.hud_text_hook = Some(Box::new(move || {
            let s = state_clone.lock().unwrap();
            format!(
                "=== Large Scene Stress ===\nScene objects: {}\n[1]1K [2]2K [3]5K [4]10K [5]20K",
                s.drawable_objects,
            )
        }));
    });
}

fn build_large_scene(target_count: usize) -> (SceneGraph, usize) {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(0.0, 12.0, 21.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.3, -1.0, -0.2).normalize(),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 1.0,
            light_group: None,
        }),
    );

    let colors = [
        Vec3::new(0.85, 0.25, 0.25),
        Vec3::new(0.25, 0.85, 0.25),
        Vec3::new(0.25, 0.35, 0.95),
        Vec3::new(0.90, 0.80, 0.15),
        Vec3::new(0.85, 0.25, 0.75),
        Vec3::new(0.25, 0.85, 0.80),
        Vec3::new(0.70, 0.70, 0.70),
        Vec3::new(0.95, 0.55, 0.15),
    ];

    let layers = 6;
    let per_layer = target_count / layers;
    let cols = (per_layer as f32).sqrt().ceil() as usize;
    let spacing = 0.35f32;
    let half = cols as f32 * spacing * 0.5;
    let size = 0.18f32;
    let mut drawable_count = 0usize;

    for layer in 0..layers {
        let y = layer as f32 * 0.35;
        for idx in 0..per_layer {
            let row = idx / cols;
            let col = idx % cols;
            let x = col as f32 * spacing - half;
            let z = row as f32 * spacing - half;
            let color_idx = idx % colors.len();
            let shape_idx = idx % 5;

            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(x, y, z))),
            );
            graph.add_child(
                sep,
                NodeData::Material(MaterialNode {
                    base_color: colors[color_idx],
                    diffuse_color: colors[color_idx],
                    metallic: (shape_idx as f32) * 0.2,
                    roughness: 0.25 + (color_idx as f32) * 0.08,
                    opacity: 1.0,
                    ..Default::default()
                }),
            );

            let shape = match shape_idx {
                0 => NodeData::Cube(CubeNode {
                    width: size,
                    height: size,
                    depth: size,
                }),
                1 => NodeData::Sphere(SphereNode {
                    radius: size * 0.55,
                }),
                2 => NodeData::Cone(ConeNode {
                    bottom_radius: size * 0.55,
                    height: size,
                }),
                _ => NodeData::Cylinder(CylinderNode {
                    radius: size * 0.4,
                    height: size,
                }),
            };
            graph.add_child(sep, shape);
            drawable_count += 1;
        }
    }

    // Floor
    let floor = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        floor,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, -2.0, 0.0))),
    );
    graph.add_child(
        floor,
        NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.15, 0.15, 0.15),
            diffuse_color: Vec3::new(0.15, 0.15, 0.15),
            metallic: 0.0,
            roughness: 0.95,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(
        floor,
        NodeData::Cube(CubeNode {
            width: cols as f32 * spacing + 2.0,
            height: 0.15,
            depth: cols as f32 * spacing + 2.0,
        }),
    );
    drawable_count += 1;

    (graph, drawable_count)
}
