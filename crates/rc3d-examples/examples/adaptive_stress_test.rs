//! Adaptive quality stress test — exercises the 5-level quality controller.
//!
//! Creates a scene complex enough to trigger quality downgrades, then
//! demonstrates the EMA + hysteresis recovery behavior.
//!
//! Keys:
//!   +/-: Increase/decrease scene complexity (add/remove objects)
//!   Q: Show current quality level
//!   L: Lock/unlock quality level
//!   Mouse drag: orbit camera
//!   Escape: clear selection | close window to quit

use std::sync::{Arc, Mutex};

use rc3d_engine_api::{CameraController, Engine};
use rc3d_core::math::Vec3;
use rc3d_render::AdaptiveControl;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use rc3d_examples::common::run_example_with_hooks;
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Default)]
struct StressState {
    object_count: usize,
    locked: bool,
    current_quality: String,
    frame_time_ms: f64,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();

    println!("Adaptive quality stress test");
    println!("Usage: cargo run -p rc3d-examples --example adaptive_stress_test");
    println!("Keys: +/- objects | Q quality | L lock | Escape clears selection");

    let state = Arc::new(Mutex::new(StressState {
        object_count: 800,
        locked: false,
        current_quality: String::from("Ultra"),
        frame_time_ms: 16.6,
    }));

    let scene = build_dense_scene(800);
    let state_clone = state.clone();

    let ctrl = CameraController::new(Vec3::new(0.0, 8.0, 15.0), 20.0);

    run_example_with_hooks("Adaptive Quality Stress Test", move |engine| {
        engine.load_scene(scene);
        engine.controller = ctrl;
        engine.continuous_redraw = true;
        engine.set_adaptive_quality(AdaptiveControl::Dynamic {
            allow_downgrade: true,
        });
        engine.hud_text_hook = Some(Box::new(move || {
            let s = state_clone.lock().unwrap();
            let fps = 1000.0 / s.frame_time_ms.max(0.01);
            let lines = vec![
                "=== Adaptive Quality Stress ===".to_string(),
                format!(
                    "Objects: {}  |  Frame: {:.2}ms ({:.0}fps)",
                    s.object_count, s.frame_time_ms, fps
                ),
                format!("Quality: {}  |  Locked: {}", s.current_quality, s.locked),
                "".to_string(),
                "Quality levels:".to_string(),
                "  Ultra   — >=60fps, full effects, 4 cascades @ 2048x2048".to_string(),
                "  High    — 45-60fps, full effects, 4 cascades @ 2048x2048".to_string(),
                "  Medium  — 30-45fps, reduced SSAO, 2 cascades @ 1024x1024".to_string(),
                "  Low     — 20-30fps, no SSAO/Bloom, 1 cascade @ 512x512".to_string(),
                "  Minimal — <20fps, no post, flat shading, 1 cascade @ 256x256".to_string(),
                "".to_string(),
                "[+/-] Complexity  [L] Quality Lock  [Q] Quality  [Esc] Clear selection"
                    .to_string(),
            ];
            lines.join("\n")
        }));
        engine.panel_overlay_key_hook = Some(Box::new({
            let state = state.clone();
            move |key: PhysicalKey| -> bool {
                if let Ok(mut s) = state.lock() {
                    match key {
                        PhysicalKey::Code(KeyCode::Equal)
                        | PhysicalKey::Code(KeyCode::NumpadAdd) => {
                            s.object_count = (s.object_count + 200).min(5000);
                            println!(
                                "Scene complexity: {} objects (restart to apply)",
                                s.object_count
                            );
                            return true;
                        }
                        PhysicalKey::Code(KeyCode::Minus)
                        | PhysicalKey::Code(KeyCode::NumpadSubtract) => {
                            s.object_count =
                                s.object_count.saturating_sub(200).max(100);
                            println!(
                                "Scene complexity: {} objects (restart to apply)",
                                s.object_count
                            );
                            return true;
                        }
                        PhysicalKey::Code(KeyCode::KeyL) => {
                            s.locked = !s.locked;
                            println!(
                                "Quality lock: {}",
                                if s.locked { "ON" } else { "OFF" }
                            );
                            return true;
                        }
                        PhysicalKey::Code(KeyCode::KeyQ) => {
                            println!(
                                "Current quality: {} @ {:.2}ms",
                                s.current_quality, s.frame_time_ms
                            );
                            return true;
                        }
                        _ => {}
                    }
                }
                false
            }
        }));
    });
}

fn build_dense_scene(count: usize) -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(0.0, 8.0, 15.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Directional light for CSM shadow stress
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.7, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 1.2,
            light_group: None,
        }),
    );

    // Point lights for cluster lighting stress
    for i in 0..4 {
        let angle = i as f32 * std::f32::consts::FRAC_PI_2;
        let light_sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            light_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(
                angle.cos() * 8.0,
                4.0,
                angle.sin() * 8.0,
            ))),
        );
        graph.add_child(
            light_sep,
            NodeData::PointLight(PointLightNode {
                location: Vec3::ZERO,
                color: Vec3::new(0.8, 0.6, 0.4),
                intensity: 50.0,
                light_group: None,
            }),
        );
    }

    // Dense grid of objects
    let layers = 3;
    let per_layer = count / layers;
    let cols = (per_layer as f32).sqrt().ceil() as i32;
    let spacing = 1.2f32;

    for layer in 0..layers {
        let y_base = layer as f32 * 1.5;
        for i in 0..per_layer.min(cols as usize * cols as usize) {
            let row = i as i32 / cols;
            let col = i as i32 % cols;
            let x = col as f32 * spacing - cols as f32 * spacing * 0.5;
            let z = row as f32 * spacing - cols as f32 * spacing * 0.5;

            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(
                    x, y_base, z,
                ))),
            );
            graph.add_child(
                sep,
                NodeData::Material(MaterialNode {
                    base_color: Vec3::new(
                        (x + 5.0) / 15.0,
                        (z + 5.0) / 15.0,
                        y_base / 5.0,
                    ),
                    diffuse_color: Vec3::new(0.5, 0.5, 0.5),
                    metallic: (i % 3) as f32 * 0.5,
                    roughness: 0.2 + (i % 7) as f32 * 0.1,
                    opacity: 1.0,
                    ..Default::default()
                }),
            );
            let _ = match i % 4 {
                0 => graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 0.35 })),
                1 => graph.add_child(sep, NodeData::Cube(CubeNode::default())),
                2 => graph.add_child(sep, NodeData::Cone(ConeNode::default())),
                _ => graph.add_child(sep, NodeData::Cylinder(CylinderNode::default())),
            };
        }
    }

    // Floor
    let floor = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        floor,
        NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, -1.0, 0.0))),
    );
    graph.add_child(
        floor,
        NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.25, 0.25, 0.25),
            metallic: 0.0,
            roughness: 0.95,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(
        floor,
        NodeData::Cube(CubeNode {
            width: cols as f32 * spacing + 4.0,
            height: 0.15,
            depth: cols as f32 * spacing + 4.0,
        }),
    );

    graph
}
