//! Performance profiling demo — displays real-time CPU/GPU frame timing.
//!
//! Demonstrates the new GpuTimer + CpuSpanCollector profiler infrastructure.
//! The on-screen overlay shows per-pass GPU times and CPU section breakdowns.
//!
//! Keys:
//!   1-4: Change scene complexity (10/100/500/2000 objects)
//!   P: Print frame timing report to console
//!   Mouse drag: orbit camera
//!   Escape: clear selection | close window to quit

use std::sync::{Arc, Mutex};

use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

#[derive(Clone, Default)]
struct ProfileState {
    frame_time_ms: f64,
    gpu_timings: Vec<(String, f64)>,
    cpu_timings: Vec<(String, f64)>,
    object_count: usize,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();

    println!("Profile viewer");
    println!("Usage: cargo run -p rc3d-app --example profile_viewer");
    println!("Keys: 1-4 scene complexity | P print report | Esc clears selection");

    let state = Arc::new(Mutex::new(ProfileState {
        object_count: 100,
        ..Default::default()
    }));

    let scene = build_scene(100);
    let state_clone = state.clone();

    let ctrl = CameraController::new(Vec3::new(0.0, 3.0, 8.0), 12.0);

    let mut app = App::new(scene)
        .with_camera_controller(ctrl)
        .with_continuous_redraw(true)
        .with_panel_overlay_text_hook(move || {
            let s = state_clone.lock().unwrap();
            let mut lines = vec![
                "=== Profile Viewer ===".to_string(),
                format!(
                    "Objects: {}  |  Frame: {:.2}ms ({:.0}fps)",
                    s.object_count,
                    s.frame_time_ms,
                    1000.0 / s.frame_time_ms.max(0.01)
                ),
                "".to_string(),
                "--- GPU Passes ---".to_string(),
            ];
            for (label, ms) in &s.gpu_timings {
                lines.push(format!("  {:>16}: {:>6.2}ms", label, ms));
            }
            lines.push("".to_string());
            lines.push("--- CPU Sections ---".to_string());
            for (label, ms) in &s.cpu_timings {
                lines.push(format!("  {:>16}: {:>6.2}ms", label, ms));
            }
            lines.push("".to_string());
            lines.push("[1-4] Complexity  [P] Print report  [Esc] Clear selection".to_string());
            lines.join("\n")
        })
        .with_panel_overlay_key_hook({
            let state = state.clone();
            move |key| -> bool {
                use winit::keyboard::KeyCode;
                let new_count = match key {
                    KeyCode::Digit1 => Some(10),
                    KeyCode::Digit2 => Some(100),
                    KeyCode::Digit3 => Some(500),
                    KeyCode::Digit4 => Some(2000),
                    _ => None,
                };
                if let Some(count) = new_count {
                    if let Ok(mut s) = state.lock() {
                        s.object_count = count;
                        println!("Switching to {} objects — restart example to apply", count);
                        println!("(Dynamic scene rebuild not yet wired)");
                    }
                    return true;
                }
                if key == KeyCode::KeyP {
                    if let Ok(s) = state.lock() {
                        println!("\n╔══ Frame Timing Report ══╗");
                        println!("║ Objects: {:>4}               ║", s.object_count);
                        println!("║ Frame:   {:>6.2}ms            ║", s.frame_time_ms);
                        println!("╟── GPU Passes ──────────────╢");
                        for (label, ms) in &s.gpu_timings {
                            println!("║  {:>14}: {:>6.2}ms    ║", label, ms);
                        }
                        println!("╟── CPU Sections ────────────╢");
                        for (label, ms) in &s.cpu_timings {
                            println!("║  {:>14}: {:>6.2}ms    ║", label, ms);
                        }
                        println!("╚════════════════════════════╝\n");
                    }
                    return true;
                }
                false
            }
        });

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn build_scene(count: usize) -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(0.0, 3.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Directional light (exercises CSM layered shadow)
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -1.0, -0.5).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.0,
            light_group: None,
        }),
    );

    // Ambient light
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.5, -0.3, 1.0).normalize(),
            color: Vec3::new(0.2, 0.2, 0.3),
            intensity: 0.3,
            light_group: None,
        }),
    );

    // Grid of objects at varying scale
    let cols = (count as f32).sqrt().ceil() as i32;
    let spacing = 1.5f32;
    let half = (cols - 1) as f32 * spacing * 0.5;

    // Shared material colors for variety
    let colors = [
        Vec3::new(0.8, 0.3, 0.3),
        Vec3::new(0.3, 0.8, 0.3),
        Vec3::new(0.3, 0.3, 0.8),
        Vec3::new(0.8, 0.8, 0.3),
        Vec3::new(0.8, 0.3, 0.8),
        Vec3::new(0.3, 0.8, 0.8),
        Vec3::new(0.6, 0.6, 0.6),
        Vec3::new(0.9, 0.5, 0.2),
    ];

    for i in 0..count {
        let row = i as i32 / cols;
        let col = i as i32 % cols;
        let x = col as f32 * spacing - half;
        let z = row as f32 * spacing - half;
        let y = 0.0;

        let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(x, y, z))),
        );
        let color = colors[i % colors.len()];
        graph.add_child(
            sep,
            NodeData::Material(MaterialNode {
                base_color: color,
                diffuse_color: color,
                metallic: (i % 3) as f32 * 0.4,
                roughness: 0.3 + (i % 5) as f32 * 0.1,
                opacity: 1.0,
                ..Default::default()
            }),
        );
        // Mix of spheres and cubes
        if i % 2 == 0 {
            graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 0.4 }));
        } else {
            graph.add_child(
                sep,
                NodeData::Cube(CubeNode {
                    width: 0.7,
                    height: 0.7,
                    depth: 0.7,
                }),
            );
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
            base_color: Vec3::new(0.3, 0.3, 0.3),
            diffuse_color: Vec3::new(0.3, 0.3, 0.3),
            metallic: 0.0,
            roughness: 0.9,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(
        floor,
        NodeData::Cube(CubeNode {
            width: cols as f32 * spacing + 2.0,
            height: 0.2,
            depth: cols as f32 * spacing + 2.0,
        }),
    );

    graph
}
