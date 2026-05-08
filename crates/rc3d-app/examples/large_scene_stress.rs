//! Large scene stress test — 10,000 objects exercising all optimized paths.
//!
//! Tests: CSM cascade frustum pre-cull, incremental BVH, material bind group
//! cache, flat draw cache, adaptive quality at scale.
//!
//! Keys:
//!   1-5: Object count (1K/2K/5K/10K/20K)
//!   F: Toggle frustum culling stats
//!   Q: Show quality level
//!   Mouse drag: orbit camera
//!   ESC: exit

use std::sync::{Arc, Mutex};

use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

#[derive(Clone, Default)]
struct StressState {
    object_count: usize,
    frame_time_ms: f64,
    quality: String,
    draws: usize,
    tris: u64,
    culled: usize,
    target_count: usize,
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d=info"),
    )
    .init();

    println!("Large scene stress test — 10,000 objects");
    println!("Usage: cargo run -p rc3d-app --example large_scene_stress --release");
    println!("Keys: 1-5 density | F cull stats | Q quality | ESC exit");

    let state = Arc::new(Mutex::new(StressState {
        target_count: 10000,
        ..Default::default()
    }));

    let scene = build_large_scene(10000);
    let state_clone = state.clone();

    let ctrl = CameraController::new(Vec3::new(0.0, 30.0, 50.0), 60.0);
    let mut app = App::new(scene)
        .with_camera_controller(ctrl)
        .with_continuous_redraw(true)
        .with_panel_overlay_text_hook(move || {
            let s = state_clone.lock().unwrap();
            let fps = 1000.0 / s.frame_time_ms.max(0.01);
            let percentage = if s.draws > 0 {
                s.culled * 100 / (s.draws + s.culled)
            } else {
                0
            };
            let lines = vec![
                "=== Large Scene Stress Test ===".to_string(),
                format!("Objects: {}  |  Frame: {:.2}ms ({:.0}fps)",
                    s.object_count, s.frame_time_ms, fps),
                format!("Quality: {}  |  Draws: {}  |  Tris: {}",
                    s.quality, s.draws, s.tris),
                format!("Culled: {} ({}%)  |  Target: {}",
                    s.culled, percentage, s.target_count),
                "".to_string(),
                "[1] 1K  [2] 2K  [3] 5K  [4] 10K  [5] 20K".to_string(),
                "[F] Cull stats  [Q] Quality  [ESC] Exit".to_string(),
            ];
            lines.join("\n")
        });

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn build_large_scene(count: usize) -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera — far back to see the whole scene
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(0.0, 30.0, 50.0),
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Directional light for CSM shadow stress
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 1.0,
            light_group: None,
        }),
    );

    // Grid of objects in XZ plane, stacked in layers
    let layers = 5;
    let per_layer = count / layers;
    let cols = (per_layer as f32).sqrt().ceil() as i32;
    let spacing = 2.5f32;

    let colors = [
        Vec3::new(0.8, 0.3, 0.3), Vec3::new(0.3, 0.8, 0.3),
        Vec3::new(0.3, 0.3, 0.8), Vec3::new(0.8, 0.8, 0.3),
        Vec3::new(0.8, 0.3, 0.8), Vec3::new(0.3, 0.8, 0.8),
        Vec3::new(0.6, 0.6, 0.6), Vec3::new(0.9, 0.5, 0.2),
    ];

    for layer in 0..layers {
        let y_base = layer as f32 * 3.0;
        for i in 0..per_layer.min(cols as usize * cols as usize) {
            let row = i as i32 / cols;
            let col = i as i32 % cols;
            let x = col as f32 * spacing - cols as f32 * spacing * 0.5;
            let z = row as f32 * spacing - cols as f32 * spacing * 0.5;

            let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
            graph.add_child(
                sep,
                NodeData::Transform(TransformNode::from_translation(Vec3::new(x, y_base, z))),
            );
            let color = colors[i % colors.len()];
            graph.add_child(
                sep,
                NodeData::Material(MaterialNode {
                    base_color: color,
                    diffuse_color: color,
                    metallic: (i % 4) as f32 * 0.3,
                    roughness: 0.2 + (i % 8) as f32 * 0.1,
                    opacity: 1.0,
                    ..Default::default()
                }),
            );
            // Mix of geometry types
            let _ = match i % 5 {
                0 => graph.add_child(sep, NodeData::Cube(CubeNode::default())),
                1 => graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 0.8 })),
                2 => graph.add_child(sep, NodeData::Cone(ConeNode::default())),
                3 => graph.add_child(
                    sep,
                    NodeData::Cube(CubeNode { width: 0.6, height: 2.0, depth: 0.6 }),
                ),
                _ => graph.add_child(sep, NodeData::Cylinder(CylinderNode::default())),
            };
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
            width: cols as f32 * spacing + 5.0,
            height: 0.2,
            depth: cols as f32 * spacing + 5.0,
        }),
    );

    graph
}
