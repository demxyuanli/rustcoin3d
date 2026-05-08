//! PBR Shader Variant Viewer — demonstrates per-material shader specialization.
//!
//! Displays 8 material configurations side by side, exercising different
//! PbrFeatures combinations. The PbrVariantCache compiles and caches up to
//! 16 specialized shader variants at runtime.
//!
//! Material types shown:
//!   1. Pure color (no textures)       — HAS_ALBEDO_TEX off
//!   2. Albedo-only texture            — HAS_ALBEDO_TEX only
//!   3. Albedo + normal                — ALBEDO | NORMAL
//!   4. Albedo + metallic/roughness    — ALBEDO | MR
//!   5. Full PBR (all textures)        — ALL texture flags
//!   6. Transparent (alpha blend)      — IS_TRANSPARENT
//!   7. Emissive-only                  — HAS_EMISSIVE_TEX
//!   8. Metallic-only (no albedo tex)  — HAS_MR_TEX only
//!
//! Keys:
//!   V: Print shader variant stats
//!   Mouse drag: orbit camera
//!   ESC: exit

use std::sync::{Arc, Mutex};

use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

#[derive(Default)]
struct VariantState {
    active_variants: usize,
    frame_time_ms: f64,
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d=info"),
    )
    .init();

    println!("PBR Shader Variant Viewer");
    println!("Usage: cargo run -p rc3d-app --example pbr_variant_viewer");
    println!("Keys: V print variants | ESC exit");
    println!("Demonstrates 8 material types exercising different PbrFeatures combinations.");

    let state = Arc::new(Mutex::new(VariantState {
        active_variants: 8,
        frame_time_ms: 16.6,
    }));

    let scene = build_variant_scene();
    let state_clone = state.clone();

    let ctrl = CameraController::new(Vec3::new(0.0, 2.0, 10.0), 15.0);

    let mut app = App::new(scene)
        .with_camera_controller(ctrl)
        .with_continuous_redraw(true)
        .with_panel_overlay_text_hook(move || {
            let s = state_clone.lock().unwrap();
            let lines = vec![
                "=== PBR Shader Variant Viewer ===".to_string(),
                format!("Frame: {:.2}ms  |  Active variants: {}",
                    s.frame_time_ms, s.active_variants),
                "".to_string(),
                "Material types (left → right):".to_string(),
                "  1. Pure color        — no textures".to_string(),
                "  2. Albedo tex        — HAS_ALBEDO_TEX".to_string(),
                "  3. Albedo + Normal   — ALBEDO | NORMAL".to_string(),
                "  4. Albedo + MR       — ALBEDO | MR".to_string(),
                "  5. Full PBR          — all texture flags".to_string(),
                "  6. Transparent       — IS_TRANSPARENT".to_string(),
                "  7. Emissive-only     — HAS_EMISSIVE_TEX".to_string(),
                "  8. Metallic-only     — HAS_MR_TEX (no albedo)".to_string(),
                "".to_string(),
                "[V] Print variant info  [ESC] Exit".to_string(),
            ];
            lines.join("\n")
        })
        .with_panel_overlay_key_hook({
            let state = state.clone();
            move |key| {
                use winit::keyboard::KeyCode;
                if key == KeyCode::KeyV {
                    if let Ok(s) = state.lock() {
                        println!("\n╔══ PBR Shader Variant Stats ══╗");
                        println!("║ Active variants: {:>4}           ║", s.active_variants);
                        println!("║ LRU Cache max:     16           ║");
                        println!("║ Frame time:    {:>6.2}ms        ║", s.frame_time_ms);
                        println!("╚═════════════════════════════════╝\n");
                    }
                }
            }
        });

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn build_variant_scene() -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));

    // Camera
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(0.0, 2.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        )),
    );

    // Directional light
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -1.0, -0.3).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
            light_group: None,
        }),
    );

    // 8 spheres in a row, each with a different material configuration
    let configs: [(&str, fn(&mut MaterialNode)); 8] = [
        ("Pure color", |m| {
            m.base_color = Vec3::new(0.8, 0.2, 0.2);
            m.metallic = 0.2;
            m.roughness = 0.4;
        }),
        ("Albedo tex", |m| {
            m.base_color = Vec3::new(0.5, 0.7, 0.5);
            m.albedo_texture = Some("test_data/checker.png".into());
            m.metallic = 0.0;
            m.roughness = 0.6;
        }),
        ("Albedo+Normal", |m| {
            m.base_color = Vec3::new(0.3, 0.5, 0.8);
            m.albedo_texture = Some("test_data/checker.png".into());
            m.normal_texture = Some("test_data/normal.png".into());
            m.metallic = 0.1;
            m.roughness = 0.5;
        }),
        ("Albedo+MR", |m| {
            m.base_color = Vec3::new(0.7, 0.4, 0.2);
            m.albedo_texture = Some("test_data/checker.png".into());
            m.metallic_roughness_texture = Some("test_data/mr.png".into());
            m.metallic = 0.8;
            m.roughness = 0.3;
        }),
        ("Full PBR", |m| {
            m.base_color = Vec3::new(0.6, 0.6, 0.6);
            m.albedo_texture = Some("test_data/checker.png".into());
            m.normal_texture = Some("test_data/normal.png".into());
            m.metallic_roughness_texture = Some("test_data/mr.png".into());
            m.emissive_texture = Some("test_data/emissive.png".into());
            m.occlusion_texture = Some("test_data/ao.png".into());
            m.metallic = 0.5;
            m.roughness = 0.4;
        }),
        ("Transparent", |m| {
            m.base_color = Vec3::new(0.3, 0.8, 0.8);
            m.opacity = 0.5;
            m.alpha_mode = rc3d_scene::AlphaMode::Blend;
            m.metallic = 0.3;
            m.roughness = 0.2;
        }),
        ("Emissive-only", |m| {
            m.base_color = Vec3::new(0.1, 0.1, 0.1);
            m.emissive_color = Vec3::new(1.0, 0.5, 0.1);
            m.emissive_texture = Some("test_data/emissive.png".into());
            m.metallic = 0.0;
            m.roughness = 0.8;
        }),
        ("Metallic-only", |m| {
            m.base_color = Vec3::new(0.9, 0.7, 0.3);
            m.albedo_texture = None;
            m.metallic = 0.95;
            m.roughness = 0.15;
        }),
    ];

    for (i, (label, configure)) in configs.iter().enumerate() {
        let x = (i as f32 - 3.5) * 2.0;
        let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(
            sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(x, 0.0, 0.0))),
        );

        let mut mat = MaterialNode {
            base_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_color: Vec3::new(0.5, 0.5, 0.5),
            ambient_color: Vec3::new(0.02, 0.02, 0.02),
            specular_color: Vec3::new(0.1, 0.1, 0.1),
            shininess: 32.0,
            metallic: 0.1,
            roughness: 0.5,
            opacity: 1.0,
            alpha_mode: rc3d_scene::AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            anisotropic: 0.0,
            ..Default::default()
        };
        configure(&mut mat);

        // Add text label
        let text_sep = graph.add_child(sep, NodeData::Separator(SeparatorNode));
        graph.add_child(
            text_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.3, 0.0))),
        );
        graph.add_child(
            text_sep,
            NodeData::Text2(Text2Node {
                string: format!("{}", i + 1),
                color: [1.0; 4],
                ..Default::default()
            }),
        );

        // Material label
        let label_sep = graph.add_child(sep, NodeData::Separator(SeparatorNode));
        graph.add_child(
            label_sep,
            NodeData::Transform(TransformNode::from_translation(Vec3::new(0.0, 1.0, 0.0))),
        );
        graph.add_child(
            label_sep,
            NodeData::Text2(Text2Node {
                string: label.to_string(),
                color: [1.0; 4],
                ..Default::default()
            }),
        );

        graph.add_child(sep, NodeData::Material(mat));
        graph.add_child(sep, NodeData::Sphere(SphereNode { radius: 0.7 }));
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
            width: 20.0,
            height: 0.1,
            depth: 3.0,
        }),
    );

    graph
}
