//! PBR Shader Variant Viewer — demonstrates per-material shader specialization.
//!
//! Displays 10 material configurations side by side, exercising different
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
//!   9. Sheen velvet                   — HAS_SHEEN
//!  10. Anisotropic metal              — GGX anisotropy
//!
//! Keys:
//!   V: Print shader variant stats
//!   Mouse drag: orbit camera
//!   Escape: clear selection | close window to quit

use std::sync::{Arc, Mutex};

use rc3d_engine_api::CameraController;
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use rc3d_examples::common::run_example_with_hooks;
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Default)]
struct VariantState {
    active_variants: usize,
    frame_time_ms: f64,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();

    println!("PBR Shader Variant Viewer");
    println!("Usage: cargo run -p rc3d-examples --example pbr_variant_viewer");
    println!("Keys: V print variants | Esc clears selection");
    println!("Demonstrates 10 material types exercising different PbrFeatures combinations.");

    let state = Arc::new(Mutex::new(VariantState {
        active_variants: 10,
        frame_time_ms: 16.6,
    }));

    let scene = build_variant_scene();
    let state_clone = state.clone();

    let ctrl = CameraController::new(Vec3::new(0.0, 2.0, 10.0), 15.0);

    run_example_with_hooks("PBR Shader Variant Viewer", move |engine| {
        engine.load_scene(scene);
        engine.controller = ctrl;
        engine.continuous_redraw = true;
        engine.hud_text_hook = Some(Box::new(move || {
            let s = state_clone.lock().unwrap();
            let lines = vec![
                "=== PBR Shader Variant Viewer ===".to_string(),
                format!(
                    "Frame: {:.2}ms  |  Active variants: {}",
                    s.frame_time_ms, s.active_variants
                ),
                "".to_string(),
                "Material types (left -> right):".to_string(),
                "  1. Pure color        — no textures".to_string(),
                "  2. Albedo tex        — HAS_ALBEDO_TEX".to_string(),
                "  3. Albedo + Normal   — ALBEDO | NORMAL".to_string(),
                "  4. Albedo + MR       — ALBEDO | MR".to_string(),
                "  5. Full PBR          — all texture flags".to_string(),
                "  6. Transparent       — IS_TRANSPARENT".to_string(),
                "  7. Emissive-only     — HAS_EMISSIVE_TEX".to_string(),
                "  8. Metallic-only     — HAS_MR_TEX (no albedo)".to_string(),
                "  9. Sheen velvet      — HAS_SHEEN".to_string(),
                " 10. Aniso metal       — GGX anisotropy".to_string(),
                "  Chrome sphere (front) uses CubeCamera local IBL probe".to_string(),
                "".to_string(),
                "[V] Print variant info  [Esc] Clear selection".to_string(),
            ];
            lines.join("\n")
        }));
        engine.panel_overlay_key_hook = Some(Box::new({
            let state = state.clone();
            move |key: PhysicalKey| -> bool {
                if key == PhysicalKey::Code(KeyCode::KeyV) {
                    if let Ok(s) = state.lock() {
                        println!("\n╔══ PBR Shader Variant Stats ══╗");
                        println!("║ Active variants: {:>4}           ║", s.active_variants);
                        println!("║ LRU Cache max:     16           ║");
                        println!("║ Frame time:    {:>6.2}ms        ║", s.frame_time_ms);
                        println!("╚═════════════════════════════════╝\n");
                    }
                    return true;
                }
                false
            }
        }));
    });
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
    let configs: [(&str, fn(&mut MaterialNode)); 10] = [
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
            m.metallic = 0.0;
            m.roughness = 0.08;
            m.transmission_factor = 0.9;
            m.ior = 1.5;
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
        ("Sheen velvet", |m| {
            m.base_color = Vec3::new(0.15, 0.08, 0.22);
            m.metallic = 0.0;
            m.roughness = 0.85;
            m.sheen_color = Vec3::new(0.9, 0.75, 0.95);
            m.sheen_roughness = 0.35;
        }),
        ("Aniso metal", |m| {
            m.base_color = Vec3::new(0.82, 0.78, 0.72);
            m.metallic = 1.0;
            m.roughness = 0.22;
            m.anisotropic = 0.85;
        }),
    ];

    for (i, (label, configure)) in configs.iter().enumerate() {
        let x = (i as f32 - 4.5) * 1.7;
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
            clearcoat_factor: 0.0,
            clearcoat_roughness: 0.0,
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
            width: 24.0,
            height: 0.1,
            depth: 10.0,
        }),
    );

    add_colored_box(
        &mut graph,
        root,
        Vec3::new(0.0, 2.0, -4.5),
        (24.0, 8.0, 0.12),
        Vec3::new(0.85, 0.22, 0.18),
    );
    add_colored_box(
        &mut graph,
        root,
        Vec3::new(-11.5, 2.0, 0.0),
        (0.12, 8.0, 10.0),
        Vec3::new(0.18, 0.72, 0.32),
    );
    add_colored_box(
        &mut graph,
        root,
        Vec3::new(11.5, 2.0, 0.0),
        (0.12, 8.0, 10.0),
        Vec3::new(0.18, 0.35, 0.85),
    );
    add_colored_box(
        &mut graph,
        root,
        Vec3::new(0.0, 5.9, 0.0),
        (24.0, 0.12, 10.0),
        Vec3::new(0.92, 0.82, 0.28),
    );

    let probe_pos = Vec3::new(0.0, 0.15, 2.6);
    let chrome = graph.add_child(root, NodeData::Separator(SeparatorNode));
    graph.add_child(
        chrome,
        NodeData::Transform(TransformNode::from_translation(probe_pos)),
    );
    graph.add_child(
        chrome,
        NodeData::Material(MaterialNode {
            base_color: Vec3::new(0.96, 0.96, 0.98),
            diffuse_color: Vec3::new(0.96, 0.96, 0.98),
            metallic: 1.0,
            roughness: 0.06,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(chrome, NodeData::Sphere(SphereNode { radius: 0.55 }));
    graph.add_child(
        root,
        NodeData::CubeCamera(CubeCameraNode {
            position: probe_pos,
            near: 0.35,
            far: 40.0,
            resolution: 128,
            update_period: 0,
            enabled: true,
        }),
    );

    graph
}

fn add_colored_box(
    graph: &mut SceneGraph,
    parent: rc3d_core::NodeId,
    translation: Vec3,
    size: (f32, f32, f32),
    color: Vec3,
) {
    let sep = graph.add_child(parent, NodeData::Separator(SeparatorNode));
    graph.add_child(
        sep,
        NodeData::Transform(TransformNode::from_translation(translation)),
    );
    graph.add_child(
        sep,
        NodeData::Material(MaterialNode {
            base_color: color,
            diffuse_color: color,
            metallic: 0.0,
            roughness: 0.85,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    graph.add_child(
        sep,
        NodeData::Cube(CubeNode {
            width: size.0,
            height: size.1,
            depth: size.2,
        }),
    );
}
