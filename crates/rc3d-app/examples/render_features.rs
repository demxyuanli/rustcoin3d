//! Render features demo — toggles post-processing effects via keyboard.
//!
//! Keys:
//!   1-9: Toggle effects (HDR/TAA/MotionBlur/SSR/DOF/Fog/LUT/Shadows/AO)
//!   W/A/S/D/Q/E: Camera movement
//!   Mouse drag: Orbit
//!   F: Cycle display mode (Shaded/ShadedWithEdges/Wireframe/HiddenLine)
//!   L: Cycle IBL preset
//!   R: Reset exposure
//!   +/-: Adjust exposure
//!   V: Toggle vsync
//!   H: Toggle HUD
//!   ESC: Exit
//!
//! Usage: render_features <file.gltf|file.glb|file.obj|file.stl>

use std::env;
use std::path::Path;

use rc3d_app::App;
use rc3d_app::camera_controller::CameraController;
use rc3d_core::{math::Vec3, DisplayMode};
use rc3d_core::NodeId;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d=info"),
    )
    .init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: render_features <file.gltf|file.glb|file.obj|file.stl|file.iv>");
        eprintln!();
        eprintln!("Keyboard controls:");
        eprintln!("  1: Toggle HDR post-processing");
        eprintln!("  2: Toggle TAA (temporal anti-aliasing)");
        eprintln!("  3: Toggle Motion Blur");
        eprintln!("  4: Toggle SSR (screen-space reflections)");
        eprintln!("  5: Toggle Depth of Field");
        eprintln!("  6: Toggle Volumetric Fog");
        eprintln!("  7: Toggle Color Grading LUT");
        eprintln!("  8: Toggle Shadows");
        eprintln!("  9: Toggle SSAO");
        eprintln!("  F: Cycle display mode");
        eprintln!("  L: Cycle IBL preset");
        eprintln!("  +/-: Adjust DOF focus distance");
        eprintln!("  V: Toggle vsync");
        eprintln!("  H: Toggle HUD");
        return;
    }

    let path = Path::new(&args[1]);
    let mut graph = match rc3d_io::import_file(path) {
        Ok(g) => {
            println!("Loaded: {} ({} roots, format auto-detected)", path.display(), g.roots().len());
            g
        }
        Err(e) => {
            eprintln!("Import error: {e}");
            return;
        }
    };

    // Add camera and lights if missing
    graph = ensure_scene_setup(graph);

    // Build camera controller
    let ctrl = CameraController::new(
        find_camera_node(&graph).unwrap_or(graph.roots()[0]),
        Vec3::ZERO,
        5.0,
    );

    let mut app = App::new(graph)
        .with_camera_controller(ctrl)
        .with_initial_display_mode(DisplayMode::ShadedWithEdges)
        .with_hdr_post_processing(true);

    println!("Render features demo ready:");
    println!("  HDR:ON  TAA:ON  MotionBlur:ON  SSR:ON  DOF:ON  Fog:ON  Shadows:ON");
    println!("  Press 1-9 to toggle effects");

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn ensure_scene_setup(mut graph: SceneGraph) -> SceneGraph {
    let has_camera = has_node_type(&graph, |d| {
        matches!(d, NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_))
    });
    let has_light = has_node_type(&graph, |d| {
        matches!(d, NodeData::DirectionalLight(_) | NodeData::PointLight(_) | NodeData::SpotLight(_))
    });

    let root = graph.roots().first().copied().unwrap_or_else(|| {
        graph.add_root(NodeData::Separator(SeparatorNode))
    });

    if !has_camera {
        let cam = PerspectiveCameraNode::look_at(
            Vec3::new(5.0, 4.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
        );
        graph.insert_child(root, 0, NodeData::PerspectiveCamera(cam));
    }

    if !has_light {
        // Key light
        graph.insert_child(root, 1, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-1.0, -0.8, -0.6).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.2,
        }));
        // Fill light
        graph.insert_child(root, 2, NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(0.8, -0.4, 1.0).normalize(),
            color: Vec3::new(0.7, 0.8, 1.0),
            intensity: 0.5,
        }));
        // Point light for dynamic shadow demo
        graph.insert_child(root, 3, NodeData::PointLight(PointLightNode {
            location: Vec3::new(2.0, 3.0, 2.0),
            color: Vec3::new(1.0, 0.6, 0.3),
            intensity: 10.0,
        }));
    }

    // Ensure material
    if !has_node_type(&graph, |d| matches!(d, NodeData::Material(_))) {
        graph.insert_child(root, 0, NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.8, 0.8, 0.8),
            ambient_color: Vec3::new(0.1, 0.1, 0.1),
            specular_color: Vec3::new(0.5, 0.5, 0.5),
            shininess: 32.0,
            base_color: Vec3::new(0.8, 0.8, 0.8),
            metallic: 0.1,
            roughness: 0.5,
            albedo_texture: None,
        }));
    }

    graph
}

fn has_node_type(graph: &SceneGraph, pred: impl Fn(&NodeData) -> bool) -> bool {
    for &root in graph.roots() {
        if has_node_type_recursive(graph, root, &pred) {
            return true;
        }
    }
    false
}

fn has_node_type_recursive(graph: &SceneGraph, node: NodeId, pred: &impl Fn(&NodeData) -> bool) -> bool {
    let Some(entry) = graph.get(node) else { return false };
    if pred(&entry.data) {
        return true;
    }
    for &child in &entry.children {
        if has_node_type_recursive(graph, child, pred) {
            return true;
        }
    }
    false
}

fn find_camera_node(graph: &SceneGraph) -> Option<NodeId> {
    for &root in graph.roots() {
        if let Some(id) = find_camera_recursive(graph, root) {
            return Some(id);
        }
    }
    None
}

fn find_camera_recursive(graph: &SceneGraph, node: NodeId) -> Option<NodeId> {
    let entry = graph.get(node)?;
    if matches!(entry.data, NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_)) {
        return Some(node);
    }
    for &child in &entry.children {
        if let Some(id) = find_camera_recursive(graph, child) {
            return Some(id);
        }
    }
    None
}
