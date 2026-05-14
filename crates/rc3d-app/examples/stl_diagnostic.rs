//! STL rendering diagnostic with CAD display tier testing.
//!
//! Usage: stl_diagnostic <file.stl> [--tier=0..3]
//!
//! Keys:
//!   1/2/3/4 — switch CAD tier (0=DesignCreation, 1=Visualization,
//!             2=IndustrialDisplay, 3=ProductRendering)
//!   Ctrl+1 — Flat         (display mode override)
//!   Ctrl+2 — Shaded       (display mode override)
//!   Ctrl+3 — Shaded+HDR   (display mode override)
//!   Ctrl+4 — ShadedWithEdges+HDR (display mode override)
//!
//! Observe degradation: orbit camera during Tier 2/3 — HUD shows tier drop.
//! Press Space to dump diagnostics to console.

use std::env;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rc3d_actions::{fit_camera_to_scene, CameraFitConfig};
use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_render::renderer::CadDisplayTier;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = env::args().collect();
    let Some(path_arg) = args.iter().skip(1).find(|a| !a.starts_with("--")) else {
        eprintln!("Usage: stl_diagnostic <file.stl> [--tier=0..3]");
        return;
    };
    let initial_tier: u32 = args
        .iter()
        .find_map(|a| a.strip_prefix("--tier="))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .clamp(0, 3);

    let path = Path::new(path_arg);
    let t0 = Instant::now();
    let graph = match rc3d_io::import_file(path) {
        Ok(g) => {
            let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let node_count = count_nodes(&g);
            println!(
                "[DIAG] Loaded {} in {:.0}ms ({} scene nodes)",
                path.display(), load_ms, node_count
            );
            g
        }
        Err(e) => {
            eprintln!("Import error: {e}");
            return;
        }
    };

    let mut graph = setup_scene(graph);
    let (target, orbit_radius) = fit_camera_to_scene(&mut graph, CameraFitConfig::default());
    let ctrl = CameraController::new(target, orbit_radius);

    let tier_atom = Arc::new(AtomicU32::new(initial_tier));
    let tier_text = tier_atom.clone();
    let tier_render = tier_atom.clone();
    let tier_key = tier_atom.clone();

    println!("[DIAG] Starting at tier {}", initial_tier);
    println!("Keys 1-4: switch tier | Ctrl+1-4: display mode");
    println!("Orbit camera to observe tier degradation during interaction");

    let mut app = App::new(graph)
        .with_camera_controller(ctrl)
        .with_initial_display_mode(DisplayMode::Shaded)
        .with_hdr_post_processing(true)
        .with_continuous_redraw(true)
        .with_pre_render_hook(move |renderer| {
            let t = tier_render.load(Ordering::Relaxed);
            renderer.set_display_tier(CadDisplayTier::from_u32(t));
        })
        .with_panel_overlay_text_hook(move || {
            let t = tier_text.load(Ordering::Relaxed);
            let tier = CadDisplayTier::from_u32(t);
            format!(
                "+--- CAD Tier: {:<16} ---+\n\
                 | 1=DesignCreation 2=Visualization |\n\
                 | 3=IndustrialDisplay 4=ProductRen |\n\
                 | Orbit camera → observe degrade   |\n\
                 +-----------------------------------+",
                tier_name(tier)
            )
        })
        .with_panel_overlay_key_hook(move |key| -> bool {
            use winit::keyboard::KeyCode;
            let ti: u32 = match key {
                KeyCode::Digit1 => 0,
                KeyCode::Digit2 => 1,
                KeyCode::Digit3 => 2,
                KeyCode::Digit4 => 3,
                _ => return false,
            };
            tier_key.store(ti, Ordering::Relaxed);
            let tier = CadDisplayTier::from_u32(ti);
            println!("[DIAG] Tier: {tier:?}");
            true
        });

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn tier_name(t: CadDisplayTier) -> &'static str {
    match t {
        CadDisplayTier::DesignCreation => "DesignCreation",
        CadDisplayTier::Visualization => "Visualization",
        CadDisplayTier::IndustrialDisplay => "IndustrialDisplay",
        CadDisplayTier::ProductRendering => "ProductRendering",
    }
}

fn setup_scene(mut graph: rc3d_scene::SceneGraph) -> rc3d_scene::SceneGraph {
    let target_root = find_geometry_root(&graph);

    let has_camera = has_node_type_recursive(&graph, target_root, |d| {
        matches!(
            d,
            rc3d_scene::NodeData::PerspectiveCamera(_)
                | rc3d_scene::NodeData::OrthographicCamera(_)
        )
    });
    if !has_camera {
        graph.insert_child(
            target_root,
            0,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(5.0, 5.0, 8.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );
        graph.insert_child(
            target_root, 1,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
                color: Vec3::ONE,
                intensity: 2.2,
                light_group: None,
            }),
        );
        graph.insert_child(
            target_root, 2,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(1.0, -0.6, 0.8).normalize(),
                color: Vec3::new(0.9, 0.92, 1.0),
                intensity: 1.2,
                light_group: None,
            }),
        );
        graph.insert_child(
            target_root, 3,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(0.3, 0.8, 1.0).normalize(),
                color: Vec3::new(0.85, 0.88, 0.95),
                intensity: 0.8,
                light_group: None,
            }),
        );
    }

    let has_material =
        has_node_type_recursive(&graph, target_root, |d| matches!(d, NodeData::Material(_)));
    if !has_material {
        let insert_idx = if has_camera { 0 } else { 4 };
        graph.insert_child(
            target_root, insert_idx,
            NodeData::Material(MaterialNode {
                diffuse_color: Vec3::splat(0.9),
                ambient_color: Vec3::splat(0.4),
                specular_color: Vec3::splat(0.5),
                shininess: 48.0,
                base_color: Vec3::splat(0.92),
                metallic: 0.0,
                roughness: 0.4,
                opacity: 1.0,
                ..Default::default()
            }),
        );
    }

    for &root in graph.roots().to_vec().iter() {
        boost_materials(&mut graph, root);
    }
    graph
}

fn boost_materials(graph: &mut rc3d_scene::SceneGraph, node: rc3d_core::NodeId) {
    let children = graph.children(node).unwrap_or(&[]).to_vec();
    if let Some(entry) = graph.get_mut(node) {
        if let NodeData::Material(mat) = &mut entry.data {
            mat.base_color = mat.base_color.max(Vec3::splat(0.75));
            mat.ambient_color = mat.ambient_color.max(Vec3::splat(0.35));
            mat.roughness = mat.roughness.min(0.65);
        }
        if let NodeData::DirectionalLight(light) = &mut entry.data {
            light.intensity = light.intensity.max(2.0);
        }
    }
    for child in children {
        boost_materials(graph, child);
    }
}

fn find_geometry_root(graph: &rc3d_scene::SceneGraph) -> rc3d_core::NodeId {
    for &root in graph.roots() {
        if let Some(entry) = graph.get(root) {
            if matches!(entry.data, NodeData::Separator(_)) {
                return root;
            }
        }
    }
    graph.roots()[0]
}

fn has_node_type_recursive(
    graph: &rc3d_scene::SceneGraph,
    node: rc3d_core::NodeId,
    pred: impl Fn(&rc3d_scene::NodeData) -> bool + Copy,
) -> bool {
    let Some(entry) = graph.get(node) else { return false };
    if pred(&entry.data) { return true; }
    for &child in &entry.children {
        if has_node_type_recursive(graph, child, pred) { return true; }
    }
    false
}

fn count_nodes(graph: &rc3d_scene::SceneGraph) -> usize {
    let mut count = 0;
    for &root in graph.roots() {
        count += count_nodes_recursive(graph, root);
    }
    count
}

fn count_nodes_recursive(graph: &rc3d_scene::SceneGraph, node: rc3d_core::NodeId) -> usize {
    let Some(entry) = graph.get(node) else { return 0 };
    let mut n = 1;
    for &child in &entry.children {
        n += count_nodes_recursive(graph, child);
    }
    n
}
