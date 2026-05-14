//! STL rendering diagnostic — stages features one-by-one to isolate issues.
//!
//! Usage: stl_diagnostic <file.stl> [--stage=1..6]
//!
//! Stages (toggle with keys 1-6) + Tiers (keys 7 8 9 0):
//!   1: Flat (Tier 0 DesignCreation) — no PBR/HDR
//!   2: Shaded (Tier 1 Visualization) — PBR, no HDR
//!   3: Shaded+HDR (Tier 2 IndustrialDisplay) — PBR+HDR
//!   4: ShadedWithEdges+HDR (Tier 2 IndustrialDisplay) — +edges
//!   5: FlatWithEdge+HDR (Tier 2 IndustrialDisplay) — +edges+flat
//!   6: Full (Tier 3 ProductRendering) — all passes
//!   7-0: Direct tier override (7=Tier0, 8=Tier1, 9=Tier2, 0=Tier3)
//!
//! Observe degradation: orbit camera during Tier 2/3 — HUD shows tier drop.
//! Press Space to dump diagnostics to console

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
        eprintln!("Usage: stl_diagnostic <file.stl> [--stage=1..6]");
        return;
    };
    let initial_stage: u32 = args
        .iter()
        .find_map(|a| a.strip_prefix("--stage="))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
        .clamp(1, 6);

    let path = Path::new(path_arg);
    let t0 = Instant::now();
    let graph = match rc3d_io::import_file(path) {
        Ok(g) => {
            let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let node_count = count_nodes(&g);
            println!(
                "[DIAG] Loaded {} in {:.0}ms ({} scene nodes)",
                path.display(),
                load_ms,
                node_count
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

    println!("[DIAG] Starting at stage {initial_stage}");
    print_stage_help();

    let (display_mode, hdr) = stage_config(initial_stage);
    let stage = Arc::new(AtomicU32::new(initial_stage));
    let tier_override = Arc::new(AtomicU32::new(u32::MAX)); // MAX = no override
    let stage_for_text = stage.clone();
    let stage_for_key = stage.clone();
    let stage_for_render = stage.clone();
    let tier_for_text = tier_override.clone();
    let tier_for_render = tier_override.clone();
    let tier_for_key = tier_override.clone();

    let mut app = App::new(graph)
        .with_camera_controller(ctrl)
        .with_initial_display_mode(display_mode)
        .with_hdr_post_processing(hdr)
        .with_continuous_redraw(true)
        .with_pre_render_hook(move |renderer| {
            let s = stage_for_render.load(Ordering::Relaxed);
            let to = tier_for_render.load(Ordering::Relaxed);
            let tier = if to != u32::MAX {
                CadDisplayTier::from_u32(to)
            } else {
                tier_for_stage(s)
            };
            let (mode, hdr) = stage_config(s);
            renderer.set_display_mode(mode);
            renderer.set_hdr_post_processing(hdr);
            renderer.set_display_tier(tier);
        })
        .with_panel_overlay_text_hook(move || {
            let s = stage_for_text.load(Ordering::Relaxed);
            let to = tier_for_text.load(Ordering::Relaxed);
            let tier = if to != u32::MAX {
                CadDisplayTier::from_u32(to)
            } else {
                tier_for_stage(s)
            };
            let (mode, hdr) = stage_config(s);
            format!(
                "+- STL Diagnostic {mode:?} HDR:{hdr} -+\n\
                 | Tier: {:<16}              |\n\
                 | Keys 1-6: stage 7-0: tier       |\n\
                 | Orbit camera → observe degrade  |\n\
                 +--------------------------------+",
                tier_name(tier)
            )
        })
        .with_panel_overlay_key_hook(move |key| -> bool {
            use winit::keyboard::KeyCode;
            match key {
                KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3
                | KeyCode::Digit4 | KeyCode::Digit5 | KeyCode::Digit6 => {
                    let s = match key {
                        KeyCode::Digit1 => 1, KeyCode::Digit2 => 2, KeyCode::Digit3 => 3,
                        KeyCode::Digit4 => 4, KeyCode::Digit5 => 5, KeyCode::Digit6 => 6,
                        _ => unreachable!(),
                    };
                    let prev = stage_for_key.swap(s, Ordering::Relaxed);
                    if prev != s {
                        tier_for_key.store(u32::MAX, Ordering::Relaxed);
                        let (mode, hdr) = stage_config(s);
                        let tier = tier_for_stage(s);
                        println!("[DIAG] Stage {s}: {mode:?} HDR={hdr} Tier={tier:?}");
                    }
                    true
                }
                KeyCode::Digit7 | KeyCode::Digit8 | KeyCode::Digit9 | KeyCode::Digit0 => {
                    let ti: u32 = match key {
                        KeyCode::Digit7 => 0, KeyCode::Digit8 => 1,
                        KeyCode::Digit9 => 2, KeyCode::Digit0 => 3,
                        _ => unreachable!(),
                    };
                    let tier = CadDisplayTier::from_u32(ti);
                    tier_for_key.store(ti, Ordering::Relaxed);
                    println!("[DIAG] Tier override: {tier:?}");
                    true
                }
                _ => false,
            }
        });

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn stage_config(stage: u32) -> (DisplayMode, bool) {
    match stage {
        1 => (DisplayMode::Flat, false),
        2 => (DisplayMode::Shaded, false),
        3 => (DisplayMode::Shaded, true),
        4 => (DisplayMode::ShadedWithEdges, true),
        5 => (DisplayMode::FlatWithEdge, true),
        _ => (DisplayMode::FlatWithEdge, true),
    }
}

fn tier_for_stage(stage: u32) -> CadDisplayTier {
    match stage {
        1 => CadDisplayTier::DesignCreation,
        2 => CadDisplayTier::Visualization,
        3|4|5 => CadDisplayTier::IndustrialDisplay,
        _ => CadDisplayTier::ProductRendering,
    }
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
            target_root,
            1,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
                color: Vec3::ONE,
                intensity: 2.2,
                light_group: None,
            }),
        );
        graph.insert_child(
            target_root,
            2,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(1.0, -0.6, 0.8).normalize(),
                color: Vec3::new(0.9, 0.92, 1.0),
                intensity: 1.2,
                light_group: None,
            }),
        );
        graph.insert_child(
            target_root,
            3,
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
            target_root,
            insert_idx,
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
    let Some(entry) = graph.get(node) else {
        return false;
    };
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

fn count_nodes(graph: &rc3d_scene::SceneGraph) -> usize {
    let mut count = 0;
    for &root in graph.roots() {
        count += count_nodes_recursive(graph, root);
    }
    count
}

fn count_nodes_recursive(graph: &rc3d_scene::SceneGraph, node: rc3d_core::NodeId) -> usize {
    let Some(entry) = graph.get(node) else {
        return 0;
    };
    let mut n = 1;
    for &child in &entry.children {
        n += count_nodes_recursive(graph, child);
    }
    n
}

fn print_stage_help() {
    println!("Keys 1-6: switch rendering stage");
    println!("  1: Flat color (geometry test)");
    println!("  2: PBR Shaded (no HDR)");
    println!("  3: PBR Shaded + HDR");
    println!("  4: ShadedWithEdges + HDR");
    println!("  5: Full quality");
    println!("  6: FlatWithEdge + HDR");
}
