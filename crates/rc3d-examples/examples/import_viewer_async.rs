//! Async import viewer — loads a model on a background thread and applies it
//! once ready.
//!
//! Usage: cargo run -p rc3d-examples --example import_viewer_async -- <file.stl|file.obj|file.iv>

use std::env;
use std::path::Path;
use std::sync::mpsc;
use std::thread;

use rc3d_actions::{fit_camera_to_scene, CameraFitConfig};
use rc3d_engine_api::{CameraController, Engine};
use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_core::NodeId;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowAttributes;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();
    print_import_viewer_async_help();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: import_viewer_async <file.stl|file.obj|file.iv>"
        );
        return;
    }
    let path = args[1].clone();
    let path_buf = Path::new(&path).to_path_buf();
    let high_contrast = path_buf
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("stl"))
        .unwrap_or(false);

    // Show STEP statistics for .step/.stp files
    if ext.eq_ignore_ascii_case("step") || ext.eq_ignore_ascii_case("stp") {
        if let Ok(text) = std::fs::read_to_string(&path_buf) {
            if let Ok(exchange) = rc3d_io::step::parser::parse_exchange(&text) {
                let report = rc3d_io::validate_step(&exchange.entities);
                println!(
                    "[STEP] {} entities | {} shells | {} faces | {} points",
                    report.entity_count,
                    report.topology_info.shells,
                    report.topology_info.faces,
                    report.topology_info.points,
                );
            }
        }
    }

    let (tx, rx) = mpsc::channel::<rc3d_core::EngineResult<SceneGraph>>();
    thread::spawn(move || {
        let r = rc3d_io::import_file(&path_buf)
            .map(|g| ensure_camera_and_light(g, high_contrast))
            .map_err(|e| rc3d_core::EngineError::Parse(e.to_string()));
        let _ = tx.send(r);
    });

    println!("Loading in background: {}", path);

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = event_loop
        .create_window(
            WindowAttributes::default().with_title("Async Import Viewer"),
        )
        .expect("failed to create window");

    let mut engine = Engine::new(&window);
    engine.set_display_mode(DisplayMode::Shaded);
    let load_rx = Some(rx);
    let mut graph_loaded = false;

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::RedrawRequested => {
                // Check for async load completion
                if let Some(ref rx) = load_rx {
                    if let Ok(result) = rx.try_recv() {
                        match result {
                            Ok(mut graph) => {
                                let (target, orbit_radius) =
                                    fit_camera_to_scene(
                                        &mut graph,
                                        CameraFitConfig::default(),
                                    );
                                engine.load_scene(graph);
                                engine.controller = CameraController::new(
                                    target,
                                    orbit_radius,
                                );
                                println!(
                                    "Scene loaded: camera at {:?}, orbit={:.1}",
                                    target, orbit_radius
                                );
                            }
                            Err(e) => {
                                eprintln!("Async load error: {e}");
                            }
                        }
                        graph_loaded = true;
                    }
                }
                engine.render();
                window.request_redraw();
            }
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(size) => {
                engine.resize(size.width, size.height);
            }
            _ => {}
        },
        Event::AboutToWait => {
            // Keep polling until graph is loaded, then stop continuous redraw
            if !graph_loaded {
                window.request_redraw();
            }
        }
        _ => {}
    });
}

fn print_import_viewer_async_help() {
    println!("Async import viewer example");
    println!("Usage: cargo run -p rc3d-examples --example import_viewer_async -- <file.stl|file.obj|file.iv>");
    println!("Controls:");
    println!("  Mouse drag: orbit camera after scene is loaded");
    println!("  Escape: clear selection | Close window to quit");
    println!("Feature switches:");
    println!("  Background loading + deferred graph apply");
    println!("  STL input enables high-contrast light preset by default");
}

fn find_geometry_root(graph: &SceneGraph) -> NodeId {
    let roots = graph.roots();
    for &root in roots {
        if let Some(entry) = graph.get(root) {
            if matches!(entry.data, NodeData::Separator(_))
                && has_geometry_recursive(graph, root)
            {
                return root;
            }
        }
    }
    roots[0]
}

fn has_geometry_recursive(graph: &SceneGraph, node: NodeId) -> bool {
    let Some(entry) = graph.get(node) else {
        return false;
    };
    match &entry.data {
        NodeData::Coordinate3(_)
        | NodeData::TextureCoordinate2(_)
        | NodeData::IndexedFaceSet(_)
        | NodeData::Cube(_)
        | NodeData::Sphere(_)
        | NodeData::Cone(_)
        | NodeData::Cylinder(_)
        | NodeData::Triangle(_) => return true,
        _ => {}
    }
    for &child in &entry.children {
        if has_geometry_recursive(graph, child) {
            return true;
        }
    }
    false
}

fn has_camera_recursive(graph: &SceneGraph, node: NodeId) -> bool {
    let Some(entry) = graph.get(node) else {
        return false;
    };
    if matches!(
        entry.data,
        NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_)
    ) {
        return true;
    }
    for &child in &entry.children {
        if has_camera_recursive(graph, child) {
            return true;
        }
    }
    false
}

fn has_material_recursive(graph: &SceneGraph, node: NodeId) -> bool {
    let Some(entry) = graph.get(node) else {
        return false;
    };
    if matches!(entry.data, NodeData::Material(_)) {
        return true;
    }
    for &child in &entry.children {
        if has_material_recursive(graph, child) {
            return true;
        }
    }
    false
}

fn ensure_camera_and_light(
    mut graph: SceneGraph,
    high_contrast: bool,
) -> SceneGraph {
    let has_camera = graph
        .roots()
        .iter()
        .any(|&root| has_camera_recursive(&graph, root));

    if !has_camera {
        let target_root = find_geometry_root(&graph);
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
                intensity: 1.0,
                light_group: None,
            }),
        );
        graph.insert_child(
            target_root,
            2,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(1.0, -0.6, 0.8).normalize(),
                color: Vec3::new(0.9, 0.92, 1.0),
                intensity: 0.5,
                light_group: None,
            }),
        );
        graph.insert_child(
            target_root,
            3,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(0.3, 0.8, 1.0).normalize(),
                color: Vec3::new(0.85, 0.88, 0.95),
                intensity: 0.4,
                light_group: None,
            }),
        );
    }

    let has_material = graph
        .roots()
        .iter()
        .any(|&root| has_material_recursive(&graph, root));
    if !has_material {
        let target_root = find_geometry_root(&graph);
        let cam_count = if has_camera { 0 } else { 4 };
        graph.insert_child(
            target_root,
            cam_count,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(
                0.7, 0.7, 0.7,
            ))),
        );
    }

    if high_contrast {
        apply_high_contrast_mode(&mut graph);
        log::info!("High-contrast import mode enabled (async)");
    }

    graph
}

fn apply_high_contrast_mode(graph: &mut SceneGraph) {
    for &root in graph.roots().to_vec().iter() {
        boost_contrast_recursive(graph, root);
    }
}

fn boost_contrast_recursive(graph: &mut SceneGraph, node: NodeId) {
    let children = graph.children(node).unwrap_or(&[]).to_vec();
    if let Some(entry) = graph.get_mut(node) {
        match &mut entry.data {
            NodeData::DirectionalLight(light) => {
                light.intensity = light.intensity.max(2.2);
                light.color = Vec3::ONE;
            }
            NodeData::Material(mat) => {
                mat.diffuse_color = mat.diffuse_color.max(Vec3::splat(0.75));
                mat.base_color = mat.base_color.max(Vec3::splat(0.75));
                mat.ambient_color = mat.ambient_color.max(Vec3::splat(0.4));
                mat.specular_color = mat.specular_color.max(Vec3::splat(0.6));
                mat.shininess = mat.shininess.max(48.0);
                mat.roughness = mat.roughness.min(0.65);
            }
            _ => {}
        }
    }
    for child in children {
        boost_contrast_recursive(graph, child);
    }
}
