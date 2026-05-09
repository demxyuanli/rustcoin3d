//! Loads a model on a background thread; main thread uploads GPU resources on redraw.
//! Usage: import_viewer_async <file.stl|file.obj|file.iv>

use std::env;
use std::path::Path;
use std::sync::mpsc;
use std::thread;

use rc3d_actions::{fit_camera_to_scene, CameraFitConfig};
use rc3d_app::camera_controller::CameraController;
use rc3d_app::App;
use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_core::NodeId;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info"))
        .init();
    print_import_viewer_async_help();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: import_viewer_async <file.stl|file.obj|file.iv>");
        return;
    }
    let path = args[1].clone();
    let path_buf = Path::new(&path).to_path_buf();
    let high_contrast = path_buf
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("stl"))
        .unwrap_or(false);

    let (tx, rx) = mpsc::channel::<rc3d_core::EngineResult<SceneGraph>>();
    thread::spawn(move || {
        let r = rc3d_io::import_file(&path_buf)
            .map(|g| ensure_camera_and_light(g, high_contrast))
            .map_err(|e| rc3d_core::EngineError::Parse(e.to_string()));
        let _ = tx.send(r);
    });

    println!("Loading in background: {}", path);

    let mut app =
        App::new(SceneGraph::new()).with_initial_display_mode(DisplayMode::ShadedWithEdges);
    app.set_pending_graph_receiver(rx);
    app.set_graph_load_hook(move |app| {
        let (target, orbit_radius) = fit_camera_to_scene(&mut app.state.world.graph, CameraFitConfig::default());
        let _controller_root = find_first_camera_node(&app.state.world.graph)
            .or_else(|| app.state.world.graph.roots().first().copied())
            .expect("non-empty graph after load");
        app.state.camera_controller = Some(CameraController::new(target, orbit_radius));
    });

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn print_import_viewer_async_help() {
    println!("Async import viewer example");
    println!(
        "Usage: cargo run -p rc3d-app --example import_viewer_async -- <file.stl|file.obj|file.iv>"
    );
    println!("Controls:");
    println!("  Mouse drag: orbit camera after scene is loaded");
    println!("  Escape: clear selection | Close window to quit");
    println!("Feature switches:");
    println!("  Background loading + deferred graph apply");
    println!("  STL input enables high-contrast light preset by default");
}

fn find_geometry_root(graph: &rc3d_scene::SceneGraph) -> NodeId {
    let roots = graph.roots();
    for &root in roots {
        if let Some(entry) = graph.get(root) {
            if matches!(entry.data, NodeData::Separator(_)) && has_geometry_recursive(graph, root) {
                return root;
            }
        }
    }
    roots[0]
}

fn has_geometry_recursive(graph: &rc3d_scene::SceneGraph, node: NodeId) -> bool {
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

fn has_camera_recursive(graph: &rc3d_scene::SceneGraph, node: NodeId) -> bool {
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

fn has_material_recursive(graph: &rc3d_scene::SceneGraph, node: NodeId) -> bool {
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
    mut graph: rc3d_scene::SceneGraph,
    high_contrast: bool,
) -> rc3d_scene::SceneGraph {
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
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.7, 0.7, 0.7))),
        );
    }

    if high_contrast {
        apply_high_contrast_mode(&mut graph);
        log::info!("High-contrast import mode enabled (async)");
    }

    graph
}

fn apply_high_contrast_mode(graph: &mut rc3d_scene::SceneGraph) {
    for &root in graph.roots().to_vec().iter() {
        boost_contrast_recursive(graph, root);
    }
}

fn boost_contrast_recursive(graph: &mut rc3d_scene::SceneGraph, node: NodeId) {
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

fn find_first_camera_node(graph: &rc3d_scene::SceneGraph) -> Option<NodeId> {
    for &root in graph.roots() {
        if let Some(id) = find_camera_recursive(graph, root) {
            return Some(id);
        }
    }
    None
}

fn find_camera_recursive(graph: &rc3d_scene::SceneGraph, node: NodeId) -> Option<NodeId> {
    let entry = graph.get(node)?;
    if matches!(
        entry.data,
        NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_)
    ) {
        return Some(node);
    }
    for &child in &entry.children {
        if let Some(id) = find_camera_recursive(graph, child) {
            return Some(id);
        }
    }
    None
}
