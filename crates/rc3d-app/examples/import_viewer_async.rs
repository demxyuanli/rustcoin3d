//! Loads a model on a background thread; main thread uploads GPU resources on redraw.
//! Usage: import_viewer_async <file.stl|file.obj|file.iv>

use std::env;
use std::path::Path;
use std::sync::mpsc;
use std::thread;

use rc3d_actions::GetBoundingBoxAction;
use rc3d_app::App;
use rc3d_app::camera_controller::CameraController;
use rc3d_core::math::{Mat4, Vec3};
use rc3d_core::DisplayMode;
use rc3d_core::NodeId;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rc3d=info")).init();

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

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let r = rc3d_io::import_file(&path_buf)
            .map(|g| ensure_camera_and_light(g, high_contrast))
            .map_err(|e| e.to_string());
        let _ = tx.send(r);
    });

    println!("Loading in background: {}", path);

    let mut app = App::new(SceneGraph::new())
        .with_initial_display_mode(DisplayMode::ShadedWithEdges);
    app.set_pending_graph_receiver(rx);
    app.set_graph_load_hook(move |app| {
        let (target, orbit_radius) = fit_camera_to_scene(&mut app.graph);
        let controller_root = find_first_camera_node(&app.graph)
            .or_else(|| app.graph.roots().first().copied())
            .expect("non-empty graph after load");
        app.camera_controller = Some(CameraController::new(controller_root, target, orbit_radius));
    });

    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
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

fn ensure_camera_and_light(mut graph: rc3d_scene::SceneGraph, high_contrast: bool) -> rc3d_scene::SceneGraph {
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
            }),
        );
        graph.insert_child(
            target_root,
            2,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(1.0, -0.6, 0.8).normalize(),
                color: Vec3::new(0.9, 0.92, 1.0),
                intensity: 0.5,
            }),
        );
    }

    let has_material = graph
        .roots()
        .iter()
        .any(|&root| has_material_recursive(&graph, root));
    if !has_material {
        let target_root = find_geometry_root(&graph);
        let cam_count = if has_camera { 0 } else { 3 };
        graph.insert_child(
            target_root,
            cam_count,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.7, 0.7, 0.7))),
        );
    }

    if high_contrast && !has_directional_light(&graph) {
        let target_root = find_geometry_root(&graph);
        graph.insert_child(
            target_root,
            0,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -0.8, -0.6).normalize(),
                color: Vec3::ONE,
                intensity: 2.4,
            }),
        );
    }

    graph
}

fn has_directional_light(graph: &rc3d_scene::SceneGraph) -> bool {
    for &root in graph.roots() {
        if has_directional_light_recursive(graph, root) {
            return true;
        }
    }
    false
}

fn has_directional_light_recursive(graph: &rc3d_scene::SceneGraph, node: NodeId) -> bool {
    let Some(entry) = graph.get(node) else {
        return false;
    };
    if matches!(entry.data, NodeData::DirectionalLight(_)) {
        return true;
    }
    for &child in &entry.children {
        if has_directional_light_recursive(graph, child) {
            return true;
        }
    }
    false
}

fn fit_camera_to_scene(graph: &mut rc3d_scene::SceneGraph) -> (Vec3, f32) {
    let mut bbox_action = GetBoundingBoxAction::new();
    for &root in graph.roots() {
        bbox_action.apply(graph, root);
    }
    let bbox = bbox_action.bounding_box;
    if !bbox.min.x.is_finite() || !bbox.max.x.is_finite() {
        return (Vec3::ZERO, 10.0);
    }
    let center = bbox.center();
    let extent = bbox.size();
    let radius = extent.length().max(1.0) * 0.5;
    let eye = center + Vec3::new(radius * 1.5, radius * 1.1, radius * 2.0);
    let near = (radius * 0.001).max(0.01);
    let far = (radius * 20.0).max(100.0);

    for &root in graph.roots().to_vec().iter() {
        apply_camera_fit_recursive(graph, root, eye, center, near, far);
    }
    (center, radius * 2.2)
}

fn apply_camera_fit_recursive(
    graph: &mut rc3d_scene::SceneGraph,
    node: NodeId,
    eye: Vec3,
    target: Vec3,
    near: f32,
    far: f32,
) {
    let children = graph.children(node).to_vec();
    if let Some(entry) = graph.get_mut(node) {
        match &mut entry.data {
            NodeData::PerspectiveCamera(cam) => {
                let fov = cam.fov;
                let aspect = cam.aspect;
                *cam = PerspectiveCameraNode::look_at(eye, target, Vec3::Y, fov, aspect);
                cam.near = near;
                cam.far = far;
            }
            NodeData::OrthographicCamera(cam) => {
                cam.position = eye;
                cam.orientation = Mat4::look_at_rh(eye, target, Vec3::Y);
                cam.near = near;
                cam.far = far;
                cam.height = (target - eye).length().max(1.0);
            }
            _ => {}
        }
    }
    for child in children {
        apply_camera_fit_recursive(graph, child, eye, target, near, far);
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
