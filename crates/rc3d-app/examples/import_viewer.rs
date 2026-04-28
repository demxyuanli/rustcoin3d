use std::env;
use std::path::Path;

use rc3d_actions::GetBoundingBoxAction;
use rc3d_app::App;
use rc3d_app::camera_controller::CameraController;
use rc3d_core::{math::{Mat4, Vec3}, DisplayMode};
use rc3d_core::NodeId;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d=info"),
    ).init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: import_viewer <file.stl|file.obj|file.iv> [--high-contrast=on|off]");
        return;
    }

    let path = Path::new(&args[1]);
    let high_contrast = parse_high_contrast(&args, path);
    let graph = match rc3d_io::import_file(path) {
        Ok(g) => {
            println!("Loaded: {}", path.display());
            g
        }
        Err(e) => {
            eprintln!("Import error: {e}");
            return;
        }
    };

    let mut graph = ensure_camera_and_light(graph, high_contrast);
    let (target, orbit_radius) = fit_camera_to_scene(&mut graph);
    let controller_root = find_first_camera_node(&graph).unwrap_or(graph.roots()[0]);
    let ctrl = CameraController::new(controller_root, target, orbit_radius);
    let mut app = App::new(graph)
        .with_camera_controller(ctrl)
        .with_initial_display_mode(DisplayMode::ShadedWithEdges);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn parse_high_contrast(args: &[String], path: &Path) -> bool {
    for arg in args.iter().skip(2) {
        if let Some(value) = arg.strip_prefix("--high-contrast=") {
            return matches!(value, "on" | "true" | "1");
        }
    }
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("stl"))
        .unwrap_or(false)
}

/// Find the first Separator root that contains geometry, or the first root.
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
    let Some(entry) = graph.get(node) else { return false };
    match &entry.data {
        NodeData::Coordinate3(_)
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

fn ensure_camera_and_light(mut graph: rc3d_scene::SceneGraph, high_contrast: bool) -> rc3d_scene::SceneGraph {
    let has_camera = graph.roots().iter().any(|&root| {
        has_camera_recursive(&graph, root)
    });

    if !has_camera {
        // Insert camera + light at index 0 so they are visited BEFORE geometry.
        // RenderCollector processes children in order; VP matrix must be set first.
        let target_root = find_geometry_root(&graph);
        graph.insert_child(
            target_root, 0,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(5.0, 5.0, 8.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );
        // Main key light: upper-left-front
        graph.insert_child(
            target_root, 1,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
                color: Vec3::ONE,
                intensity: 1.0,
            }),
        );
        // Fill light: upper-right, softer intensity to brighten shadows
        graph.insert_child(
            target_root, 2,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(1.0, -0.6, 0.8).normalize(),
                color: Vec3::new(0.9, 0.92, 1.0),
                intensity: 0.5,
            }),
        );
    }

    let has_material = graph.roots().iter().any(|&root| {
        has_material_recursive(&graph, root)
    });
    if !has_material {
        let target_root = find_geometry_root(&graph);
        let cam_count = if has_camera { 0 } else { 3 };
        graph.insert_child(
            target_root, cam_count,
            NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.7, 0.7, 0.7))),
        );
    }

    if high_contrast {
        apply_high_contrast_mode(&mut graph);
        log::info!("High-contrast import mode enabled");
    }

    graph
}

fn apply_high_contrast_mode(graph: &mut rc3d_scene::SceneGraph) {
    if !has_directional_light(graph) {
        let target_root = find_geometry_root(graph);
        graph.insert_child(
            target_root, 0,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -0.8, -0.6).normalize(),
                color: Vec3::ONE,
                intensity: 2.4,
            }),
        );
        graph.insert_child(
            target_root, 1,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(1.0, -0.5, 0.7).normalize(),
                color: Vec3::new(0.9, 0.92, 1.0),
                intensity: 1.0,
            }),
        );
    }
    for &root in graph.roots().to_vec().iter() {
        boost_contrast_recursive(graph, root);
    }
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
    let Some(entry) = graph.get(node) else { return false };
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

fn boost_contrast_recursive(graph: &mut rc3d_scene::SceneGraph, node: NodeId) {
    let children = graph.children(node).to_vec();
    if let Some(entry) = graph.get_mut(node) {
        match &mut entry.data {
            NodeData::DirectionalLight(light) => {
                light.intensity = light.intensity.max(2.2);
                light.color = Vec3::ONE;
            }
            NodeData::Material(mat) => {
                mat.diffuse_color = mat.diffuse_color.max(Vec3::splat(0.75));
                mat.ambient_color = mat.ambient_color.max(Vec3::splat(0.25));
                mat.specular_color = mat.specular_color.max(Vec3::splat(0.6));
                mat.shininess = mat.shininess.max(48.0);
            }
            _ => {}
        }
    }
    for child in children {
        boost_contrast_recursive(graph, child);
    }
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

fn has_camera_recursive(graph: &rc3d_scene::SceneGraph, node: rc3d_core::NodeId) -> bool {
    let Some(entry) = graph.get(node) else { return false };
    match &entry.data {
        NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_) => return true,
        _ => {}
    }
    for &child in &entry.children {
        if has_camera_recursive(graph, child) {
            return true;
        }
    }
    false
}

fn has_material_recursive(graph: &rc3d_scene::SceneGraph, node: rc3d_core::NodeId) -> bool {
    let Some(entry) = graph.get(node) else { return false };
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
