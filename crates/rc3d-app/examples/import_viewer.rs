use std::env;
use std::path::Path;

use rc3d_app::App;
use rc3d_app::camera_controller::CameraController;
use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: import_viewer <file.stl|file.obj|file.iv>");
        return;
    }

    let path = Path::new(&args[1]);
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

    // Inject camera + light if the scene lacks them
    let graph = ensure_camera_and_light(graph);

    let ctrl = CameraController::new(graph.roots()[0], Vec3::ZERO, 10.0);
    let mut app = App::new(graph).with_camera_controller(ctrl);
    winit::event_loop::EventLoop::new()
        .unwrap()
        .run_app(&mut app)
        .expect("event loop error");
}

fn ensure_camera_and_light(mut graph: rc3d_scene::SceneGraph) -> rc3d_scene::SceneGraph {
    // Check if any root already has a camera
    let has_camera = graph.roots().iter().any(|&root| {
        has_camera_recursive(&graph, root)
    });

    if !has_camera {
        // Create a new root separator with camera + light
        let sep = graph.add_root(NodeData::Separator(SeparatorNode));
        graph.add_child(
            sep,
            NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
                Vec3::new(5.0, 5.0, 8.0),
                Vec3::ZERO,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                800.0 / 600.0,
            )),
        );
        graph.add_child(
            sep,
            NodeData::DirectionalLight(DirectionalLightNode {
                direction: Vec3::new(-1.0, -1.0, -1.0).normalize(),
                color: Vec3::ONE,
                intensity: 1.0,
            }),
        );
    }

    // Ensure at least one material exists
    let has_material = graph.roots().iter().any(|&root| {
        has_material_recursive(&graph, root)
    });
    if !has_material {
        // Add a default material to the first root separator that has geometry
        for &root in graph.roots() {
            let entry = graph.get(root);
            if let Some(entry) = entry {
                if matches!(entry.data, NodeData::Separator(_)) {
                    graph.add_child(
                        root,
                        NodeData::Material(MaterialNode::from_diffuse(Vec3::new(0.7, 0.7, 0.7))),
                    );
                    break;
                }
            }
        }
    }

    graph
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
