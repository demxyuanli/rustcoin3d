//! STL rendering diagnostic — import and display an STL file.
//!
//! Usage: cargo run -p rc3d-examples --example stl_diagnostic <file.stl>

use std::env;
use std::path::Path;

use rc3d_actions::{fit_camera_to_scene, CameraFitConfig};
use rc3d_core::math::Vec3;
use rc3d_core::{DisplayMode, NodeId};
use rc3d_engine_api::CameraController;
use rc3d_examples::common::run_example;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path_arg = args.iter().skip(1).next().cloned();

    let Some(ref path_str) = path_arg else {
        eprintln!("Usage: stl_diagnostic <file.stl>");
        return;
    };

    let path = Path::new(path_str);
    let path_buf = path.to_path_buf();

    run_example("STL Diagnostic", |engine| {
        engine.set_display_mode(DisplayMode::Shaded);

        match engine.import(&path_buf) {
            Ok(_root_id) => {
                // Ensure camera, lights, and material exist (STL files only contain geometry)
                ensure_camera_lights_material(engine.scene_mut());

                // Fit camera to the imported geometry
                let (target, orbit_radius) =
                    fit_camera_to_scene(engine.scene_mut(), CameraFitConfig::default());
                engine.controller = CameraController::new(target, orbit_radius);

                log::info!("Imported STL: {}", path.display());
            }
            Err(e) => {
                eprintln!("Import error: {e}");
            }
        }
    });
}

/// Add PerspectiveCamera, DirectionalLights, and Material if the scene graph
/// doesn't already have them (STL files carry no camera/material data).
fn ensure_camera_lights_material(graph: &mut SceneGraph) {
    let has_camera = graph
        .roots()
        .iter()
        .any(|&root| has_camera_recursive(graph, root));
    let has_material = graph
        .roots()
        .iter()
        .any(|&root| has_material_recursive(graph, root));

    let target_root = find_geometry_root(graph);

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
    }

    if !has_material {
        let insert_idx = if has_camera { 0 } else { 3 };
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
}

fn find_geometry_root(graph: &SceneGraph) -> NodeId {
    for &root in graph.roots() {
        if let Some(entry) = graph.get(root) {
            if matches!(entry.data, NodeData::Separator(_)) {
                return root;
            }
        }
    }
    graph.roots()[0]
}

fn has_camera_recursive(graph: &SceneGraph, node: NodeId) -> bool {
    let Some(entry) = graph.get(node) else { return false };
    if matches!(entry.data, NodeData::PerspectiveCamera(_) | NodeData::OrthographicCamera(_)) {
        return true;
    }
    for &child in &entry.children {
        if has_camera_recursive(graph, child) { return true; }
    }
    false
}

fn has_material_recursive(graph: &SceneGraph, node: NodeId) -> bool {
    let Some(entry) = graph.get(node) else { return false };
    if matches!(entry.data, NodeData::Material(_)) { return true; }
    for &child in &entry.children {
        if has_material_recursive(graph, child) { return true; }
    }
    false
}
