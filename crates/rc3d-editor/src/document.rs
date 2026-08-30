use std::path::{Path, PathBuf};

use rc3d_core::math::Vec3;
use rc3d_scene::node_data::*;
use rc3d_scene::SceneGraph;

pub fn is_native_scene(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase()),
        Some(ext) if ext == "json"
    )
}

pub fn with_json_extension(path: PathBuf) -> PathBuf {
    match path.extension().and_then(|e| e.to_str()) {
        Some(_) => path,
        None => path.with_extension("json"),
    }
}

pub fn load_native_scene(path: &Path) -> Result<SceneGraph, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

pub fn save_native_scene(graph: &SceneGraph, path: &Path) -> Result<(), String> {
    let text = serde_json::to_string_pretty(graph).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| e.to_string())
}

pub fn pick_open_scene() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Scene JSON", &["json"])
        .pick_file()
}

pub fn pick_save_scene() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Scene JSON", &["json"])
        .set_file_name("untitled.json")
        .save_file()
        .map(with_json_extension)
}

pub fn pick_import_mesh() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Meshes", &["stl", "obj", "gltf", "glb", "fbx"])
        .add_filter("All files", &["*"])
        .pick_file()
}

/// Empty editable scene: camera + light, no geometry.
pub fn blank_scene() -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    graph.add_child(
        root,
        NodeData::PerspectiveCamera(PerspectiveCameraNode::look_at(
            Vec3::new(5.0, 4.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            1.0,
        )),
    );
    graph.add_child(
        root,
        NodeData::DirectionalLight(DirectionalLightNode {
            direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.9),
            intensity: 1.2,
            light_group: None,
        }),
    );
    graph
}

pub fn document_display_name(path: Option<&Path>) -> &str {
    path.and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("Untitled")
}

pub fn window_title(path: Option<&Path>, dirty: bool) -> String {
    let name = document_display_name(path);
    if dirty {
        format!("{name}* - rustcoin3d Studio")
    } else {
        format!("{name} - rustcoin3d Studio")
    }
}
