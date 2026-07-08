//! File import with auto-format detection and FileNode resolution.
//!
//! Delegates to the low-level `rc3d_io` parsers and wraps imported geometry
//! in a `Separator` node before adding it to the scene graph.

use std::path::Path;
use std::sync::Arc;

use rc3d_core::{EngineError, EngineResult, NodeId};
use rc3d_render::SceneLoadFn;
use rc3d_scene::node_data::{NodeData, SeparatorNode};
use rc3d_scene::SceneGraph;

/// Import a 3D file into the scene graph.
///
/// The file format is auto-detected from the extension (stl, obj, gltf, glb).
/// All imported geometry is wrapped in a root-level `Separator`
/// node whose [`NodeId`] is returned.
///
/// # Errors
///
/// Returns [`EngineError::Parse`] if the file format is not recognized or
/// parsing fails. Returns [`EngineError::Io`] if the file cannot be read.
pub fn import_file(graph: &mut SceneGraph, path: impl AsRef<Path>) -> EngineResult<NodeId> {
    let path = path.as_ref();
    let imported = rc3d_io::import_file(path).map_err(|e| EngineError::Parse(e.to_string()))?;

    let sep = graph.add_root(NodeData::Separator(SeparatorNode));
    for &root in imported.roots() {
        copy_subtree(&imported, graph, root, sep);
    }
    Ok(sep)
}

/// Resolve all `FileNode` references in the scene graph by loading the
/// referenced files (glTF, OBJ, STL) and merging their content in-place.
///
/// Walks the graph recursively; loaded files may contain further `FileNode`
/// references which are resolved in turn.
///
/// Returns the number of files loaded. Missing or unreadable files are
/// logged as warnings and the `FileNode` is left in-place.
pub fn resolve_file_nodes(graph: &mut SceneGraph) -> usize {
    let mut count = 0usize;
    let roots: Vec<NodeId> = graph.roots().to_vec();
    let mut to_resolve: Vec<(NodeId, String)> = Vec::new();

    for &root in &roots {
        collect_file_nodes(graph, root, &mut to_resolve);
    }

    for (node_id, path) in to_resolve {
        log::info!("[FileNode] loading: {}", path);
        match rc3d_io::import_file(Path::new(&path)) {
            Ok(imported) => {
                let parent_id = graph.get(node_id).and_then(|e| e.parent);
                if let Some(pid) = parent_id {
                    let sep = graph.add_child(pid, NodeData::Separator(SeparatorNode));
                    for &root in imported.roots() {
                        copy_subtree(&imported, graph, root, sep);
                    }
                    graph.remove(node_id);
                    count += 1 + resolve_file_nodes_in_subtree(graph, sep);
                }
            }
            Err(e) => {
                log::warn!("[FileNode] failed to load '{}': {}", path, e);
            }
        }
    }

    count
}

fn collect_file_nodes(graph: &SceneGraph, node: NodeId, out: &mut Vec<(NodeId, String)>) {
    if let Some(entry) = graph.get(node) {
        if let NodeData::File(f) = &entry.data {
            out.push((node, f.path.clone()));
        }
    }
    if let Some(children) = graph.children(node) {
        for &child in children {
            collect_file_nodes(graph, child, out);
        }
    }
}

fn resolve_file_nodes_in_subtree(graph: &mut SceneGraph, root: NodeId) -> usize {
    let mut count = 0usize;
    let mut to_resolve: Vec<(NodeId, String)> = Vec::new();

    if let Some(children) = graph.children(root) {
        for &child in children {
            collect_file_nodes(graph, child, &mut to_resolve);
        }
    }

    for (node_id, path) in to_resolve {
        log::info!("[FileNode] loading (nested): {}", path);
        match rc3d_io::import_file(Path::new(&path)) {
            Ok(imported) => {
                let parent_id = graph.get(node_id).and_then(|e| e.parent);
                if let Some(pid) = parent_id {
                    let sep = graph.add_child(pid, NodeData::Separator(SeparatorNode));
                    for &root in imported.roots() {
                        copy_subtree(&imported, graph, root, sep);
                    }
                    graph.remove(node_id);
                    count += 1 + resolve_file_nodes_in_subtree(graph, sep);
                }
            }
            Err(e) => {
                log::warn!("[FileNode] failed to load '{}': {}", path, e);
            }
        }
    }

    count
}

/// Scene loader callback for [`rc3d_render::AsyncAssetManager`].
pub fn default_scene_loader() -> SceneLoadFn {
    Arc::new(|path| {
        rc3d_io::import_file(path).map_err(|e| e.to_string())
    })
}

/// Deep-copy a node and all its descendants from `src` to `dst` under `dst_parent`.
pub(crate) fn copy_subtree(
    src: &SceneGraph,
    dst: &mut SceneGraph,
    src_id: NodeId,
    dst_parent: NodeId,
) -> NodeId {
    let entry = src.get(src_id).expect("copy_subtree: source node exists");
    let new_id = dst.add_child(dst_parent, entry.data.clone());
    if let Some(children) = src.children(src_id) {
        for &child in children {
            copy_subtree(src, dst, child, new_id);
        }
    }
    new_id
}
