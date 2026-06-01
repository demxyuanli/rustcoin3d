//! File import with auto-format detection.
//!
//! Delegates to the low-level `rc3d_io` parsers and wraps imported geometry
//! in a `Separator` node before adding it to the scene graph.

use std::path::Path;
use std::sync::Arc;

use rc3d_core::{EngineError, EngineResult, NodeId};
use rc3d_io::{import_step_file_with_options, StepImportOptions, StepImportResult};
use rc3d_render::SceneLoadFn;
use rc3d_scene::node_data::{NodeData, SeparatorNode};
use rc3d_scene::SceneGraph;

/// Import a 3D file into the scene graph.
///
/// The file format is auto-detected from the extension (stl, obj, gltf, glb,
/// fbx, iv). All imported geometry is wrapped in a root-level `Separator`
/// node whose [`NodeId`] is returned.
///
/// # Errors
///
/// Returns [`EngineError::Parse`] if the file format is not recognized or
/// parsing fails. Returns [`EngineError::Io`] if the file cannot be read.
pub fn import_file(graph: &mut SceneGraph, path: impl AsRef<Path>) -> EngineResult<NodeId> {
    let path = path.as_ref();
    let imported = rc3d_io::import_file(path).map_err(|e| EngineError::Parse(e.to_string()))?;

    // Wrap all imported roots under a new Separator
    let sep = graph.add_root(NodeData::Separator(SeparatorNode));
    for &root in imported.roots() {
        copy_subtree(&imported, graph, root, sep);
    }
    Ok(sep)
}

/// Import a STEP file and return the full result including [`StepImportResult::document`].
pub fn import_step_with_document(path: impl AsRef<Path>) -> EngineResult<StepImportResult> {
    import_step_file_with_options(path.as_ref(), &StepImportOptions::default())
        .map_err(|e| EngineError::Parse(e.to_string()))
}

/// Import STEP into an existing graph; returns the wrapper separator and full import result.
pub fn import_step_into_graph(
    graph: &mut SceneGraph,
    path: impl AsRef<Path>,
) -> EngineResult<(NodeId, StepImportResult)> {
    let result = import_step_with_document(path)?;
    let sep = graph.add_root(NodeData::Separator(SeparatorNode));
    for &root in result.graph.roots() {
        copy_subtree(&result.graph, graph, root, sep);
    }
    Ok((sep, result))
}

/// Scene loader callback for [`rc3d_render::AsyncAssetManager`].
pub fn default_scene_loader() -> SceneLoadFn {
    Arc::new(|path| {
        rc3d_io::import_file(path).map_err(|e| e.to_string())
    })
}

/// Deep-copy a node and all its descendants from `src` to `dst` under `dst_parent`.
///
/// The two graphs must be independent (different `SlotMap` allocations).
/// Node data is cloned, so the original graph is unchanged.
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
