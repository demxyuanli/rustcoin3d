//! # rc3d-io — File Format Import/Export
//!
//! Minimal I/O layer: B-Rep binary serialization and basic mesh export.
//! Geometry parsing (STEP, IGES) has been removed.

pub mod gltf;
pub mod obj;
pub mod stl;

pub use gltf::{parse_gltf_file, GltfError};
pub use obj::{parse_obj, parse_obj_file, ObjError};
pub use stl::{
    parse_stl, parse_stl_file, parse_stl_triangles, write_ascii_stl, write_binary_stl, StlError,
};

use std::path::Path;
use rc3d_scene::SceneGraph;

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("STL error: {0}")]
    Stl(#[from] StlError),
    #[error("OBJ error: {0}")]
    Obj(#[from] ObjError),
    #[error("glTF error: {0}")]
    Gltf(#[from] GltfError),
    #[error("Unknown format: {0}")]
    UnknownFormat(String),
}

pub fn import_file(path: &Path) -> Result<SceneGraph, ImportError> {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "stl" => Ok(parse_stl_file(path)?),
        "obj" => Ok(parse_obj_file(path)?),
        "gltf" | "glb" => Ok(parse_gltf_file(path)?),
        _ => Err(ImportError::UnknownFormat(ext)),
    }
}
