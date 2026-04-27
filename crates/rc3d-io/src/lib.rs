pub mod iv;
pub mod obj;
pub mod stl;

pub use iv::{parse_iv, write_iv, IvError};
pub use obj::{parse_obj, parse_obj_file, ObjError};
pub use stl::{parse_stl, parse_stl_file, StlError};

use std::path::Path;
use rc3d_scene::SceneGraph;

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("IV error: {0}")]
    Iv(#[from] IvError),
    #[error("STL error: {0}")]
    Stl(#[from] StlError),
    #[error("OBJ error: {0}")]
    Obj(#[from] ObjError),
    #[error("Unknown format: {0}")]
    UnknownFormat(String),
}

pub fn import_file(path: &Path) -> Result<SceneGraph, ImportError> {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "iv" => {
            let content = std::fs::read_to_string(path)?;
            Ok(parse_iv(&content)?)
        }
        "stl" => {
            Ok(parse_stl_file(path)?)
        }
        "obj" => {
            Ok(parse_obj_file(path)?)
        }
        _ => Err(ImportError::UnknownFormat(ext)),
    }
}
