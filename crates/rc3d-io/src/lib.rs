pub mod brep_binary;
pub mod mesh_export;
pub mod fbx;
pub mod gltf;
pub mod iges;
pub mod iv;
pub mod obj;
pub mod step;
pub mod stl;

pub use fbx::{parse_fbx_file, FbxError};
pub use gltf::{parse_gltf_file, GltfError};
pub use iv::{parse_iv, write_iv, IvError};
pub use obj::{parse_obj, parse_obj_file, ObjError};
pub use step::{
    decode_step_bytes, import_step_file_with_options, import_step_with_options, parse_step,
    parse_step_file, parse_step_file_with_options, parse_step_with_options, write_step_file,
    write_step_entities_file, AdapterMode, StepError, StepImportMode, StepImportOptions,
    StepImportReport, StepImportResult,
};
pub use step::write::{write_step_from_entities, write_step_from_graph};
pub use step::validate::{validate as validate_step, quick_check as quick_check_step, ValidationReport};
pub use step::xml::{write_xml_step, parse_xml_step};
pub use step::bool::BoolOp;
pub use step::brep;
pub use step::tree::{AssemblyTree, AssemblyNode, ProductMetadata};
pub use step::header::HeaderInfo;
pub use step::lod;
pub use stl::{
    parse_stl, parse_stl_file, parse_stl_triangles, write_ascii_stl, write_binary_stl, StlError,
};
pub use mesh_export::{
    convert_stl_to_ascii, default_stl_output, export_file_to_ascii_stl,
    export_step_per_face_ascii_stl, export_step_to_ascii_stl, mesh_step_file, ExportSummary,
    MeshExportError, MeshExportOptions, StepMeshResult,
};

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
    #[error("glTF error: {0}")]
    Gltf(#[from] GltfError),
    #[error("FBX error: {0}")]
    Fbx(#[from] FbxError),
    #[error("STEP error: {0}")]
    Step(#[from] StepError),
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
        "gltf" | "glb" => {
            Ok(parse_gltf_file(path)?)
        }
        "fbx" => {
            Ok(parse_fbx_file(path)?)
        }
        "step" | "stp" => {
            Ok(parse_step_file(path)?)
        }
        _ => Err(ImportError::UnknownFormat(ext)),
    }
}
