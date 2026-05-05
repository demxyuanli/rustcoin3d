use std::io::BufReader;
use std::path::Path;

use rc3d_scene::SceneGraph;

use self::parser::FbxParser;
use self::scene_builder::build_scene;

pub mod animation;
pub mod parser;
pub mod scene_builder;
pub mod skeleton;
pub mod types;

#[derive(Debug, thiserror::Error)]
pub enum FbxError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("FBX parse error: {0}")]
    Parse(String),
    #[error("Unsupported FBX version")]
    UnsupportedVersion,
}

/// Parse an FBX file into a SceneGraph.
pub fn parse_fbx_file(path: &Path) -> Result<SceneGraph, FbxError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    match fbxcel::pull_parser::any::AnyParser::from_seekable_reader(reader) {
        Ok(fbxcel::pull_parser::any::AnyParser::V7400(parser)) => {
            let mut fbxp = FbxParser::new(parser);
            let data = fbxp.parse()?;
            build_scene(&data)
        }
        Ok(_) => Err(FbxError::UnsupportedVersion),
        Err(e) => Err(FbxError::Parse(format!("Failed to create FBX parser: {e}"))),
    }
}
