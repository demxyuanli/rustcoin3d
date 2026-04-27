use std::path::Path;

use rc3d_core::math::Vec3;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::{Coordinate3Node, IndexedFaceSetNode, SeparatorNode};

#[derive(Debug, thiserror::Error)]
pub enum ObjError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },
}

pub fn parse_obj_file(path: &Path) -> Result<SceneGraph, ObjError> {
    let text = std::fs::read_to_string(path)?;
    parse_obj(&text)
}

pub fn parse_obj(text: &str) -> Result<SceneGraph, ObjError> {
    let mut positions = Vec::new();
    // Face groups: each face is a list of vertex indices (0-based after conversion)
    let mut faces: Vec<Vec<u32>> = Vec::new();

    for (line_num, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(rest) = line.strip_prefix("v ") {
            let v = parse_vec3(rest, line_num)?;
            positions.push(v);
        } else if let Some(rest) = line.strip_prefix("f ") {
            let face = parse_face(rest, line_num, positions.len())?;
            faces.push(face);
        }
        // Ignore vn, vt, mtllib, usemtl, g, o, s, etc.
    }

    if positions.is_empty() || faces.is_empty() {
        return Err(ObjError::Parse { line: 0, message: "no geometry found".into() });
    }

    // Fan-triangulate polygon faces and build coord_index
    let mut coord_index = Vec::new();
    for face in &faces {
        if face.len() < 3 {
            continue;
        }
        for j in 1..face.len() - 1 {
            coord_index.push(face[0] as i32);
            coord_index.push(face[j] as i32);
            coord_index.push(face[j + 1] as i32);
            coord_index.push(-1);
        }
    }

    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    graph.add_child(root, NodeData::Coordinate3(Coordinate3Node::from_points(positions)));
    graph.add_child(root, NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index }));
    Ok(graph)
}

fn parse_vec3(s: &str, line_num: usize) -> Result<Vec3, ObjError> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(ObjError::Parse { line: line_num, message: "expected 3 floats for vertex".into() });
    }
    let x = parts[0].parse::<f32>().map_err(|_| ObjError::Parse { line: line_num, message: "invalid x".into() })?;
    let y = parts[1].parse::<f32>().map_err(|_| ObjError::Parse { line: line_num, message: "invalid y".into() })?;
    let z = parts[2].parse::<f32>().map_err(|_| ObjError::Parse { line: line_num, message: "invalid z".into() })?;
    Ok(Vec3::new(x, y, z))
}

fn parse_face(s: &str, line_num: usize, vertex_count: usize) -> Result<Vec<u32>, ObjError> {
    let mut indices = Vec::new();
    for part in s.split_whitespace() {
        // Formats: "v", "v/vt", "v/vt/vn", "v//vn"
        let v_str = part.split('/').next().unwrap();
        let idx = if v_str.starts_with('-') {
            // Negative index: relative to end
            let n: i32 = v_str.parse().map_err(|_| ObjError::Parse { line: line_num, message: format!("invalid face index: {part}") })?;
            (vertex_count as i32 + n) as u32
        } else {
            let n: u32 = v_str.parse().map_err(|_| ObjError::Parse { line: line_num, message: format!("invalid face index: {part}") })?;
            n - 1 // OBJ is 1-indexed
        };
        indices.push(idx);
    }
    if indices.len() < 3 {
        return Err(ObjError::Parse { line: line_num, message: "face needs at least 3 vertices".into() });
    }
    Ok(indices)
}
