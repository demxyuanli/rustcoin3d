use std::collections::HashMap;
use std::path::Path;

use rc3d_core::math::Vec3;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::{Coordinate3Node, IndexedFaceSetNode, SeparatorNode, TextureCoordinate2Node};

#[derive(Debug, Clone, Copy)]
struct FaceCorner {
    v: u32,
    vt: Option<u32>,
}

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
    let mut texcoords = Vec::new();
    let mut faces: Vec<Vec<FaceCorner>> = Vec::new();

    for (line_num, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(rest) = line.strip_prefix("v ") {
            let v = parse_vec3(rest, line_num)?;
            positions.push(v);
        } else if let Some(rest) = line.strip_prefix("vt ") {
            let uv = parse_vt(rest, line_num)?;
            texcoords.push(uv);
        } else if let Some(rest) = line.strip_prefix("f ") {
            let face = parse_face_with_uv(rest, line_num, positions.len(), texcoords.len())?;
            faces.push(face);
        }
    }

    if positions.is_empty() || faces.is_empty() {
        return Err(ObjError::Parse { line: 0, message: "no geometry found".into() });
    }

    let (exp_positions, exp_tex, coord_index) = expand_faces_with_uv(&positions, &texcoords, &faces)?;

    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    graph.add_child(root, NodeData::Coordinate3(Coordinate3Node::from_points(exp_positions)));
    graph.add_child(
        root,
        NodeData::TextureCoordinate2(TextureCoordinate2Node::from_points(exp_tex)),
    );
    graph.add_child(root, NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index }));
    Ok(graph)
}

fn expand_faces_with_uv(
    positions: &[Vec3],
    texcoords: &[[f32; 2]],
    faces: &[Vec<FaceCorner>],
) -> Result<(Vec<Vec3>, Vec<[f32; 2]>, Vec<i32>), ObjError> {
    let mut key_to_new: HashMap<(u32, u32), i32> = HashMap::new();
    let mut exp_pos = Vec::new();
    let mut exp_tex = Vec::new();
    let mut coord_index = Vec::new();

    let corner_key = |v: u32, vt: Option<u32>| -> (u32, u32) { (v, vt.unwrap_or(u32::MAX)) };

    let mut map_corner = |v: u32, vt: Option<u32>| -> Result<i32, ObjError> {
        let key = corner_key(v, vt);
        if let Some(&i) = key_to_new.get(&key) {
            return Ok(i);
        }
        let i = exp_pos.len() as i32;
        exp_pos.push(positions[v as usize]);
        let uv = match vt {
            Some(ti) => texcoords[ti as usize],
            None => [0.0, 0.0],
        };
        exp_tex.push(uv);
        key_to_new.insert(key, i);
        Ok(i)
    };

    for face in faces {
        if face.len() < 3 {
            continue;
        }
        for j in 1..face.len() - 1 {
            let i0 = map_corner(face[0].v, face[0].vt)?;
            let i1 = map_corner(face[j].v, face[j].vt)?;
            let i2 = map_corner(face[j + 1].v, face[j + 1].vt)?;
            coord_index.push(i0);
            coord_index.push(i1);
            coord_index.push(i2);
            coord_index.push(-1);
        }
    }

    if coord_index.is_empty() {
        return Err(ObjError::Parse { line: 0, message: "no valid faces after triangulation".into() });
    }

    Ok((exp_pos, exp_tex, coord_index))
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

fn parse_vt(s: &str, line_num: usize) -> Result<[f32; 2], ObjError> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() {
        return Err(ObjError::Parse { line: line_num, message: "expected u [v] for vt".into() });
    }
    let u = parts[0].parse::<f32>().map_err(|_| ObjError::Parse { line: line_num, message: "invalid vt u".into() })?;
    let v = if parts.len() >= 2 {
        parts[1].parse::<f32>().map_err(|_| ObjError::Parse { line: line_num, message: "invalid vt v".into() })?
    } else {
        0.0
    };
    Ok([u, v])
}

fn parse_face_with_uv(
    s: &str,
    line_num: usize,
    vertex_count: usize,
    tex_count: usize,
) -> Result<Vec<FaceCorner>, ObjError> {
    let mut corners = Vec::new();
    for part in s.split_whitespace() {
        corners.push(parse_face_corner(part, line_num, vertex_count, tex_count)?);
    }
    if corners.len() < 3 {
        return Err(ObjError::Parse { line: line_num, message: "face needs at least 3 vertices".into() });
    }
    Ok(corners)
}

fn parse_face_corner(
    part: &str,
    line_num: usize,
    vertex_count: usize,
    tex_count: usize,
) -> Result<FaceCorner, ObjError> {
    let mut it = part.split('/');
    let v_str = it.next().filter(|s| !s.is_empty()).ok_or_else(|| ObjError::Parse {
        line: line_num,
        message: format!("invalid face token: {part}"),
    })?;
    let vt_str = it.next();
    let vt_str = vt_str.and_then(|s| if s.is_empty() { None } else { Some(s) });

    let v = parse_obj_index(v_str, vertex_count, line_num, "vertex")?;
    let vt = if let Some(ts) = vt_str {
        Some(parse_obj_index(ts, tex_count, line_num, "texture coordinate")?)
    } else {
        None
    };
    Ok(FaceCorner { v, vt })
}

fn parse_obj_index(s: &str, count: usize, line_num: usize, kind: &str) -> Result<u32, ObjError> {
    if count == 0 {
        return Err(ObjError::Parse {
            line: line_num,
            message: format!("no {kind} data for index"),
        });
    }
    if s.starts_with('-') {
        let n: i32 = s.parse().map_err(|_| ObjError::Parse {
            line: line_num,
            message: format!("invalid relative {kind} index: {s}"),
        })?;
        let idx = count as i32 + n;
        if idx < 0 || idx >= count as i32 {
            return Err(ObjError::Parse {
                line: line_num,
                message: format!("{kind} index out of range: {s}"),
            });
        }
        Ok(idx as u32)
    } else {
        let n: u32 = s.parse().map_err(|_| ObjError::Parse {
            line: line_num,
            message: format!("invalid {kind} index: {s}"),
        })?;
        if n < 1 || n as usize > count {
            return Err(ObjError::Parse {
                line: line_num,
                message: format!("{kind} index out of range: {s}"),
            });
        }
        Ok(n - 1)
    }
}
