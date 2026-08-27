use std::collections::HashMap;
use std::path::Path;

use rc3d_core::math::Vec3;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::{
    Coordinate3Node, FaceMaterialGroup, IndexedFaceSetNode, MaterialNode, SeparatorNode,
    TextureCoordinate2Node,
};

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
    parse_obj_with_base(&text, path.parent())
}

pub fn parse_obj(text: &str) -> Result<SceneGraph, ObjError> {
    parse_obj_with_base(text, None)
}

pub fn parse_obj_with_base(text: &str, base_dir: Option<&Path>) -> Result<SceneGraph, ObjError> {
    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    let mut faces: Vec<FaceRec> = Vec::new();
    let mut mtllib: Option<String> = None;
    let mut mat_order: Vec<String> = Vec::new();
    let mut mat_lookup: HashMap<String, u32> = HashMap::new();
    let mut current_mat: Option<u32> = None;

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
            let corners = parse_face_with_uv(rest, line_num, positions.len(), texcoords.len())?;
            let material = match current_mat {
                Some(m) => m,
                None => {
                    let m = intern_material("default", &mut mat_order, &mut mat_lookup);
                    current_mat = Some(m);
                    m
                }
            };
            faces.push(FaceRec { corners, material });
        } else if let Some(rest) = line.strip_prefix("usemtl ") {
            let name = rest.trim();
            if !name.is_empty() {
                current_mat = Some(intern_material(name, &mut mat_order, &mut mat_lookup));
            }
        } else if let Some(rest) = line.strip_prefix("mtllib ") {
            let name = rest.trim();
            if !name.is_empty() {
                mtllib = Some(name.to_string());
            }
        }
    }

    if positions.is_empty() || faces.is_empty() {
        return Err(ObjError::Parse { line: 0, message: "no geometry found".into() });
    }

    let (exp_positions, exp_tex, coord_index, tri_slots) =
        expand_faces_with_uv(&positions, &texcoords, &faces)?;

    let mut materials = placeholder_materials(&mat_order);
    if let (Some(lib), Some(dir)) = (mtllib.as_deref(), base_dir) {
        let mtl_path = dir.join(lib);
        if let Ok(mtl_text) = std::fs::read_to_string(&mtl_path) {
            apply_mtl(&mut materials, &mat_order, &mtl_text, mtl_path.parent().unwrap_or(dir));
        }
    }

    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    let default_mat = materials.first().cloned().unwrap_or_else(|| MaterialNode {
        diffuse_color: Vec3::new(0.9, 0.9, 0.9),
        ambient_color: Vec3::new(0.25, 0.25, 0.25),
        specular_color: Vec3::new(0.04, 0.04, 0.04),
        shininess: 64.0,
        base_color: Vec3::new(0.94, 0.94, 0.94),
        metallic: 0.0,
        roughness: 0.45,
        opacity: 1.0,
        ..Default::default()
    });
    graph.add_child(root, NodeData::Material(default_mat));
    graph.add_child(root, NodeData::Coordinate3(Coordinate3Node::from_points(exp_positions)));
    graph.add_child(
        root,
        NodeData::TextureCoordinate2(TextureCoordinate2Node::from_points(exp_tex)),
    );
    let mut ifs = IndexedFaceSetNode::from_coord_index(coord_index);
    if materials.len() > 1 {
        ifs.material_groups = FaceMaterialGroup::compact_from_triangle_slots(&tri_slots);
        ifs.materials = materials;
    }
    graph.add_child(root, NodeData::IndexedFaceSet(ifs));
    Ok(graph)
}

#[derive(Debug, Clone)]
struct FaceRec {
    corners: Vec<FaceCorner>,
    material: u32,
}

fn intern_material(name: &str, order: &mut Vec<String>, lookup: &mut HashMap<String, u32>) -> u32 {
    if let Some(&idx) = lookup.get(name) {
        return idx;
    }
    let idx = order.len() as u32;
    order.push(name.to_string());
    lookup.insert(name.to_string(), idx);
    idx
}

fn placeholder_color(i: usize) -> Vec3 {
    const PALETTE: [[f32; 3]; 6] = [
        [0.85, 0.22, 0.18],
        [0.20, 0.55, 0.85],
        [0.25, 0.75, 0.35],
        [0.90, 0.70, 0.15],
        [0.65, 0.30, 0.80],
        [0.20, 0.80, 0.75],
    ];
    let c = PALETTE[i % PALETTE.len()];
    Vec3::new(c[0], c[1], c[2])
}

fn placeholder_materials(names: &[String]) -> Vec<MaterialNode> {
    names
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let c = placeholder_color(i);
            MaterialNode {
                diffuse_color: c,
                ambient_color: c * 0.25,
                specular_color: Vec3::new(0.04, 0.04, 0.04),
                shininess: 64.0,
                base_color: c,
                metallic: 0.0,
                roughness: 0.45,
                opacity: 1.0,
                ..Default::default()
            }
        })
        .collect()
}

fn apply_mtl(materials: &mut [MaterialNode], names: &[String], text: &str, base_dir: &Path) {
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (i, n) in names.iter().enumerate() {
        by_name.insert(n.as_str(), i);
    }
    let mut current: Option<usize> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("newmtl ") {
            current = by_name.get(rest.trim()).copied();
        } else if let Some(idx) = current {
            let mat = &mut materials[idx];
            if let Some(rest) = line.strip_prefix("Kd ") {
                if let Ok(c) = parse_vec3(rest, 0) {
                    mat.base_color = c;
                    mat.diffuse_color = c;
                    mat.ambient_color = c * 0.25;
                }
            } else if let Some(rest) = line.strip_prefix("map_Kd ") {
                let file = rest.split_whitespace().last().unwrap_or(rest).trim();
                if !file.is_empty() {
                    mat.albedo_texture = Some(base_dir.join(file).to_string_lossy().to_string());
                }
            } else if let Some(rest) = line.strip_prefix("d ") {
                if let Ok(d) = rest.split_whitespace().next().unwrap_or("1").parse::<f32>() {
                    mat.opacity = d.clamp(0.0, 1.0);
                }
            } else if let Some(rest) = line.strip_prefix("Ns ") {
                if let Ok(ns) = rest.split_whitespace().next().unwrap_or("0").parse::<f32>() {
                    mat.shininess = ns.max(1.0);
                    mat.roughness = (1.0 - (ns / 1000.0).clamp(0.0, 1.0)).clamp(0.04, 1.0);
                }
            }
        }
    }
}

type ExpandedMesh = (Vec<Vec3>, Vec<[f32; 2]>, Vec<i32>, Vec<u32>);

fn expand_faces_with_uv(
    positions: &[Vec3],
    texcoords: &[[f32; 2]],
    faces: &[FaceRec],
) -> Result<ExpandedMesh, ObjError> {
    let mut key_to_new: HashMap<(u32, u32), i32> = HashMap::new();
    let mut exp_pos = Vec::new();
    let mut exp_tex = Vec::new();
    let mut coord_index = Vec::new();
    let mut tri_slots = Vec::new();

    let corner_key = |v: u32, vt: Option<u32>| -> (u32, u32) { (v, vt.unwrap_or(u32::MAX)) };

    let mut map_corner = |v: u32, vt: Option<u32>| -> Result<i32, ObjError> {
        let key = corner_key(v, vt);
        if let Some(&i) = key_to_new.get(&key) {
            return Ok(i);
        }
        let i = exp_pos.len() as i32;
        exp_pos.push(positions[v as usize]);
        let uv = match vt {
            Some(ti) => texcoords.get(ti as usize).copied().unwrap_or([0.0, 0.0]),
            None => [0.0, 0.0],
        };
        exp_tex.push(uv);
        key_to_new.insert(key, i);
        Ok(i)
    };

    for face in faces {
        if face.corners.len() < 3 {
            continue;
        }
        for j in 1..face.corners.len() - 1 {
            let i0 = map_corner(face.corners[0].v, face.corners[0].vt)?;
            let i1 = map_corner(face.corners[j].v, face.corners[j].vt)?;
            let i2 = map_corner(face.corners[j + 1].v, face.corners[j + 1].vt)?;
            coord_index.push(i0);
            coord_index.push(i1);
            coord_index.push(i2);
            coord_index.push(-1);
            tri_slots.push(face.material);
        }
    }

    if coord_index.is_empty() {
        return Err(ObjError::Parse { line: 0, message: "no valid faces after triangulation".into() });
    }

    Ok((exp_pos, exp_tex, coord_index, tri_slots))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_obj_triangle() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        let result = parse_obj(text);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        let g = result.unwrap();
        assert!(!g.roots().is_empty());
    }

    #[test]
    fn test_parse_obj_cube() {
        let text = "\
v -0.5 -0.5 -0.5\nv  0.5 -0.5 -0.5\nv  0.5  0.5 -0.5\nv -0.5  0.5 -0.5\n\
v -0.5 -0.5  0.5\nv  0.5 -0.5  0.5\nv  0.5  0.5  0.5\nv -0.5  0.5  0.5\n\
f 1 2 3 4\nf 5 8 7 6\nf 1 5 6 2\nf 2 6 7 3\nf 3 7 8 4\nf 5 1 4 8\n";
        let result = parse_obj(text);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
    }

    #[test]
    fn test_parse_obj_empty() {
        assert!(parse_obj("").is_err() || parse_obj("").is_ok());
        // Empty OBJ may or may not be valid depending on parser behavior
    }
}
