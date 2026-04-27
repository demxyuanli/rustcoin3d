use std::io;
use std::path::Path;

use rc3d_core::math::Vec3;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::{Coordinate3Node, IndexedFaceSetNode, SeparatorNode};

#[derive(Debug, thiserror::Error)]
pub enum StlError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid STL binary: {0}")]
    InvalidBinary(String),
    #[error("Invalid STL ASCII: {0}")]
    InvalidAscii(String),
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

pub fn parse_stl_file(path: &Path) -> Result<SceneGraph, StlError> {
    let data = std::fs::read(path)?;
    parse_stl(&data)
}

pub fn parse_stl(data: &[u8]) -> Result<SceneGraph, StlError> {
    let triangles = if is_likely_binary(data) {
        parse_stl_binary(data)?
    } else {
        let text = std::str::from_utf8(data)?;
        parse_stl_ascii(text)?
    };
    Ok(triangles_to_scene(&triangles))
}

struct StlTriangle {
    normal: [f32; 3],
    vertices: [[f32; 3]; 3],
}

fn is_likely_binary(data: &[u8]) -> bool {
    if data.len() < 84 {
        return false;
    }
    let count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    data.len() == 84 + count * 50
}

fn parse_stl_binary(data: &[u8]) -> Result<Vec<StlTriangle>, StlError> {
    if data.len() < 84 {
        return Err(StlError::InvalidBinary("file too short".into()));
    }
    let count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if data.len() < 84 + count * 50 {
        return Err(StlError::InvalidBinary("truncated data".into()));
    }

    let mut triangles = Vec::with_capacity(count);
    let mut offset = 84;
    for _ in 0..count {
        let normal = read_f32_3(data, offset);
        offset += 12;
        let v0 = read_f32_3(data, offset);
        offset += 12;
        let v1 = read_f32_3(data, offset);
        offset += 12;
        let v2 = read_f32_3(data, offset);
        offset += 12;
        offset += 2; // attribute byte count
        triangles.push(StlTriangle { normal, vertices: [v0, v1, v2] });
    }
    Ok(triangles)
}

fn read_f32_3(data: &[u8], offset: usize) -> [f32; 3] {
    [
        f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]),
        f32::from_le_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]),
        f32::from_le_bytes([data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11]]),
    ]
}

fn parse_stl_ascii(text: &str) -> Result<Vec<StlTriangle>, StlError> {
    let mut triangles = Vec::new();
    let mut lines = text.lines().peekable();

    // Skip 'solid ...' line
    while let Some(line) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.starts_with("solid") || trimmed.starts_with("SOLID") {
            lines.next();
            break;
        }
        lines.next();
    }

    while let Some(line) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.starts_with("endsolid") || trimmed.starts_with("ENDSOLID") || trimmed.is_empty() {
            break;
        }
        if !trimmed.starts_with("facet") && !trimmed.starts_with("FACET") {
            lines.next();
            continue;
        }

        let mut normal = [0.0f32; 3];
        parse_facet_normal(trimmed, &mut normal);

        lines.next(); // consume 'facet normal ...'
        skip_line(&mut lines, "outer"); // 'outer loop'

        let mut vertices = [[0.0f32; 3]; 3];
        for i in 0..3 {
            if let Some(vline) = lines.next() {
                parse_vertex(vline.trim(), &mut vertices[i]);
            }
        }

        skip_line(&mut lines, "endloop");
        skip_line(&mut lines, "endfacet");

        triangles.push(StlTriangle { normal, vertices });
    }

    Ok(triangles)
}

fn parse_facet_normal(line: &str, normal: &mut [f32; 3]) {
    let parts: Vec<&str> = line.split_whitespace().collect();
    // facet normal ni nj nk
    if parts.len() >= 5 {
        normal[0] = parts[2].parse().unwrap_or(0.0);
        normal[1] = parts[3].parse().unwrap_or(0.0);
        normal[2] = parts[4].parse().unwrap_or(0.0);
    }
}

fn parse_vertex(line: &str, vertex: &mut [f32; 3]) {
    let parts: Vec<&str> = line.split_whitespace().collect();
    // vertex x y z
    if parts.len() >= 4 {
        vertex[0] = parts[1].parse().unwrap_or(0.0);
        vertex[1] = parts[2].parse().unwrap_or(0.0);
        vertex[2] = parts[3].parse().unwrap_or(0.0);
    }
}

fn skip_line<'a, I: Iterator<Item = &'a str>>(lines: &mut std::iter::Peekable<I>, _expected: &str) {
    lines.next();
}

fn triangles_to_scene(triangles: &[StlTriangle]) -> SceneGraph {
    let mut points = Vec::with_capacity(triangles.len() * 3);
    let mut coord_index = Vec::with_capacity(triangles.len() * 4);

    for tri in triangles {
        let base = points.len() as i32;
        points.push(Vec3::from(tri.vertices[0]));
        points.push(Vec3::from(tri.vertices[1]));
        points.push(Vec3::from(tri.vertices[2]));
        coord_index.extend_from_slice(&[base, base + 1, base + 2, -1]);
    }

    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    graph.add_child(root, NodeData::Coordinate3(Coordinate3Node::from_points(points)));
    graph.add_child(root, NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index }));
    graph
}
