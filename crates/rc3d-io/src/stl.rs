use std::io;
use std::path::Path;
use std::collections::HashMap;

use rc3d_core::math::Vec3;
use rc3d_core::DisplayMode;
use rc3d_scene::{NodeData, SceneGraph};
use rc3d_scene::node_data::{Coordinate3Node, IndexedFaceSetNode, MaterialNode, SeparatorNode};

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
    let triangles = parse_stl_triangles(data)?;
    Ok(triangles_to_scene(&triangles))
}

/// Parse STL into triangle list (for T4 reference mesh comparison).
pub fn parse_stl_triangles(data: &[u8]) -> Result<Vec<StlTriangle>, StlError> {
    if is_likely_binary(data) {
        parse_stl_binary(data)
    } else {
        let text = std::str::from_utf8(data)?;
        parse_stl_ascii(text)
    }
}

fn mesh_to_stl_triangles(vertices: &[Vec3], indices: &[i32]) -> Vec<StlTriangle> {
    let mut tris = Vec::new();
    for chunk in indices.chunks(4) {
        if chunk.len() < 3 {
            continue;
        }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];
        let n = (v1 - v0).cross(v2 - v0);
        let normal = if n.length_squared() > 1e-20 {
            n.normalize()
        } else {
            Vec3::Z
        };
        tris.push(StlTriangle {
            normal: [normal.x, normal.y, normal.z],
            vertices: [
                [v0.x, v0.y, v0.z],
                [v1.x, v1.y, v1.z],
                [v2.x, v2.y, v2.z],
            ],
        });
    }
    tris
}

/// Write engine mesh as ASCII STL (indices layout: i0, i1, i2, -1 per triangle).
pub fn write_ascii_stl(path: &Path, vertices: &[Vec3], indices: &[i32]) -> Result<(), StlError> {
    let tris = mesh_to_stl_triangles(vertices, indices);
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("mesh");
    let mut out = String::with_capacity(tris.len().saturating_mul(200) + 64);
    out.push_str(&format!("solid {name}\n"));
    for tri in &tris {
        let [nx, ny, nz] = tri.normal;
        out.push_str(&format!("  facet normal {nx} {ny} {nz}\n"));
        out.push_str("    outer loop\n");
        for v in &tri.vertices {
            out.push_str(&format!(
                "      vertex {} {} {}\n",
                v[0], v[1], v[2]
            ));
        }
        out.push_str("    endloop\n");
        out.push_str("  endfacet\n");
    }
    out.push_str(&format!("endsolid {name}\n"));
    std::fs::write(path, out)?;
    Ok(())
}

/// Write engine mesh as binary STL (OCC reference export workflow).
pub fn write_binary_stl(path: &Path, vertices: &[Vec3], indices: &[i32]) -> Result<(), StlError> {
    let tris = mesh_to_stl_triangles(vertices, indices);
    let mut out = Vec::with_capacity(84 + tris.len() * 50);
    let mut header = [0u8; 80];
    let label = b"rc3d-io binary STL";
    header[..label.len()].copy_from_slice(label);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for tri in &tris {
        for f in tri.normal.iter().chain(tri.vertices[0].iter()).chain(tri.vertices[1].iter()).chain(tri.vertices[2].iter()) {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Single STL triangle (binary/ASCII parse output).
pub struct StlTriangle {
    pub normal: [f32; 3],
    pub vertices: [[f32; 3]; 3],
}

fn is_likely_binary(data: &[u8]) -> bool {
    if data.len() < 84 {
        return false;
    }
    let count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    data.len() >= 84 + count * 50
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
        let normal = read_f32_3(data, offset)?;
        offset += 12;
        let v0 = read_f32_3(data, offset)?;
        offset += 12;
        let v1 = read_f32_3(data, offset)?;
        offset += 12;
        let v2 = read_f32_3(data, offset)?;
        offset += 12;
        offset += 2; // attribute byte count
        triangles.push(StlTriangle { normal, vertices: [v0, v1, v2] });
    }
    Ok(triangles)
}

fn read_f32_3(data: &[u8], offset: usize) -> Result<[f32; 3], StlError> {
    if offset + 12 > data.len() {
        return Err(StlError::InvalidBinary(format!("offset {} out of bounds", offset)));
    }
    Ok([
        f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]),
        f32::from_le_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]),
        f32::from_le_bytes([data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11]]),
    ])
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
        parse_facet_normal(trimmed, &mut normal)?;

        lines.next(); // consume 'facet normal ...'
        skip_line(&mut lines, "outer"); // 'outer loop'

        let mut vertices = [[0.0f32; 3]; 3];
        for vertex in &mut vertices {
            if let Some(vline) = lines.next() {
                parse_vertex(vline.trim(), vertex)?;
            }
        }

        skip_line(&mut lines, "endloop");
        skip_line(&mut lines, "endfacet");

        triangles.push(StlTriangle { normal, vertices });
    }

    Ok(triangles)
}

fn parse_facet_normal(line: &str, normal: &mut [f32; 3]) -> Result<(), StlError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    // facet normal ni nj nk
    if parts.len() >= 5 {
        normal[0] = parts[2].parse().map_err(|_| StlError::InvalidAscii(format!("invalid normal x in: {}", line)))?;
        normal[1] = parts[3].parse().map_err(|_| StlError::InvalidAscii(format!("invalid normal y in: {}", line)))?;
        normal[2] = parts[4].parse().map_err(|_| StlError::InvalidAscii(format!("invalid normal z in: {}", line)))?;
    }
    Ok(())
}

fn parse_vertex(line: &str, vertex: &mut [f32; 3]) -> Result<(), StlError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    // vertex x y z
    if parts.len() >= 4 {
        vertex[0] = parts[1].parse().map_err(|_| StlError::InvalidAscii(format!("invalid vertex x in: {}", line)))?;
        vertex[1] = parts[2].parse().map_err(|_| StlError::InvalidAscii(format!("invalid vertex y in: {}", line)))?;
        vertex[2] = parts[3].parse().map_err(|_| StlError::InvalidAscii(format!("invalid vertex z in: {}", line)))?;
    }
    Ok(())
}

fn skip_line<'a, I: Iterator<Item = &'a str>>(lines: &mut std::iter::Peekable<I>, _expected: &str) {
    lines.next();
}

fn triangles_to_scene(triangles: &[StlTriangle]) -> SceneGraph {
    let mut points = Vec::with_capacity(triangles.len() * 3);
    let mut coord_index = Vec::with_capacity(triangles.len() * 4);
    let mut remap: std::collections::HashMap<[u32; 3], i32> =
        std::collections::HashMap::with_capacity(triangles.len() * 2);

    for tri in triangles {
        for v in tri.vertices {
            let key = rc3d_core::utils::hash::f32x3_to_bits(v);
            let idx = if let Some(&i) = remap.get(&key) {
                i
            } else {
                let i = points.len() as i32;
                points.push(Vec3::from(v));
                remap.insert(key, i);
                i
            };
            coord_index.push(idx);
        }
        coord_index.push(-1);
    }

    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    // STL usually has no material payload; inject a readable default material for contrast.
    graph.add_child(
        root,
        NodeData::Material(MaterialNode {
            diffuse_color: Vec3::new(0.9, 0.9, 0.9),
            ambient_color: Vec3::new(0.35, 0.35, 0.35),
            specular_color: Vec3::new(0.0, 0.0, 0.0),
            shininess: 0.0,
            base_color: Vec3::new(0.94, 0.94, 0.94),
            metallic: 0.0,
            roughness: 0.35,
            opacity: 1.0,
            ..Default::default()
        }),
    );
    const MAX_VERTICES_PER_CHUNK: usize = 2_000_000;
    let chunks = split_indexed_face_set_into_chunks(&points, &coord_index, MAX_VERTICES_PER_CHUNK);
    for (chunk_points, chunk_indices) in chunks {
        graph.add_child(root, NodeData::Coordinate3(Coordinate3Node::from_points(chunk_points)));
        graph.add_child(root, NodeData::IndexedFaceSet(IndexedFaceSetNode { coord_index: chunk_indices }));
    }
    if let Some(root_entry) = graph.get_mut(root) {
        root_entry.display_mode = Some(DisplayMode::ShadedWithEdges);
    }
    graph
}

fn split_indexed_face_set_into_chunks(
    points: &[Vec3],
    coord_index: &[i32],
    max_vertices_per_chunk: usize,
) -> Vec<(Vec<Vec3>, Vec<i32>)> {
    let mut out: Vec<(Vec<Vec3>, Vec<i32>)> = Vec::new();
    let mut chunk_points: Vec<Vec3> = Vec::new();
    let mut chunk_indices: Vec<i32> = Vec::new();
    let mut remap: HashMap<i32, i32> = HashMap::new();
    let mut face: Vec<i32> = Vec::with_capacity(3);

    let flush_chunk = |out: &mut Vec<(Vec<Vec3>, Vec<i32>)>,
                       chunk_points: &mut Vec<Vec3>,
                       chunk_indices: &mut Vec<i32>,
                       remap: &mut HashMap<i32, i32>| {
        if !chunk_indices.is_empty() {
            out.push((std::mem::take(chunk_points), std::mem::take(chunk_indices)));
            remap.clear();
        }
    };

    for &idx in coord_index {
        if idx >= 0 {
            face.push(idx);
            continue;
        }
        if face.len() != 3 {
            face.clear();
            continue;
        }

        let mut new_vertices_needed = 0usize;
        for &src in &face {
            if !remap.contains_key(&src) {
                new_vertices_needed += 1;
            }
        }

        if !chunk_indices.is_empty() && chunk_points.len() + new_vertices_needed > max_vertices_per_chunk {
            flush_chunk(&mut out, &mut chunk_points, &mut chunk_indices, &mut remap);
        }

        for &src in &face {
            let mapped = if let Some(&m) = remap.get(&src) {
                m
            } else {
                let m = chunk_points.len() as i32;
                remap.insert(src, m);
                chunk_points.push(points[src as usize]);
                m
            };
            chunk_indices.push(mapped);
        }
        chunk_indices.push(-1);
        face.clear();
    }

    flush_chunk(&mut out, &mut chunk_points, &mut chunk_indices, &mut remap);

    if out.is_empty() {
        out.push((points.to_vec(), coord_index.to_vec()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_binary_stl(triangles: &[[f32; 12]]) -> Vec<u8> {
        let mut buf = vec![0u8; 84]; // 80 header + 4 count
        let count = triangles.len() as u32;
        buf[80..84].copy_from_slice(&count.to_le_bytes());
        for tri in triangles {
            // 12 floats: nx,ny,nz, v1x,v1y,v1z, v2x,v2y,v2z, v3x,v3y,v3z
            for v in tri {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            buf.extend_from_slice(&[0u8; 2]); // attribute byte count
        }
        buf
    }

    #[test]
    fn test_write_binary_stl_roundtrip() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2, -1];
        let dir = std::env::temp_dir().join("rc3d_stl_roundtrip_test.stl");
        write_binary_stl(&dir, &verts, &indices).expect("write");
        let data = std::fs::read(&dir).expect("read");
        assert_eq!(data.len(), 84 + 50, "one triangle binary STL size");
        assert_eq!(&data[80..84], &1u32.to_le_bytes(), "tri count at byte 80");
        let tris = parse_stl_triangles(&data).expect("parse written STL");
        assert_eq!(tris.len(), 1);
        let _ = std::fs::remove_file(dir);
    }

    #[test]
    fn test_parse_binary_stl_one_triangle() {
        let data = make_binary_stl(&[[
            0.0, 0.0, 1.0,  // normal
            0.0, 0.0, 0.0,  // v1
            1.0, 0.0, 0.0,  // v2
            0.0, 1.0, 0.0,  // v3
        ]]);
        let result = parse_stl(&data);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        let g = result.unwrap();
        assert!(!g.roots().is_empty());
    }

    #[test]
    fn test_parse_binary_stl_empty() {
        let data = make_binary_stl(&[]);
        let result = parse_stl(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_ascii_stl_minimal() {
        let text = "solid test\n\
                    facet normal 0 0 1\n\
                    outer loop\n\
                    vertex 0 0 0\n\
                    vertex 1 0 0\n\
                    vertex 0 1 0\n\
                    endloop\n\
                    endfacet\n\
                    endsolid test\n";
        let result = parse_stl(text.as_bytes());
        assert!(result.is_ok());
    }
}
