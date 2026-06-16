//! Path A: SceneGraph → STEP (mesh export with coplanar face merging).
//!
//! Algorithm:
//! 1. Walk SceneGraph collecting per-Separator meshes (Coordinate3 + IndexedFaceSet)
//! 2. Group triangles by quantized face normal
//! 3. Union-find on shared-edge adjacency within each group
//! 4. Extract outer boundary loop per connected component
//! 5. Emit STEP entities → ISO 10303-21 text

use std::collections::HashMap;
use rc3d_core::math::{Real, PVec3;
use rc3d_scene::{NodeData, SceneGraph};
use super::format::format_header;

/// A triangle reference: three vertex indices into a flat vertex list.
type Triangle = [usize; 3];

/// A merged face: boundary polygon as vertex indices, plus normal.
struct MergedFace {
    normal: PVec3,
    boundary: Vec<usize>, // CCW vertex indices around outer loop
}

/// Edge key for adjacency detection (sorted pair of vertex indices).
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

pub fn write_scene(graph: &SceneGraph) -> Result<String, String> {
    // ── Phase 1: Collect all mesh data ──
    let mut all_verts: Vec<PVec3> = Vec::new();
    let mut all_tris: Vec<Triangle> = Vec::new();
    // Track per-component vertex ranges for assembly hierarchy
    let mut components: Vec<(Option<String>, usize, usize)> = Vec::new();
    // (name, vert_start, vert_end)

    let roots = graph.roots().to_vec();
    for &root in &roots {
        collect_mesh_data(graph, root, &mut all_verts, &mut all_tris, &mut components);
        // Entry for root-level geometry
        let vert_start = components.last().map(|c| c.2).unwrap_or(0);
        if all_verts.len() > vert_start {
            components.push((None, vert_start, all_verts.len()));
        }
    }

    // Recalculate components properly
    components.clear();
    for &root in &roots {
        collect_mesh_data(graph, root, &mut all_verts, &mut all_tris, &mut components);
        // Also collect from children
        if let Some(entry) = graph.get(root) {
            for &child in &entry.children {
                collect_mesh_data(graph, child, &mut all_verts, &mut all_tris, &mut components);
            }
        }
    }

    if all_verts.is_empty() || all_tris.is_empty() {
        return Err("No mesh geometry found in SceneGraph".into());
    }

    // ── Phase 2: Merge coplanar triangles ──
    let faces = merge_coplanar_triangles(&all_verts, &all_tris);
    if faces.is_empty() {
        return Err("No faces produced after merging".into());
    }

    // ── Phase 3: Emit STEP entities ──
    let mut out = String::new();
    out.push_str(&format_header("AP203_CONFIGURATION_CONTROLLED_3D_DESIGN_OF_MECHANICAL_PARTS_AND_ASSEMBLIES_MIM_LF"));
    out.push_str("DATA;\n");

    let mut next_id: u64 = 1;

    // Helper macro to allocate an ID and format an entity line
    macro_rules! emit {
        ($out:expr, $id:expr, $name:expr, $params:expr) => {{
            let eid = $id;
            $id += 1;
            $out.push_str(&format!("#{} = {}({});\n", eid, $name, $params));
            eid
        }};
    }

    // Unique vertices → CARTESIAN_POINT
    let mut vert_ids: Vec<u64> = Vec::with_capacity(all_verts.len());
    let mut vert_map: HashMap<[u32; 3], u64> = HashMap::new();
    for v in &all_verts {
        let key = rc3d_core::utils::hash::f64x3_quantized_bits([v.x, v.y, v.z]);
        if let Some(&id) = vert_map.get(&key) {
            vert_ids.push(id);
        } else {
            let id = emit!(out, next_id, "CARTESIAN_POINT",
                format!("'',({:.6},{:.6},{:.6})", v.x, v.y, v.z));
            vert_map.insert(key, id);
            vert_ids.push(id);
        }
    }

    // For each merged face: AXIS2_PLACEMENT_3D, PLANE, EDGE_LOOP edges, FACE_OUTER_BOUND, ADVANCED_FACE
    let mut face_bound_ids: Vec<u64> = Vec::new();
    for face in &faces {
        let boundary = &face.boundary;
        if boundary.len() < 3 { continue; }

        // AXIS2_PLACEMENT_3D: origin = first vertex, Z = face normal, X = first edge direction
        let x_dir = (all_verts[boundary[1]] - all_verts[boundary[0]]).normalize();
        let (z_dir_id, z_dir_text) = make_dir(&mut next_id, face.normal);
        let (x_dir_id, x_dir_text) = make_dir(&mut next_id, x_dir);
        out.push_str(&z_dir_text);
        out.push_str(&x_dir_text);
        let axis_params = format!("'',#{},#{},#{}", vert_ids[boundary[0]], z_dir_id, x_dir_id);
        let axis_id = emit!(out, next_id, "AXIS2_PLACEMENT_3D", axis_params);

        // PLANE
        let plane_id = emit!(out, next_id, "PLANE", format!("'',#{}", axis_id));

        // LINE + EDGE_CURVE per boundary edge
        let mut edge_ids: Vec<u64> = Vec::new();
        let n = boundary.len();
        for i in 0..n {
            let v0 = boundary[i];
            let v1 = boundary[(i + 1) % n];
            let (dir_id, dir_text) = make_dir(&mut next_id, all_verts[v1] - all_verts[v0]);
            out.push_str(&dir_text);
            let line_id = emit!(out, next_id, "LINE",
                format!("'',#{},#{}", vert_ids[v0], dir_id));
            let ec_id = emit!(out, next_id, "EDGE_CURVE",
                format!("'',#{},#{},#{},.T.", vert_ids[v0], vert_ids[v1], line_id));
            edge_ids.push(ec_id);
        }

        // EDGE_LOOP
        let edge_list: Vec<String> = edge_ids.iter().map(|id| format!("#{}", id)).collect();
        let loop_id = emit!(out, next_id, "EDGE_LOOP",
            format!("'',({})", edge_list.join(",")));

        // FACE_OUTER_BOUND
        let fob_id = emit!(out, next_id, "FACE_OUTER_BOUND",
            format!("'',#{},.T.", loop_id));

        // ADVANCED_FACE: include plane reference for proper B-rep
        let face_id = emit!(out, next_id, "ADVANCED_FACE",
            format!("'',(#{}),#{},.T.", fob_id, plane_id));
        face_bound_ids.push(face_id);
    }

    // CLOSED_SHELL
    let face_list: Vec<String> = face_bound_ids.iter().map(|id| format!("#{}", id)).collect();
    let shell_id = emit!(out, next_id, "CLOSED_SHELL",
        format!("'',({})", face_list.join(",")));

    // MANIFOLD_SOLID_BREP
    out.push_str(&format!(
        "#{} = MANIFOLD_SOLID_BREP('',#{});\n",
        next_id, shell_id
    ));

    out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
    Ok(out)
}

/// Allocate a DIRECTION entity ID and return (id, formatted entity text).
fn make_dir(next_id: &mut u64, v: PVec3) -> (u64, String) {
    let id = *next_id;
    *next_id += 1;
    (id, format!("#{} = DIRECTION('',({:.6},{:.6},{:.6}));\n", id, v.x, v.y, v.z))
}

/// Merge coplanar triangles into polygonal faces.
fn merge_coplanar_triangles(verts: &[PVec3], tris: &[Triangle]) -> Vec<MergedFace> {
    // Step 1: Compute per-triangle normal, group by quantized normal
    let mut normals: Vec<PVec3> = Vec::with_capacity(tris.len());
    let mut groups: HashMap<[i32; 3], Vec<usize>> = HashMap::new();

    for (ti, tri) in tris.iter().enumerate() {
        let v0 = verts[tri[0]];
        let v1 = verts[tri[1]];
        let v2 = verts[tri[2]];
        let n = (v1 - v0).cross(v2 - v0);
        if n.length() < 1e-10 { continue; } // degenerate
        let n = n.normalize();
        // Quantize to ~0.001 radian precision
        let key = [
            (n.x * 1000.0).round() as i32,
            (n.y * 1000.0).round() as i32,
            (n.z * 1000.0).round() as i32,
        ];
        normals.push(n);
        groups.entry(key).or_default().push(ti);
    }

    // Step 2: Within each group, union-find on shared-edge adjacency
    let mut parent: Vec<usize> = (0..tris.len()).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        if p[x] != x { p[x] = find(p, p[x]); }
        p[x]
    }
    fn union(p: &mut [usize], a: usize, b: usize) {
        let ra = find(p, a);
        let rb = find(p, b);
        if ra != rb { p[ra] = rb; }
    }

    for indices in groups.values() {
        // Build edge → triangle map within this group
        let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for &ti in indices {
            let tri = tris[ti];
            for i in 0..3 {
                let key = edge_key(tri[i], tri[(i + 1) % 3]);
                edge_tris.entry(key).or_default().push(ti);
            }
        }
        // Union triangles that share an edge
        for tris_set in edge_tris.values() {
            if tris_set.len() >= 2 {
                for w in tris_set.windows(2) {
                    union(&mut parent, w[0], w[1]);
                }
            }
        }
    }

    // Step 3: Build merged faces (one per connected component)
    let mut root_to_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for ti in 0..tris.len() {
        if ti < normals.len() {
            let r = find(&mut parent, ti);
            root_to_indices.entry(r).or_default().push(ti);
        }
    }

    let mut faces: Vec<MergedFace> = Vec::new();
    for tris_set in root_to_indices.values() {
        if tris_set.is_empty() { continue; }
        let normal = normals[*tris_set.first().unwrap()];
        let boundary = extract_boundary(verts, tris, tris_set);
        if boundary.len() >= 3 {
            faces.push(MergedFace { normal, boundary });
        }
    }

    faces
}

/// Extract the outer boundary polygon from a set of connected triangles.
/// Uses the edge-count method: interior edges appear twice, boundary edges appear once.
fn extract_boundary(_verts: &[PVec3], tris: &[Triangle], tri_indices: &[usize]) -> Vec<usize> {
    let mut edge_count: HashMap<(usize, usize), u32> = HashMap::new();
    for &ti in tri_indices {
        let tri = tris[ti];
        for i in 0..3 {
            let key = edge_key(tri[i], tri[(i + 1) % 3]);
            *edge_count.entry(key).or_default() += 1;
        }
    }

    // Collect boundary edges (count == 1)
    let mut boundary_edges: Vec<(usize, usize)> = edge_count.iter()
        .filter(|(_, &count)| count == 1)
        .map(|(&key, _)| key)
        .collect();

    if boundary_edges.len() < 3 { return vec![]; }

    // Chain boundary edges into a loop
    let mut boundary: Vec<usize> = Vec::new();
    let (first, mut current) = boundary_edges[0];
    boundary.push(first);
    boundary.push(current);
    boundary_edges.remove(0);

    while !boundary_edges.is_empty() && current != first {
        let mut found = false;
        for i in 0..boundary_edges.len() {
            let (a, b) = boundary_edges[i];
            if a == current {
                current = b;
                boundary.push(current);
                boundary_edges.remove(i);
                found = true;
                break;
            } else if b == current {
                current = a;
                boundary.push(current);
                boundary_edges.remove(i);
                found = true;
                break;
            }
        }
        if !found { break; }
    }

    // Remove duplicate closing vertex
    if boundary.len() > 1 && boundary[0] == boundary[boundary.len() - 1] {
        boundary.pop();
    }

    boundary
}

/// Recursively collect vertex data and triangle indices from the scene graph.
fn collect_mesh_data(
    graph: &SceneGraph,
    node_id: rc3d_core::NodeId,
    vertices: &mut Vec<PVec3>,
    triangles: &mut Vec<Triangle>,
    _components: &mut Vec<(Option<String>, usize, usize)>,
) {
    let entry = match graph.get(node_id) {
        Some(e) => e,
        None => return,
    };

    if let NodeData::Coordinate3(coord) = &entry.data {
        let base_idx = vertices.len();
        for pt in &coord.point {
            vertices.push(*pt);
        }
        // Look for sibling IndexedFaceSet
        for &child in &entry.children {
            if let Some(child_entry) = graph.get(child) {
                if let NodeData::IndexedFaceSet(ifs) = &child_entry.data {
                    // Parse indices: triples separated by -1
                    let mut i = 0;
                    while i + 2 < ifs.coord_index.len() {
                        let a = ifs.coord_index[i] as usize;
                        let b = ifs.coord_index[i + 1] as usize;
                        let c = ifs.coord_index[i + 2] as usize;
                        if a < coord.point.len() && b < coord.point.len() && c < coord.point.len() {
                            triangles.push([base_idx + a, base_idx + b, base_idx + c]);
                        }
                        i += 3;
                        // Skip sentinel
                        while i < ifs.coord_index.len() && ifs.coord_index[i] == -1 {
                            i += 1;
                        }
                    }
                }
            }
        }
    }

    // Recurse into children
    let children = entry.children.clone();
    for child in &children {
        collect_mesh_data(graph, *child, vertices, triangles, _components);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_coplanar_quad() {
        // Two coplanar triangles forming a square
        let verts = vec![
            PVec3::new(0.0, 0.0, 0.0),
            PVec3::new(1.0, 0.0, 0.0),
            PVec3::new(1.0, 1.0, 0.0),
            PVec3::new(0.0, 1.0, 0.0),
        ];
        let tris = vec![
            [0, 1, 2],
            [0, 2, 3],
        ];
        let faces = merge_coplanar_triangles(&verts, &tris);
        assert_eq!(faces.len(), 1, "two coplanar triangles should merge into one face");
        assert_eq!(faces[0].boundary.len(), 4, "merged face should have 4 boundary vertices");
    }

    #[test]
    fn test_merge_non_coplanar() {
        // Two non-coplanar triangles (different normals)
        let verts = vec![
            PVec3::new(0.0, 0.0, 0.0),
            PVec3::new(1.0, 0.0, 0.0),
            PVec3::new(1.0, 1.0, 0.0),
            PVec3::new(0.0, 0.0, 1.0),
        ];
        let tris = vec![
            [0, 1, 2],  // XY plane
            [0, 1, 3],  // tilted
        ];
        let faces = merge_coplanar_triangles(&verts, &tris);
        assert_eq!(faces.len(), 2, "non-coplanar triangles should stay separate");
    }

    #[test]
    fn test_extract_boundary_square() {
        let verts = vec![
            PVec3::new(0.0, 0.0, 0.0),
            PVec3::new(1.0, 0.0, 0.0),
            PVec3::new(1.0, 1.0, 0.0),
            PVec3::new(0.0, 1.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        let boundary = extract_boundary(&verts, &tris, &[0, 1]);
        assert_eq!(boundary.len(), 4);
    }
}
