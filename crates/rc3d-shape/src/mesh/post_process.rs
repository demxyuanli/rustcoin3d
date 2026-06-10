use rc3d_core::math::Vec3;
use crate::mesh_result::MeshResult;

/// Recompute normals from triangle faces, preserving specified vertex normals.
///
/// Vertices in `preserve` keep their existing normals. All others are
/// recomputed from adjacent triangle face normals (area-weighted average).
pub fn recompute_normals_from_tris_preserving(
    vertices: &[Vec3], indices: &[i32], normals: &mut Vec<Vec3>, preserve: &[usize],
) {
    use std::collections::HashSet;
    let preserved: HashSet<usize> = preserve.iter().copied().collect();

    if normals.len() != vertices.len() {
        normals.resize(vertices.len(), Vec3::ZERO);
    }
    // Only zero out normals for non-preserved vertices
    for (i, n) in normals.iter_mut().enumerate() {
        if !preserved.contains(&i) {
            *n = Vec3::ZERO;
        }
    }
    for chunk in indices.chunks(4) {
        if chunk.len() < 4 || chunk[3] != -1 {
            continue;
        }
        let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let n = (vertices[i1] - vertices[i0]).cross(vertices[i2] - vertices[i0]);
        if n.length_squared() < 1e-20 {
            continue;
        }
        // Only accumulate for non-preserved vertices
        if !preserved.contains(&i0) { normals[i0] += n; }
        if !preserved.contains(&i1) { normals[i1] += n; }
        if !preserved.contains(&i2) { normals[i2] += n; }
    }
    for (i, n) in normals.iter_mut().enumerate() {
        if preserved.contains(&i) {
            // Ensure preserved normals are unit length
            let len = n.length();
            if len > 1e-10 {
                *n *= (1.0 / len);
            }
        } else {
            let len = n.length();
            if len > 1e-10 {
                *n *= (1.0 / len);
            } else {
                *n = Vec3::Z; // zero → downstream should handle
            }
        }
    }
}

pub fn cull_degenerate_tris(indices: &mut Vec<i32>, vertices: &[Vec3]) -> usize {
    let mut out = Vec::with_capacity(indices.len());
    let mut removed = 0usize;
    for chunk in indices.chunks(4) {
        if chunk.len() < 4 || chunk[3] != -1 {
            out.extend_from_slice(chunk);
            continue;
        }
        let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            out.extend_from_slice(chunk);
            continue;
        }
        let area = (vertices[i0] - vertices[i1])
            .cross(vertices[i0] - vertices[i2])
            .length();
        if area > 1e-12 {
            out.extend_from_slice(chunk);
        } else {
            removed += 1;
        }
    }
    *indices = out;
    removed
}

/// Drop vertices not referenced by any triangle and remap indices (keeps STL/export clean).
pub fn compact_mesh_vertices(mesh: &mut MeshResult) {
    let mut used = vec![false; mesh.vertices.len()];
    for chunk in mesh.indices.chunks(4) {
        if chunk.len() < 3 {
            continue;
        }
        for k in 0..3 {
            let gi = chunk[k];
            if gi >= 0 && (gi as usize) < used.len() {
                used[gi as usize] = true;
            }
        }
    }
    let live: usize = used.iter().filter(|&&u| u).count();
    if live == mesh.vertices.len() {
        return;
    }
    let mut remap = vec![-1i32; mesh.vertices.len()];
    let mut new_verts = Vec::with_capacity(live);
    let mut new_normals = Vec::with_capacity(live);
    for (old, &is_used) in used.iter().enumerate() {
        if !is_used {
            continue;
        }
        remap[old] = new_verts.len() as i32;
        new_verts.push(mesh.vertices[old]);
        let n = mesh.normals.get(old).copied().unwrap_or(Vec3::Y);
        new_normals.push(n);
    }
    for idx in mesh.indices.iter_mut() {
        if *idx >= 0 {
            *idx = remap[*idx as usize];
        }
    }
    mesh.vertices = new_verts;
    mesh.normals = new_normals;
}
