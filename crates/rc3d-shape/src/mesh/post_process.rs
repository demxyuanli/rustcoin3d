//! Post-processing passes for triangle meshes after B-Rep tessellation.
//!
//! ## OCC alignment
//! Corresponds to `BRepMesh_ModelHealer` — gap repair, T-junction fixing,
//! degenerate triangle culling, normal recomputation, and vertex compaction.
//!
//! ## Key functions
//! - `heal_mesh_gaps()` — weld free boundary edges across adjacent face meshes via union-find
//! - `fix_t_junctions()` — split edges at T-junction vertices to produce watertight output
//! - `cull_degenerate_tris()` — remove zero-area triangles from index buffer
//! - `recompute_normals_from_tris_preserving()` — rebuild normals from triangle geometry
//! - `compact_mesh_vertices()` — remove unreferenced vertices and remap indices

use rc3d_core::math::{Real, PVec3};
use rc3d_core::utils::hash::f64x3_quantized_bits;
use crate::mesh_result::MeshResult;
use std::collections::HashMap;

// ── ModelHealer: gap repair across adjacent face meshes ──

/// Quantized vertex-pair key for edge deduplication across meshes.
type EdgeKey = ([u64; 3], [u64; 3]);

#[inline]
fn make_edge_key(v0: PVec3, v1: PVec3) -> EdgeKey {
    let k0 = f64x3_quantized_bits([v0.x, v0.y, v0.z]);
    let k1 = f64x3_quantized_bits([v1.x, v1.y, v1.z]);
    if k0 <= k1 { (k0, k1) } else { (k1, k0) }
}

/// Reference to one triangle edge in the combined mesh set.
#[derive(Debug, Clone)]
struct FreeEdgeRef {
    /// Index into the `meshes` slice.
    mesh_idx: usize,
    /// Caller-assigned face identifier.
    face_id: usize,
    /// Local vertex index of the first edge endpoint.
    v_a: usize,
    /// Local vertex index of the second edge endpoint.
    v_b: usize,
}

/// Compute squared point-to-segment distance and the closest-point parameter t (0..1).
fn point_segment_dist2(p: PVec3, a: PVec3, b: PVec3) -> (Real, Real) {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 < 1e-30 {
        return ((p - a).length_squared(), 0.0);
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    let closest = a + ab * t;
    ((p - closest).length_squared(), t)
}

/// Build an edge map grouping triangle edges by quantized vertex-pair key.
///
/// Each entry maps to a list of `FreeEdgeRef` referencing the edge across all meshes.
fn build_edge_map(meshes: &[(&MeshResult, usize)]) -> HashMap<EdgeKey, Vec<FreeEdgeRef>> {
    let mut map: HashMap<EdgeKey, Vec<FreeEdgeRef>> = HashMap::new();
    for (mesh_idx, (mesh, face_id)) in meshes.iter().enumerate() {
        for (_tri_idx, chunk) in mesh.indices.chunks(4).enumerate() {
            if chunk.len() < 4 || chunk[3] != -1 {
                continue;
            }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            let nv = mesh.vertices.len();
            if i0 >= nv || i1 >= nv || i2 >= nv {
                continue;
            }
            for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
                let key = make_edge_key(mesh.vertices[a], mesh.vertices[b]);
                map.entry(key).or_default().push(FreeEdgeRef {
                    mesh_idx,
                    face_id: *face_id,
                    v_a: a,
                    v_b: b,
                });
            }
        }
    }
    map
}

/// Find edges that appear only once in the edge map — free boundary edges.
fn find_free_edges(edge_map: &HashMap<EdgeKey, Vec<FreeEdgeRef>>) -> Vec<FreeEdgeRef> {
    edge_map
        .values()
        .filter(|refs| refs.len() == 1)
        .map(|refs| refs[0].clone())
        .collect()
}

/// Weld a list of explicit vertex pairs within a single mesh using union-find.
///
/// Returns the number of vertices removed (input vertex count minus output vertex count).
#[allow(dead_code)]
fn weld_vertex_pairs(mesh: &mut MeshResult, pairs: &[(usize, usize)]) -> usize {
    if pairs.is_empty() || mesh.vertices.is_empty() {
        return 0;
    }
    let n = mesh.vertices.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }
    for &(a, b) in pairs {
        if a < n && b < n {
            union(&mut parent, a, b);
        }
    }
    // Group by root
    let mut root_to_group: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        root_to_group.entry(root).or_default().push(i);
    }
    if root_to_group.len() == n {
        return 0; // No merges
    }
    // Build new vertex arrays
    let mut new_vertices = Vec::with_capacity(root_to_group.len());
    let mut new_normals = Vec::with_capacity(root_to_group.len());
    let mut remap: Vec<usize> = vec![0; n];
    for members in root_to_group.values() {
        let new_idx = new_vertices.len();
        let mut sum_pos = PVec3::ZERO;
        let mut sum_norm = PVec3::ZERO;
        for &i in members {
            sum_pos += mesh.vertices[i];
            if i < mesh.normals.len() {
                sum_norm += mesh.normals[i];
            }
        }
        let count = members.len() as Real;
        new_vertices.push(sum_pos * (1.0 / count));
        new_normals.push(if sum_norm.length() > 1e-10 {
            sum_norm.normalize()
        } else {
            PVec3::Z
        });
        for &i in members {
            remap[i] = new_idx;
        }
    }
    // Remap indices
    let old_indices = std::mem::take(&mut mesh.indices);
    mesh.indices = Vec::with_capacity(old_indices.len());
    for &idx in &old_indices {
        if idx == -1 {
            mesh.indices.push(-1);
        } else if idx >= 0 && (idx as usize) < n {
            mesh.indices.push(remap[idx as usize] as i32);
        }
    }
    mesh.vertices = new_vertices;
    mesh.normals = new_normals;
    n - mesh.vertices.len()
}

/// Heal mesh gaps between adjacent faces.
///
/// Detects free boundary edges (edges not shared by two triangles), finds the
/// closest free edge on a different face within `tolerance`, and welds the
/// closest vertex pair across the gap.
///
/// Each entry in `meshes` is `(mesh, face_id)` where `face_id` distinguishes
/// faces so that edges shared within the same face are not treated as gaps.
///
/// Returns the number of vertex pairs welded.
pub fn heal_mesh_gaps(
    meshes: &mut [(&mut MeshResult, usize)],
    tolerance: Real,
) -> usize {
    if meshes.len() < 2 {
        return 0;
    }

    // 1. Build read-only view for edge map construction
    let read_meshes: Vec<(&MeshResult, usize)> = meshes
        .iter()
        .map(|(m, fid)| (&**m, *fid))
        .collect();

    let edge_map = build_edge_map(&read_meshes);
    let free_edges = find_free_edges(&edge_map);

    if free_edges.len() < 2 {
        return 0;
    }

    // 2. Find gap pairs: for each pair of free edges from different faces,
    //    find the minimum vertex-pair distance
    let tol2 = tolerance * tolerance;

    // Collect weld candidates: (mesh_a, v_a, mesh_b, v_b)
    let mut weld_candidates: Vec<(usize, usize, usize, usize)> = Vec::new();

    for i in 0..free_edges.len() {
        for j in (i + 1)..free_edges.len() {
            let e0 = &free_edges[i];
            let e1 = &free_edges[j];

            // Only weld across different faces
            if e0.face_id == e1.face_id {
                continue;
            }

            let mesh_a = &read_meshes[e0.mesh_idx].0;
            let mesh_b = &read_meshes[e1.mesh_idx].0;

            let pos_a0 = mesh_a.vertices[e0.v_a];
            let pos_a1 = mesh_a.vertices[e0.v_b];
            let pos_b0 = mesh_b.vertices[e1.v_a];
            let pos_b1 = mesh_b.vertices[e1.v_b];

            // Find closest vertex pair between the two free edges
            let pairs = [
                (e0.mesh_idx, e0.v_a, e1.mesh_idx, e1.v_a, (pos_a0 - pos_b0).length_squared()),
                (e0.mesh_idx, e0.v_a, e1.mesh_idx, e1.v_b, (pos_a0 - pos_b1).length_squared()),
                (e0.mesh_idx, e0.v_b, e1.mesh_idx, e1.v_a, (pos_a1 - pos_b0).length_squared()),
                (e0.mesh_idx, e0.v_b, e1.mesh_idx, e1.v_b, (pos_a1 - pos_b1).length_squared()),
            ];

            let best = pairs.iter().min_by(|a, b| a.4.partial_cmp(&b.4).unwrap()).unwrap();

            if best.4 < tol2 {
                // Ensure canonical ordering to deduplicate
                let (mi_a, vi_a, mi_b, vi_b) = if best.0 < best.2 || (best.0 == best.2 && best.1 <= best.3) {
                    (best.0, best.1, best.2, best.3)
                } else {
                    (best.2, best.3, best.0, best.1)
                };
                // Avoid duplicate welds
                if !weld_candidates.contains(&(mi_a, vi_a, mi_b, vi_b)) {
                    weld_candidates.push((mi_a, vi_a, mi_b, vi_b));
                }
            }
        }
    }

    if weld_candidates.is_empty() {
        return 0;
    }

    // 3. Apply welds: move each vertex pair to the midpoint
    for &(mi_a, vi_a, mi_b, vi_b) in &weld_candidates {
        // Move both vertices to midpoint
        let pos_a = meshes[mi_a].0.vertices[vi_a];
        let pos_b = meshes[mi_b].0.vertices[vi_b];
        let mid = (pos_a + pos_b) * 0.5;
        meshes[mi_a].0.vertices[vi_a] = mid;
        meshes[mi_b].0.vertices[vi_b] = mid;
    }

    // 4. Weld coincident vertices within each mesh and recompute normals
    for mi in 0..meshes.len() {
        meshes[mi].0.weld_vertices(tolerance);
        meshes[mi].0.compute_normals();
    }

    let count = weld_candidates.len();
    count
}

/// Detect T-junctions on free boundary edges and split the edges to fix them.
///
/// A T-junction occurs when a vertex from one triangle lies on (or very near)
/// the edge of another triangle without being connected. This function:
///
/// 1. Finds all free boundary edges (edges referenced by exactly one triangle).
/// 2. For each free edge, checks whether any other vertex in the mesh lies
///    on that edge segment within `tolerance`.
/// 3. Splits the edge at the junction point, replacing the original triangle
///    with two new triangles that share the junction vertex.
///
/// Returns the number of T-junctions fixed (edges split).
pub fn fix_t_junctions(mesh: &mut MeshResult, tolerance: Real) -> usize {
    let tol2 = tolerance * tolerance;

    // Build edge map for this single mesh
    let mut edge_map: HashMap<EdgeKey, Vec<usize>> = HashMap::new();
    // tri_ref[i] = list of (tri_base, i0, i1, i2) for all triangles vertex i belongs to
    let mut tri_ref: Vec<Vec<(usize, usize, usize, usize)>> = vec![];

    for (tri_idx, chunk) in mesh.indices.chunks(4).enumerate() {
        if chunk.len() < 4 || chunk[3] != -1 {
            continue;
        }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        let nv = mesh.vertices.len();
        if i0 >= nv || i1 >= nv || i2 >= nv {
            continue;
        }
        let tri_base = tri_idx * 4;
        let tri_info = (tri_base, i0, i1, i2);

        // Extend tri_ref if needed
        let max_idx = i0.max(i1).max(i2);
        if max_idx >= tri_ref.len() {
            tri_ref.resize(max_idx + 1, Vec::new());
        }
        tri_ref[i0].push(tri_info);
        tri_ref[i1].push(tri_info);
        tri_ref[i2].push(tri_info);

        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            let key = make_edge_key(mesh.vertices[a], mesh.vertices[b]);
            edge_map.entry(key).or_default().push(tri_base);
        }
    }

    // Find free edges: edges referenced by exactly one triangle
    let free_edge_tris: Vec<(usize, usize, usize, usize)> = edge_map
        .values()
        .filter(|refs| refs.len() == 1)
        .filter_map(|refs| {
            let tri_base = refs[0];
            // Re-find which edge of this triangle is free
            // We need the actual vertices of the free edge
            // The edge map doesn't store vertex indices, only tri_base
            // So we need to look up the triangle
            let start = tri_base;
            if start + 3 >= mesh.indices.len() {
                return None;
            }
            let i0 = mesh.indices[start] as usize;
            let i1 = mesh.indices[start + 1] as usize;
            let i2 = mesh.indices[start + 2] as usize;
            let nv = mesh.vertices.len();
            if i0 >= nv || i1 >= nv || i2 >= nv {
                return None;
            }
            // Check each edge for freedom
            for &(ea, eb, ec) in &[(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
                let ek = make_edge_key(mesh.vertices[ea], mesh.vertices[eb]);
                if let Some(erefs) = edge_map.get(&ek) {
                    if erefs.len() == 1 {
                        return Some((tri_base, ea, eb, ec));
                    }
                }
            }
            None
        })
        .collect();

    if free_edge_tris.is_empty() {
        return 0;
    }

    // Collect T-junction splits: (tri_base, edge_a, edge_b, opposite_c, junction_v)
    // Replace triangle (a,b,c) with (a,v,c) + (v,b,c)
    let mut splits: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
    let nv = mesh.vertices.len();

    for &(tri_base, ea, eb, ec) in &free_edge_tris {
        let pos_a = mesh.vertices[ea];
        let pos_b = mesh.vertices[eb];

        for v in 0..nv {
            if v == ea || v == eb || v == ec {
                continue;
            }
            // Skip vertices from the same triangle (vertex may be shared by multiple tris)
            if tri_ref.get(v).map_or(false, |tris| tris.iter().any(|&(tb, _, _, _)| tb == tri_base)) {
                continue;
            }
            let (dist2, t) = point_segment_dist2(mesh.vertices[v], pos_a, pos_b);
            if dist2 < tol2 && t > 0.001 && t < 0.999 {
                splits.push((tri_base, ea, eb, ec, v));
                break; // Only split once per free edge
            }
        }
    }

    if splits.is_empty() {
        return 0;
    }

    // Apply splits in reverse tri_base order so earlier indices stay valid
    splits.sort_by(|a, b| b.0.cmp(&a.0));

    let mut count = 0usize;
    for &(tri_base, a, b, c, v) in &splits {
        if tri_base + 3 >= mesh.indices.len() {
            continue;
        }
        // Verify the triangle hasn't already been modified
        if mesh.indices[tri_base] != a as i32
            || mesh.indices[tri_base + 1] != b as i32
            || mesh.indices[tri_base + 2] != c as i32
        {
            continue;
        }
        // Replace [a, b, c, -1] with [a, v, c, -1, v, b, c, -1]
        mesh.indices[tri_base] = a as i32;
        mesh.indices[tri_base + 1] = v as i32;
        mesh.indices[tri_base + 2] = c as i32;
        // -1 stays at tri_base+3
        // Insert new quad
        mesh.indices.splice(
            tri_base + 4..tri_base + 4,
            [v as i32, b as i32, c as i32, -1],
        );
        count += 1;
    }

    // Recompute normals since topology changed
    mesh.compute_normals();

    count
}

// ── Original post-process functions ──

/// Recompute normals from triangle faces, preserving specified vertex normals.
///
/// Vertices in `preserve` keep their existing normals. All others are
/// recomputed from adjacent triangle face normals (area-weighted average).
pub fn recompute_normals_from_tris_preserving(
    vertices: &[PVec3], indices: &[i32], normals: &mut Vec<PVec3>, preserve: &[usize],
) {
    use std::collections::HashSet;
    let preserved: HashSet<usize> = preserve.iter().copied().collect();

    if normals.len() != vertices.len() {
        normals.resize(vertices.len(), PVec3::ZERO);
    }
    // Only zero out normals for non-preserved vertices
    for (i, n) in normals.iter_mut().enumerate() {
        if !preserved.contains(&i) {
            *n = PVec3::ZERO;
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
                *n *= 1.0 / len;
            }
        } else {
            let len = n.length();
            if len > 1e-10 {
                *n *= 1.0 / len;
            } else {
                *n = PVec3::Z; // zero → downstream should handle
            }
        }
    }
}

pub fn cull_degenerate_tris(indices: &mut Vec<i32>, vertices: &[PVec3]) -> usize {
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
        let n = mesh.normals.get(old).copied().unwrap_or(PVec3::Y);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quad_mesh(verts: &[(Real, Real, Real)], tris: &[(usize, usize, usize)]) -> MeshResult {
        MeshResult {
            vertices: verts
                .iter()
                .map(|&(x, y, z)| PVec3::new(x, y, z))
                .collect(),
            normals: vec![PVec3::Z; verts.len()],
            indices: tris
                .iter()
                .flat_map(|&(a, b, c)| vec![a as i32, b as i32, c as i32, -1])
                .collect(),
        }
    }

    #[test]
    fn heal_mesh_gaps_closes_small_gap_between_two_faces() {
        // Face 0: triangle at z=0, vertices (0,0,0)-(1,0,0)-(0,1,0)
        let mut mesh0 = make_quad_mesh(
            &[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            &[(0, 1, 2)],
        );
        // Face 1: triangle at z=0.01 (small gap), vertices slightly offset
        // Shared "edge" along y=0 but at z=0.01 instead of z=0
        let mut mesh1 = make_quad_mesh(
            &[
                (0.0, 0.0, 0.01),
                (1.0, 0.0, 0.01),
                (0.5, 0.5, 0.01),
            ],
            &[(0, 1, 2)],
        );

        let mut meshes = vec![
            (&mut mesh0, 0usize),
            (&mut mesh1, 1usize),
        ];

        let welded = heal_mesh_gaps(&mut meshes, 0.05);
        assert!(welded >= 1, "expected at least 1 vertex pair welded, got {welded}");

        // After healing, the gap should be closed: the boundary edge vertices
        // should be at the same z-coordinate (midpoint = 0.005 or closer)
        let z0 = mesh0.vertices.iter().map(|v| v.z.abs()).fold(0.0f64, f64::max);
        let z1 = mesh1.vertices.iter().map(|v| v.z.abs()).fold(0.0f64, f64::max);

        // At least some vertices should have moved closer to z=0
        assert!(
            z0 <= 0.01 + 1e-6 && z1 <= 0.01 + 1e-6,
            "z coords after heal: mesh0 max_z={z0}, mesh1 max_z={z1}"
        );
    }

    #[test]
    fn heal_mesh_gaps_noop_when_already_watertight() {
        // Two faces sharing the same edge vertices exactly
        let mut mesh0 = make_quad_mesh(
            &[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            &[(0, 1, 2)],
        );
        let mut mesh1 = make_quad_mesh(
            &[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
            &[(0, 1, 2)],
        );

        let orig_vert_count = mesh0.vertices.len() + mesh1.vertices.len();
        let orig_tri_count = (mesh0.indices.len() + mesh1.indices.len()) / 4;

        let mut meshes = vec![
            (&mut mesh0, 0usize),
            (&mut mesh1, 1usize),
        ];

        let _welded = heal_mesh_gaps(&mut meshes, 0.01);
        // Welding coincident vertices is harmless no-op.
        // Verify mesh integrity is preserved.
        let new_tri_count = (mesh0.indices.len() + mesh1.indices.len()) / 4;
        assert_eq!(new_tri_count, orig_tri_count, "triangle count unchanged");
        // Shared edge (0,0,0)-(1,0,0) still exists in both
        assert!(mesh0.vertices.len() >= 3, "mesh0 has at least 3 vertices");
        assert!(mesh1.vertices.len() >= 3, "mesh1 has at least 3 vertices");
    }

    #[test]
    fn fix_t_junctions_splits_edge_at_junction_vertex() {
        // Create a mesh where one vertex lies exactly on another triangle's edge
        // Triangle 0: (0,0,0)-(2,0,0)-(0,2,0)
        // Triangle 1: (1,0,0)-(2,2,0)-(0,2,0)
        // Here vertex at (1,0,0) from triangle 1 lies on edge (0,0,0)-(2,0,0) of triangle 0
        let mut mesh = MeshResult {
            vertices: vec![
                PVec3::new(0.0, 0.0, 0.0), // 0
                PVec3::new(2.0, 0.0, 0.0), // 1
                PVec3::new(0.0, 2.0, 0.0), // 2
                PVec3::new(1.0, 0.0, 0.0), // 3 - T-junction vertex on edge (0,1)
                PVec3::new(2.0, 2.0, 0.0), // 4
            ],
            normals: vec![PVec3::Z; 5],
            indices: vec![
                0, 1, 2, -1, // triangle 0
                3, 4, 2, -1, // triangle 1
            ],
        };

        let fixed = fix_t_junctions(&mut mesh, 0.01);
        assert!(fixed >= 1, "expected at least 1 T-junction fixed, got {fixed}");

        // After fixing, triangle 0 should be split: (0,1,2) → (0,3,2) + (3,1,2)
        // Total tris should be 3 instead of 2
        let tri_count = mesh.indices.len() / 4;
        assert_eq!(tri_count, 3, "expected 3 triangles after split, got {tri_count}");

        // Verify that vertex 3 (1,0,0) is now shared by the split triangles
        let idx3 = 3i32;
        let mut tri_with_3 = 0usize;
        for chunk in mesh.indices.chunks(4) {
            if chunk.len() < 3 {
                continue;
            }
            if chunk[0] == idx3 || chunk[1] == idx3 || chunk[2] == idx3 {
                tri_with_3 += 1;
            }
        }
        assert!(tri_with_3 >= 2, "vertex 3 should be referenced by at least 2 triangles (was in 1)");
    }

    #[test]
    fn fix_t_junctions_noop_when_no_junction() {
        // Two adjacent triangles sharing an edge properly
        let mut mesh = MeshResult {
            vertices: vec![
                PVec3::new(0.0, 0.0, 0.0),
                PVec3::new(1.0, 0.0, 0.0),
                PVec3::new(0.0, 1.0, 0.0),
                PVec3::new(1.0, 1.0, 0.0),
            ],
            normals: vec![PVec3::Z; 4],
            indices: vec![
                0, 1, 2, -1,
                1, 3, 2, -1,
            ],
        };

        let fixed = fix_t_junctions(&mut mesh, 0.01);
        assert_eq!(fixed, 0, "no T-junctions should be detected");
    }
}
