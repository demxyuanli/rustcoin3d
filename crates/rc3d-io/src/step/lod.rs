//! Level-of-detail generation via edge collapse simplification.

use rc3d_core::math::Real;
use rc3d_core::math::PVec3;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// A candidate edge collapse, ordered by cost (quadric error).
#[derive(Debug, Clone)]
struct CollapseCandidate {
    v0: usize,
    v1: usize,
    cost: Real,
}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits()
    }
}
impl Eq for CollapseCandidate {}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cost.partial_cmp(&other.cost).map(|o| o.reverse())
    }
}
impl Ord for CollapseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Simplify a mesh to approximately `target_vertices` count using edge collapse.
/// Returns simplified (vertices, indices) or None if already simple enough.
pub fn simplify_mesh(
    vertices: &[PVec3],
    indices: &[i32],
    target_vertices: usize,
) -> Option<(Vec<PVec3>, Vec<i32>)> {
    if vertices.len() <= target_vertices || target_vertices < 3 {
        return None;
    }

    let mut verts = vertices.to_vec();
    let idx = indices.to_vec();

    // Build edge list with collapse costs
    let mut heap = BinaryHeap::new();
    for chunk in indices.chunks(4) {
        if chunk.len() < 3 { continue; }
        for (a, b) in [(chunk[0] as usize, chunk[1] as usize),
                       (chunk[1] as usize, chunk[2] as usize),
                       (chunk[2] as usize, chunk[0] as usize)] {
            if a < verts.len() && b < verts.len() && a < b {
                let cost = (verts[a] - verts[b]).length();
                heap.push(CollapseCandidate { v0: a, v1: b, cost });
            }
        }
    }

    // Collapse until target reached
    let mut remap: Vec<usize> = (0..verts.len()).collect();
    let mut removed = 0usize;
    let target = verts.len() - target_vertices;

    while removed < target {
        let candidate = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        // Both vertices still valid (not already collapsed)?
        let r0 = find_root(&remap, candidate.v0);
        let r1 = find_root(&remap, candidate.v1);
        if r0 == r1 { continue; }

        // Collapse v1 into v0
        let mid = (verts[r0] + verts[r1]) * 0.5;
        verts[r0] = mid;
        remap[r1] = r0;
        removed += 1;
    }

    // Compact vertices and remap indices
    let mut compact_idx = 0usize;
    let mut new_idx: Vec<Option<usize>> = vec![None; verts.len()];
    let mut new_verts = Vec::new();

    for i in 0..verts.len() {
        let r = find_root(&remap, i);
        if let Some(&ci) = new_idx.get(r).and_then(|o| o.as_ref()) {
            // already mapped
            let _ = ci;
        } else if new_idx[r].is_none() {
            new_idx[r] = Some(compact_idx);
            new_verts.push(verts[r]);
            compact_idx += 1;
        }
    }

    // Build final index lookup: old idx → new compact idx
    let final_idx: Vec<usize> = (0..verts.len())
        .map(|i| new_idx[find_root(&remap, i)].unwrap_or(0))
        .collect();

    // Remap indices, deduplicate degenerate triangles
    let mut new_indices = Vec::new();
    for chunk in idx.chunks(4) {
        if chunk.len() < 3 { continue; }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        if i0 >= final_idx.len() || i1 >= final_idx.len() || i2 >= final_idx.len() { continue; }
        let a = final_idx[i0] as i32;
        let b = final_idx[i1] as i32;
        let c = final_idx[i2] as i32;
        if a == b || b == c || a == c { continue; } // skip degenerate
        new_indices.extend_from_slice(&[a, b, c, -1]);
    }

    Some((new_verts, new_indices))
}

fn find_root(parent: &[usize], x: usize) -> usize {
    if x >= parent.len() { return x; }
    let mut cur = x;
    while parent[cur] != cur {
        cur = parent[cur];
    }
    cur
}

/// A single LOD level mesh.
#[derive(Debug, Clone)]
pub struct LodLevel {
    pub vertices: Vec<PVec3>,
    pub normals: Vec<PVec3>,
    pub indices: Vec<i32>,
    pub vertex_count: usize,
    pub triangle_count: usize,
}

/// Multi-level LOD mesh with transition distances.
#[derive(Debug, Clone)]
pub struct LodMesh {
    /// Mesh levels from highest to lowest detail.
    pub levels: Vec<LodLevel>,
    /// Camera distances at which to transition between levels.
    pub transitions: Vec<Real>,
}

/// Generate multi-level LOD from a high-detail mesh.
///
/// # Arguments
/// * `vertices` - High-detail vertices
/// * `indices` - High-detail indices (chunks of 4: i0,i1,i2,-1)
/// * `levels` - Number of LOD levels (e.g., 3 = high/medium/low)
/// * `reduction_ratios` - Vertex reduction ratio per level (e.g., [1.0, 0.5, 0.1])
pub fn generate_lod(
    vertices: &[PVec3],
    indices: &[i32],
    levels: usize,
    reduction_ratios: &[Real],
) -> LodMesh {
    let levels = levels.max(1);
    let mut lod = LodMesh {
        levels: Vec::with_capacity(levels),
        transitions: Vec::new(),
    };

    // Level 0: original mesh
    lod.levels.push(LodLevel {
        vertex_count: vertices.len(),
        triangle_count: indices.len() / 4,
        vertices: vertices.to_vec(),
        normals: Vec::new(),
        indices: indices.to_vec(),
    });

    // Subsequent levels: progressively simplified
    for i in 1..levels {
        let ratio = reduction_ratios.get(i).copied().unwrap_or(0.5_f64.powi(i as i32));
        let target = (vertices.len() as Real * ratio) as usize;
        let target = target.max(12); // Minimum: 4 triangles

        if let Some((simplified_v, simplified_i)) = simplify_mesh(vertices, indices, target) {
            lod.levels.push(LodLevel {
                vertex_count: simplified_v.len(),
                triangle_count: simplified_i.len() / 4,
                vertices: simplified_v,
                normals: Vec::new(),
                indices: simplified_i,
            });
        } else {
            // Can't simplify further — duplicate last level
            if let Some(last) = lod.levels.last() {
                lod.levels.push(last.clone());
            }
        }
    }

    // Compute transition distances based on bounding sphere
    let bbox_diag = compute_bbox_diagonal(vertices);
    for i in 0..levels.saturating_sub(1) {
        let dist = bbox_diag * 2.0_f64.powi(i as i32 + 1);
        lod.transitions.push(dist);
    }

    lod
}

/// Compute the bounding box diagonal length for a vertex set.
fn compute_bbox_diagonal(vertices: &[PVec3]) -> Real {
    if vertices.is_empty() { return 1.0; }
    let mut min = vertices[0];
    let mut max = vertices[0];
    for v in vertices {
        min = PVec3::new(min.x.min(v.x), min.y.min(v.y), min.z.min(v.z));
        max = PVec3::new(max.x.max(v.x), max.y.max(v.y), max.z.max(v.z));
    }
    (max - min).length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_already_small() {
        let verts = vec![
            PVec3::new(0.0, 0.0, 0.0),
            PVec3::new(1.0, 0.0, 0.0),
            PVec3::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2, -1];
        // Target >= current count → no simplification needed
        assert!(simplify_mesh(&verts, &indices, 10).is_none());
    }

    #[test]
    fn test_simplify_reduces_count() {
        let verts: Vec<PVec3> = (0..20).map(|i| {
            PVec3::new(i as Real, 0.0, 0.0)
        }).collect();
        let mut indices = Vec::new();
        for i in 0..18 {
            indices.extend_from_slice(&[i, i + 1, i + 2, -1]);
        }
        let result = simplify_mesh(&verts, &indices, 10);
        assert!(result.is_some());
        let (new_v, _) = result.unwrap();
        assert!(new_v.len() <= 10);
    }

    fn generate_test_mesh(n: usize) -> (Vec<PVec3>, Vec<i32>) {
        let side = (n as Real).sqrt().ceil() as usize;
        let mut verts = Vec::new();
        for i in 0..=side {
            for j in 0..=side {
                verts.push(PVec3::new(i as Real, j as Real, 0.0));
            }
        }
        let mut indices = Vec::new();
        let w = side + 1;
        for i in 0..side {
            for j in 0..side {
                let a = i * w + j;
                let b = a + 1;
                let c = a + w;
                let d = c + 1;
                indices.extend_from_slice(&[a as i32, b as i32, d as i32, -1]);
                indices.extend_from_slice(&[a as i32, d as i32, c as i32, -1]);
            }
        }
        (verts, indices)
    }

    #[test]
    fn test_generate_lod_3_levels() {
        let (verts, idx) = generate_test_mesh(400);
        let lod = generate_lod(&verts, &idx, 3, &[1.0, 0.5, 0.1]);
        assert_eq!(lod.levels.len(), 3);
        assert!(lod.levels[0].vertex_count >= lod.levels[1].vertex_count);
        assert!(lod.levels[1].vertex_count >= lod.levels[2].vertex_count);
        assert_eq!(lod.transitions.len(), 2);
    }

    #[test]
    fn test_generate_lod_single_level() {
        let (verts, idx) = generate_test_mesh(100);
        let lod = generate_lod(&verts, &idx, 1, &[1.0]);
        assert_eq!(lod.levels.len(), 1);
        assert!(lod.transitions.is_empty());
    }

    #[test]
    fn test_compute_bbox_diagonal_unit_cube() {
        let verts = vec![
            PVec3::ZERO, PVec3::X, PVec3::Y, PVec3::Z,
            PVec3::new(1.0, 1.0, 1.0),
        ];
        let diag = compute_bbox_diagonal(&verts);
        assert!((diag - 3.0_f64.sqrt()).abs() < 1e-5);
    }
}
