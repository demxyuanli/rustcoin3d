//! Level-of-detail generation via edge collapse simplification.

use rc3d_core::math::Vec3;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// A candidate edge collapse, ordered by cost (quadric error).
#[derive(Debug, Clone)]
struct CollapseCandidate {
    v0: usize,
    v1: usize,
    cost: f32,
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
    vertices: &[Vec3],
    indices: &[i32],
    target_vertices: usize,
) -> Option<(Vec<Vec3>, Vec<i32>)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_already_small() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2, -1];
        // Target >= current count → no simplification needed
        assert!(simplify_mesh(&verts, &indices, 10).is_none());
    }

    #[test]
    fn test_simplify_reduces_count() {
        let verts: Vec<Vec3> = (0..20).map(|i| {
            Vec3::new(i as f32, 0.0, 0.0)
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
}
