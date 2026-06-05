//! Mesh result type — output of all tessellation pipelines.

use std::collections::{HashMap, HashSet};
use rc3d_core::math::Vec3;

#[derive(Debug, Default, Clone)]
pub struct MeshResult {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<i32>,
}

impl MeshResult {
    /// Compute per-vertex normals by averaging normals of adjacent triangles.
    pub fn compute_normals(&mut self) {
        if self.vertices.is_empty() || self.indices.is_empty() { return; }
        let mut normals_acc = vec![Vec3::ZERO; self.vertices.len()];
        for chunk in self.indices.chunks(4) {
            if chunk.len() < 3 { continue; }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }
            let n = (self.vertices[i1] - self.vertices[i0]).cross(self.vertices[i2] - self.vertices[i0]);
            normals_acc[i0] = normals_acc[i0] + n;
            normals_acc[i1] = normals_acc[i1] + n;
            normals_acc[i2] = normals_acc[i2] + n;
        }
        self.normals = normals_acc.into_iter()
            .map(|n| if n.length() > 1e-10 { n * (1.0 / n.length()) } else { Vec3::Z })
            .collect();
    }

    /// Flip triangle winding and negate stored normals (solid-level Reverse).
    pub fn reverse_winding(&mut self) {
        for chunk in self.indices.chunks_mut(4) {
            if chunk.len() >= 3 {
                chunk.swap(1, 2);
            }
        }
        for n in &mut self.normals {
            *n = -*n;
        }
    }

    /// Append another mesh into this one (indices remapped).
    pub fn append_from(&mut self, other: &MeshResult) {
        if other.vertices.is_empty() || other.indices.is_empty() {
            return;
        }
        let base = self.vertices.len() as i32;
        self.vertices.extend_from_slice(&other.vertices);
        if other.normals.len() == other.vertices.len() {
            if self.normals.len() < self.vertices.len() - other.vertices.len() {
                self.normals.resize(self.vertices.len() - other.vertices.len(), Vec3::ZERO);
            }
            self.normals.extend_from_slice(&other.normals);
        } else if !self.vertices.is_empty() && self.normals.len() != self.vertices.len() {
            self.normals.resize(self.vertices.len(), Vec3::ZERO);
        }
        for chunk in other.indices.chunks(4) {
            if chunk.len() < 3 {
                continue;
            }
            self.indices.extend_from_slice(&[
                chunk[0] + base,
                chunk[1] + base,
                chunk[2] + base,
                -1,
            ]);
        }
    }

    /// Weld vertices within tolerance using spatial hash + union-find.
    pub fn weld_vertices(&mut self, tolerance: f32) {
        if self.vertices.is_empty() { return; }
        let n = self.vertices.len();
        let cell_size = tolerance.max(1e-4);
        let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, v) in self.vertices.iter().enumerate() {
            let cell = (
                (v.x / cell_size).floor() as i64,
                (v.y / cell_size).floor() as i64,
                (v.z / cell_size).floor() as i64,
            );
            grid.entry(cell).or_default().push(i);
        }
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x { parent[x] = find(parent, parent[x]); }
            parent[x]
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb { parent[ra] = rb; }
        }
        for (&(cx, cy, cz), indices) in &grid {
            for dx in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dz in -1i64..=1 {
                        let key = (cx + dx, cy + dy, cz + dz);
                        if let Some(other_indices) = grid.get(&key) {
                            for &i in indices {
                                for &j in other_indices {
                                    if i < j && find(&mut parent, i) != find(&mut parent, j) {
                                        let d = (self.vertices[i] - self.vertices[j]).length();
                                        if d < tolerance { union(&mut parent, i, j); }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut root_to_group: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n { let root = find(&mut parent, i); root_to_group.entry(root).or_default().push(i); }
        let mut new_vertices = Vec::with_capacity(root_to_group.len());
        let mut new_normals = Vec::with_capacity(root_to_group.len());
        let mut remap: Vec<usize> = vec![0; n];
        for (_root, members) in &root_to_group {
            let new_idx = new_vertices.len();
            let mut sum_pos = Vec3::ZERO;
            let mut sum_norm = Vec3::ZERO;
            for &i in members {
                sum_pos = sum_pos + self.vertices[i];
                if i < self.normals.len() { sum_norm = sum_norm + self.normals[i]; }
            }
            let count = members.len();
            new_vertices.push(sum_pos * (1.0 / count as f32));
            new_normals.push(if sum_norm.length() > 1e-10 { sum_norm.normalize() } else { Vec3::Z });
            for &i in members { remap[i] = new_idx; }
        }
        let mut new_indices = Vec::with_capacity(self.indices.len());
        for &idx in &self.indices {
            if idx == -1 { new_indices.push(-1); }
            else if idx >= 0 && (idx as usize) < n { new_indices.push(remap[idx as usize] as i32); }
        }
        self.vertices = new_vertices;
        self.normals = new_normals;
        self.indices = new_indices;
    }

    /// Weld vertices within tolerance, preserving protected vertices as anchors.
    ///
    /// Protected vertices (e.g. discretized edge boundary points) absorb nearby
    /// interior vertices during welding but never move themselves. This ensures
    /// cross-face boundary watertightness after parallel face meshing.
    ///
    /// OCC alignment: BRepMesh_FastDiscret locks discretized edge vertices;
    /// internal Steiner points can move but boundary vertices are fixed.
    pub fn weld_vertices_protected(&mut self, tolerance: f32, protected: &HashSet<usize>) -> usize {
        if self.vertices.len() < 2 || tolerance <= 0.0 {
            return 0;
        }
        let n = self.vertices.len();
        let cell_size = tolerance.max(1e-4);

        // Build spatial hash grid
        let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, v) in self.vertices.iter().enumerate() {
            let cell = (
                (v.x / cell_size).floor() as i64,
                (v.y / cell_size).floor() as i64,
                (v.z / cell_size).floor() as i64,
            );
            grid.entry(cell).or_default().push(i);
        }

        // Union-find for vertex clustering
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x { parent[x] = find(parent, parent[x]); }
            parent[x]
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb { parent[ra] = rb; }
        }

        // Cluster non-protected vertices with their neighbors
        for (&(cx, cy, cz), indices) in &grid {
            for dx in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dz in -1i64..=1 {
                        let key = (cx + dx, cy + dy, cz + dz);
                        if let Some(other_indices) = grid.get(&key) {
                            for &i in indices {
                                for &j in other_indices {
                                    if i >= j { continue; }
                                    let ri = find(&mut parent, i);
                                    let rj = find(&mut parent, j);
                                    if ri == rj { continue; }

                                    // Never merge two protected vertices together
                                    if protected.contains(&i) && protected.contains(&j) {
                                        continue;
                                    }

                                    let d = (self.vertices[i] - self.vertices[j]).length();
                                    if d < tolerance {
                                        // Protected vertex becomes the canonical root
                                        if protected.contains(&i) {
                                            parent[rj] = ri;
                                        } else if protected.contains(&j) {
                                            parent[ri] = rj;
                                        } else {
                                            parent[ri] = rj;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Collect clusters, counting protected members
        let mut root_to_group: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            root_to_group.entry(root).or_default().push(i);
        }

        // Build remap: each cluster → one output vertex
        let mut new_vertices = Vec::with_capacity(root_to_group.len());
        let mut new_normals = Vec::with_capacity(root_to_group.len());
        let mut remap: Vec<usize> = vec![0; n];

        for (_root, members) in &root_to_group {
            let new_idx = new_vertices.len();

            // Find the protected member in this cluster (if any) to use as anchor
            let anchor = members.iter().find(|&&m| protected.contains(&m));

            if let Some(&a) = anchor {
                // Protected vertex: use its exact position, don't average
                new_vertices.push(self.vertices[a]);
                new_normals.push(if a < self.normals.len() {
                    self.normals[a]
                } else {
                    Vec3::Z
                });
            } else {
                // No protected vertex: average all members
                let mut sum_pos = Vec3::ZERO;
                let mut sum_norm = Vec3::ZERO;
                for &i in members {
                    sum_pos = sum_pos + self.vertices[i];
                    if i < self.normals.len() {
                        sum_norm = sum_norm + self.normals[i];
                    }
                }
                let count = members.len();
                new_vertices.push(sum_pos * (1.0 / count as f32));
                new_normals.push(
                    if sum_norm.length() > 1e-10 { sum_norm.normalize() } else { Vec3::Z }
                );
            }

            for &i in members {
                remap[i] = new_idx;
            }
        }

        // Remap indices
        let mut new_indices = Vec::with_capacity(self.indices.len());
        for &idx in &self.indices {
            if idx == -1 {
                new_indices.push(-1);
            } else if idx >= 0 && (idx as usize) < n {
                new_indices.push(remap[idx as usize] as i32);
            }
        }

        self.vertices = new_vertices;
        self.normals = new_normals;
        self.indices = new_indices;

        n - self.vertices.len()
    }

    /// Finalize normals: use accumulated analytical normals where available,
    /// fall back to face-averaged normals.
    pub fn finalize_normals(&mut self) {
        if self.vertices.is_empty() || self.indices.is_empty() { return; }
        if self.normals.len() != self.vertices.len() {
            self.normals = vec![Vec3::ZERO; self.vertices.len()];
        }
        let mut face_normals = vec![Vec3::ZERO; self.vertices.len()];
        for chunk in self.indices.chunks(4) {
            if chunk.len() < 3 { continue; }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() { continue; }
            let n = (self.vertices[i1] - self.vertices[i0]).cross(self.vertices[i2] - self.vertices[i0]);
            face_normals[i0] = face_normals[i0] + n;
            face_normals[i1] = face_normals[i1] + n;
            face_normals[i2] = face_normals[i2] + n;
        }
        for i in 0..self.vertices.len() {
            let ana = self.normals[i];
            if ana.length() > 1e-10 { self.normals[i] = ana.normalize(); }
            else {
                let n = face_normals[i];
                self.normals[i] = if n.length() > 1e-10 { n.normalize() } else { Vec3::Z };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weld_vertices_protected_preserves_boundary() {
        let mut mesh = MeshResult {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),       // boundary vertex 0
                Vec3::new(1.0, 0.0, 0.0),       // boundary vertex 1
                Vec3::new(0.001, 0.0, 0.0),     // near-duplicate interior (close to 0)
                Vec3::new(0.5, 0.5, 0.0),       // interior
            ],
            normals: vec![Vec3::Z; 4],
            indices: vec![0, 1, 3, -1, 0, 3, 2, -1],
        };
        let mut protected: HashSet<usize> = HashSet::new();
        protected.insert(0);
        protected.insert(1);

        let merged = mesh.weld_vertices_protected(0.01, &protected);
        assert!(merged > 0, "should merge at least one vertex pair");
        // Boundary vertex 0 must still exist at origin (unchanged)
        let v0 = mesh.vertices.iter().position(|v| {
            (v.x - 0.0).abs() < 1e-6 && (v.y - 0.0).abs() < 1e-6 && (v.z - 0.0).abs() < 1e-6
        });
        assert!(v0.is_some(), "boundary vertex 0 must survive at origin");

        // Boundary vertex 1 must still exist at (1,0,0)
        let v1 = mesh.vertices.iter().position(|v| {
            (v.x - 1.0).abs() < 1e-6 && (v.y - 0.0).abs() < 1e-6 && (v.z - 0.0).abs() < 1e-6
        });
        assert!(v1.is_some(), "boundary vertex 1 must survive at (1,0,0)");
    }

    #[test]
    fn test_weld_vertices_protected_does_not_merge_two_protected() {
        let mut mesh = MeshResult {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),   // protected
                Vec3::new(0.001, 0.0, 0.0), // also protected, close to 0
            ],
            normals: vec![Vec3::Z; 2],
            indices: vec![0, 1, 0, -1],
        };
        let mut protected: HashSet<usize> = HashSet::new();
        protected.insert(0);
        protected.insert(1);

        let merged = mesh.weld_vertices_protected(0.01, &protected);
        assert_eq!(merged, 0, "two protected vertices must NOT merge");
        assert_eq!(mesh.vertices.len(), 2, "both protected vertices survive");
    }
}
