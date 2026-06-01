//! Mesh result type — output of all tessellation pipelines.

use std::collections::HashMap;
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
