use std::collections::HashMap;

use rc3d_core::math::Vec3;

/// Packed edge key: canonical form where lo < hi.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeKey(u64);

impl EdgeKey {
    pub fn new(v0: u32, v1: u32) -> Self {
        let (lo, hi) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        Self((lo as u64) | ((hi as u64) << 32))
    }
    pub fn vertices(&self) -> (u32, u32) {
        ((self.0 & 0xFFFF_FFFF) as u32, (self.0 >> 32) as u32)
    }
}

/// Face identifier: index into the face list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceId(pub u32);

/// Edge identifier: index into the edge list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32);

/// A triangle face with topology.
#[derive(Clone, Debug)]
pub struct Face {
    pub vertices: [u32; 3],
    pub normal: Vec3,
    pub edges: [EdgeId; 3],
    pub adjacent_faces: [Option<FaceId>; 3],
}

/// An edge shared by one or two faces.
#[derive(Clone, Debug)]
pub struct Edge {
    pub vertices: [u32; 2],
    pub faces: [Option<FaceId>; 2],
}

/// Topology-rich triangle mesh.
#[derive(Clone, Debug)]
pub struct TriangleMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub faces: Vec<Face>,
    pub edges: Vec<Edge>,
    pub edge_map: HashMap<EdgeKey, EdgeId>,
}

impl TriangleMesh {
    /// Build from a triangle soup (flat position array, 3 per triangle).
    pub fn from_tris(positions: &[Vec3]) -> Self {
        let face_count = positions.len() / 3;
        let mut mesh = Self {
            positions: positions.to_vec(),
            normals: vec![Vec3::ZERO; positions.len()],
            faces: Vec::with_capacity(face_count),
            edges: Vec::new(),
            edge_map: HashMap::new(),
        };
        mesh.build_topology();
        mesh.compute_face_normals();
        mesh.compute_vertex_normals();
        mesh
    }

    /// Build from indexed triangle list.
    pub fn from_indexed(positions: &[Vec3], indices: &[u32]) -> Self {
        let mut all_positions = Vec::with_capacity(indices.len());
        for chunk in indices.chunks(3) {
            if chunk.len() == 3 {
                all_positions.push(positions[chunk[0] as usize]);
                all_positions.push(positions[chunk[1] as usize]);
                all_positions.push(positions[chunk[2] as usize]);
            }
        }
        Self::from_tris(&all_positions)
    }

    /// Build from an Open Inventor indexed face set (negative index = face end, fan-triangulated).
    pub fn from_indexed_face_set(positions: &[Vec3], coord_index: &[i32]) -> Self {
        let mut all_positions = Vec::new();
        let mut face_verts = Vec::new();
        for &idx in coord_index {
            if idx < 0 {
                // Fan-triangulate the polygon
                if face_verts.len() >= 3 {
                    for j in 1..face_verts.len() - 1 {
                        all_positions.push(positions[face_verts[0]]);
                        all_positions.push(positions[face_verts[j]]);
                        all_positions.push(positions[face_verts[j + 1]]);
                    }
                }
                face_verts.clear();
            } else {
                face_verts.push(idx as usize);
            }
        }
        // Handle trailing face without terminator
        if face_verts.len() >= 3 {
            for j in 1..face_verts.len() - 1 {
                all_positions.push(positions[face_verts[0]]);
                all_positions.push(positions[face_verts[j]]);
                all_positions.push(positions[face_verts[j + 1]]);
            }
        }
        Self::from_tris(&all_positions)
    }

    /// Empty mesh.
    pub fn empty() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            faces: Vec::new(),
            edges: Vec::new(),
            edge_map: HashMap::new(),
        }
    }

    fn build_topology(&mut self) {
        self.edge_map.clear();
        self.edges.clear();
        self.faces.clear();

        let tri_count = self.positions.len() / 3;
        for fi in 0..tri_count {
            let v0 = (fi * 3) as u32;
            let v1 = (fi * 3 + 1) as u32;
            let v2 = (fi * 3 + 2) as u32;
            let verts = [v0, v1, v2];
            let mut edge_ids = [EdgeId(0), EdgeId(0), EdgeId(0)];
            let mut adj = [None, None, None];

            for i in 0..3 {
                let ek = EdgeKey::new(verts[i], verts[(i + 1) % 3]);
                let eid = match self.edge_map.get(&ek) {
                    Some(&id) => {
                        // Second face sharing this edge
                        let ev = ek.vertices();
                        adj[i] = self.edges[id.0 as usize].faces[0];
                        self.edges[id.0 as usize].faces[1] = Some(FaceId(fi as u32));
                        id
                    }
                    None => {
                    let id = EdgeId(self.edges.len() as u32);
                    self.edges.push(Edge {
                        vertices: [ek.vertices().0, ek.vertices().1],
                        faces: [Some(FaceId(fi as u32)), None],
                    });
                    self.edge_map.insert(ek, id);
                    id
                }
                };
                edge_ids[i] = eid;
            }

            self.faces.push(Face {
                vertices: verts,
                normal: Vec3::ZERO,
                edges: edge_ids,
                adjacent_faces: adj,
            });
        }
    }

    fn compute_face_normals(&mut self) {
        for fi in 0..self.faces.len() {
            let v = self.faces[fi].vertices;
            let p0 = self.positions[v[0] as usize];
            let p1 = self.positions[v[1] as usize];
            let p2 = self.positions[v[2] as usize];
            self.faces[fi].normal = (p1 - p0).cross(p2 - p0).normalize();
        }
    }

    fn compute_vertex_normals(&mut self) {
        // Area-weighted average of adjacent face normals
        for (i, n) in self.normals.iter_mut().enumerate() {
            *n = Vec3::ZERO;
        }
        for fi in 0..self.faces.len() {
            let f = &self.faces[fi];
            let area = {
                let p0 = self.positions[f.vertices[0] as usize];
                let p1 = self.positions[f.vertices[1] as usize];
                let p2 = self.positions[f.vertices[2] as usize];
                (p1 - p0).cross(p2 - p0).length()
            };
            for &v in &f.vertices {
                self.normals[v as usize] += f.normal * area;
            }
        }
        for n in self.normals.iter_mut() {
            let len = n.length();
            if len > 1e-10 {
                *n = *n / len;
            }
        }
    }

    /// All unique edges.
    pub fn all_edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Edges with only one adjacent face (boundary edges).
    pub fn boundary_edges(&self) -> Vec<EdgeId> {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.faces[1].is_none())
            .map(|(i, _)| EdgeId(i as u32))
            .collect()
    }

    /// Silhouette edges: where adjacent face normals have opposite facing relative to view direction.
    pub fn silhouette_edges(&self, view_dir: Vec3) -> Vec<EdgeId> {
        let mut result = Vec::new();
        for (i, edge) in self.edges.iter().enumerate() {
            if let (Some(f0), Some(f1)) = (edge.faces[0], edge.faces[1]) {
                let n0 = self.faces[f0.0 as usize].normal;
                let n1 = self.faces[f1.0 as usize].normal;
                if n0.dot(view_dir) * n1.dot(view_dir) < 0.0 {
                    result.push(EdgeId(i as u32));
                }
            }
        }
        result
    }

    /// Generate line-list indices from a set of edge IDs.
    pub fn edge_line_indices(&self, edge_ids: &[EdgeId]) -> Vec<u32> {
        let mut indices = Vec::with_capacity(edge_ids.len() * 2);
        for eid in edge_ids {
            let e = &self.edges[eid.0 as usize];
            indices.push(e.vertices[0]);
            indices.push(e.vertices[1]);
        }
        indices
    }

    /// Generate all edges as line-list indices.
    pub fn all_edge_line_indices(&self) -> Vec<u32> {
        let mut indices = Vec::with_capacity(self.edges.len() * 2);
        for edge in &self.edges {
            indices.push(edge.vertices[0]);
            indices.push(edge.vertices[1]);
        }
        indices
    }

    /// Flat buffers for GPU: (positions as [f32;3] array, indices).
    pub fn triangle_buffers(&self) -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions: Vec<[f32; 3]> = self.positions.iter().map(|p| p.to_array()).collect();
        let indices: Vec<u32> = (0..self.positions.len() as u32).collect();
        (positions, indices)
    }

    /// Interleaved position+normal buffers for Phong shading.
    pub fn phong_buffers(&self) -> (Vec<[f32; 6]>, Vec<u32>) {
        let vertices: Vec<[f32; 6]> = self
            .positions
            .iter()
            .zip(self.normals.iter())
            .map(|(p, n)| {
                let mut v = [0f32; 6];
                v[0..3].copy_from_slice(&p.to_array());
                v[3..6].copy_from_slice(&n.to_array());
                v
            })
            .collect();
        let indices: Vec<u32> = (0..self.positions.len() as u32).collect();
        (vertices, indices)
    }

    /// Bounding box in local space.
    pub fn bounding_box(&self) -> rc3d_actions::Aabb {
        if self.positions.is_empty() {
            return rc3d_actions::Aabb::empty();
        }
        let mut aabb = rc3d_actions::Aabb::from_point(self.positions[0]);
        for p in &self.positions[1..] {
            aabb = aabb.union(&rc3d_actions::Aabb::from_point(*p));
        }
        aabb
    }
}
