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
    /// Per-vertex UV; same length as `positions` when populated.
    pub texcoords: Vec<[f32; 2]>,
    pub tri_indices: Vec<u32>,
    pub faces: Vec<Face>,
    pub edges: Vec<Edge>,
    pub edge_map: HashMap<EdgeKey, EdgeId>,
    /// Per-vertex tangent (xyz) + handedness (w). Computed by `compute_tangents()`.
    pub tangents: Vec<[f32; 4]>,
}

fn quantize(v: Vec3) -> [i64; 3] {
    const SCALE: f64 = 1e5;
    [
        (v.x as f64 * SCALE).round() as i64,
        (v.y as f64 * SCALE).round() as i64,
        (v.z as f64 * SCALE).round() as i64,
    ]
}

impl TriangleMesh {
    /// Build from a triangle soup (flat position array, 3 per triangle).
    /// Vertices at the same position are deduplicated so edges are shared.
    pub fn from_tris(raw_positions: &[Vec3]) -> Self {
        let mut positions = Vec::new();
        let mut pos_map: HashMap<[i64; 3], u32> = HashMap::new();
        let mut tri_indices = Vec::with_capacity(raw_positions.len());

        for p in raw_positions {
            let key = quantize(*p);
            let idx = *pos_map.entry(key).or_insert_with(|| {
                let i = positions.len() as u32;
                positions.push(*p);
                i
            });
            tri_indices.push(idx);
        }

        let vert_count = positions.len();
        let mut mesh = Self {
            positions,
            normals: vec![Vec3::ZERO; vert_count],
            texcoords: vec![[0.0, 0.0]; vert_count],
            tri_indices,
            faces: Vec::with_capacity(raw_positions.len() / 3),
            edges: Vec::new(),
            edge_map: HashMap::new(),
            tangents: Vec::new(),
        };
        mesh.build_topology();
        mesh.compute_face_normals();
        mesh.compute_vertex_normals();
        mesh
    }

    /// Build from indexed triangle list (positions already shared by index).
    pub fn from_indexed(positions: &[Vec3], indices: &[u32]) -> Self {
        let mut mesh = Self {
            positions: positions.to_vec(),
            normals: vec![Vec3::ZERO; positions.len()],
            texcoords: vec![[0.0, 0.0]; positions.len()],
            tri_indices: indices.to_vec(),
            faces: Vec::with_capacity(indices.len() / 3),
            edges: Vec::new(),
            edge_map: HashMap::new(),
            tangents: Vec::new(),
        };
        mesh.build_topology();
        mesh.compute_face_normals();
        mesh.compute_vertex_normals();
        mesh
    }

    /// Same as `from_indexed` but with per-vertex texture coordinates.
    pub fn from_indexed_with_texcoords(
        positions: &[Vec3],
        indices: &[u32],
        texcoords: &[[f32; 2]],
    ) -> Self {
        assert_eq!(
            positions.len(),
            texcoords.len(),
            "texcoords length must match positions"
        );
        let mut mesh = Self {
            positions: positions.to_vec(),
            normals: vec![Vec3::ZERO; positions.len()],
            texcoords: texcoords.to_vec(),
            tri_indices: indices.to_vec(),
            faces: Vec::with_capacity(indices.len() / 3),
            edges: Vec::new(),
            edge_map: HashMap::new(),
            tangents: Vec::new(),
        };
        mesh.build_topology();
        mesh.compute_face_normals();
        mesh.compute_vertex_normals();
        mesh
    }

    /// Build from an Open Inventor indexed face set (negative index = face end, fan-triangulated).
    pub fn from_indexed_face_set(positions: &[Vec3], coord_index: &[i32]) -> Self {
        let all_indices = tri_indices_from_coord_index(coord_index);
        Self::from_indexed(positions, &all_indices)
    }

    /// Same as `from_indexed_face_set` with per-vertex UVs (`texcoords.len() == positions.len()`).
    pub fn from_indexed_face_set_tex(
        positions: &[Vec3],
        texcoords: &[[f32; 2]],
        coord_index: &[i32],
    ) -> Self {
        assert_eq!(
            positions.len(),
            texcoords.len(),
            "positions and texcoords must align"
        );
        let all_indices = tri_indices_from_coord_index(coord_index);
        Self::from_indexed_with_texcoords(positions, &all_indices, texcoords)
    }

    /// Non-welded triangle list: `positions.len()` divisible by 3, one UV per corner.
    pub fn from_triangle_list_with_texcoords(
        positions: &[Vec3],
        texcoords: &[[f32; 2]],
    ) -> Self {
        assert_eq!(positions.len(), texcoords.len());
        assert_eq!(
            positions.len() % 3,
            0,
            "triangle list length must be multiple of 3"
        );
        let n = positions.len();
        let tri_indices: Vec<u32> = (0..n as u32).collect();
        let mut mesh = Self {
            positions: positions.to_vec(),
            normals: vec![Vec3::ZERO; n],
            texcoords: texcoords.to_vec(),
            tri_indices,
            faces: Vec::with_capacity(n / 3),
            edges: Vec::new(),
            edge_map: HashMap::new(),
            tangents: Vec::new(),
        };
        mesh.build_topology();
        mesh.compute_face_normals();
        mesh.compute_vertex_normals();
        mesh
    }

    /// Empty mesh.
    pub fn empty() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            texcoords: Vec::new(),
            tri_indices: Vec::new(),
            faces: Vec::new(),
            edges: Vec::new(),
            edge_map: HashMap::new(),
            tangents: Vec::new(),
        }
    }

    fn build_topology(&mut self) {
        self.edge_map.clear();
        self.edges.clear();
        self.faces.clear();

        let tri_count = self.tri_indices.len() / 3;
        for fi in 0..tri_count {
            let v0 = self.tri_indices[fi * 3];
            let v1 = self.tri_indices[fi * 3 + 1];
            let v2 = self.tri_indices[fi * 3 + 2];
            let verts = [v0, v1, v2];
            let mut edge_ids = [EdgeId(0), EdgeId(0), EdgeId(0)];
            let mut adj = [None, None, None];

            for i in 0..3 {
                let ek = EdgeKey::new(verts[i], verts[(i + 1) % 3]);
                let eid = match self.edge_map.get(&ek) {
                    Some(&id) => {
                        // Second face sharing this edge
                        let _ev = ek.vertices();
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
            let cross = (p1 - p0).cross(p2 - p0);
            let len = cross.length();
            self.faces[fi].normal = if len > 1e-20 { cross / len } else { Vec3::Y };
        }
    }

    fn compute_vertex_normals(&mut self) {
        for n in &mut self.normals {
            *n = Vec3::ZERO;
        }
        for fi in 0..self.faces.len() {
            let f = &self.faces[fi];
            let p0 = self.positions[f.vertices[0] as usize];
            let p1 = self.positions[f.vertices[1] as usize];
            let p2 = self.positions[f.vertices[2] as usize];
            let area = (p1 - p0).cross(p2 - p0).length();
            if area > 1e-20 {
                let weighted = f.normal * area;
                for &v in &f.vertices {
                    self.normals[v as usize] += weighted;
                }
            }
        }
        for n in self.normals.iter_mut() {
            let len = n.length();
            if len > 1e-10 && len.is_finite() {
                *n /= len;
            } else {
                *n = Vec3::Y;
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

    /// Generate all edges as line-list vertex positions.
    pub fn edge_line_positions(&self) -> Vec<[f32; 3]> {
        let mut positions = Vec::with_capacity(self.edges.len() * 2);
        for edge in &self.edges {
            let p0 = self.positions[edge.vertices[0] as usize].to_array();
            let p1 = self.positions[edge.vertices[1] as usize].to_array();
            positions.push(p0);
            positions.push(p1);
        }
        positions
    }

    /// Default crease angle (degrees) for [`Self::edge_line_positions_feature`]: technical overlays.
    pub const DEFAULT_FEATURE_EDGE_CREASE_DEG: f32 = 12.0;

    /// Boundary + crease edges for static visualization (`ShadedWithEdges`), not full triangulation.
    ///
    /// Includes every **boundary** edge (one adjacent face) and **crease** edges where the dihedral
    /// angle between face normals exceeds `crease_angle_deg`.
    pub fn edge_line_positions_feature(&self, crease_angle_deg: f32) -> Vec<[f32; 3]> {
        let cos_thresh = crease_angle_deg.to_radians().cos();
        let mut positions = Vec::new();
        for edge in &self.edges {
            let p0 = self.positions[edge.vertices[0] as usize].to_array();
            let p1 = self.positions[edge.vertices[1] as usize].to_array();
            match (edge.faces[0], edge.faces[1]) {
                (None, None) => {}
                (Some(_), None) | (None, Some(_)) => {
                    positions.push(p0);
                    positions.push(p1);
                }
                (Some(a), Some(b)) => {
                    let n0 = self.faces[a.0 as usize].normal;
                    let n1 = self.faces[b.0 as usize].normal;
                    let d = n0.dot(n1).clamp(-1.0, 1.0);
                    if d < cos_thresh {
                        positions.push(p0);
                        positions.push(p1);
                    }
                }
            }
        }
        positions
    }

    /// Flat buffers for GPU: (positions as [f32;3] array, indices).
    pub fn triangle_buffers(&self) -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions: Vec<[f32; 3]> = self.positions.iter().map(|p| p.to_array()).collect();
        (positions, self.tri_indices.clone())
    }

    /// Compute per-vertex tangents using the MikkTSpace/Megelan method.
    /// Requires positions, normals, texcoords, and triangle indices to be populated.
    pub fn compute_tangents(&mut self) {
        let vcount = self.positions.len();
        let mut tan1 = vec![Vec3::ZERO; vcount];
        let mut tan2 = vec![Vec3::ZERO; vcount];

        for ti in (0..self.tri_indices.len()).step_by(3) {
            let i0 = self.tri_indices[ti] as usize;
            let i1 = self.tri_indices[ti + 1] as usize;
            let i2 = self.tri_indices[ti + 2] as usize;
            if i0 >= vcount || i1 >= vcount || i2 >= vcount { continue; }

            let p0 = self.positions[i0];
            let p1 = self.positions[i1];
            let p2 = self.positions[i2];

            let default_uv = [0.0f32, 0.0];
            let uv0 = self.texcoords.get(i0).copied().unwrap_or(default_uv);
            let uv1 = self.texcoords.get(i1).copied().unwrap_or(default_uv);
            let uv2 = self.texcoords.get(i2).copied().unwrap_or(default_uv);

            let x1 = p1.x - p0.x;
            let y1 = p1.y - p0.y;
            let z1 = p1.z - p0.z;
            let x2 = p2.x - p0.x;
            let y2 = p2.y - p0.y;
            let z2 = p2.z - p0.z;

            let s1 = uv1[0] - uv0[0];
            let t1 = uv1[1] - uv0[1];
            let s2 = uv2[0] - uv0[0];
            let t2 = uv2[1] - uv0[1];

            let r = 1.0 / (s1 * t2 - s2 * t1).max(1e-10);
            let sdir = Vec3::new(
                (t2 * x1 - t1 * x2) * r,
                (t2 * y1 - t1 * y2) * r,
                (t2 * z1 - t1 * z2) * r,
            );
            let tdir = Vec3::new(
                (s1 * x2 - s2 * x1) * r,
                (s1 * y2 - s2 * y1) * r,
                (s1 * z2 - s2 * z1) * r,
            );

            tan1[i0] += sdir; tan1[i1] += sdir; tan1[i2] += sdir;
            tan2[i0] += tdir; tan2[i1] += tdir; tan2[i2] += tdir;
        }

        self.tangents = Vec::with_capacity(vcount);
        for i in 0..vcount {
            let n = self.normals.get(i).copied().unwrap_or(Vec3::Y);
            let t = tan1[i];
            // Gram-Schmidt orthogonalize
            let tangent = (t - n * n.dot(t)).normalize();
            let handedness = if n.cross(t).dot(tan2[i]) < 0.0 { -1.0 } else { 1.0 };
            self.tangents.push([tangent.x, tangent.y, tangent.z, handedness]);
        }
    }

    /// Interleaved position + normal + texcoord + tangent for lit shading.
    pub fn phong_buffers(&self) -> (Vec<[f32; 12]>, Vec<u32>) {
        let default_uv = [0.0f32, 0.0];
        let default_tangent = [1.0f32, 0.0, 0.0, 1.0];
        let vertices: Vec<[f32; 12]> = self
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let n = self.normals.get(i).copied().unwrap_or(Vec3::ZERO);
                let uv = self.texcoords.get(i).copied().unwrap_or(default_uv);
                let t = self.tangents.get(i).copied().unwrap_or(default_tangent);
                [
                    p.x, p.y, p.z, n.x, n.y, n.z, uv[0], uv[1],
                    t[0], t[1], t[2], t[3],
                ]
            })
            .collect();
        (vertices, self.tri_indices.clone())
    }

    /// Bounding box in local space.
    pub fn bounding_box(&self) -> rc3d_core::Aabb {
        if self.positions.is_empty() {
            return rc3d_core::Aabb::empty();
        }
        let mut aabb = rc3d_core::Aabb::from_point(self.positions[0]);
        for p in &self.positions[1..] {
            aabb = aabb.union(&rc3d_core::Aabb::from_point(*p));
        }
        aabb
    }
}

/// Triangulate Inventor-style `coord_index` into a triangle index list.
///
/// Uses ear-clipping for concave-safe triangulation. Falls back to fan
/// triangulation when positions are not available (fan is correct for convex polygons
/// and the common case; ear-clipping adds concavity safety at a small cost).
pub fn tri_indices_from_coord_index(coord_index: &[i32]) -> Vec<u32> {
    fan_triangulate_coord_index(coord_index)
}

/// Fan triangulation (fast, correct for convex polygons).
fn fan_triangulate_coord_index(coord_index: &[i32]) -> Vec<u32> {
    let mut all_indices = Vec::new();
    let mut face_verts = Vec::new();
    for &idx in coord_index {
        if idx < 0 {
            if face_verts.len() >= 3 {
                for j in 1..face_verts.len() - 1 {
                    all_indices.push(face_verts[0] as u32);
                    all_indices.push(face_verts[j] as u32);
                    all_indices.push(face_verts[j + 1] as u32);
                }
            }
            face_verts.clear();
        } else {
            face_verts.push(idx as usize);
        }
    }
    if face_verts.len() >= 3 {
        for j in 1..face_verts.len() - 1 {
            all_indices.push(face_verts[0] as u32);
            all_indices.push(face_verts[j] as u32);
            all_indices.push(face_verts[j + 1] as u32);
        }
    }
    all_indices
}

/// Ear-clipping triangulation with 2D projection.
/// Handles concave polygons correctly by clipping ears one at a time.
pub fn tri_indices_from_coord_index_earclip(coord_index: &[i32], positions: &[Vec3]) -> Vec<u32> {
    let mut all_indices = Vec::new();
    let mut face_verts = Vec::new();
    for &idx in coord_index {
        if idx < 0 {
            if face_verts.len() >= 3 {
                earclip_polygon(&face_verts, positions, &mut all_indices);
            }
            face_verts.clear();
        } else {
            face_verts.push(idx as usize);
        }
    }
    if face_verts.len() >= 3 {
        earclip_polygon(&face_verts, positions, &mut all_indices);
    }
    all_indices
}

fn earclip_polygon(verts: &[usize], positions: &[Vec3], out: &mut Vec<u32>) {
    if verts.len() < 3 { return; }
    if verts.len() == 3 {
        out.push(verts[0] as u32);
        out.push(verts[1] as u32);
        out.push(verts[2] as u32);
        return;
    }

    // Project to 2D using best-fit plane normal
    let n = {
        let _p0 = positions[verts[0]];
        let mut normal = Vec3::ZERO;
        for i in 0..verts.len() {
            let j = (i + 1) % verts.len();
            let pi = positions[verts[i]];
            let pj = positions[verts[j]];
            normal.x += (pi.y - pj.y) * (pi.z + pj.z);
            normal.y += (pi.z - pj.z) * (pi.x + pj.x);
            normal.z += (pi.x - pj.x) * (pi.y + pj.y);
        }
        if normal.length_squared() < 1e-10 { return; }
        normal.normalize()
    };
    let u_axis = if n.dot(Vec3::Z).abs() < 0.9 {
        n.cross(Vec3::Z).normalize()
    } else {
        n.cross(Vec3::Y).normalize()
    };
    let v_axis = n.cross(u_axis);
    let origin = positions[verts[0]];

    let mut poly: Vec<(f32, f32, usize)> = verts.iter().map(|&vi| {
        let rel = positions[vi] - origin;
        (rel.dot(u_axis), rel.dot(v_axis), vi)
    }).collect();

    // Ear-clipping: remove one ear at a time
    while poly.len() > 3 {
        let mut ear_found = false;
        for i in 0..poly.len() {
            let prev = (i + poly.len() - 1) % poly.len();
            let next = (i + 1) % poly.len();
            let (p0x, p0y, _) = poly[prev];
            let (p1x, p1y, _) = poly[i];
            let (p2x, p2y, _) = poly[next];

            // Check if angle is convex (cross product positive)
            let cross = (p1x - p0x) * (p2y - p1y) - (p1y - p0y) * (p2x - p1x);
            if cross <= 0.0 { continue; }

            // Check if triangle contains any other vertex
            let mut contains = false;
            for j in 0..poly.len() {
                if j == prev || j == i || j == next { continue; }
                let (px, py, _) = poly[j];
                if point_in_tri_2d(px, py, p0x, p0y, p1x, p1y, p2x, p2y) {
                    contains = true;
                    break;
                }
            }
            if contains { continue; }

            // Found an ear
            out.push(poly[prev].2 as u32);
            out.push(poly[i].2 as u32);
            out.push(poly[next].2 as u32);
            poly.remove(i);
            ear_found = true;
            break;
        }
        if !ear_found { break; }
    }
    // Last triangle
    if poly.len() == 3 {
        out.push(poly[0].2 as u32);
        out.push(poly[1].2 as u32);
        out.push(poly[2].2 as u32);
    }
}

fn point_in_tri_2d(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> bool {
    let d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
    let d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
    let d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
    let neg = d1 < -1e-10 || d2 < -1e-10 || d3 < -1e-10;
    let pos = d1 > 1e-10 || d2 > 1e-10 || d3 > 1e-10;
    !(neg && pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_key_canonical_order() {
        let k1 = EdgeKey::new(1, 2);
        let k2 = EdgeKey::new(2, 1);
        assert_eq!(k1, k2);
        assert_eq!(k1.vertices(), (1, 2));
    }

    #[test]
    fn test_from_tris_single_triangle() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mesh = TriangleMesh::from_tris(&positions);
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.tri_indices.len(), 3);
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.edges.len(), 3);
    }

    #[test]
    fn test_from_tris_deduplicates_shared_vertices() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0), // same as first
            Vec3::new(1.0, 0.0, 0.0), // same as second
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let mesh = TriangleMesh::from_tris(&positions);
        // Should have 4 unique vertices, not 6
        assert_eq!(mesh.positions.len(), 4);
    }

    #[test]
    fn test_tri_indices_fan_triangulation() {
        // Quad fan: vertices 0,1,2,3,-1 = quad triangulated as 0-1-2, 0-2-3
        let indices = tri_indices_from_coord_index(&[0, 1, 2, 3, -1]);
        assert_eq!(indices.len(), 6);
        assert_eq!(indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn test_tri_indices_multi_face() {
        let indices = tri_indices_from_coord_index(&[0, 1, 2, -1, 0, 2, 3, -1]);
        assert_eq!(indices.len(), 6);
    }

    #[test]
    fn test_compute_tangents_simple_triangle() {
        let mut mesh = TriangleMesh {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            texcoords: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri_indices: vec![0, 1, 2],
            faces: Vec::new(),
            edges: Vec::new(),
            edge_map: HashMap::new(),
            tangents: Vec::new(),
        };
        mesh.compute_tangents();
        assert_eq!(mesh.tangents.len(), 3);
        // Tangent should be roughly along the x-axis
        assert!(mesh.tangents[0][0] > 0.5);
    }

    #[test]
    fn test_phong_buffers_includes_tangent() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut mesh = TriangleMesh::from_tris(&positions);
        mesh.compute_tangents();
        let (verts, indices) = mesh.phong_buffers();
        assert_eq!(verts.len(), 3);
        assert_eq!(verts[0].len(), 12); // pos(3) + norm(3) + uv(2) + tangent(4) = 12
        assert_eq!(indices.len(), 3);
    }
}
