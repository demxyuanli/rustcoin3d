//! Mesh-level void shell subtraction for BREP_WITH_VOIDS.
//!
//! Legacy centroid ray-cast removal. Production paths use
//! [`solid_mesh::mesh_solid_with_voids`] (OCC oriented multi-shell merge).

use rc3d_core::math::Vec3;
use crate::mesh_result::MeshResult;

/// Result of void subtraction.
#[derive(Debug)]
#[allow(dead_code)]
pub struct VoidSubtractResult {
    /// Mesh with void triangles removed.
    pub mesh: MeshResult,
    /// Number of triangles removed.
    pub removed_tris: usize,
}

/// Subtract void shell meshes from the outer shell mesh.
///
/// Keeps triangles whose centroid is NOT inside any void.
#[allow(dead_code)]
pub fn subtract_void_meshes(
    outer_mesh: &MeshResult,
    void_meshes: &[MeshResult],
) -> VoidSubtractResult {
    if void_meshes.is_empty() || outer_mesh.indices.is_empty() {
        return VoidSubtractResult {
            mesh: outer_mesh.clone(),
            removed_tris: 0,
        };
    }

    // Collect void triangles as (v0, v1, v2) for ray-cast tests
    let void_tris: Vec<[Vec3; 3]> = void_meshes
        .iter()
        .flat_map(|m| {
            m.indices.chunks(4).filter_map(|chunk| {
                if chunk.len() < 3 { return None; }
                let i0 = chunk[0] as usize;
                let i1 = chunk[1] as usize;
                let i2 = chunk[2] as usize;
                Some([
                    m.vertices[i0],
                    m.vertices[i1],
                    m.vertices[i2],
                ])
            })
        })
        .collect();

    if void_tris.is_empty() {
        return VoidSubtractResult {
            mesh: outer_mesh.clone(),
            removed_tris: 0,
        };
    }

    // Spatial grid for void triangles: cell size = avg void edge length * 2
    let cell_size = void_tris.iter()
        .flat_map(|t| {
            let e01 = (t[1] - t[0]).length();
            let e12 = (t[2] - t[1]).length();
            let e20 = (t[0] - t[2]).length();
            vec![e01, e12, e20]
        })
        .fold(1e-6f32, f32::max)
        .max(1e-3) * 2.0;

    let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::new();

    for (ti, tri) in void_tris.iter().enumerate() {
        let min_x = tri[0].x.min(tri[1].x).min(tri[2].x);
        let max_x = tri[0].x.max(tri[1].x).max(tri[2].x);
        let min_y = tri[0].y.min(tri[1].y).min(tri[2].y);
        let max_y = tri[0].y.max(tri[1].y).max(tri[2].y);
        let min_z = tri[0].z.min(tri[1].z).min(tri[2].z);
        let max_z = tri[0].z.max(tri[1].z).max(tri[2].z);
        let ci0 = (min_x / cell_size).floor() as i32;
        let ci1 = (max_x / cell_size).ceil() as i32;
        let cj0 = (min_y / cell_size).floor() as i32;
        let cj1 = (max_y / cell_size).ceil() as i32;
        let ck0 = (min_z / cell_size).floor() as i32;
        let ck1 = (max_z / cell_size).ceil() as i32;
        for ci in ci0..=ci1 {
            for cj in cj0..=cj1 {
                for ck in ck0..=ck1 {
                    grid.entry((ci, cj, ck)).or_default().push(ti);
                }
            }
        }
    }

    let cell_size_inv = 1.0 / cell_size;
    let max_x = grid.keys().map(|k| k.0).max().unwrap_or(i32::MIN);
    let mut kept_vertices = Vec::new();
    let mut kept_normals = Vec::new();
    let mut kept_indices = Vec::new();
    let mut removed = 0usize;

    let has_normals = outer_mesh.normals.len() == outer_mesh.vertices.len();

    for chunk in outer_mesh.indices.chunks(4) {
        if chunk.len() < 3 { continue; }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        let v0 = outer_mesh.vertices[i0];
        let v1 = outer_mesh.vertices[i1];
        let v2 = outer_mesh.vertices[i2];
        let centroid = Vec3::new(
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0,
        );

        let inside = point_inside_void_mesh(&centroid, &void_tris, &grid, cell_size_inv, max_x);
        if inside {
            removed += 1;
        } else {
            let base = kept_vertices.len() as i32;
            kept_vertices.extend_from_slice(&[v0, v1, v2]);
            kept_indices.extend_from_slice(&[
                base, base + 1, base + 2, -1,
            ]);
            if has_normals {
                kept_normals.push(outer_mesh.normals[i0]);
                kept_normals.push(outer_mesh.normals[i1]);
                kept_normals.push(outer_mesh.normals[i2]);
            }
        }
    }

    VoidSubtractResult {
        mesh: MeshResult {
            vertices: kept_vertices,
            indices: kept_indices,
            normals: kept_normals,
        },
        removed_tris: removed,
    }
}

/// Ray-cast test: is `point` inside the void mesh?
///
/// Casts a ray along +X and counts triangle intersections using
/// DDA grid traversal. Odd count → inside, even count → outside.
#[allow(dead_code)]
fn point_inside_void_mesh(
    point: &Vec3,
    void_tris: &[[Vec3; 3]],
    grid: &std::collections::HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size_inv: f32,
    max_x: i32,
) -> bool {
    let cx = (point.x * cell_size_inv).floor() as i32;
    let cy = (point.y * cell_size_inv).floor() as i32;
    let cz = (point.z * cell_size_inv).floor() as i32;

    let mut seen = std::collections::HashSet::<usize>::new();
    // DDA walk along +X through grid cells
    for ci in cx..=max_x {
        // Tube check: current Y,Z cell + immediate neighbors
        for &(dcj, dck) in &[(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
            let cj = cy + dcj;
            let ck = cz + dck;
            if let Some(candidates) = grid.get(&(ci, cj, ck)) {
                seen.extend(candidates);
            }
        }
    }

    // Möller–Trumbore ray-triangle intersection along +X
    let ray_origin = *point;
    let ray_dir = Vec3::X;
    let mut count = 0u32;

    for &ti in &seen {
        let tri = &void_tris[ti];
        if ray_triangle_intersect(&ray_origin, &ray_dir, tri) {
            count += 1;
        }
    }

    count % 2 == 1
}

/// Möller–Trumbore ray-triangle intersection.
#[allow(dead_code)]
fn ray_triangle_intersect(origin: &Vec3, dir: &Vec3, tri: &[Vec3; 3]) -> bool {
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);

    if det.abs() < 1e-12 {
        return false;
    }

    let inv_det = 1.0 / det;
    let tvec = *origin - tri[0];
    let u = tvec.dot(pvec) * inv_det;
    if u < 0.0 || u > 1.0 {
        return false;
    }

    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = e2.dot(qvec) * inv_det;
    t > 1e-8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_voids_returns_unchanged() {
        let outer = MeshResult {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            indices: vec![0, 1, 2, -1],
            normals: vec![],
        };
        let result = subtract_void_meshes(&outer, &[]);
        assert_eq!(result.mesh.indices.len(), 4);
        assert_eq!(result.removed_tris, 0);
    }

    #[test]
    fn test_void_subtraction_removes_inner_tris() {
        // Outer: large square (two tris) at z=0 from [-2,-2] to [2,2]
        let outer = MeshResult {
            vertices: vec![
                Vec3::new(-2.0, -2.0, 0.0),
                Vec3::new(2.0, -2.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(-2.0, 2.0, 0.0),
            ],
            indices: vec![
                0, 1, 2, -1,  // centroid (0.667, -0.667) — inside void
                0, 2, 3, -1,  // centroid (-0.667, 0.667) — outside void
            ],
            normals: vec![],
        };
        // Void: box covering bottom-right quadrant only
        let void = MeshResult {
            vertices: vec![
                Vec3::new(0.5, -2.5, 0.1),
                Vec3::new(2.5, -2.5, 0.1),
                Vec3::new(2.5, -0.5, 0.1),
                Vec3::new(0.5, -0.5, 0.1),
                Vec3::new(0.5, -2.5, -0.1),
                Vec3::new(2.5, -2.5, -0.1),
                Vec3::new(2.5, -0.5, -0.1),
                Vec3::new(0.5, -0.5, -0.1),
            ],
            indices: vec![
                0, 1, 2, -1,  0, 2, 3, -1,  // top
                7, 6, 5, -1,  7, 5, 4, -1,  // bottom
                0, 4, 5, -1,  0, 5, 1, -1,  // front
                1, 5, 6, -1,  1, 6, 2, -1,  // right
                2, 6, 7, -1,  2, 7, 3, -1,  // back
                3, 7, 4, -1,  3, 4, 0, -1,  // left
            ],
            normals: vec![],
        };
        let result = subtract_void_meshes(&outer, &[void]);
        // First tri (centroid 0.667, -0.667) is inside void, second is outside
        assert_eq!(result.removed_tris, 1);
        assert_eq!(result.mesh.indices.len(), 4); // one tri remains: 3 indices + -1
    }

    #[test]
    fn test_ray_triangle_hit() {
        let tri = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(ray_triangle_intersect(
            &Vec3::new(-1.0, 0.3, 0.3),
            &Vec3::X,
            &tri
        ));
    }

    #[test]
    fn test_ray_triangle_miss() {
        let tri = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(!ray_triangle_intersect(
            &Vec3::new(-1.0, 2.0, 2.0),
            &Vec3::X,
            &tri
        ));
    }
}
