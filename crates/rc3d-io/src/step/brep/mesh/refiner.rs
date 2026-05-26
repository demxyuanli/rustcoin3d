//! Deflection-driven mesh refinement. T2.5-T2.6

use std::collections::{HashMap, HashSet};

use rc3d_core::math::Vec3;
use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::topo::BRepFace;
use crate::step::mesh_result::MeshResult;
use super::face_fill::FaceMeshRange;

#[derive(Debug, Clone)]
pub struct RefineConfig {
    /// Post-fill refine pass (OCC does deflection during node insertion, not after).
    pub enable_post_refine: bool,
    pub max_deflection: f32,
    pub max_iterations: usize,
    /// Skip refine when fill already produced at least this many tris.
    pub skip_refine_above: usize,
    /// Hard cap on local tri count during iterative refinement.
    pub max_tris: usize,
    /// Angular deflection threshold for edge-split decisions.
    pub angular_deflection: f32,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            enable_post_refine: true,
            max_deflection: 0.01,
            max_iterations: 4,
            skip_refine_above: 512,
            max_tris: 8192,
            angular_deflection: 0.2,
        }
    }
}

/// Refine mesh by subdividing triangles where deflection exceeds threshold.
/// Returns refined (vertices, indices, normals). Normals are recomputed from
/// the surface (not averaged) for best accuracy.
pub fn refine_mesh(
    mesh: &MeshResult,
    surface: &SurfaceGeom,
    same_sense: bool,
    config: &RefineConfig,
) -> MeshResult {
    let mut verts = mesh.vertices.clone();
    let mut idx = mesh.indices.clone();
    let mut norms = mesh.normals.clone();

    for _iter in 0..config.max_iterations {
        let mut new_idx = Vec::with_capacity(idx.len());
        let mut any_split = false;

        for chunk in idx.chunks(4) {
            if chunk.len() < 4 || chunk[3] != -1 {
                if chunk.len() == 4 { new_idx.extend_from_slice(chunk); }
                continue;
            }
            let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
            if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() {
                new_idx.extend_from_slice(chunk);
                continue;
            }

            let v0 = verts[i0]; let v1 = verts[i1]; let v2 = verts[i2];

            // Check deviation at each edge midpoint
            let mut max_dev = 0.0f32;
            for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
                let mid_3d = (verts[a] + verts[b]) * 0.5;
                // Project to UV (use surface.project for available types)
                if let Some((u, v)) = surface.project(mid_3d) {
                    let surface_pt = surface.d0_native(u, v);
                    let dev = (mid_3d - surface_pt).length();
                    max_dev = max_dev.max(dev);
                }
            }

            if max_dev > config.max_deflection {
                // Split at centroid
                let mid_pt = (v0 + v1 + v2) * (1.0 / 3.0);
                let mid_idx = verts.len() as i32;
                verts.push(mid_pt);

                let mut mid_n = Vec3::Z;
                // Try to project centroid to UV for better normal
                if let Some((u, v)) = surface.project(mid_pt) {
                    mid_n = surface.normal_native(u, v);
                    if !same_sense { mid_n = -mid_n; }
                } else {
                    // Fallback: use normal from derivatives at domain center
                    if let Some((u, v)) = try_get_uv(mid_pt, surface) {
                        mid_n = surface.normal_native(u, v);
                        if !same_sense { mid_n = -mid_n; }
                    }
                }
                norms.push(mid_n);

                new_idx.extend_from_slice(&[
                    i0 as i32, i1 as i32, mid_idx, -1,
                    i1 as i32, i2 as i32, mid_idx, -1,
                    i2 as i32, i0 as i32, mid_idx, -1,
                ]);
                any_split = true;
            } else {
                new_idx.extend_from_slice(chunk);
            }
        }

        idx = new_idx;
        if !any_split { break; }
    }

    MeshResult { vertices: verts, indices: idx, normals: norms }
}

/// Refine only triangles whose vertices are all non-boundary locals (watertight rule).
pub fn refine_mesh_interior(
    mesh: &MeshResult,
    face: &BRepFace,
    local_boundary: &HashSet<usize>,
    config: &RefineConfig,
) -> MeshResult {
    let tri_count = mesh.indices.len() / 4;
    if tri_count == 0 || tri_count >= config.skip_refine_above {
        return mesh.clone();
    }
    let mut refined = mesh.clone();
    for _iter in 0..config.max_iterations {
        if refined.indices.len() / 4 >= config.max_tris {
            break;
        }
        let mut new_idx = Vec::new();
        let mut any_split = false;
        for chunk in refined.indices.chunks(4) {
            if chunk.len() < 4 || chunk[3] != -1 {
                if chunk.len() == 4 {
                    new_idx.extend_from_slice(chunk);
                }
                continue;
            }
            let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
            if local_boundary.contains(&i0)
                || local_boundary.contains(&i1)
                || local_boundary.contains(&i2)
            {
                new_idx.extend_from_slice(chunk);
                continue;
            }

            let v0 = refined.vertices[i0];
            let v1 = refined.vertices[i1];
            let v2 = refined.vertices[i2];
            let mut max_dev = 0.0f32;
            for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
                let mid_3d = (refined.vertices[a] + refined.vertices[b]) * 0.5;
                if let Some((u, v)) = face.surface.project(mid_3d) {
                    let surface_pt = face.surface.d0_native(u, v);
                    max_dev = max_dev.max((mid_3d - surface_pt).length());
                }
            }
            if max_dev <= config.max_deflection {
                new_idx.extend_from_slice(chunk);
                continue;
            }

            let mut mid_pt = (v0 + v1 + v2) * (1.0 / 3.0);
            if let Some((u, v)) = face.surface.project(mid_pt) {
                mid_pt = face.surface.d0_native(u, v);
            }
            let mid_idx = refined.vertices.len() as i32;
            refined.vertices.push(mid_pt);
            let mut mid_n = Vec3::Z;
            if let Some((u, v)) = face.surface.project(mid_pt) {
                mid_n = face.surface.normal_native(u, v);
                if !face.same_sense {
                    mid_n = -mid_n;
                }
            }
            refined.normals.push(mid_n);
            new_idx.extend_from_slice(&[
                i0 as i32, i1 as i32, mid_idx, -1,
                i1 as i32, i2 as i32, mid_idx, -1,
                i2 as i32, i0 as i32, mid_idx, -1,
            ]);
            any_split = true;
        }
        refined.indices = new_idx;
        if !any_split {
            break;
        }
    }
    refined
}

pub fn extract_face_mesh_with_map(
    global_vertices: &[Vec3],
    global_normals: &[Vec3],
    all_indices: &[i32],
    range: &FaceMeshRange,
) -> (MeshResult, HashMap<usize, usize>) {
    let start = range.first_tri * 4;
    let end = start + range.tri_count * 4;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut local_to_global: HashMap<usize, usize> = HashMap::new();
    let mut gi_to_local: HashMap<i32, usize> = HashMap::new();
    let mut indices = Vec::new();

    for chunk in all_indices[start..end.min(all_indices.len())].chunks(4) {
        if chunk.len() < 4 {
            continue;
        }
        let mut local_tri = [0i32; 3];
        for (j, &gi) in chunk[0..3].iter().enumerate() {
            let local_i = *gi_to_local.entry(gi).or_insert_with(|| {
                let li = vertices.len();
                let g = gi as usize;
                local_to_global.insert(li, g);
                vertices.push(global_vertices[g]);
                normals.push(global_normals.get(g).copied().unwrap_or(Vec3::Z));
                li
            });
            local_tri[j] = local_i as i32;
        }
        indices.extend_from_slice(&[local_tri[0], local_tri[1], local_tri[2], -1]);
    }

    (
        MeshResult {
            vertices,
            indices,
            normals,
        },
        local_to_global,
    )
}

pub fn merge_refined_face(
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    all_indices: &mut Vec<i32>,
    range: &FaceMeshRange,
    refined: &MeshResult,
    local_to_global: &HashMap<usize, usize>,
) {
    let start = range.first_tri * 4;
    let end = start + range.tri_count * 4;

    let mut local_remap: HashMap<usize, i32> = HashMap::new();
    for (&local_i, &global_i) in local_to_global {
        local_remap.insert(local_i, global_i as i32);
    }
    for local_i in local_to_global.len()..refined.vertices.len() {
        let gi = global_vertices.len();
        global_vertices.push(refined.vertices[local_i]);
        global_normals.push(refined.normals.get(local_i).copied().unwrap_or(Vec3::Z));
        local_remap.insert(local_i, gi as i32);
    }

    let mut new_tris = Vec::with_capacity(refined.indices.len());
    for chunk in refined.indices.chunks(4) {
        if chunk.len() < 4 {
            continue;
        }
        let i0 = *local_remap.get(&(chunk[0] as usize)).unwrap_or(&chunk[0]);
        let i1 = *local_remap.get(&(chunk[1] as usize)).unwrap_or(&chunk[1]);
        let i2 = *local_remap.get(&(chunk[2] as usize)).unwrap_or(&chunk[2]);
        new_tris.extend_from_slice(&[i0, i1, i2, -1]);
    }

    all_indices.splice(start..end.min(all_indices.len()), new_tris);
}

/// Try to get UV coordinates for a 3D point. Returns None if impossible.
fn try_get_uv(point: Vec3, surface: &SurfaceGeom) -> Option<(f32, f32)> {
    surface.project(point)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refine_plane_no_change() {
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let mesh = MeshResult {
            vertices: vec![Vec3::new(0.,0.,0.), Vec3::new(1.,0.,0.), Vec3::new(0.,1.,0.)],
            indices: vec![0, 1, 2, -1],
            normals: vec![Vec3::Z, Vec3::Z, Vec3::Z],
        };
        let result = refine_mesh(
            &mesh,
            &surface,
            true,
            &RefineConfig {
                enable_post_refine: true,
                max_iterations: 4,
                ..Default::default()
            },
        );
        // Plane should not need refinement (exact representation)
        assert_eq!(result.vertices.len(), 3);
    }
}
