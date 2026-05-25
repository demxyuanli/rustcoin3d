//! Deflection-driven mesh refinement. T2.5-T2.6

use rc3d_core::math::Vec3;
use crate::step::brep::geom::SurfaceGeom;
use crate::step::tessellate::MeshResult;

#[derive(Debug, Clone)]
pub struct RefineConfig {
    pub max_deflection: f32,
    pub max_iterations: usize,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self { max_deflection: 0.05, max_iterations: 4 }
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
                    let surface_pt = surface.d0(u, v);
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
                    mid_n = surface.normal(u, v);
                    if !same_sense { mid_n = -mid_n; }
                } else {
                    // Fallback: use normal from derivatives at domain center
                    if let Some((u, v)) = try_get_uv(mid_pt, surface) {
                        mid_n = surface.normal(u, v);
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
        let result = refine_mesh(&mesh, &surface, true, &RefineConfig::default());
        // Plane should not need refinement (exact representation)
        assert_eq!(result.vertices.len(), 3);
    }
}
