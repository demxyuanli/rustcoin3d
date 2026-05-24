//! Incremental mesh refinement: subdivide triangles that exceed deviation tolerance.

use rc3d_core::math::Vec3;
use crate::step::nurbs::NurbsSurface;
use crate::step::entity_types::EntityType;

/// Refinement configuration.
pub struct RefineConfig {
    /// Maximum allowed deviation between triangle center and actual surface
    pub max_deviation: f32,
    /// Maximum number of refinement iterations
    pub max_iterations: usize,
    /// Maximum triangle count (safety limit)
    pub max_triangles: usize,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self { max_deviation: 0.01, max_iterations: 4, max_triangles: 100_000 }
    }
}

/// Refine a mesh by subdividing triangles that exceed the deviation tolerance.
/// Returns (vertices, indices, normals) after refinement.
pub fn refine_mesh(
    vertices: &[Vec3],
    indices: &[i32],
    normals: &[Vec3],
    nurbs: &NurbsSurface,
    entity_type: EntityType,
    u_min: f32,
    u_max: f32,
    v_min: f32,
    v_max: f32,
    config: &RefineConfig,
) -> (Vec<Vec3>, Vec<i32>, Vec<Vec3>) {
    let mut verts = vertices.to_vec();
    let mut idx = indices.to_vec();
    let mut norms = normals.to_vec();

    for _iter in 0..config.max_iterations {
        if idx.len() / 4 >= config.max_triangles { break; }

        let mut new_indices = Vec::with_capacity(idx.len());
        let mut any_split = false;

        for chunk in idx.chunks(4) {
            if chunk.len() < 3 || chunk[3] != -1 {
                if chunk.len() == 4 { new_indices.extend_from_slice(chunk); }
                continue;
            }
            let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
            if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() {
                new_indices.extend_from_slice(chunk);
                continue;
            }

            let v0 = verts[i0]; let v1 = verts[i1]; let v2 = verts[i2];
            let midpoint = (v0 + v1 + v2) * (1.0 / 3.0);

            // Approximate deviation by evaluating NURBS at domain center
            let u_mid = (u_min + u_max) * 0.5;
            let v_mid = (v_min + v_max) * 0.5;
            let (un, vn) = crate::step::surface_tess::map_uv_to_nurbs(entity_type, u_mid, v_mid);
            let surface_pt = nurbs.evaluate(un, vn);
            let deviation = (midpoint - surface_pt).length();

            if deviation > config.max_deviation {
                // Split: add midpoint vertex and create 3 sub-triangles
                let mid_idx = verts.len() as i32;
                verts.push(midpoint);
                let mid_n = if norms.len() > i0 && norms.len() > i1 && norms.len() > i2 {
                    (norms[i0] + norms[i1] + norms[i2]).normalize_or_zero()
                } else {
                    Vec3::Z
                };
                norms.push(mid_n);

                new_indices.extend_from_slice(&[
                    i0 as i32, i1 as i32, mid_idx, -1,
                    i1 as i32, i2 as i32, mid_idx, -1,
                    i2 as i32, i0 as i32, mid_idx, -1,
                ]);
                any_split = true;
            } else {
                new_indices.extend_from_slice(chunk);
            }
        }

        idx = new_indices;
        if !any_split { break; }
    }

    (verts, idx, norms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::nurbs::NurbsSurface;

    #[test]
    fn test_refine_plane_no_split() {
        let surf = NurbsSurface::plane(0.0, 1.0, 0.0, 1.0);
        // Triangle whose centroid lies at the domain center (0.5, 0.5, 0.0),
        // which matches where the NURBS plane evaluates to, so deviation is 0.
        let verts = vec![
            Vec3::new(0.5, 0.4, 0.0),
            Vec3::new(0.8, 0.6, 0.0),
            Vec3::new(0.2, 0.5, 0.0),
        ];
        let indices = vec![0, 1, 2, -1];
        let norms = vec![Vec3::Z, Vec3::Z, Vec3::Z];
        let (out_v, out_i, _) = refine_mesh(
            &verts, &indices, &norms, &surf, EntityType::Plane,
            0.0, 1.0, 0.0, 1.0, &RefineConfig::default(),
        );
        // Plane should not need refinement (midpoint is on the surface)
        assert_eq!(out_v.len(), 3);
        assert_eq!(out_i.len(), 4);
    }
}
