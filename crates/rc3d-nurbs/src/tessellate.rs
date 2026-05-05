use glam::Vec3;
use crate::surface::NurbsSurface;

/// Adaptive quadtree-based surface tessellation.
/// Subdivides a coarse grid where curvature exceeds tolerance.
pub fn tessellate_surface_adaptive(
    surface: &NurbsSurface,
    tolerance: f32,
) -> rc3d_mesh::TriangleMesh {
    let initial = 4;
    let max_depth = 5;

    // Build initial coarse mesh
    let mut positions = Vec::new();
    let n = initial + 1;
    for i in 0..n {
        for j in 0..n {
            let u = i as f32 / initial as f32;
            let v = j as f32 / initial as f32;
            positions.push(surface.evaluate(u, v));
        }
    }

    // Recursively subdivide each quad cell
    let mut indices = Vec::new();
    let stride = n;
    for ci in 0..initial {
        for cj in 0..initial {
            subdivide_cell(
                surface,
                &mut positions,
                &mut indices,
                ci as f32 / initial as f32,
                (ci + 1) as f32 / initial as f32,
                cj as f32 / initial as f32,
                (cj + 1) as f32 / initial as f32,
                ci * stride + cj,
                (ci + 1) * stride + cj,
                ci * stride + cj + 1,
                (ci + 1) * stride + cj + 1,
                tolerance,
                0,
                max_depth,
            );
        }
    }

    rc3d_mesh::TriangleMesh::from_indexed(&positions, &indices)
}

#[allow(clippy::too_many_arguments)]
fn subdivide_cell(
    surface: &NurbsSurface,
    positions: &mut Vec<Vec3>,
    indices: &mut Vec<u32>,
    u0: f32,
    u1: f32,
    v0: f32,
    v1: f32,
    idx00: usize,
    idx10: usize,
    idx01: usize,
    idx11: usize,
    tolerance: f32,
    depth: usize,
    max_depth: usize,
) {
    if depth >= max_depth {
        // Emit two triangles for this quad
        indices.extend_from_slice(&[
            idx00 as u32, idx10 as u32, idx11 as u32,
            idx00 as u32, idx11 as u32, idx01 as u32,
        ]);
        return;
    }

    let um = (u0 + u1) * 0.5;
    let vm = (v0 + v1) * 0.5;

    // Evaluate edge midpoints and center
    let p_u0 = surface.evaluate(um, v0);
    let p_u1 = surface.evaluate(um, v1);
    let p_v0 = surface.evaluate(u0, vm);
    let p_v1 = surface.evaluate(u1, vm);
    let p_center = surface.evaluate(um, vm);

    // Deviation: distance of surface center from geometric diagonal center
    let p00 = positions[idx00];
    let p11 = positions[idx11];
    let center = (p00 + p11) * 0.5;
    let deviation = (p_center - center).length();
    let diag = (p11 - p00).length();

    if diag < 1e-6 {
        indices.extend_from_slice(&[
            idx00 as u32, idx10 as u32, idx11 as u32,
            idx00 as u32, idx11 as u32, idx01 as u32,
        ]);
        return;
    }

    if deviation / diag > tolerance {
        // Add new vertices at midpoints and center
        let idx_um0 = positions.len();
        positions.push(p_u0);
        let idx_um1 = positions.len();
        positions.push(p_u1);
        let idx_0vm = positions.len();
        positions.push(p_v0);
        let idx_1vm = positions.len();
        positions.push(p_v1);
        let idx_center = positions.len();
        positions.push(p_center);

        // Recurse into 4 sub-quads
        let next_depth = depth + 1;
        subdivide_cell(surface, positions, indices, u0, um, v0, vm, idx00, idx_um0, idx_0vm, idx_center, tolerance, next_depth, max_depth);
        subdivide_cell(surface, positions, indices, um, u1, v0, vm, idx_um0, idx10, idx_center, idx_1vm, tolerance, next_depth, max_depth);
        subdivide_cell(surface, positions, indices, u0, um, vm, v1, idx_0vm, idx_center, idx01, idx_um1, tolerance, next_depth, max_depth);
        subdivide_cell(surface, positions, indices, um, u1, vm, v1, idx_center, idx_1vm, idx_um1, idx11, tolerance, next_depth, max_depth);
    } else {
        indices.extend_from_slice(&[
            idx00 as u32, idx10 as u32, idx11 as u32,
            idx00 as u32, idx11 as u32, idx01 as u32,
        ]);
    }
}
