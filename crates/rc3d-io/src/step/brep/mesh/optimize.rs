//! Delaunay edge-flip mesh optimization. T2.7

use rc3d_core::math::Vec3;
use crate::step::mesh_result::MeshResult;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    pub min_angle_degrees: f32,
    pub max_iterations: usize,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self { min_angle_degrees: 15.0, max_iterations: 3 }
    }
}

/// Improve mesh quality by flipping non-Delaunay interior edges.
pub fn optimize_mesh(mesh: &mut MeshResult, config: &OptimizeConfig) {
    for _iter in 0..config.max_iterations {
        let flipped = run_flip_pass(&mut mesh.vertices, &mut mesh.indices, &mut mesh.normals);
        if flipped == 0 { break; }
    }
}

/// One pass of edge flipping. Returns number of edges flipped.
fn run_flip_pass(verts: &mut Vec<Vec3>, indices: &mut Vec<i32>, _normals: &mut Vec<Vec3>) -> usize {
    // Build edge->triangle adjacency
    let mut edge_to_tris: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    let mut flipped = 0usize;

    for (ti, chunk) in indices.chunks(4).enumerate() {
        if chunk.len() < 3 { continue; }
        let a = chunk[0]; let b = chunk[1]; let c = chunk[2];
        for (x, y) in [(a,b), (b,c), (c,a)] {
            let key = if x < y { (x, y) } else { (y, x) };
            edge_to_tris.entry(key).or_default().push(ti);
        }
    }

    // Find interior edges (shared by exactly 2 triangles)
    let mut edges_to_flip: HashSet<(i32, i32)> = HashSet::new();
    for (&(a, b), tris) in &edge_to_tris {
        if tris.len() == 2 {
            let (t0, t1) = (tris[0], tris[1]);
            // Get the 4 vertices of the quadrilateral
            let v_a = verts[a as usize];
            let v_b = verts[b as usize];
            // Find the opposite vertices
            let chunk0 = &indices[t0*4..t0*4+3];
            let chunk1 = &indices[t1*4..t1*4+3];
            let opp0 = chunk0.iter().find(|&&v| v != a && v != b).copied().unwrap_or(-1);
            let opp1 = chunk1.iter().find(|&&v| v != a && v != b).copied().unwrap_or(-1);
            if opp0 < 0 || opp1 < 0 { continue; }

            // Delaunay criterion: current diagonal should be flipped if
            // opp0 and opp1 are inside the circumcircle of (a, b, opp0)
            // Simplified: flip if makes triangles more equiangular
            let current_min = min_angle_quad(v_a, v_b, verts[opp0 as usize], verts[opp1 as usize]);
            let flipped_min = min_angle_quad(verts[opp0 as usize], verts[opp1 as usize], v_a, v_b);

            if flipped_min > current_min {
                edges_to_flip.insert((a, b));
            }
        }
    }

    // Apply flips
    for &(a, b) in &edges_to_flip {
        if let Some(tris) = edge_to_tris.get(&(a, b)) {
            if tris.len() == 2 {
                let t0 = tris[0]; let t1 = tris[1];
                let c0 = &indices[t0*4..t0*4+3];
                let c1 = &indices[t1*4..t1*4+3];
                let opp0 = c0.iter().find(|&&v| v != a && v != b).copied().unwrap_or(-1);
                let opp1 = c1.iter().find(|&&v| v != a && v != b).copied().unwrap_or(-1);
                if opp0 >= 0 && opp1 >= 0 {
                    // Replace: (a,b,opp0) + (a,opp1,b) -> (a,opp1,opp0) + (b,opp0,opp1)
                    indices[t0*4] = a; indices[t0*4+1] = opp1; indices[t0*4+2] = opp0; indices[t0*4+3] = -1;
                    indices[t1*4] = b; indices[t1*4+1] = opp0; indices[t1*4+2] = opp1; indices[t1*4+3] = -1;
                    flipped += 1;
                }
            }
        }
    }

    flipped
}

fn min_angle_quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f32 {
    // Compute min angle in triangles (a,b,c) and (a,c,d)
    let tri1_min = min_tri_angle(a, b, c);
    let tri2_min = min_tri_angle(a, c, d);
    tri1_min.min(tri2_min).to_degrees()
}

fn min_tri_angle(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = (b - a).length(); let bc = (c - b).length(); let ca = (a - c).length();
    if ab < 1e-10 || bc < 1e-10 || ca < 1e-10 { return 0.0; }
    let alpha = ((bc*bc + ca*ca - ab*ab) / (2.0*bc*ca)).acos();
    let beta = ((ca*ca + ab*ab - bc*bc) / (2.0*ca*ab)).acos();
    let gamma = ((ab*ab + bc*bc - ca*ca) / (2.0*ab*bc)).acos();
    alpha.min(beta).min(gamma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_does_not_crash() {
        let mut mesh = MeshResult {
            vertices: vec![
                Vec3::new(0.,0.,0.), Vec3::new(1.,0.,0.),
                Vec3::new(1.,1.,0.), Vec3::new(0.,1.,0.),
            ],
            indices: vec![0,1,2,-1, 0,2,3,-1],
            normals: vec![Vec3::Z; 4],
        };
        optimize_mesh(&mut mesh, &OptimizeConfig::default());
        // Should not crash and should still be valid
        assert!(!mesh.indices.is_empty());
    }
}
