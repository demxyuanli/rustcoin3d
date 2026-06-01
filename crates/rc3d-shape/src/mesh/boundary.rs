use std::collections::HashMap;

use rc3d_core::math::Vec3;
use rc3d_core::utils::hash::f32x3_quantized_bits;

/// Register a boundary point in the shared pool (quantized dedup; topology vertices seeded first).
pub fn register_boundary_point(
    pt: Vec3,
    global_vertices: &mut Vec<Vec3>,
    global_normals: &mut Vec<Vec3>,
    pos_to_idx: &mut HashMap<[u32; 3], usize>,
) -> usize {
    let hash = f32x3_quantized_bits([pt.x, pt.y, pt.z]);
    *pos_to_idx.entry(hash).or_insert_with(|| {
        let i = global_vertices.len();
        global_vertices.push(pt);
        global_normals.push(Vec3::ZERO);
        i
    })
}
