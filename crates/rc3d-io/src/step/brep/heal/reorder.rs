//! Wire edge reordering. T3.1

use super::super::topo::{EdgeKey, Orientation};
use super::super::registry::BRepRegistry;
use rc3d_core::math::Vec3;

/// Reorder wire edges into a connected chain. Returns None if disconnected.
pub fn reorder_wire_edges(
    edges: &[(EdgeKey, Orientation)],
    reg: &BRepRegistry,
) -> Option<Vec<(EdgeKey, Orientation)>> {
    if edges.len() <= 1 { return Some(edges.to_vec()); }

    // Build adjacency: which vertex connects edge i to edge j?
    // We approximate using edge curve endpoints
    let n = edges.len();
    let mut endpoints: Vec<Option<(u64, u64)>> = vec![None; n]; // (start_hash, end_hash)
    for (i, (ek, orient)) in edges.iter().enumerate() {
        if let Some(edge) = reg.edges.get(*ek) {
            let (p0, p1) = if *orient == Orientation::Forward {
                (edge.v_low, edge.v_high)
            } else {
                (edge.v_high, edge.v_low)
            };
            let pos0 = reg.vertices.get(p0).map(|v| v.position);
            let pos1 = reg.vertices.get(p1).map(|v| v.position);
            if let (Some(p0), Some(p1)) = (pos0, pos1) {
                let h0 = quantize(p0);
                let h1 = quantize(p1);
                endpoints[i] = Some((h0, h1));
            }
        }
    }

    let mut used = vec![false; n];
    let mut result = Vec::with_capacity(n);

    // Start from first edge, follow chain
    let mut current = 0;
    used[current] = true;
    result.push(edges[current]);

    for _ in 1..n {
        let (_, ref_prev_end) = endpoints[current]?;
        let mut found = false;
        for j in 0..n {
            if used[j] { continue; }
            let (next_start, _) = endpoints[j]?;
            if next_start == ref_prev_end {
                current = j;
                used[current] = true;
                result.push(edges[current]);
                found = true;
                break;
            }
            // Also try reversed connection
            let (_, next_end) = endpoints[j]?;
            if next_end == ref_prev_end {
                current = j;
                used[current] = true;
                result.push((edges[current].0, edges[current].1)); // keep original orientation
                found = true;
                break;
            }
        }
        if !found { return None; } // disconnected
    }

    Some(result)
}

fn quantize(p: Vec3) -> u64 {
    let x = (p.x * 1e6) as i64;
    let y = (p.y * 1e6) as i64;
    let z = (p.z * 1e6) as i64;
    ((x as u64) << 40) ^ ((y as u64) << 20) ^ (z as u64)
}
