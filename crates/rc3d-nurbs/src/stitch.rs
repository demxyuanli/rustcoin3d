//! Multi-patch NURBS surface stitching with G0/G1 continuity.

use crate::surface::{BoundaryEdge, NurbsRenderSurface, TessellatedSurface};
use glam::Vec3;

/// Continuity mode for patch stitching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StitchMode {
    /// Position continuity only (G0): shared boundary vertices are deduplicated
    /// but normals remain patch-local.
    PositionOnly,
    /// Tangent continuity (G1): normals at shared boundary vertices are averaged
    /// between adjacent patches for smoother appearance at seams.
    NormalContinuity,
}

/// Error during patch stitching.
#[derive(Clone, Debug)]
pub enum StitchError {
    BoundaryMismatch {
        edge_a: BoundaryEdge,
        edge_b: BoundaryEdge,
        max_error: f32,
        tolerance: f32,
    },
    EmptyBoundary,
}

/// Stitch two patches along specified edges.
pub fn stitch_two(
    a: &NurbsRenderSurface,
    b: &NurbsRenderSurface,
    edge_a: BoundaryEdge,
    edge_b: BoundaryEdge,
    mode: StitchMode,
    tolerance: f32,
) -> Result<TessellatedSurface, StitchError> {
    let curve_a = a.boundary_curve(edge_a);
    let curve_b = b.boundary_curve(edge_b);

    if curve_a.control_points.len() != curve_b.control_points.len()
        || curve_a.control_points.is_empty()
    {
        return Err(StitchError::EmptyBoundary);
    }
    let mut max_err = 0.0f32;
    for (pa, pb) in curve_a
        .control_points
        .iter()
        .zip(curve_b.control_points.iter())
    {
        let va = Vec3::new(pa[0] / pa[3], pa[1] / pa[3], pa[2] / pa[3]);
        let vb = Vec3::new(pb[0] / pb[3], pb[1] / pb[3], pb[2] / pb[3]);
        max_err = max_err.max((va - vb).length());
    }
    if max_err > tolerance {
        return Err(StitchError::BoundaryMismatch {
            edge_a,
            edge_b,
            max_error: max_err,
            tolerance,
        });
    }

    let (u_a, v_a) = (a.u_count() * 4, a.v_count() * 4);
    let (u_b, v_b) = (b.u_count() * 4, b.v_count() * 4);
    let ts_a = a.tessellate_uniform_with_normals(u_a, v_a);
    let ts_b = b.tessellate_uniform_with_normals(u_b, v_b);

    merge_tessellated_pair(&ts_a, &ts_b, edge_a, edge_b, u_a, v_a, u_b, v_b, mode)
}

/// Merge two already-tessellated surfaces, deduplicating shared boundary vertices.
fn merge_tessellated_pair(
    ts_a: &TessellatedSurface,
    ts_b: &TessellatedSurface,
    edge_a: BoundaryEdge,
    edge_b: BoundaryEdge,
    u_a: usize,
    v_a: usize,
    u_b: usize,
    v_b: usize,
    mode: StitchMode,
) -> Result<TessellatedSurface, StitchError> {
    let mut positions = ts_a.positions.clone();
    let mut normals = ts_a.normals.clone();
    let mut indices = ts_a.indices.clone();

    let a_count = positions.len();
    let total_b = ts_b.positions.len();
    let mut b_new_idx: Vec<usize> = vec![0; total_b];

    let (a_bdy_start, a_bdy_count, a_bdy_stride) = boundary_params(edge_a, u_a, v_a);
    let (b_bdy_start, b_bdy_count, b_bdy_stride) = boundary_params(edge_b, u_b, v_b);

    // Verify boundary counts match
    if a_bdy_count != b_bdy_count {
        return Err(StitchError::EmptyBoundary);
    }

    // Build skip set for patch B's boundary vertices
    let mut skip_set = vec![false; total_b];
    for k in 0..b_bdy_count {
        let bi = b_bdy_start + k * b_bdy_stride;
        if bi < total_b {
            skip_set[bi] = true;
        }
    }

    for (i, pos) in ts_b.positions.iter().enumerate() {
        if skip_set[i] {
            // Map to corresponding boundary vertex on patch A
            let bdy_idx = if b_bdy_stride > 0 {
                (i.saturating_sub(b_bdy_start)) / b_bdy_stride
            } else {
                0
            };
            let ref_idx = a_bdy_start + bdy_idx.min(a_bdy_count.saturating_sub(1)) * a_bdy_stride;
            let clamped = ref_idx.min(a_count.saturating_sub(1));
            b_new_idx[i] = clamped;
            if matches!(mode, StitchMode::NormalContinuity) && clamped < a_count {
                let na = normals[clamped];
                let nb = ts_b.normals[i];
                let avg = (na + nb).normalize();
                if avg.length() > 0.001 {
                    normals[clamped] = avg;
                }
            }
        } else {
            positions.push(*pos);
            normals.push(ts_b.normals[i]);
            b_new_idx[i] = positions.len() - 1;
        }
    }

    // Remap and append patch B indices
    for &idx in &ts_b.indices {
        indices.push(b_new_idx[idx as usize] as u32);
    }

    Ok(TessellatedSurface {
        positions,
        normals,
        indices,
    })
}

/// Stitch a rectangular grid of equally-sized patches.
///
/// All patches must have the same `u_count()` and `v_count()`.
pub fn stitch_grid(
    patches: &[NurbsRenderSurface],
    rows: usize,
    cols: usize,
    mode: StitchMode,
    _tolerance: f32,
) -> Result<TessellatedSurface, StitchError> {
    if patches.is_empty() || rows == 0 || cols == 0 || patches.len() != rows * cols {
        return Err(StitchError::EmptyBoundary);
    }
    let u_per = patches[0].u_count() * 4;
    let v_per = patches[0].v_count() * 4;

    // Stitch each row left-to-right
    let mut row_results: Vec<TessellatedSurface> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row_ts: Option<TessellatedSurface> = None;
        for c in 0..cols {
            let ts = patches[r * cols + c].tessellate_uniform_with_normals(u_per, v_per);
            match row_ts.take() {
                None => row_ts = Some(ts),
                Some(prev) => {
                    let merged = merge_tessellated_pair(
                        &prev, &ts,
                        BoundaryEdge::UMax, BoundaryEdge::UMin,
                        u_per, v_per, u_per, v_per, mode,
                    )?;
                    row_ts = Some(merged);
                }
            }
        }
        row_results.push(row_ts.unwrap());
    }

    // Stitch rows vertically
    let mut result = row_results.remove(0);
    for row_ts in row_results {
        let u_merged = cols * u_per;
        result = merge_tessellated_pair(
            &result, &row_ts,
            BoundaryEdge::VMax, BoundaryEdge::VMin,
            u_merged, v_per, u_merged, v_per, mode,
        )?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use BoundaryEdge::*;

    fn flat_patch(ox: f32, oy: f32) -> NurbsRenderSurface {
        let grid: Vec<Vec<Vec3>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| Vec3::new(ox + i as f32, oy + j as f32, 0.0))
                    .collect()
            })
            .collect();
        NurbsRenderSurface::from_points_grid(&grid, 2, 2)
    }

    #[test]
    fn boundary_curve_extraction() {
        let s = flat_patch(0.0, 0.0);
        let curve = s.boundary_curve(UMin);
        assert_eq!(curve.control_points.len(), 3);
        let curve = s.boundary_curve(VMax);
        assert_eq!(curve.control_points.len(), 3);
    }

    #[test]
    fn stitch_two_identical_patches() {
        let a = flat_patch(0.0, 0.0);
        let b = flat_patch(2.0, 0.0); // adjacent to the right
        let result = stitch_two(&a, &b, UMax, UMin, StitchMode::PositionOnly, 0.01);
        assert!(result.is_ok());
        let ts = result.unwrap();
        // Combined: 2x 7x7 = 98 vertices, minus 7 shared boundary = 91
        assert!(ts.positions.len() > 80);
        assert!(!ts.indices.is_empty());
    }

    #[test]
    fn stitch_two_mismatched_boundaries_errors() {
        let a = flat_patch(0.0, 0.0);
        let b = flat_patch(2.0, 1.0); // offset in y
        let result = stitch_two(&a, &b, UMax, UMin, StitchMode::PositionOnly, 0.001);
        assert!(result.is_err());
    }

    #[test]
    fn stitch_grid_2x2() {
        let patches: Vec<NurbsRenderSurface> = (0..2)
            .flat_map(|ri| {
                (0..2).map(move |ci| flat_patch(ci as f32 * 2.0, ri as f32 * 2.0))
            })
            .collect();
        let result = stitch_grid(&patches, 2, 2, StitchMode::PositionOnly, 0.01);
        assert!(result.is_ok());
    }
}

fn boundary_params(
    edge: BoundaryEdge,
    u_samples: usize,
    v_samples: usize,
) -> (usize, usize, usize) {
    let stride = v_samples + 1;
    match edge {
        BoundaryEdge::UMin => (0, v_samples + 1, 1),
        BoundaryEdge::UMax => (u_samples * stride, v_samples + 1, 1),
        BoundaryEdge::VMin => (0, u_samples + 1, stride),
        BoundaryEdge::VMax => (v_samples, u_samples + 1, stride),
    }
}
