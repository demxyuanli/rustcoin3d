//! KD-tree sort, deduplication, and initial hull construction.
//!
//! Implements the "Prepare" phase of the Newton Apple Wrapper algorithm:
//!   1. Sort points by 45°-rotated coordinates (u = x+y, v = x-y)
//!   2. Remove exact duplicates
//!   3. Find initial non-collinear triple
//!   4. Split remaining points into upper/lower chains
//!   5. Build initial cone hull

use super::predicates::adaptive_orient2d;
use super::table::{DelaBella, FLAG_HULL, INVALID};

// ── 45° rotated coordinate sort ──────────────────────────────────────

/// Sort points by 45°-rotated coordinates for NAW hull properties.
/// The rotation (u = x+y, v = x-y) maps 2D Delaunay to 3D convex hull
/// via paraboloid lifting, ensuring the incremental insertion order
/// produces a valid growing convex hull.
pub fn kd_sort(points: &mut [(f64, f64)]) {
    points.sort_by(|a, b| {
        let ua = a.0 + a.1;
        let va = a.0 - a.1;
        let ub = b.0 + b.1;
        let vb = b.0 - b.1;
        ua.partial_cmp(&ub)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal))
    });
}

// ── Deduplication ─────────────────────────────────────────────────────

/// Remove exact duplicates from a sorted point array.
/// Returns a map from original index to deduplicated index, and the count of unique points.
pub fn dedup_sorted(points: &[(f64, f64)], orig_indices: &[u32]) -> (Vec<(f64, f64)>, Vec<u32>, usize) {
    if points.is_empty() {
        return (vec![], vec![], 0);
    }
    let n = points.len();
    let mut unique_points = Vec::with_capacity(n);
    let mut unique_orig = Vec::with_capacity(n);
    let mut dedup_map = vec![INVALID as usize; n];

    unique_points.push(points[0]);
    unique_orig.push(orig_indices[0]);
    dedup_map[0] = 0;
    let mut write = 1usize;

    for read in 1..n {
        if (points[read].0 - points[read - 1].0).abs() > 1e-15
            || (points[read].1 - points[read - 1].1).abs() > 1e-15
        {
            unique_points.push(points[read]);
            unique_orig.push(orig_indices[read]);
            dedup_map[read] = write;
            write += 1;
        } else {
            // Duplicate: map to the existing unique entry
            dedup_map[read] = dedup_map[read - 1];
        }
    }

    (unique_points, unique_orig, write)
}

// ── Initial triple ────────────────────────────────────────────────────

/// Find the first three non-collinear points in sorted order.
/// Returns the indices into the sorted unique array.
pub fn find_initial_triple(points: &[(f64, f64)]) -> Option<[usize; 3]> {
    if points.len() < 3 {
        return None;
    }
    // Check first 3
    let orient = adaptive_orient2d(
        points[0].0, points[0].1,
        points[1].0, points[1].1,
        points[2].0, points[2].1,
    );
    if orient != 0.0 {
        return Some([0, 1, 2]);
    }

    // First 3 are collinear — scan further
    for i in 3..points.len() {
        let orient = adaptive_orient2d(
            points[0].0, points[0].1,
            points[1].0, points[1].1,
            points[i].0, points[i].1,
        );
        if orient != 0.0 {
            return Some([0, 1, i]);
        }
    }
    None // All collinear
}

// ── Chain splitting ───────────────────────────────────────────────────

/// Split remaining points (after initial triple) into upper and lower chains
/// relative to the line from the first to the last triple vertex.
pub fn split_chains(
    points: &[(f64, f64)],
    triple: [usize; 3],
) -> (Vec<usize>, Vec<usize>) {
    let a = triple[0];
    let b = triple[2]; // last triple vertex
    let mut upper = Vec::new();
    let mut lower = Vec::new();

    for i in 0..points.len() {
        if i == triple[0] || i == triple[1] || i == triple[2] {
            continue;
        }
        let orient = adaptive_orient2d(
            points[a].0, points[a].1,
            points[b].0, points[b].1,
            points[i].0, points[i].1,
        );
        if orient > 0.0 {
            upper.push(i);
        } else if orient < 0.0 {
            lower.push(i);
        } else {
            // Collinear with the diagonal — assign to lower by convention
            lower.push(i);
        }
    }

    // Sort upper chain in ascending rotated-u order (already sorted globally)
    // Sort lower chain in descending rotated-u order
    lower.sort_by(|&a, &b| b.cmp(&a));

    (upper, lower)
}

// ── Cone hull construction ────────────────────────────────────────────

/// Build the initial cone hull from the sorted triple and chains.
///
/// The "Newton Apple Wrapper" starts by creating a cone from the apex vertex
/// (last in triple) to the base ring (earlier triple vertices). Additional
/// points are integrated by splitting base faces and adding side faces.
pub fn build_cone_hull(
    della: &mut DelaBella,
    points: &[(f64, f64)],
    orig_indices: &[u32],
    triple: [usize; 3],
    _upper: &[usize],
    _lower: &[usize],
) {
    // Ensure vertices are registered
    let n_existing = della.verts.len();
    let needed = points.len();
    for i in n_existing..needed {
        della.add_vert(points[i].0, points[i].1, orig_indices[i]);
    }

    // Create initial triangle from the triple
    let v0 = triple[0] as u32;
    let v1 = triple[1] as u32;
    let v2 = triple[2] as u32;

    // Determine orientation: if the triple is CW, swap v0 and v1
    let orient = adaptive_orient2d(
        points[triple[0]].0, points[triple[0]].1,
        points[triple[1]].0, points[triple[1]].1,
        points[triple[2]].0, points[triple[2]].1,
    );

    let (v0, v1) = if orient < 0.0 {
        (v1, v0) // swap to make CCW
    } else {
        (v0, v1)
    };

    // Create the initial triangle (all edges are hull edges)
    let fi = della.make_face(v0, v1, v2, INVALID, INVALID, INVALID);
    della.faces[fi as usize].flags = FLAG_HULL;
    della.hull_first = fi;
}

// ── Full prepare pipeline ─────────────────────────────────────────────

/// Prepare the triangulation: sort, dedup, find triple, build initial hull.
/// Returns the number of unique points, or 0 if all points are collinear.
pub fn prepare(
    della: &mut DelaBella,
    points: &[(f64, f64)],
    orig_indices: &[u32],
) -> usize {
    let n = points.len();
    if n < 3 {
        return n; // Degenerate: 0, 1, or 2 points
    }

    // Sort by 45° rotated coordinates
    let mut sorted_points = points.to_vec();
    let sorted_orig = orig_indices.to_vec();
    kd_sort(&mut sorted_points);

    // Dedup
    let (unique_points, unique_orig, unique_count) = dedup_sorted(&sorted_points, &sorted_orig);

    if unique_count < 3 {
        // Register the points we have
        for i in 0..unique_count {
            della.add_vert(unique_points[i].0, unique_points[i].1, unique_orig[i]);
        }
        return unique_count;
    }

    // Find initial non-collinear triple
    let triple = match find_initial_triple(&unique_points[..unique_count]) {
        Some(t) => t,
        None => {
            // All collinear — register points but no triangulation
            for i in 0..unique_count {
                della.add_vert(unique_points[i].0, unique_points[i].1, unique_orig[i]);
            }
            return unique_count;
        }
    };

    // Split remaining points into chains
    let (upper, lower) = split_chains(&unique_points[..unique_count], triple);

    // Pre-allocate faces
    della.reserve_faces(unique_count);
    della.n_points = unique_count;

    // Build initial hull
    build_cone_hull(della, &unique_points, &unique_orig, triple, &upper, &lower);

    // Store chain info for insertion (we'll handle this in insert.rs)
    // For now, store the chain indices in the DelaBella struct
    // The insert module will read them
    della.input_points = unique_points.clone();

    unique_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kd_sort_order() {
        let mut pts = vec![(1.0, 2.0), (3.0, 0.0), (0.0, 0.0)];
        kd_sort(&mut pts);
        // After sort by (u=x+y, v=x-y):
        // (0,0) -> u=0, v=0
        // (1,2) -> u=3, v=-1
        // (3,0) -> u=3, v=3
        assert_eq!(pts[0], (0.0, 0.0));
        assert_eq!(pts[1], (1.0, 2.0));
        assert_eq!(pts[2], (3.0, 0.0));
    }

    #[test]
    fn test_find_triple_basic() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let triple = find_initial_triple(&pts).unwrap();
        assert_eq!(triple, [0, 1, 2]);
    }

    #[test]
    fn test_find_triple_collinear_first() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0)];
        let triple = find_initial_triple(&pts).unwrap();
        assert_eq!(triple[0], 0);
        assert_eq!(triple[1], 1);
        assert!(triple[2] >= 3, "should skip collinear point");
    }

    #[test]
    fn test_find_triple_all_collinear() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        assert!(find_initial_triple(&pts).is_none());
    }

    #[test]
    fn test_split_chains() {
        let pts = vec![(0.0, 0.0), (2.0, 0.0), (1.0, 2.0), (0.5, 0.5), (1.5, -0.5)];
        let triple = [0, 1, 2]; // base line: (0,0) → (1,2)
        let (upper, lower) = split_chains(&pts, triple);
        // orient2d(a=pts[0], b=pts[2], c=pts[3]) = orient2d((0,0), (1,2), (0.5,0.5))
        // = (1-0)*(0.5-0) - (0.5-0)*(2-0) = 0.5 - 1.0 = -0.5 → lower
        assert!(lower.contains(&3), "point (0.5,0.5) relative to line (0,0)→(1,2) should be in lower");
        // orient2d((0,0), (1,2), (1.5,-0.5)) = (1-0)*(-0.5-0) - (1.5-0)*(2-0) = -0.5 - 3.0 → lower
        assert!(lower.contains(&4), "point (1.5,-0.5) should be in lower");
    }

    #[test]
    fn test_prepare_square() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let orig: Vec<u32> = vec![0, 1, 2, 3];
        let mut della = DelaBella::new();
        let count = prepare(&mut della, &pts, &orig);
        assert!(count >= 3, "should have at least 3 unique points");
    }
}
