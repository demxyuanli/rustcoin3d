//! Polygon walking and ear-cut triangulation (OCC meshLeftPolygonOf / meshPolygon).

use std::collections::{HashMap, HashSet};

use super::geom::{robust_orient2d, Point2d};
use super::half_edge::VertIdx;
use super::triangulation::{edge_key, point_in_triangle, segments_cross_proper};

const EPS: f64 = 1e-12;

/// CCW signed area of a closed polygon.
pub fn signed_area_2d(points: &[Point2d], verts: &[VertIdx]) -> f64 {
    if verts.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..verts.len() {
        let a = points[verts[i] as usize];
        let b = points[verts[(i + 1) % verts.len()] as usize];
        area += a.x * b.y - b.x * a.y;
    }
    area * 0.5
}

/// Walk a closed polygon CCW starting from directed edge (start -> end).
///
/// Uses OCC `findNextPolygonLink` semantics: maximum turn angle, intersection
/// rejection, leprous/dead link tracking, and backtracking on dead ends.
pub fn walk_polygon_ccw(
    points: &[Point2d],
    start: VertIdx,
    end: VertIdx,
    adj: &HashMap<VertIdx, Vec<VertIdx>>,
) -> Vec<VertIdx> {
    walk_polygon_occ(points, start, end, adj)
}

fn walk_polygon_occ(
    points: &[Point2d],
    start: VertIdx,
    end: VertIdx,
    adj: &HashMap<VertIdx, Vec<VertIdx>>,
) -> Vec<VertIdx> {
    // Degenerate guard: no edges or trivially collinear input
    let total_edges: usize = adj.values().map(|v| v.len()).sum();
    if total_edges == 0 {
        return vec![start];
    }
    let mut poly = vec![start, end];
    let mut dead: HashSet<(VertIdx, VertIdx)> = HashSet::new();
    let mut leprous: HashSet<(VertIdx, VertIdx)> = HashSet::new();
    leprous.insert(edge_key(start, end));

    let max_steps = adj.values().map(|v| v.len()).sum::<usize>().saturating_mul(4) + 8;
    let mut skip_leprous = true;

    for _ in 0..max_steps {
        if poly.len() >= 3 && *poly.last().unwrap_or(&start) == start {
            poly.pop();
            break;
        }

        let n = poly.len();
        let prev = poly[n - 2];
        let pivot = poly[n - 1];
        if pivot == start {
            break;
        }

        let ref_dir = link_dir(points, prev, pivot);
        if ref_dir.0 * ref_dir.0 + ref_dir.1 * ref_dir.1 < EPS * EPS {
            break;
        }

        if let Some(next) = find_next_polygon_link(
            points,
            start,
            prev,
            pivot,
            ref_dir,
            &poly,
            adj,
            skip_leprous,
            &mut leprous,
            &dead,
        ) {
            poly.push(next);
            skip_leprous = true;
        } else if poly.len() <= 2 {
            break;
        } else {
            let tail = poly.pop().unwrap_or(start);
            let back_prev = *poly.last().unwrap_or(&start);
            dead.insert(edge_key(back_prev, tail));
            leprous.remove(&edge_key(back_prev, tail));
            skip_leprous = false;
        }
    }

    if poly.len() >= 3 && poly.first() == poly.last() {
        poly.pop();
    }
    poly
}

/// OCC findNextPolygonLink: neighbor with maximum CCW angle w.r.t. reference link.
fn find_next_polygon_link(
    points: &[Point2d],
    first: VertIdx,
    prev: VertIdx,
    pivot: VertIdx,
    ref_dir: (f64, f64),
    poly: &[VertIdx],
    adj: &HashMap<VertIdx, Vec<VertIdx>>,
    skip_leprous: bool,
    leprous: &mut HashSet<(VertIdx, VertIdx)>,
    dead: &HashSet<(VertIdx, VertIdx)>,
) -> Option<VertIdx> {
    let neighbors = adj.get(&pivot).map(|v| v.as_slice()).unwrap_or(&[]);
    let mut best_angle = f64::NEG_INFINITY;
    let mut best: Option<VertIdx> = None;

    for &other in neighbors {
        if other == prev {
            continue;
        }
        let ek = edge_key(pivot, other);
        if dead.contains(&ek) {
            continue;
        }
        if skip_leprous && leprous.contains(&ek) {
            continue;
        }

        let cur_dir = link_dir(points, pivot, other);
        if cur_dir.0 * cur_dir.0 + cur_dir.1 * cur_dir.1 < EPS * EPS {
            continue;
        }

        if !leprous.contains(&ek) {
            leprous.insert(ek);
        }

        let angle = signed_angle(ref_dir, cur_dir);
        if angle <= best_angle {
            continue;
        }

        let check_endpoints = other != first;
        if link_intersects_polygon(points, pivot, other, poly, check_endpoints) {
            continue;
        }

        best_angle = angle;
        best = Some(other);
    }

    best
}

fn link_dir(points: &[Point2d], a: VertIdx, b: VertIdx) -> (f64, f64) {
    let pa = points[a as usize];
    let pb = points[b as usize];
    (pb.x - pa.x, pb.y - pa.y)
}

fn signed_angle(ref_dir: (f64, f64), cur_dir: (f64, f64)) -> f64 {
    let (rx, ry) = ref_dir;
    let (cx, cy) = cur_dir;
    let cross = rx * cy - ry * cx;
    let dot = rx * cx + ry * cy;
    cross.atan2(dot)
}

/// Returns true if directed link (a -> b) crosses an earlier polygon edge.
fn link_intersects_polygon(
    points: &[Point2d],
    a: VertIdx,
    b: VertIdx,
    poly: &[VertIdx],
    check_endpoints: bool,
) -> bool {
    if poly.len() < 2 {
        return false;
    }
    let pa = points[a as usize];
    let pb = points[b as usize];

    // Skip the edge ending at pivot (a): poly[n-2] -> poly[n-1] == a when extending from a.
    let skip_last = poly.len().saturating_sub(2);
    for i in 0..skip_last {
        let c = poly[i];
        let d = poly[i + 1];
        if c == d {
            continue;
        }
        let pc = points[c as usize];
        let pd = points[d as usize];

        if !check_endpoints && (c == b || d == b) {
            continue;
        }
        if segments_cross_proper(pa, pb, pc, pd) {
            return true;
        }
    }
    false
}

/// Order cavity boundary loop starting from constrained edge (a -> b).
pub fn order_boundary_loop(
    points: &[Point2d],
    a: VertIdx,
    b: VertIdx,
    boundary: &HashMap<(VertIdx, VertIdx), (VertIdx, VertIdx)>,
) -> Vec<VertIdx> {
    let mut adj: HashMap<VertIdx, Vec<VertIdx>> = HashMap::new();
    for (_ek, &(v0, v1)) in boundary.iter() {
        adj.entry(v0).or_default().push(v1);
        adj.entry(v1).or_default().push(v0);
    }
    walk_polygon_ccw(points, a, b, &adj)
}

/// Ear-clipping triangulation of a simple polygon.
///
/// Implements the classic O(n²) ear-clipping algorithm using an in-place
/// `removed[]` mark array to avoid O(n) vertex shifts per ear clip.
/// For degenerate / self-intersecting polygons, no ear may be found — we
/// detect this via a consecutive-failure counter and return the partial
/// triangulation, logging a warning so callers can fall back.
pub fn earcut_polygon(points: &[Point2d], verts: &[VertIdx]) -> Vec<[VertIdx; 3]> {
    let n = verts.len();
    if n < 3 {
        return Vec::new();
    }
    if n == 3 {
        return vec![[verts[0], verts[1], verts[2]]];
    }

    let mut removed = vec![false; n];
    let mut remaining = n;
    let ccw = signed_area_2d(points, verts) > 0.0;
    let mut tris = Vec::new();
    let mut guard = 0usize;
    let mut consecutive_failures = 0usize;

    while remaining > 3 && guard < n * 3 {
        guard += 1;
        let mut clipped = false;

        // Scan for an ear among non-removed vertices.
        let mut icur = 0usize;
        // Find first non-removed vertex.
        while icur < n && removed[icur] { icur += 1; }
        let mut iprev = step_backward(&removed, icur);

        let mut scan_count = 0;
        while scan_count < remaining {
            scan_count += 1;
            let inext = step_forward(&removed, icur);

            let v0 = verts[iprev];
            let v1 = verts[icur];
            let v2 = verts[inext];
            let p0 = points[v0 as usize];
            let p1 = points[v1 as usize];
            let p2 = points[v2 as usize];

            if is_convex(p0, p1, p2, ccw)
                && !ear_contains_vertex_marked(points, verts, &removed, iprev, icur, inext)
            {
                tris.push([v0, v1, v2]);
                removed[icur] = true;
                remaining -= 1;
                clipped = true;
                consecutive_failures = 0;
                break;
            }

            iprev = icur;
            icur = inext;
        }

        if !clipped {
            consecutive_failures += 1;
            if consecutive_failures >= 2 {
                log::warn!(
                    "earcut: degenerate polygon detected ({} verts remaining of {}), triangulation incomplete",
                    remaining, n
                );
                break;
            }
        }
    }

    if remaining == 3 {
        let final_tri = collect_remaining(&removed, verts);
        tris.push(final_tri);
    } else if remaining > 3 {
        log::warn!(
            "earcut: {}/{} vertices un-triangulated — face may have holes",
            remaining - 3, n
        );
    }
    tris
}

/// Step to the next non-removed index, wrapping.
fn step_forward(removed: &[bool], from: usize) -> usize {
    let n = removed.len();
    let mut i = (from + 1) % n;
    for _ in 0..n {
        if !removed[i] { return i; }
        i = (i + 1) % n;
    }
    from
}

/// Step to the previous non-removed index, wrapping.
fn step_backward(removed: &[bool], from: usize) -> usize {
    let n = removed.len();
    let mut i = (from + n - 1) % n;
    for _ in 0..n {
        if !removed[i] { return i; }
        i = (i + n - 1) % n;
    }
    from
}

/// Collect the three remaining (non-removed) vertex indices into a triangle.
fn collect_remaining(removed: &[bool], verts: &[VertIdx]) -> [VertIdx; 3] {
    let mut result = [0u32; 3];
    let mut j = 0;
    for (i, &r) in removed.iter().enumerate() {
        if !r {
            result[j] = verts[i];
            j += 1;
        }
    }
    debug_assert_eq!(j, 3);
    result
}

fn is_convex(a: Point2d, b: Point2d, c: Point2d, ccw: bool) -> bool {
    let o = robust_orient2d(a, b, c);
    if ccw {
        o > EPS
    } else {
        o < -EPS
    }
}

fn ear_contains_vertex_marked(
    points: &[Point2d],
    verts: &[VertIdx],
    removed: &[bool],
    i0: usize,
    i1: usize,
    i2: usize,
) -> bool {
    let a = points[verts[i0] as usize];
    let b = points[verts[i1] as usize];
    let c = points[verts[i2] as usize];
    for (k, &r) in removed.iter().enumerate() {
        if r || k == i0 || k == i1 || k == i2 {
            continue;
        }
        let p = points[verts[k] as usize];
        // Zero-width bridge vertices (duplicate coordinates from hole merging)
        // lie on the ear boundary, not inside it.
        if ((p.x - a.x).abs() < EPS && (p.y - a.y).abs() < EPS)
            || ((p.x - b.x).abs() < EPS && (p.y - b.y).abs() < EPS)
            || ((p.x - c.x).abs() < EPS && (p.y - c.y).abs() < EPS)
        {
            continue;
        }
        if point_in_triangle(p, a, b, c) {
            return true;
        }
    }
    false
}

/// Build boundary vertex adjacency (edges with exactly one alive triangle).
pub fn boundary_adjacency(
    edge_map: &HashMap<(VertIdx, VertIdx), Vec<u32>>,
    tri_alive: &[bool],
) -> HashMap<VertIdx, Vec<VertIdx>> {
    let mut adj: HashMap<VertIdx, Vec<VertIdx>> = HashMap::new();
    for (key, tris) in edge_map {
        if tris.len() != 1 {
            continue;
        }
        if !tri_alive.get(tris[0] as usize).copied().unwrap_or(false) {
            continue;
        }
        let (a, b) = *key;
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    adj
}

/// OCC meshLeftPolygonOf: collect polygon on the left of directed edge and return vertex loop.
pub fn mesh_left_polygon_loop(
    points: &[Point2d],
    start: VertIdx,
    end: VertIdx,
    adj: &HashMap<VertIdx, Vec<VertIdx>>,
    _skipped: &HashSet<(VertIdx, VertIdx)>,
) -> Vec<VertIdx> {
    let mut poly = walk_polygon_ccw(points, start, end, adj);
    if poly.len() < 3 {
        return poly;
    }
    let area = signed_area_2d(points, &poly);
    if area < 0.0 {
        poly.reverse();
    }
    poly
}

/// Triangulate a UV polygon using ear-clipping.
///
/// Returns triangle vertex indices `[i0, i1, i2]` into the input `vertices` slice.
/// Returns an empty `Vec` for polygons with holes — the caller should use a fan
/// fallback for those cases.
pub fn earcut_uv_polygon(vertices: &[(f64, f64)], hole_indices: &[usize]) -> Vec<[usize; 3]> {
    if vertices.len() < 3 {
        return Vec::new();
    }
    if !hole_indices.is_empty() {
        // Holes not supported — caller should use fan fallback.
        return Vec::new();
    }
    let points: Vec<Point2d> = vertices
        .iter()
        .map(|&(u, v)| Point2d::new(u, v))
        .collect();
    let verts: Vec<VertIdx> = (0..vertices.len() as VertIdx).collect();
    earcut_polygon(&points, &verts)
        .into_iter()
        .map(|[a, b, c]| [a as usize, b as usize, c as usize])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earcut_square() {
        let points = vec![
            Point2d::new(0.0, 0.0),
            Point2d::new(1.0, 0.0),
            Point2d::new(1.0, 1.0),
            Point2d::new(0.0, 1.0),
        ];
        let verts = vec![0, 1, 2, 3];
        let tris = earcut_polygon(&points, &verts);
        assert_eq!(tris.len(), 2);
    }

    #[test]
    fn earcut_l_shape() {
        let points = vec![
            Point2d::new(0.0, 0.0),
            Point2d::new(2.0, 0.0),
            Point2d::new(2.0, 1.0),
            Point2d::new(1.0, 1.0),
            Point2d::new(1.0, 2.0),
            Point2d::new(0.0, 2.0),
        ];
        let verts: Vec<VertIdx> = (0..6).collect();
        let tris = earcut_polygon(&points, &verts);
        assert_eq!(tris.len(), 4);
    }

    #[test]
    fn walk_square_ccw() {
        let points = vec![
            Point2d::new(0.0, 0.0),
            Point2d::new(1.0, 0.0),
            Point2d::new(1.0, 1.0),
            Point2d::new(0.0, 1.0),
        ];
        let mut adj: HashMap<VertIdx, Vec<VertIdx>> = HashMap::new();
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
        let poly = walk_polygon_ccw(&points, 0, 1, &adj);
        assert_eq!(poly.len(), 4);
        assert_eq!(poly, vec![0, 1, 2, 3]);
    }

    #[test]
    fn walk_concave_with_backtrack() {
        // 4
        // |
        // 0--1--2  (0-2 closes the loop)
        // Max-angle from 1 picks 4 first (dead end); backtrack then takes 2.
        let points = vec![
            Point2d::new(0.0, 0.0),
            Point2d::new(1.0, 0.0),
            Point2d::new(2.0, 0.0),
            Point2d::new(0.5, 1.0),
        ];
        let mut adj: HashMap<VertIdx, Vec<VertIdx>> = HashMap::new();
        for (a, b) in [(0, 1), (1, 2), (1, 3), (0, 2)] {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
        let poly = walk_polygon_ccw(&points, 0, 1, &adj);
        assert_eq!(poly, vec![0, 1, 2]);
    }
}
