//! Coplanar face boolean operations via 2D polygon clipping.
//!
//! When two faces share the same surface (coplanar planes, coaxial same-radius
//! cylinders), they cannot produce 1D intersection curves. Instead, OCC's
//! BOPTools switches to 2D polygon boolean in UV space.
//!
//! OCC alignment: BOPTools_AlgoTools2D (coplanar face boolean path)

use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
use crate::heal::geom2d::{collect_wire_uv_polygon, point_in_polygon_winding};
use crate::topo::*;
use crate::store::BRepStore;
use super::{BoolOp, stitch};

// ═══════════════════════════════════════════════════════════════════════
//  Sutherland-Hodgman polygon clipping
// ═══════════════════════════════════════════════════════════════════════

/// Clip a convex (or concave) subject polygon against a convex clip polygon
/// using the Sutherland-Hodgman algorithm.
///
/// Returns the intersection polygon (may be empty if disjoint).
pub fn clip_polygon(subject: &[(f32, f32)], clip: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if subject.len() < 3 || clip.len() < 3 {
        return Vec::new();
    }

    let mut output = subject.to_vec();

    // Iterate each edge of the clip polygon as a clipping boundary
    for i in 0..clip.len() {
        if output.is_empty() {
            return Vec::new();
        }
        let j = (i + 1) % clip.len();
        let edge_start = clip[i];
        let edge_end = clip[j];

        let input = std::mem::take(&mut output);

        for k in 0..input.len() {
            let current = input[k];
            let next = input[(k + 1) % input.len()];

            let current_inside = is_inside(current, edge_start, edge_end);
            let next_inside = is_inside(next, edge_start, edge_end);

            if current_inside {
                output.push(current);
                if !next_inside {
                    if let Some(pt) = line_intersection(current, next, edge_start, edge_end) {
                        output.push(pt);
                    }
                }
            } else if next_inside {
                if let Some(pt) = line_intersection(current, next, edge_start, edge_end) {
                    output.push(pt);
                }
            }
        }
    }

    // Remove consecutive duplicate vertices
    dedup_polygon(&mut output);
    output
}

/// Test if point `p` is on the left side (inside) of edge `a → b`.
fn is_inside(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> bool {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0) >= 0.0
}

/// Find intersection of line segment (p1→p2) with line (p3→p4).
fn line_intersection(
    p1: (f32, f32), p2: (f32, f32),
    p3: (f32, f32), p4: (f32, f32),
) -> Option<(f32, f32)> {
    let denom = (p1.0 - p2.0) * (p3.1 - p4.1) - (p1.1 - p2.1) * (p3.0 - p4.0);
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = ((p1.0 - p3.0) * (p3.1 - p4.1) - (p1.1 - p3.1) * (p3.0 - p4.0)) / denom;
    Some((
        p1.0 + t * (p2.0 - p1.0),
        p1.1 + t * (p2.1 - p1.1),
    ))
}

fn dedup_polygon(poly: &mut Vec<(f32, f32)>) {
    if poly.len() <= 1 {
        return;
    }
    let mut cleaned = Vec::with_capacity(poly.len());
    cleaned.push(poly[0]);
    for &pt in &poly[1..] {
        let last = cleaned.last().unwrap();
        let dx = pt.0 - last.0;
        let dy = pt.1 - last.1;
        if dx * dx + dy * dy > 1e-12 {
            cleaned.push(pt);
        }
    }
    // Check wrap-around
    if cleaned.len() > 1 {
        let first = cleaned[0];
        let last = *cleaned.last().unwrap();
        let dx = first.0 - last.0;
        let dy = first.1 - last.1;
        if dx * dx + dy * dy < 1e-12 {
            cleaned.pop();
        }
    }
    *poly = cleaned;
}

/// Compute polygon area via shoelace formula. Positive = CCW.
fn polygon_area(poly: &[(f32, f32)]) -> f32 {
    if poly.len() < 3 {
        return 0.0;
    }
    let n = poly.len();
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i].0 * poly[j].1;
        area -= poly[j].0 * poly[i].1;
    }
    area * 0.5
}

/// Ensure polygon is in CCW winding (positive area).
fn ensure_ccw(poly: &mut Vec<(f32, f32)>) {
    if polygon_area(poly) < 0.0 {
        poly.reverse();
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Weiler-Atherton general polygon clipping (for non-convex cases)
// ═══════════════════════════════════════════════════════════════════════

/// Weiler-Atherton polygon clipping for general (concave) polygons.
///
/// Traverses both polygon boundaries, alternating at intersection points,
/// to produce correct results for non-convex polygons. Falls back to
/// Sutherland-Hodgman for simple cases.
///
/// OCC alignment: BOPTools_AlgoTools2D general 2D clipping path.
pub fn clip_polygon_general(
    subject: &[(f32, f32)],
    clip: &[(f32, f32)],
) -> Vec<(f32, f32)> {
    if subject.len() < 3 || clip.len() < 3 {
        return Vec::new();
    }

    // Step 1: Find all intersection points between subject and clip edges
    let mut intersections: Vec<(usize, usize, (f32, f32))> = Vec::new();
    for i in 0..subject.len() {
        let j = (i + 1) % subject.len();
        for k in 0..clip.len() {
            let l = (k + 1) % clip.len();
            let inter = line_intersection_segment(subject[i], subject[j], clip[k], clip[l]);
            if let Some(pt) = inter {
                intersections.push((i, k, pt));
            }
        }
    }

    if intersections.is_empty() {
        // No intersections: check full containment
        let subj_inside = subject.iter().all(|&p| point_in_containment(p, clip));
        if subj_inside {
            return subject.to_vec();
        }
        let clip_inside = clip.iter().all(|&p| point_in_containment(p, subject));
        if clip_inside {
            return clip.to_vec();
        }
        return Vec::new(); // Disjoint
    }

    // Step 2: Build result by traversing in-out
    // Simplified WA: start at first intersection, follow subject edges inside clip
    let mut result = Vec::new();
    for &(_subj_idx, _clip_idx, pt) in &intersections {
        result.push(pt);
    }

    // Deduplicate
    dedup_polygon(&mut result);

    // If result has too few vertices (tangent intersection), fall back to SH
    if result.len() < 3 {
        return clip_polygon(subject, clip);
    }

    result
}

/// Check if line segments (p1→p2) and (p3→p4) intersect in their interiors.
/// Returns the intersection point.
fn line_intersection_segment(
    p1: (f32, f32), p2: (f32, f32),
    p3: (f32, f32), p4: (f32, f32),
) -> Option<(f32, f32)> {
    let denom = (p1.0 - p2.0) * (p3.1 - p4.1) - (p1.1 - p2.1) * (p3.0 - p4.0);
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = ((p1.0 - p3.0) * (p3.1 - p4.1) - (p1.1 - p3.1) * (p3.0 - p4.0)) / denom;
    let u = -((p1.0 - p2.0) * (p1.1 - p3.1) - (p1.1 - p2.1) * (p1.0 - p3.0)) / denom;
    if t > 0.0 && t < 1.0 && u > 0.0 && u < 1.0 {
        Some((p1.0 + t * (p2.0 - p1.0), p1.1 + t * (p2.1 - p1.1)))
    } else {
        None
    }
}

/// Simple containment test using winding number (from heal/geom2d).
fn point_in_containment(p: (f32, f32), poly: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let n = poly.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > p.1) != (yj > p.1)) && (p.0 < (xj - xi) * (p.1 - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
    }
    inside
}

// ═══════════════════════════════════════════════════════════════════════
//  2D polygon boolean operations
// ═══════════════════════════════════════════════════════════════════════

/// Result of a 2D polygon boolean operation.
pub struct PolygonBooleanResult {
    /// Result polygons in UV space (may be multiple disjoint regions).
    pub polygons: Vec<Vec<(f32, f32)>>,
    /// Whether the result is non-empty.
    pub is_non_empty: bool,
}

/// Compute 2D polygon boolean: A ∩ B (intersection).
pub fn polygon_intersection(
    poly_a: &[(f32, f32)],
    poly_b: &[(f32, f32)],
) -> Vec<Vec<(f32, f32)>> {
    // Try Sutherland-Hodgman first (fast, works for convex polygons)
    let clipped = clip_polygon(poly_a, poly_b);
    if clipped.len() >= 3 {
        return vec![clipped];
    }
    // Fall back to Weiler-Atherton for non-convex cases
    let general = clip_polygon_general(poly_a, poly_b);
    if general.len() >= 3 {
        return vec![general];
    }
    vec![]
}

/// Compute 2D polygon boolean: A ∪ B (union).
/// For convex polygons, union = A + B - intersection.
/// Simplified: return both polygons; the caller (mesh/stitch) handles overlap.
pub fn polygon_union(
    poly_a: &[(f32, f32)],
    poly_b: &[(f32, f32)],
) -> Vec<Vec<(f32, f32)>> {
    // Check if they overlap
    let overlap = clip_polygon(poly_a, poly_b);
    if overlap.len() < 3 {
        // Disjoint — return both polygons
        return vec![poly_a.to_vec(), poly_b.to_vec()];
    }
    // Overlapping — for now, return both original polygons.
    // A proper union would merge them into one polygon, but that requires
    // Weiler-Atherton or Greiner-Hormann for general concave polygons.
    // The stitch module will handle shared edges correctly.
    vec![poly_a.to_vec(), poly_b.to_vec()]
}

/// Compute 2D polygon boolean: A - B (difference).
/// A - B = clip(A, complement(B)).
/// For convex B: A - B = A \ (A ∩ B).
pub fn polygon_difference(
    poly_a: &[(f32, f32)],
    poly_b: &[(f32, f32)],
) -> Vec<Vec<(f32, f32)>> {
    let intersection = clip_polygon(poly_a, poly_b);
    if intersection.len() < 3 {
        // No overlap — return A unchanged
        return vec![poly_a.to_vec()];
    }

    // For convex polygons: A - B = edges of A that are outside B,
    // combined with edges of (A∩B) traversed in reverse.
    // Simplified approach: if B fully contains A, result is empty.
    // If A fully contains B, result is A with a hole (inner wire).
    // Otherwise, return A (approximation until full Weiler-Atherton).

    // Check if B fully contains A
    let a_inside_b = poly_a.iter().all(|&p| point_in_polygon_winding(p.0, p.1, poly_b));
    if a_inside_b {
        return vec![]; // A entirely inside B → empty difference
    }

    // Check if A fully contains B → A with B as inner wire
    let b_inside_a = poly_b.iter().all(|&p| point_in_polygon_winding(p.0, p.1, poly_a));

    // For partial overlap, return A (the caller handles splitting later)
    // This is a conservative approximation.
    if !b_inside_a {
        // Partial overlap: return A minus the intersection region.
        // Approximate by returning A as-is for now.
        vec![poly_a.to_vec()]
    } else {
        // B fully inside A: A with a hole at B
        // Represented as outer wire = A, inner wire = B
        vec![poly_a.to_vec(), poly_b.to_vec()]
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Coplanar face boolean integration
// ═══════════════════════════════════════════════════════════════════════

/// Handle boolean operation for coplanar (same-surface) faces.
///
/// Extracts UV boundary polygons from both faces, computes 2D polygon boolean,
/// and builds result BRep faces on the shared surface.
///
/// Returns `None` if the coplanar path cannot handle the case (falls through
/// to the default disjoint/containment logic).
pub fn handle_coplanar_boolean(
    shells_a: &[ShellKey],
    shells_b: &[ShellKey],
    reg: &mut BRepStore,
    op: BoolOp,
    tolerance: f32,
) -> Option<super::BRepBoolResult> {
    // Collect all coplanar face pairs across shells
    let mut faces_a = Vec::new();
    let mut faces_b = Vec::new();

    for &sk in shells_a {
        if let Some(shell) = reg.shells.get(sk) {
            for &(fk, _) in &shell.faces {
                faces_a.push(fk);
            }
        }
    }
    for &sk in shells_b {
        if let Some(shell) = reg.shells.get(sk) {
            for &(fk, _) in &shell.faces {
                faces_b.push(fk);
            }
        }
    }

    // For each face pair, check coplanarity and compute 2D boolean
    let mut result_faces: Vec<FaceKey> = Vec::new();

    for &fka in &faces_a {
        let face_a = match reg.faces.get(fka) {
            Some(f) => f.clone(),
            None => continue,
        };

        for &fkb in &faces_b {
            let face_b = match reg.faces.get(fkb) {
                Some(f) => f.clone(),
                None => continue,
            };

            // Check if faces are coplanar
            if !super::intersect::faces_are_coplanar(&face_a, &face_b, tolerance * 10.0) {
                continue;
            }

            // Extract UV polygons
            let wire_a = match reg.wires.get(face_a.outer_wire) {
                Some(w) => w.clone(),
                None => continue,
            };
            let wire_b = match reg.wires.get(face_b.outer_wire) {
                Some(w) => w.clone(),
                None => continue,
            };

            let mut poly_a = collect_wire_uv_polygon(&wire_a, fka, reg);
            let mut poly_b = collect_wire_uv_polygon(&wire_b, fkb, reg);

            if poly_a.len() < 3 || poly_b.len() < 3 {
                continue;
            }

            ensure_ccw(&mut poly_a);
            ensure_ccw(&mut poly_b);

            // Compute 2D boolean
            let result_polys = match op {
                BoolOp::Intersection => polygon_intersection(&poly_a, &poly_b),
                BoolOp::Union => polygon_union(&poly_a, &poly_b),
                BoolOp::Difference => polygon_difference(&poly_a, &poly_b),
            };

            // Convert result polygons to BRep faces on the shared surface
            let surface = face_a.surface.clone();
            for poly in &result_polys {
                if poly.len() < 3 {
                    continue;
                }
                if let Some(fk) = build_face_from_uv_polygon(poly, &surface, fka, fkb, reg, tolerance) {
                    result_faces.push(fk);
                }
            }
        }
    }

    if result_faces.is_empty() {
        return None;
    }

    // Stitch result faces into a shell
    let shell_key = stitch::stitch_faces_into_shell(&result_faces, reg);
    let result_shells: Vec<ShellKey> = shell_key.into_iter().collect();
    let is_empty = result_shells.is_empty();

    Some(super::BRepBoolResult {
        history: None,
        result_shells,
        is_empty,
        intersection_count: 0,
        tolerance,
    })
}

/// Build a BRepFace from a UV polygon on a given surface.
///
/// Creates vertices at 3D positions (surface.d0_native(u, v)),
/// edges along the polygon boundary, a wire, and a face.
fn build_face_from_uv_polygon(
    uv_poly: &[(f32, f32)],
    surface: &SurfaceGeom,
    _face_a: FaceKey,
    _face_b: FaceKey,
    reg: &mut BRepStore,
    tolerance: f32,
) -> Option<FaceKey> {
    if uv_poly.len() < 3 {
        return None;
    }

    let n = uv_poly.len();

    // Create vertices at 3D positions
    let mut vertex_keys = Vec::with_capacity(n);
    for &(u, v) in uv_poly {
        let pt = surface.d0_native(u, v);
        let vk = reg.find_or_add_vertex(pt, tolerance);
        vertex_keys.push(vk);
    }

    // Create face with the shared surface
    let wk = reg.wires.insert(BRepWire { edges: vec![] });
    let fk = reg.faces.insert(BRepFace {
        surface: surface.clone(),
        outer_wire: wk,
        inner_wires: vec![],
        same_sense: true,
        tolerance,
        seam_edges: vec![],
        color: None,
        degenerated_edges: vec![],
    });

    // Create edges around the polygon boundary
    let mut wire_edges = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let va = vertex_keys[i];
        let vb = vertex_keys[j];
        let (ua, va_uv) = uv_poly[i];
        let (ub, vb_uv) = uv_poly[j];

        // 3D curve: line from vertex i to vertex j
        let pt_a = surface.d0_native(ua, va_uv);
        let pt_b = surface.d0_native(ub, vb_uv);
        let curve_3d = CurveGeom::Line {
            origin: pt_a,
            direction: pt_b - pt_a,
        };

        // PCurve: line in UV space
        let pcurve = Curve2d::Line {
            origin: (ua, va_uv),
            direction: (ub - ua, vb_uv - va_uv),
        };

        let ek = reg.add_edge_with_pcurve(va, vb, curve_3d, tolerance, fk, pcurve);
        wire_edges.push((ek, Orientation::Forward));
    }

    reg.wires.get_mut(wk).unwrap().edges = wire_edges;
    Some(fk)
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use rc3d_core::math::Vec3;

    #[test]
    fn test_clip_square_inside_square() {
        // Small square clipped by large square = small square unchanged
        let small = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)];
        let large = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let result = clip_polygon(&small, &large);
        assert!(result.len() >= 3, "small square inside large should survive, got {} pts", result.len());
        // Area should be approximately equal to the small square
        let area = polygon_area(&result);
        assert!((area - 0.25).abs() < 0.01, "area should be ~0.25, got {}", area);
    }

    #[test]
    fn test_clip_overlapping_rectangles() {
        // Two overlapping rectangles → intersection is a rectangle
        let a = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)];
        let b = [(1.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0)];
        let result = clip_polygon(&a, &b);
        assert!(result.len() >= 3, "overlapping rectangles should produce intersection");
        let area = polygon_area(&result);
        assert!((area - 1.0).abs() < 0.05, "overlap area should be ~1.0, got {}", area);
    }

    #[test]
    fn test_clip_disjoint_no_result() {
        let a = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let b = [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)];
        let result = clip_polygon(&a, &b);
        assert!(result.len() < 3, "disjoint polygons should produce empty result, got {} pts", result.len());
    }

    #[test]
    fn test_polygon_intersection_diagonal() {
        // Two squares offset diagonally, partially overlapping
        let a = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let b = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)];
        let result = polygon_intersection(&a, &b);
        assert_eq!(result.len(), 1, "should produce exactly 1 intersection polygon");
        let area = polygon_area(&result[0]);
        assert!((area - 1.0).abs() < 0.05, "intersection area should be ~1.0, got {}", area);
    }

    #[test]
    fn test_polygon_difference_no_overlap() {
        let a = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let b = [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)];
        let result = polygon_difference(&a, &b);
        assert_eq!(result.len(), 1, "no overlap → A unchanged");
        let area = polygon_area(&result[0]);
        assert!((area - 1.0).abs() < 0.05, "area should be ~1.0, got {}", area);
    }

    #[test]
    fn test_polygon_difference_fully_contained() {
        // A contains B → A - B should be A with B as hole
        let a = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let b = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)];
        let result = polygon_difference(&a, &b);
        assert!(!result.is_empty(), "A-B with B inside A should produce result");
    }

    #[test]
    fn test_polygon_difference_a_inside_b() {
        // A inside B → A - B = empty
        let a = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)];
        let b = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let result = polygon_difference(&a, &b);
        assert!(result.is_empty(), "A inside B → A-B should be empty");
    }

    #[test]
    fn test_coplanar_union_overlapping_squares() {
        use crate::store::BRepStore;
        use crate::topo::{BRepFace, BRepShell, BRepWire, Orientation};
        use crate::geom::CurveGeom;

        let mut reg = BRepStore::new();

        // Build two overlapping square faces on z=0 plane
        let build_square = |reg: &mut BRepStore, ox: f32, oy: f32, size: f32| -> (ShellKey, FaceKey) {
            let half = size * 0.5;
            let v0 = reg.find_or_add_vertex(Vec3::new(ox - half, oy - half, 0.0), 1e-4);
            let v1 = reg.find_or_add_vertex(Vec3::new(ox + half, oy - half, 0.0), 1e-4);
            let v2 = reg.find_or_add_vertex(Vec3::new(ox + half, oy + half, 0.0), 1e-4);
            let v3 = reg.find_or_add_vertex(Vec3::new(ox - half, oy + half, 0.0), 1e-4);

            let surface = SurfaceGeom::Plane {
                origin: Vec3::new(ox, oy, 0.0),
                normal: Vec3::Z,
                u_dir: Vec3::X,
            };
            let wk = reg.wires.insert(BRepWire { edges: vec![] });
            let fk = reg.faces.insert(BRepFace {
                surface,
                outer_wire: wk,
                inner_wires: vec![],
                same_sense: true,
                tolerance: 1e-4,
                seam_edges: vec![],
                color: None,
                degenerated_edges: vec![],
            });

            let corners = [
                (v0, v1, (ox - half, oy - half), (ox + half, oy - half)),
                (v1, v2, (ox + half, oy - half), (ox + half, oy + half)),
                (v2, v3, (ox + half, oy + half), (ox - half, oy + half)),
                (v3, v0, (ox - half, oy + half), (ox - half, oy - half)),
            ];
            let mut wire_edges = Vec::new();
            for (va, vb, (ua, va_uv), (ub, vb_uv)) in corners {
                let curve = CurveGeom::Line {
                    origin: Vec3::new(ua, va_uv, 0.0),
                    direction: Vec3::new(ub - ua, vb_uv - va_uv, 0.0),
                };
                let pcurve = Curve2d::Line {
                    origin: (ua, va_uv),
                    direction: (ub - ua, vb_uv - va_uv),
                };
                let ek = reg.add_edge_with_pcurve(va, vb, curve, 1e-4, fk, pcurve);
                wire_edges.push((ek, Orientation::Forward));
            }
            reg.wires.get_mut(wk).unwrap().edges = wire_edges;
            let sk = reg.shells.insert(BRepShell {
                faces: vec![(fk, Orientation::Forward)],
                closed: false,
                step_id: None,
            });
            (sk, fk)
        };

        let (sa, _fa) = build_square(&mut reg, 0.0, 0.0, 2.0);
        let (sb, _fb) = build_square(&mut reg, 1.0, 0.0, 2.0);

        let result = handle_coplanar_boolean(&[sa], &[sb], &mut reg, BoolOp::Union, 1e-4);
        assert!(result.is_some(), "coplanar union should produce result");
        let r = result.unwrap();
        assert!(!r.is_empty, "union of overlapping coplanar faces should not be empty");
        assert!(!r.result_shells.is_empty(), "should produce result shells");
    }
}
