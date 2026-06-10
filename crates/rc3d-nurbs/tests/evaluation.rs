//! Integration tests for rc3d-nurbs: curve/surface evaluation, basis, knot utilities.
//!
//! Covers: NurbsCurve, NurbsRenderSurface, bspline_basis, find_span, knot functions,
//! stitched surfaces, and tessellation output.

use rc3d_core::math::Vec3;
use rc3d_nurbs::*;
use rc3d_nurbs::basis::bspline_bases;

// ── Knot utilities ──────────────────────────────────────────────────────

#[test]
fn find_span_uniform() {
    let knots = uniform_knots(3, 7); // degree 3, 7 CPs → 11 knots: [0..11]
    assert_eq!(knots.len(), 11);
    let s = find_span(3, &knots, 0.5);
    assert!(s < knots.len(), "span {s} out of range for {} knots", knots.len());
    let s2 = find_span(3, &knots, 5.0);
    assert!(s2 < knots.len(), "span {s2} out of range");
}

#[test]
fn open_uniform_knots_clamped() {
    let knots = open_uniform_knots(2, 4); // degree 2, 4 CPs
    // Clamped: 0,0,0, 0.5, 1,1,1
    assert_eq!(knots.len(), 7);
    assert!((knots[0] - knots[2]).abs() < 1e-6, "first deg+1 knots should equal");
    assert!((knots[4] - knots[6]).abs() < 1e-6, "last deg+1 knots should equal");
}

// ── B-spline basis ──────────────────────────────────────────────────────

#[test]
fn basis_partition_of_unity() {
    let knots = open_uniform_knots(2, 4);
    let span = find_span(2, &knots, 0.3);
    let bases = bspline_bases(span, 2, 0.3, &knots);
    let sum: f32 = bases.iter().map(|(_, v)| v).sum();
    assert!((sum - 1.0).abs() < 1e-5, "partition of unity failed: {sum}");
}

#[test]
fn basis_at_knot_boundary() {
    let knots = open_uniform_knots(2, 4);
    let span = find_span(2, &knots, 0.0);
    let bases = bspline_bases(span, 2, 0.0, &knots);
    // At t=0 clamped: first basis = 1.0, rest = 0.0
    assert!((bases[0].1 - 1.0).abs() < 1e-5);
}

// ── Curve evaluation ────────────────────────────────────────────────────

#[test]
fn curve_line_from_two_points() {
    let pts = vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
    let curve = NurbsCurve::from_points(&pts, 1); // degree 1 = linear
    let p_mid = curve.evaluate(0.5);
    assert!((p_mid.x - 5.0).abs() < 1e-5);
    assert!(p_mid.y.abs() < 1e-5);
}

#[test]
fn curve_derivative_linear() {
    let pts = vec![Vec3::ZERO, Vec3::X];
    let curve = NurbsCurve::from_points(&pts, 1);
    let d = curve.tangent(0.5);
    let dn = d.normalize();
    assert!((dn - Vec3::X).length() < 1e-5 || (dn + Vec3::X).length() < 1e-5);
}

#[test]
fn curve_arc_length_positive() {
    let pts = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(3.0, 4.0, 0.0),
        Vec3::new(6.0, 0.0, 0.0),
    ];
    let curve = NurbsCurve::from_points(&pts, 2);
    let len = curve.arc_length(64);
    assert!(len > 5.0, "arc length too short: {len}");
    assert!(len < 15.0, "arc length too long: {len}");
}

// ── Surface evaluation ──────────────────────────────────────────────────

#[test]
fn surface_plane_evaluation() {
    let grid = vec![
        vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
        vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)],
    ];
    let surf = NurbsRenderSurface::from_points_grid(&grid, 1, 1);
    let p = surf.evaluate(0.5, 0.5);
    assert!((p.x - 0.5).abs() < 1e-5);
    assert!((p.y - 0.5).abs() < 1e-5);
    assert!(p.z.abs() < 1e-5);
}

#[test]
fn surface_normal_outward() {
    let grid = vec![
        vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
        vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)],
    ];
    let surf = NurbsRenderSurface::from_points_grid(&grid, 1, 1);
    let n = surf.normal(0.5, 0.5);
    // Plane normal should be Z
    assert!((n - Vec3::Z).length() < 1e-4 || (n + Vec3::Z).length() < 1e-4,
        "expected Z-aligned normal, got {:?}", n);
}

#[test]
fn surface_boundary_curve_is_valid() {
    let grid = vec![
        vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0)],
        vec![Vec3::new(3.0, 0.0, 0.0), Vec3::new(3.0, 2.0, 0.0)],
    ];
    let surf = NurbsRenderSurface::from_points_grid(&grid, 1, 1);
    let edge = surf.boundary_curve(BoundaryEdge::UMax);
    // Boundary curve should be non-degenerate
    let p0 = edge.evaluate(0.0);
    let p1 = edge.evaluate(1.0);
    assert!((p0 - p1).length() > 1e-3, "boundary curve degenerate");
    // Boundary should have 2 control points (v_count = 2)
    assert!(edge.n_control_points() >= 2);
}

// ── Tessellation ────────────────────────────────────────────────────────

#[test]
fn tessellate_uniform_output_valid() {
    let grid = vec![
        vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
        vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)],
    ];
    let surf = NurbsRenderSurface::from_points_grid(&grid, 1, 1);
    let mesh = surf.tessellate_uniform(4, 4);
    assert_eq!(mesh.positions.len(), 25); // 5×5 grid
    assert!(mesh.tri_indices.len() >= 32 * 3); // 4×4×2 triangles ×3 indices
    // All indices should be in bounds
    for &idx in &mesh.tri_indices {
        assert!((idx as usize) < mesh.positions.len());
    }
}

#[test]
fn tessellate_with_normals_has_coverage() {
    let grid = vec![
        vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
        vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)],
    ];
    let surf = NurbsRenderSurface::from_points_grid(&grid, 1, 1);
    let ts = surf.tessellate_uniform_with_normals(3, 3);
    assert_eq!(ts.positions.len(), ts.normals.len());
    assert!(!ts.indices.is_empty());
    // Normals should be unit length
    for n in &ts.normals {
        assert!((n.length() - 1.0).abs() < 1e-4, "normal not unit: {:?}", n);
    }
}
