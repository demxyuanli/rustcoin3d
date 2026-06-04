//! SameParameter PCurve re-parameterization.
//!
//! Fixes PCurves so that C(t) ≈ S(pcurve(t)) for all t, where C is the 3D edge
//! curve, S is the surface, and pcurve is the 2D UV curve on the surface.

use rc3d_core::math::Vec3;

use crate::geom::project::project_point_on_surface;
use crate::geom::{CurveGeom, SurfaceGeom, eval_pcurve_on_surface};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, WireKey};

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub(crate) struct SameParamReport {
    pub pcurves_fixed: usize,
    pub max_deviation_before: f32,
    pub max_deviation_after: f32,
}

// ---------------------------------------------------------------------------
// Core per-edge fix
// ---------------------------------------------------------------------------

/// Fix a single (edge, face) pair so the PCurve satisfies SameParameter.
///
/// Returns a report with before/after max deviation.
pub(crate) fn fix_same_parameter_edge(
    ek: EdgeKey,
    face_key: FaceKey,
    reg: &mut BRepStore,
    tolerance: f32,
    max_iterations: usize,
) -> SameParamReport {
    let (curve_3d, pcurve, surface) = {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => return SameParamReport::default(),
        };
        // Skip degenerate edges
        if edge.v_low == edge.v_high {
            return SameParamReport::default();
        }
        let pcurve = match edge.pcurves.get(&face_key) {
            Some(pc) => pc.clone(),
            None => return SameParamReport::default(),
        };
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => return SameParamReport::default(),
        };
        (edge.curve.clone(), pcurve, face.surface.clone())
    };

    // Initial deviation
    let (mut best_max_dev, _samples) = sample_deviation(&curve_3d, &pcurve, &surface, tolerance);
    if best_max_dev < tolerance {
        return SameParamReport {
            max_deviation_before: best_max_dev,
            max_deviation_after: best_max_dev,
            ..Default::default()
        };
    }

    let before_max_dev = best_max_dev;
    let mut best_pcurve = pcurve;
    let mut fixed_count = 0usize;

    for _iter in 0..max_iterations {
        // Adaptive samples from 3D curve
        let curve_samples = curve_3d.sample_adaptive(0.0, 1.0, tolerance * 0.1);
        if curve_samples.len() < 2 {
            break;
        }

        // Ensure endpoints are included
        let mut t_points: Vec<(f32, Vec3)> = Vec::with_capacity(curve_samples.len());
        let has_t0 = curve_samples.first().map(|(t, _)| *t).unwrap_or(-1.0).abs() < 1e-9;
        let has_t1 = curve_samples
            .last()
            .map(|(t, _)| (*t - 1.0).abs())
            .unwrap_or(1.0)
            < 1e-9;
        if !has_t0 {
            t_points.push((0.0, curve_3d.d0(0.0)));
        }
        t_points.extend_from_slice(&curve_samples);
        if !has_t1 {
            t_points.push((1.0, curve_3d.d0(1.0)));
        }

        // Project each 3D point onto the surface to get corrected UV
        let mut uv_pairs: Vec<(f32, Vec3)> = Vec::with_capacity(t_points.len());
        for (t_i, p3d) in &t_points {
            let current_uv = best_pcurve.d0(*t_i);
            let corrected_uv = match project_to_uv(&surface, *p3d, current_uv, tolerance) {
                Some(uv) => uv,
                None => {
                    log::debug!(
                        "[SameParam] edge {:?} face {:?}: projection failed at t={:.4}",
                        ek,
                        face_key,
                        t_i
                    );
                    uv_pairs.push((*t_i, current_uv));
                    continue;
                }
            };
            let wrapped = normalize_periodic_uv(corrected_uv, current_uv, &surface);
            uv_pairs.push((*t_i, wrapped));
        }

        if uv_pairs.len() < 2 {
            break;
        }

        // Fit new PCurve
        let new_pcurve = fit_new_pcurve(&uv_pairs, tolerance);

        // Verify improvement
        let (new_max_dev, _) = sample_deviation(&curve_3d, &new_pcurve, &surface, tolerance);

        if new_max_dev < best_max_dev {
            best_max_dev = new_max_dev;
            best_pcurve = new_pcurve;
            fixed_count += 1;
        }

        if new_max_dev < tolerance {
            break;
        }
    }

    // Replace pcurve only if improved
    if fixed_count > 0 {
        reg.set_pcurve(ek, face_key, best_pcurve);
    }

    SameParamReport {
        pcurves_fixed: if fixed_count > 0 { 1 } else { 0 },
        max_deviation_before: before_max_dev,
        max_deviation_after: best_max_dev,
    }
}

// ---------------------------------------------------------------------------
// UV projection helper
// ---------------------------------------------------------------------------

/// Project a 3D point onto the surface and return native (u, v) as Vec3(u, v, 0).
fn project_to_uv(
    surface: &SurfaceGeom,
    point: Vec3,
    _current_uv: Vec3,
    tolerance: f32,
) -> Option<Vec3> {
    // First try analytical inverse
    if let Some((u, v)) = surface.inverse_native_uv(point, tolerance * 100.0) {
        return Some(Vec3::new(u, v, 0.0));
    }
    // Fallback: Newton projection
    let results = project_point_on_surface(surface, point);
    if results.is_empty() {
        return None;
    }
    // Take best by distance
    let (u, v, _) = results
        .into_iter()
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))?;
    Some(Vec3::new(u, v, 0.0))
}

// ---------------------------------------------------------------------------
// Periodic UV normalization
// ---------------------------------------------------------------------------

/// Shift uv by +/- period to be closest to reference in UV space.
pub(crate) fn normalize_periodic_uv(uv: Vec3, reference: Vec3, surface: &SurfaceGeom) -> Vec3 {
    let mut result = uv;
    if let Some(pu) = surface.native_u_period() {
        let du = result.x - reference.x;
        if du.abs() > pu * 0.5 {
            let shifts = if du > 0.0 { -pu } else { pu };
            result.x += shifts;
        }
    }
    if let Some(pv) = surface.native_v_period() {
        let dv = result.y - reference.y;
        if dv.abs() > pv * 0.5 {
            let shifts = if dv > 0.0 { -pv } else { pv };
            result.y += shifts;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Deviation sampling
// ---------------------------------------------------------------------------

/// Sample deviation between 3D curve and surface(pcurve(t)) at adaptive points.
/// Returns (max_deviation, [(t, p3d, deviation)]).
fn sample_deviation(
    curve_3d: &CurveGeom,
    pcurve: &CurveGeom,
    surface: &SurfaceGeom,
    tolerance: f32,
) -> (f32, Vec<(f32, Vec3, f32)>) {
    let samples = curve_3d.sample_adaptive(0.0, 1.0, tolerance * 0.1);
    let mut max_dev = 0.0f32;
    let mut results = Vec::with_capacity(samples.len());
    for (t, p3d) in &samples {
        let via_surface = eval_pcurve_on_surface(pcurve, surface, *t);
        let dev = (*p3d - via_surface).length();
        if dev > max_dev {
            max_dev = dev;
        }
        results.push((*t, *p3d, dev));
    }
    (max_dev, results)
}

// ---------------------------------------------------------------------------
// PCurve fitting dispatcher
// ---------------------------------------------------------------------------

/// Decide between line and BSpline fit based on collinearity.
fn fit_new_pcurve(samples: &[(f32, Vec3)], tolerance: f32) -> CurveGeom {
    if samples.len() <= 2 {
        let start = samples.first().map(|&(_, uv)| uv).unwrap_or(Vec3::ZERO);
        let end = samples.last().map(|&(_, uv)| uv).unwrap_or(Vec3::ZERO);
        return fit_pcurve_line(start, end);
    }
    if all_collinear_uv(samples, tolerance) {
        let start = samples[0].1;
        let end = samples[samples.len() - 1].1;
        return fit_pcurve_line(start, end);
    }
    fit_pcurve_bspline(samples, 3)
}

/// Check if all UV points are collinear within tolerance.
fn all_collinear_uv(samples: &[(f32, Vec3)], tolerance: f32) -> bool {
    if samples.len() <= 2 {
        return true;
    }
    let p0 = samples[0].1;
    let p1 = samples[samples.len() - 1].1;
    let dir = p1 - p0;
    let len = dir.length();
    if len < 1e-12 {
        // All points should be near p0
        return samples.iter().all(|&(_, uv)| (uv - p0).length() < tolerance);
    }
    let dir_n = dir / len;
    for &(_, uv) in &samples[1..samples.len() - 1] {
        let v = uv - p0;
        let proj = v.dot(dir_n);
        let perp = v - dir_n * proj;
        if perp.length() > tolerance {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Line fit
// ---------------------------------------------------------------------------

fn fit_pcurve_line(uv_start: Vec3, uv_end: Vec3) -> CurveGeom {
    CurveGeom::Line {
        origin: Vec3::new(uv_start.x, uv_start.y, 0.0),
        direction: Vec3::new(uv_end.x - uv_start.x, uv_end.y - uv_start.y, 0.0),
    }
}

// ---------------------------------------------------------------------------
// BSpline fit (interpolation through sample points)
// ---------------------------------------------------------------------------

fn fit_pcurve_bspline(samples: &[(f32, Vec3)], degree: usize) -> CurveGeom {
    let n = samples.len();
    if n <= 2 || n < degree + 1 {
        return CurveGeom::Polyline {
            points: samples.iter().map(|&(_, uv)| Vec3::new(uv.x, uv.y, 0.0)).collect(),
        };
    }

    // Clamp degree to n-1
    let deg = degree.min(n - 1);

    // Build clamped knot vector: deg+1 copies of t[0], deg+1 copies of t[n-1],
    // interior knots by averaging consecutive deg parameter values.
    let t_vals: Vec<f32> = samples.iter().map(|(t, _)| *t).collect();
    let mut knots = Vec::with_capacity(n + deg + 1);
    for _ in 0..=deg {
        knots.push(t_vals[0]);
    }
    // Interior knots: j = 1 .. n-deg-1, knot[deg+j] = avg(t[j..j+deg])
    for j in 1..n - deg {
        let avg: f32 = t_vals[j..j + deg].iter().sum::<f32>() / deg as f32;
        knots.push(avg);
    }
    for _ in 0..=deg {
        knots.push(t_vals[n - 1]);
    }

    // Build n x n basis matrix and solve for control points (u coords and v coords separately)
    let mut mat_u = vec![vec![0.0f32; n]; n];
    let mut rhs_u = vec![0.0f32; n];
    let mut rhs_v = vec![0.0f32; n];

    for i in 0..n {
        for j in 0..n {
            mat_u[i][j] = bspline_basis(deg, j, &knots, t_vals[i]);
        }
        rhs_u[i] = samples[i].1.x;
        rhs_v[i] = samples[i].1.y;
    }

    let cp_u = match solve_linear_system(&mat_u, &rhs_u, n) {
        Some(sol) => sol,
        None => {
            return CurveGeom::Polyline {
                points: samples.iter().map(|&(_, uv)| Vec3::new(uv.x, uv.y, 0.0)).collect(),
            };
        }
    };
    let cp_v = match solve_linear_system(&mat_u, &rhs_v, n) {
        Some(sol) => sol,
        None => {
            return CurveGeom::Polyline {
                points: samples.iter().map(|&(_, uv)| Vec3::new(uv.x, uv.y, 0.0)).collect(),
            };
        }
    };

    let control_points: Vec<Vec3> = cp_u
        .into_iter()
        .zip(cp_v.into_iter())
        .map(|(u, v)| Vec3::new(u, v, 0.0))
        .collect();

    CurveGeom::BSpline {
        degree: deg,
        control_points,
        knots,
        weights: None,
    }
}

// ---------------------------------------------------------------------------
// Gaussian elimination
// ---------------------------------------------------------------------------

fn solve_linear_system(mat: &[Vec<f32>], rhs: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut a: Vec<Vec<f32>> = mat.to_vec();
    let mut b: Vec<f32> = rhs.to_vec();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = a[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-10 {
            return None; // Singular
        }
        // Swap rows
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }
        // Eliminate below
        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        if a[i][i].abs() < 1e-10 {
            return None;
        }
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// B-spline basis (Cox-de Boor)
// ---------------------------------------------------------------------------

/// Evaluate all B-spline basis functions of given degree at parameter t.
///
/// Implements Algorithm A2.2 from "The NURBS Book" (Piegl & Tiller).
/// Returns a vector of n_basis values where n_basis = knots.len() - degree - 1.
fn bspline_basis_all(degree: usize, knots: &[f32], t: f32) -> Vec<f32> {
    let n_knots = knots.len();
    let n_basis = n_knots.saturating_sub(degree + 1);
    if n_basis == 0 {
        return vec![];
    }

    let t_min = knots[degree];
    let t_max = knots[n_knots - degree - 1];
    let t = t.clamp(t_min, t_max);

    // Special case: degree 0
    if degree == 0 {
        let mut result = vec![0.0f32; n_basis];
        result[0] = 1.0;
        return result;
    }

    // Find knot span: knots[span] <= t < knots[span+1]
    // For a clamped knot vector, span is in [degree, n_knots-degree-2]
    let mut span = degree;
    for i in degree..n_knots - degree - 1 {
        if t < knots[i + 1] {
            span = i;
            break;
        }
    }

    // Compute non-zero basis functions using the triangular table (Algorithm A2.2)
    let mut n_table = vec![0.0f32; degree + 1];
    n_table[0] = 1.0;

    let mut left = vec![0.0f32; degree + 1];
    let mut right = vec![0.0f32; degree + 1];

    for j in 1..=degree {
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;

        let mut saved = 0.0f32;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom.abs() > 1e-12 {
                n_table[r] / denom
            } else {
                0.0
            };
            n_table[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n_table[j] = saved;
    }

    // Map triangular table entries into the full basis vector
    let mut result = vec![0.0f32; n_basis];
    let start = span as isize - degree as isize;
    for r in 0..=degree {
        let idx = start + r as isize;
        if idx >= 0 && (idx as usize) < n_basis {
            result[idx as usize] = n_table[r];
        }
    }

    result
}

/// Standard Cox-de Boor: returns the value of the i-th B-spline basis function at t.
fn bspline_basis(degree: usize, knot_index: usize, knots: &[f32], t: f32) -> f32 {
    let all = bspline_basis_all(degree, knots, t);
    if knot_index < all.len() {
        all[knot_index]
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Wire driver
// ---------------------------------------------------------------------------

/// Fix SameParameter for all edges in a wire.
/// Returns the count of PCurves that were fixed.
pub(crate) fn fix_same_parameter_wire(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepStore,
    tolerance: f32,
) -> usize {
    let edge_keys: Vec<EdgeKey> = reg
        .wires
        .get(wire_key)
        .map(|w| w.edges.iter().map(|(ek, _)| *ek).collect())
        .unwrap_or_default();

    let mut fixed = 0usize;
    for ek in edge_keys {
        let report = fix_same_parameter_edge(ek, face_key, reg, tolerance, 3);
        fixed += report.pcurves_fixed;
    }
    fixed
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::geom::SurfaceGeom;
    use crate::topo::{BRepEdge, BRepFace, BRepWire, Orientation};

    /// Helper: build a simple test setup with one edge on a planar face.
    fn build_plane_edge(
        pcurve: CurveGeom,
        degenerate: bool,
    ) -> (BRepStore, EdgeKey, FaceKey, WireKey) {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = if degenerate {
            v0 // same vertex → degenerate
        } else {
            reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4)
        };
        let curve_3d = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
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
        let ek = reg.edges.insert(BRepEdge {
            v_low: v0.min(v1),
            v_high: v0.max(v1),
            curve: curve_3d,
            tolerance: 1e-4,
            pcurves: HashMap::from([(fk, pcurve)]),
        });
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        (reg, ek, fk, wk)
    }

    #[test]
    fn test_no_fix_when_already_same_parameter() {
        // Aligned line pcurve on a plane → no fix needed
        let good_pcurve = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let (mut reg, ek, fk, _wk) = build_plane_edge(good_pcurve, false);
        let report = fix_same_parameter_edge(ek, fk, &mut reg, 1e-3, 3);
        assert_eq!(report.pcurves_fixed, 0);
        assert!(report.max_deviation_before < 1e-3);
        assert!(report.max_deviation_after < 1e-3);
    }

    #[test]
    fn test_fix_misaligned_line_pcurve() {
        // PCurve offset in V by 0.1 → should be corrected
        let bad_pcurve = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.1, 0.0),
            direction: Vec3::X,
        };
        let (mut reg, ek, fk, _wk) = build_plane_edge(bad_pcurve, false);
        let report = fix_same_parameter_edge(ek, fk, &mut reg, 1e-3, 3);
        assert_eq!(report.pcurves_fixed, 1);
        assert!(report.max_deviation_after < report.max_deviation_before);
        // After fix the deviation should be well within tolerance
        assert!(
            report.max_deviation_after < 1e-2,
            "expected deviation < 1e-2 after fix, got {}",
            report.max_deviation_after
        );
    }

    #[test]
    fn test_degenerate_edge_skipped() {
        let pcurve = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let (mut reg, ek, fk, _wk) = build_plane_edge(pcurve, true);
        let report = fix_same_parameter_edge(ek, fk, &mut reg, 1e-3, 3);
        assert_eq!(report.pcurves_fixed, 0);
    }

    #[test]
    fn test_bspline_fitting_collinear() {
        // Collinear points → should produce a Line
        let samples = vec![
            (0.0f32, Vec3::new(0.0, 0.0, 0.0)),
            (0.5, Vec3::new(0.5, 0.0, 0.0)),
            (1.0, Vec3::new(1.0, 0.0, 0.0)),
        ];
        let curve = fit_new_pcurve(&samples, 1e-3);
        assert!(
            matches!(curve, CurveGeom::Line { .. }),
            "expected Line for collinear points, got {:?}",
            curve
        );
    }

    #[test]
    fn test_bspline_fitting_curved() {
        // Non-collinear points → should produce a BSpline
        let samples = vec![
            (0.0f32, Vec3::new(0.0, 0.0, 0.0)),
            (0.25, Vec3::new(0.25, 0.5, 0.0)),
            (0.5, Vec3::new(0.5, 0.5, 0.0)),
            (0.75, Vec3::new(0.75, 0.5, 0.0)),
            (1.0, Vec3::new(1.0, 0.0, 0.0)),
        ];
        let curve = fit_new_pcurve(&samples, 1e-3);
        assert!(
            matches!(curve, CurveGeom::BSpline { .. }),
            "expected BSpline for non-collinear points, got {:?}",
            curve
        );
        // The BSpline should interpolate endpoints
        let p0 = curve.d0(0.0);
        let p1 = curve.d0(1.0);
        assert!((p0.x - 0.0).abs() < 0.05, "start u ~ 0, got {}", p0.x);
        assert!((p1.x - 1.0).abs() < 0.05, "end u ~ 1, got {}", p1.x);
    }

    #[test]
    fn test_periodic_uv_normalization() {
        let surface = SurfaceGeom::Cylinder {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 1.0,
            x_dir: Vec3::X,
            y_dir: Vec3::Y,
        };
        // UV at TAU+0.1, reference at 0.1 → should wrap to 0.1
        let uv = Vec3::new(std::f32::consts::TAU + 0.1, 1.0, 0.0);
        let reference = Vec3::new(0.1, 1.0, 0.0);
        let normalized = normalize_periodic_uv(uv, reference, &surface);
        assert!(
            (normalized.x - 0.1).abs() < 1e-4,
            "expected u ~ 0.1, got {}",
            normalized.x
        );

        // UV at -0.1, reference at TAU-0.1 → should wrap to TAU-0.1
        let uv2 = Vec3::new(-0.1, 1.0, 0.0);
        let reference2 = Vec3::new(std::f32::consts::TAU - 0.1, 1.0, 0.0);
        let normalized2 = normalize_periodic_uv(uv2, reference2, &surface);
        assert!(
            (normalized2.x - (std::f32::consts::TAU - 0.1)).abs() < 1e-4,
            "expected u ~ TAU-0.1, got {}",
            normalized2.x
        );
    }

    #[test]
    fn test_fix_cylinder_shifted_u() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Cylinder {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 1.0,
            x_dir: Vec3::X,
            y_dir: Vec3::Y,
        };
        let v0 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 1.0), 1e-4);
        // 3D curve: vertical line on cylinder at (1,0,z)
        let curve_3d = CurveGeom::Line {
            origin: Vec3::new(1.0, 0.0, 0.0),
            direction: Vec3::new(0.0, 0.0, 1.0),
        };
        // Bad pcurve: shifted U by TAU/4 (90 degrees)
        let bad_pcurve = CurveGeom::Line {
            origin: Vec3::new(std::f32::consts::TAU * 0.25, 0.0, 0.0),
            direction: Vec3::new(0.0, 0.0, 1.0),
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
        let ek = reg.edges.insert(BRepEdge {
            v_low: v0.min(v1),
            v_high: v0.max(v1),
            curve: curve_3d,
            tolerance: 1e-4,
            pcurves: HashMap::from([(fk, bad_pcurve)]),
        });
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];

        let report = fix_same_parameter_edge(ek, fk, &mut reg, 1e-3, 5);
        assert!(
            report.max_deviation_after < report.max_deviation_before,
            "expected improvement: before={}, after={}",
            report.max_deviation_before,
            report.max_deviation_after
        );
    }

    #[test]
    fn test_wire_driver() {
        let bad_pcurve = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.1, 0.0),
            direction: Vec3::X,
        };
        let (mut reg, _ek, fk, wk) = build_plane_edge(bad_pcurve, false);
        let fixed = fix_same_parameter_wire(wk, fk, &mut reg, 1e-3);
        assert_eq!(fixed, 1);
    }
}
