//! Curve derivative computation for STEP curve entities.
//!
//! Computes dC/dt for LINE, CIRCLE, ELLIPSE, and B_SPLINE_CURVE types
//! at a given parameter t in [0, 1].

use std::f32::consts::PI;
use rc3d_core::math::Vec3;
use super::super::parser::EntityIndex;
use super::super::topology;
use super::super::geom;
use super::super::value::StepValue;

/// Compute the derivative dC/dt of a curve entity at parameter `t` in [0, 1].
pub fn curve_derivative(curve_id: u64, entities: &EntityIndex, t: f32) -> Option<Vec3> {
    let record = entities.get(&curve_id)?;
    match record.name.as_str() {
        "LINE" => line_derivative(&record.params, entities),
        "CIRCLE" => circle_derivative(&record.params, entities, t),
        "ELLIPSE" => ellipse_derivative(&record.params, entities, t),
        "B_SPLINE_CURVE_WITH_KNOTS" | "B_SPLINE_CURVE" | "RATIONAL_B_SPLINE_CURVE" => {
            bspline_derivative(&record.params, entities, t)
        }
        "TRIMMED_CURVE" => {
            // Unwrap to the underlying curve
            let inner_id = geom::nth_ref(&record.params, 1)?;
            curve_derivative(inner_id, entities, t)
        }
        "OFFSET_CURVE_3D" => {
            let inner_id = geom::nth_ref(&record.params, 1)?;
            curve_derivative(inner_id, entities, t)
        }
        "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE" => {
            let inner_id = geom::nth_ref(&record.params, 1)?;
            curve_derivative(inner_id, entities, t)
        }
        "COMPOSITE_CURVE" => {
            // For composite curves, find the segment that covers t
            let seg_ids = geom::nth_list_refs(&record.params, 1).unwrap_or_default();
            if seg_ids.is_empty() {
                return None;
            }
            let n = seg_ids.len();
            let seg_idx = ((t * n as f32) as usize).min(n - 1);
            let seg_t = (t * n as f32) - seg_idx as f32;
            if let Some(seg) = entities.get(&seg_ids[seg_idx]) {
                if let Some(parent_id) = geom::nth_ref(&seg.params, 3) {
                    return curve_derivative(parent_id, entities, seg_t);
                }
            }
            None
        }
        _ => None,
    }
}

/// LINE derivative: constant direction vector.
/// LINE('', #pnt, #dir) → dC/dt = direction
fn line_derivative(params: &StepValue, entities: &EntityIndex) -> Option<Vec3> {
    let dir_id = geom::nth_ref(params, 2)?;
    topology::resolve_direction(dir_id, entities)
}

/// CIRCLE derivative: dC/dt = 2πr·(-sin(θ)·X + cos(θ)·Y)
/// where θ = 2πt. At t=0, derivative points in +Y direction.
fn circle_derivative(params: &StepValue, entities: &EntityIndex, t: f32) -> Option<Vec3> {
    let pos_id = geom::nth_ref(params, 1)?;
    let radius = geom::nth_real(params, 2).unwrap_or(1.0) as f32;
    let (_origin, x_axis, z_axis) = topology::resolve_placement(pos_id, entities)?;
    let y_axis = z_axis.cross(x_axis).normalize();

    let theta = t * 2.0 * PI;
    let tangent = -x_axis * (radius * theta.sin()) + y_axis * (radius * theta.cos());
    Some(tangent * 2.0 * PI)
}

/// ELLIPSE derivative: dC/dt = 2π·(-a·sin(θ)·X + b·cos(θ)·Y)
/// where θ = 2πt, a and b are semi-axes.
fn ellipse_derivative(params: &StepValue, entities: &EntityIndex, t: f32) -> Option<Vec3> {
    let pos_id = geom::nth_ref(params, 1)?;
    let a = geom::nth_real(params, 2).unwrap_or(1.0) as f32;
    let b = geom::nth_real(params, 3).unwrap_or(1.0) as f32;
    let (_origin, x_axis, z_axis) = topology::resolve_placement(pos_id, entities)?;
    let y_axis = z_axis.cross(x_axis).normalize();

    let theta = t * 2.0 * PI;
    let tangent = -x_axis * (a * theta.sin()) + y_axis * (b * theta.cos());
    Some(tangent * 2.0 * PI)
}

/// B-spline derivative using basis function derivative recurrence.
///
/// N'_{i,p} = p/(k_{i+p}-k_i)·N_{i,p-1} - p/(k_{i+p+1}-k_{i+1})·N_{i+1,p-1}
///
/// For rational curves, the quotient rule is applied.
fn bspline_derivative(params: &StepValue, entities: &EntityIndex, t: f32) -> Option<Vec3> {
    let degree = geom::nth_int(params, 1).unwrap_or(3) as usize;
    let ctrl_pts = geom::resolve_bspline_ctrl_pts(params, 2, entities);

    let knots_raw = geom::nth_list_reals(params, 7);
    let multiplicities = geom::nth_list_ints(params, 6);

    if ctrl_pts.is_empty() || knots_raw.is_empty() {
        return None;
    }

    // Build knot vector
    let knot_vec = build_knot_vector(&knots_raw, &multiplicities, degree, ctrl_pts.len());

    // Extract rational weights
    let weights: Vec<f32> = geom::find_weights_list(params, ctrl_pts.len())
        .unwrap_or_else(|| vec![1.0; ctrl_pts.len()]);

    let n = ctrl_pts.len();
    if n < degree + 1 {
        return None;
    }

    // Clamp t to valid knot range
    let u_min = knot_vec[degree];
    let u_max = knot_vec[n];
    let u = u_min + t * (u_max - u_min);

    let span = geom::find_span(degree, &knot_vec, u);

    // Compute basis functions N_{i,p}(u) for i = span-degree .. span
    let bases = geom::bspline_bases(span, degree, u, &knot_vec);

    // Compute basis function derivatives N'_{i,p}(u) using recurrence
    let bases_deriv = bspline_basis_derivatives(span, degree, u, &knot_vec);

    // Build the set of all basis indices from both bases and bases_deriv
    let max_idx = (span + 1).min(n);
    let min_idx = span.saturating_sub(degree + 1);

    // Rational derivative via quotient rule:
    // C = A/W where A = Σ w_i·P_i·N_i, W = Σ w_i·N_i
    // C' = (A'·W - A·W') / W²
    let mut a = Vec3::ZERO;
    let mut w = 0.0f32;
    let mut a_deriv = Vec3::ZERO;
    let mut w_deriv = 0.0f32;

    for i in min_idx..max_idx.min(n) {
        let wi = *weights.get(i).unwrap_or(&1.0);
        let pi = ctrl_pts[i];

        // Find N_i(u) in bases list
        let n_i = bases.iter()
            .find(|(k, _)| *k == i)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);

        // Find N'_i(u) in bases_deriv list
        let n_deriv_i = bases_deriv.iter()
            .find(|(k, _)| *k == i)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);

        a += pi * wi * n_i;
        w += wi * n_i;
        a_deriv += pi * wi * n_deriv_i;
        w_deriv += wi * n_deriv_i;
    }

    if w.abs() < 1e-10 {
        return None;
    }

    // C' = (A'·W - A·W') / W²
    Some((a_deriv * w - a * w_deriv) / (w * w))
}

/// Compute B-spline basis function derivatives N'_{i,p}(u) at parameter u.
///
/// Recurrence: N'_{i,p} = p/(k_{i+p}-k_i)·N_{i,p-1} - p/(k_{i+p+1}-k_{i+1})·N_{i+1,p-1}
/// with the convention that terms with zero denominator vanish.
fn bspline_basis_derivatives(
    span: usize,
    degree: usize,
    u: f32,
    knots: &[f32],
) -> Vec<(usize, f32)> {
    if degree == 0 {
        // N'_{i,0} = 0 everywhere (step function, derivative is a Dirac delta)
        return vec![];
    }

    let mut result = Vec::new();
    let n = knots.len();

    // N'_{i,p}(u) depends only on N_{i,p-1}(u) and N_{i+1,p-1}(u)
    // The non-zero N_{i,p-1} are for i in [span-(p-1), span]
    for i in span.saturating_sub(degree)..=span {
        if i + degree >= n {
            continue;
        }

        // Left term: p/(k_{i+p} - k_i) * N_{i,p-1}(u)
        let left = if i + degree < n {
            let denom = knots[i + degree] - knots[i];
            if denom > 1e-10 {
                let n_i_pm1 = basis_value(i, degree - 1, u, knots);
                degree as f32 / denom * n_i_pm1
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Right term: p/(k_{i+p+1} - k_{i+1}) * N_{i+1,p-1}(u)
        let right = if i + degree + 1 < n {
            let denom = knots[i + degree + 1] - knots[i + 1];
            if denom > 1e-10 {
                let n_ip1_pm1 = basis_value(i + 1, degree - 1, u, knots);
                degree as f32 / denom * n_ip1_pm1
            } else {
                0.0
            }
        } else {
            0.0
        };

        let val = left - right;
        if val.abs() > 1e-10 {
            result.push((i, val));
        }
    }

    result
}

/// Evaluate a single basis function N_{i,p}(u) using recursive definition.
fn basis_value(i: usize, p: usize, u: f32, knots: &[f32]) -> f32 {
    if p == 0 {
        // Indicator: 1 if u in [k_i, k_{i+1}), 0 otherwise.
        // At the last knot, include the right endpoint.
        if i + 1 < knots.len() {
            if u >= knots[i] && u < knots[i + 1] {
                return 1.0;
            }
            // Handle the very last knot span: include right endpoint
            if i + 2 == knots.len() && (u - knots[i + 1]).abs() < 1e-10 {
                return 1.0;
            }
        }
        return 0.0;
    }

    let mut val = 0.0;

    // Left recursive term
    if i + p < knots.len() {
        let denom = knots[i + p] - knots[i];
        if denom > 1e-10 {
            val += (u - knots[i]) / denom * basis_value(i, p - 1, u, knots);
        }
    }

    // Right recursive term
    if i + p + 1 < knots.len() {
        let denom = knots[i + p + 1] - knots[i + 1];
        if denom > 1e-10 {
            val += (knots[i + p + 1] - u) / denom * basis_value(i + 1, p - 1, u, knots);
        }
    }

    val
}

// ── B-spline helpers (reuse logic from geom via public access) ────────

/// Build the full knot vector from knot values and multiplicities.
/// Mirrors the logic in geom::sample_bspline.
fn build_knot_vector(
    knots: &[StepValue],
    multiplicities: &[i64],
    degree: usize,
    num_ctrl_pts: usize,
) -> Vec<f32> {
    let mut knot_vec: Vec<f32> = Vec::new();

    if !multiplicities.is_empty() && !knots.is_empty() {
        for (i, &mult) in multiplicities.iter().enumerate() {
            if i < knots.len() {
                let k = knots[i].as_real().unwrap_or(0.0) as f32;
                for _ in 0..mult {
                    knot_vec.push(k);
                }
            }
        }
    }

    // Expand if needed
    if knot_vec.len() < num_ctrl_pts + degree + 1 {
        let mut expanded: Vec<f32> = Vec::new();
        for (i, k) in knots.iter().enumerate() {
            let kval = k.as_real().unwrap_or(i as f64) as f32;
            let mult = if i < multiplicities.len() { multiplicities[i] as usize } else { degree };
            for _ in 0..mult {
                expanded.push(kval);
            }
        }
        if expanded.len() < num_ctrl_pts + degree + 1 {
            let min_k = *expanded.first().unwrap_or(&0.0);
            let max_k = *expanded.last().unwrap_or(&1.0);
            let needed = num_ctrl_pts + degree + 1 - expanded.len();
            for i in 0..needed {
                let t = (i + 1) as f32 / (needed + 1) as f32;
                expanded.push(min_k + (max_k - min_k) * t);
            }
        }
        knot_vec = expanded;
    }

    knot_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    knot_vec
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::parser;

    fn make_entities(data_section: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data_section
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_line_derivative_is_direction() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (1.0, 0.0, 0.0));
#10 = LINE('', #1, #2);\
",
        );
        let deriv = curve_derivative(10, &entities, 0.5).unwrap();
        assert!((deriv - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4,
            "line derivative should equal the direction vector");
        // Derivative should be the same at any t
        let deriv2 = curve_derivative(10, &entities, 0.0).unwrap();
        assert!((deriv - deriv2).length() < 1e-4,
            "line derivative should be constant");
    }

    #[test]
    fn test_circle_derivative_orthogonal_to_radius() {
        // Circle in XY plane: origin at (0,0,0), Z axis up, X axis right
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#10 = CIRCLE('', #4, 1.0);\
",
        );
        // At t=0, the point is at (1, 0, 0) on the circle.
        // The tangent (derivative) should be pointing in +Y direction.
        let deriv = curve_derivative(10, &entities, 0.0).unwrap();
        assert!((deriv.x).abs() < 1e-4, "tangent at t=0 should have zero x component");
        assert!(deriv.y > 0.0, "tangent at t=0 should point in +Y direction");
        assert!((deriv.z).abs() < 1e-4, "tangent at t=0 should have zero z component");
        // Magnitude should be 2πr = 2π for unit circle
        assert!((deriv.length() - 2.0 * PI).abs() < 1e-3,
            "circle derivative magnitude should be 2πr, got {}", deriv.length());
    }

    #[test]
    fn test_circle_derivative_at_quarter() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#10 = CIRCLE('', #4, 1.0);\
",
        );
        // At t=0.25, angle=π/2. Point is at (0, 1, 0). Tangent should point -X.
        let deriv = curve_derivative(10, &entities, 0.25).unwrap();
        assert!(deriv.x < 0.0, "tangent at t=0.25 should point in -X direction");
        assert!((deriv.y).abs() < 1e-4, "tangent at t=0.25 should have zero y component");
    }

    #[test]
    fn test_ellipse_derivative() {
        // Ellipse with semi-axes a=2, b=1 in XY plane
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#10 = ELLIPSE('', #4, 2.0, 1.0);\
",
        );
        // At t=0, the point is at (2, 0, 0). Tangent should be in +Y with magnitude 2πb.
        let deriv = curve_derivative(10, &entities, 0.0).unwrap();
        assert!((deriv.x).abs() < 1e-4, "tangent at t=0 should have zero x component");
        assert!(deriv.y > 0.0, "tangent at t=0 should point in +Y direction");
        assert!((deriv.length() - 2.0 * PI * 1.0).abs() < 1e-2,
            "ellipse derivative magnitude at t=0 should be 2πb");
    }

    #[test]
    fn test_trimmed_curve_unwrap() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (1.0, 0.0, 0.0));
#10 = LINE('', #1, #2);
#20 = TRIMMED_CURVE('', #10, (0.0, 1.0), (0.0, 1.0), .T., .CARTESIAN.);\
",
        );
        // TRIMMED_CURVE should unwrap to the inner LINE
        let deriv = curve_derivative(20, &entities, 0.5).unwrap();
        assert!((deriv - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn test_bspline_derivative_non_zero() {
        // Simple degree-1 B-spline (line segment)
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#10 = B_SPLINE_CURVE_WITH_KNOTS('', 1, (#1, #2), .UNSPECIFIED., .F., .F., (2, 2), (0.0, 1.0), .UNSPECIFIED.);\
",
        );
        // Derivative of a linear B-spline should be positive in x direction
        let deriv = curve_derivative(10, &entities, 0.5).unwrap();
        assert!(deriv.x > 0.0, "bspline derivative should be non-zero");
        assert!((deriv.y).abs() < 1e-4);
        assert!((deriv.z).abs() < 1e-4);
    }

    #[test]
    fn test_nonexistent_curve() {
        let entities = make_entities("");
        assert!(curve_derivative(999, &entities, 0.5).is_none());
    }
}
