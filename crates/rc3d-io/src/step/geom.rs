//! Curve geometry evaluation and surface parameterization.

use std::f32::consts::PI;
use rc3d_core::math::Vec3;
use super::parser::EntityIndex;
use super::topology;
use super::value::StepValue;

/// Sample a curve entity into a list of 3D points.
pub fn sample_curve(
    curve_id: u64,
    entities: &EntityIndex,
    start: Vec3,
    end: Vec3,
    tolerance: f32,
) -> Vec<Vec3> {
    let record = match entities.get(&curve_id) {
        Some(r) => r,
        None => return vec![start, end],
    };
    match record.name.as_str() {
        "LINE" => sample_line(&record.params, entities, start, end),
        "CIRCLE" => sample_circle(&record.params, entities, start, end, tolerance),
        "ELLIPSE" => sample_ellipse(&record.params, entities, start, end, tolerance),
        "HYPERBOLA" => sample_hyperbola(&record.params, entities),
        "PARABOLA" => sample_parabola(&record.params, entities),
        "POLYLINE" => sample_polyline(&record.params, entities),
        "B_SPLINE_CURVE_WITH_KNOTS" | "B_SPLINE_CURVE" | "RATIONAL_B_SPLINE_CURVE" => {
            sample_bspline(&record.params, entities, tolerance)
        }
        "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE" => {
            // surface-bounded curve wrappers: unwrap to the underlying 3D curve
            // SURFACE_CURVE/SEAM_CURVE/INTERSECTION_CURVE('', #curve_3d, pcurves, sense)
            if let Some(geom_curve_id) = nth_ref(&record.params, 1) {
                sample_curve(geom_curve_id, entities, start, end, tolerance)
            } else {
                vec![start, end]
            }
        }
        "TRIMMED_CURVE" => {
            // TRIMMED_CURVE('', #basis_curve, (trim1), (trim2), .T., .CARTESIAN.)
            // Unwrap to the underlying curve for sampling
            if let Some(inner_id) = nth_ref(&record.params, 1) {
                sample_curve(inner_id, entities, start, end, tolerance)
            } else {
                vec![start, end]
            }
        }
        "OFFSET_CURVE_3D" => {
            // OFFSET_CURVE_3D('', #basis_curve, #ref_direction, distance, .F.)
            // Unwrap to the underlying curve (ignore offset for sampling)
            if let Some(inner_id) = nth_ref(&record.params, 1) {
                sample_curve(inner_id, entities, start, end, tolerance)
            } else {
                vec![start, end]
            }
        }
        "COMPOSITE_CURVE" => {
            // COMPOSITE_CURVE('', (#seg1, #seg2, ...), .F.)
            // Each segment is a COMPOSITE_CURVE_SEGMENT('', .CONTINUOUS., .T., #parent_curve)
            let seg_ids = nth_list_refs(&record.params, 1).unwrap_or_default();
            let mut all_pts = Vec::new();
            for seg_id in seg_ids {
                if let Some(seg) = entities.get(&seg_id) {
                    // params: [0]=name, [1]=transition, [2]=same_sense, [3]=#parent_curve
                    if let Some(parent_id) = nth_ref(&seg.params, 3) {
                        let seg_pts = sample_curve(parent_id, entities, start, end, tolerance);
                        all_pts.extend(seg_pts);
                    }
                }
            }
            if all_pts.is_empty() { vec![start, end] } else { all_pts }
        }
        _ => vec![start, end],
    }
}

fn sample_line(
    params: &StepValue,
    entities: &EntityIndex,
    start: Vec3,
    end: Vec3,
) -> Vec<Vec3> {
    // LINE args: (name, #pnt, #dir)
    let pnt_id = nth_ref(params, 1);
    let dir_id = nth_ref(params, 2);
    if let (Some(pid), Some(did)) = (pnt_id, dir_id) {
        if let Some(p) = topology::resolve_point(pid, entities) {
            // DIRECTION entities must be resolved via resolve_direction, not resolve_point
            let dir = topology::resolve_direction(did, entities)
                .unwrap_or_else(|| {
                    // Fallback: try as VECTOR or other direction-like entity
                    resolve_direction_fallback(did, entities)
                        .unwrap_or(Vec3::Z)
                });
            return vec![p, p + dir];
        }
    }
    vec![start, end]
}

/// Fallback direction resolution for VECTOR and other direction-like entities.
/// Compute adaptive sample count using chordal tolerance.
/// For arc length `L`, radius `R`, and chordal tolerance `ε`:
///   n = max(2, min(128, ceil(L / sqrt(8 * ε * R))))
fn chordal_sample_count(arc_length: f32, radius: f32, tolerance: f32) -> usize {
    if radius < 1e-6 || tolerance < 1e-10 {
        return 4;
    }
    let step = (8.0 * tolerance * radius).sqrt();
    let n = (arc_length / step.max(1e-6)).ceil() as usize;
    n.max(4).min(128)
}

fn resolve_direction_fallback(dir_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    let record = entities.get(&dir_id)?;
    match record.name.as_str() {
        "VECTOR" => {
            // VECTOR: (name, #direction, magnitude)
            let inner_dir_id = nth_ref(&record.params, 1)?;
            topology::resolve_direction(inner_dir_id, entities)
        }
        _ => None,
    }
}

fn sample_circle(
    params: &StepValue,
    entities: &EntityIndex,
    _start: Vec3,
    _end: Vec3,
    tolerance: f32,
) -> Vec<Vec3> {
    // CIRCLE args: (name, #position, radius)
    let pos_id = nth_ref(params, 1);
    let radius = nth_real(params, 2).unwrap_or(1.0) as f32;
    let (origin, x_axis, z_axis) = pos_id
        .and_then(|id| topology::resolve_placement(id, entities))
        .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));

    let circumference = radius.abs() * 2.0 * PI;
    let n = chordal_sample_count(circumference, radius.abs(), tolerance);
    let mut points = Vec::with_capacity(n + 1);
    let y_axis = z_axis.cross(x_axis).normalize();
    for i in 0..=n {
        let angle = (i as f32) * 2.0 * PI / (n as f32);
        let pt = origin + x_axis * (radius * angle.cos()) + y_axis * (radius * angle.sin());
        points.push(pt);
    }
    points
}

fn sample_ellipse(
    params: &StepValue,
    entities: &EntityIndex,
    _start: Vec3,
    _end: Vec3,
    tolerance: f32,
) -> Vec<Vec3> {
    // ELLIPSE args: (name, #position, semi_axis_1, semi_axis_2)
    let pos_id = nth_ref(params, 1);
    let a = nth_real(params, 2).unwrap_or(1.0) as f32;
    let b = nth_real(params, 3).unwrap_or(1.0) as f32;
    let (origin, x_axis, z_axis) = pos_id
        .and_then(|id| topology::resolve_placement(id, entities))
        .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));

    // Ramanujan approximation for ellipse circumference
    let circ = PI * (3.0 * (a.abs() + b.abs())
        - ((3.0 * a.abs() + b.abs()) * (a.abs() + 3.0 * b.abs())).sqrt());
    let n = chordal_sample_count(circ, a.abs().max(b.abs()), tolerance);
    let mut points = Vec::with_capacity(n + 1);
    let y_axis = z_axis.cross(x_axis).normalize();
    for i in 0..=n {
        let angle = (i as f32) * 2.0 * PI / (n as f32);
        let pt = origin + x_axis * (a * angle.cos()) + y_axis * (b * angle.sin());
        points.push(pt);
    }
    points
}

fn sample_polyline(
    params: &StepValue,
    entities: &EntityIndex,
) -> Vec<Vec3> {
    // POLYLINE args: (name, (#pnt1, #pnt2, ...))
    let pt_ids = nth_list_refs(params, 1).unwrap_or_default();
    pt_ids.iter().filter_map(|&id| topology::resolve_point(id, entities)).collect()
}

fn sample_hyperbola(
    params: &StepValue,
    entities: &EntityIndex,
) -> Vec<Vec3> {
    // HYPERBOLA: (name, #position, semi_axis, semi_imag_axis)
    let pos_id = nth_ref(params, 1);
    let a = nth_real(params, 2).unwrap_or(1.0) as f32;
    let b = nth_real(params, 3).unwrap_or(1.0) as f32;
    let (origin, x_axis, z_axis) = pos_id
        .and_then(|id| topology::resolve_placement(id, entities))
        .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
    let y_axis = z_axis.cross(x_axis).normalize();

    let n = 64;
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t_val = -3.0 + 6.0 * (i as f32 / n as f32);
        let x = a * t_val.cosh();
        let y = b * t_val.sinh();
        pts.push(origin + x_axis * x + y_axis * y);
    }
    pts
}

fn sample_parabola(
    params: &StepValue,
    entities: &EntityIndex,
) -> Vec<Vec3> {
    // PARABOLA: (name, #position, focal_dist)
    let pos_id = nth_ref(params, 1);
    let f = nth_real(params, 2).unwrap_or(1.0) as f32;
    let (origin, x_axis, z_axis) = pos_id
        .and_then(|id| topology::resolve_placement(id, entities))
        .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
    let y_axis = z_axis.cross(x_axis).normalize();

    let n = 64;
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t_val = -5.0 + 10.0 * (i as f32 / n as f32);
        let x = t_val;
        let y = t_val * t_val / (4.0 * f);
        pts.push(origin + x_axis * x + y_axis * y);
    }
    pts
}

/// Evaluate a surface entity into a grid of 3D points (rows × cols).
/// Returns None for unsupported surface types.
pub fn evaluate_surface(
    surface_id: u64,
    entities: &EntityIndex,
    samples_u: usize,
    samples_v: usize,
) -> Option<Vec<Vec<Vec3>>> {
    let record = entities.get(&surface_id)?;
    match record.name.as_str() {
        "SURFACE_OF_LINEAR_EXTRUSION" => eval_extrusion(&record.params, entities, samples_u, samples_v),
        "SURFACE_OF_REVOLUTION" => eval_revolution(&record.params, entities, samples_u, samples_v),
        _ => None,
    }
}

fn eval_extrusion(
    params: &StepValue,
    entities: &EntityIndex,
    _samples_u: usize,
    samples_v: usize,
) -> Option<Vec<Vec<Vec3>>> {
    // SURFACE_OF_LINEAR_EXTRUSION: (name, #swept_curve, #extrusion_axis)
    let curve_id = nth_ref(params, 1)?;
    let axis_id = nth_ref(params, 2)?;
    let direction = resolve_direction(axis_id, entities)?;

    // Sample the generatrix curve
    let curve_pts = sample_curve(curve_id, entities, Vec3::ZERO, Vec3::ZERO, 0.1);
    if curve_pts.is_empty() {
        return None;
    }

    let n_curve = curve_pts.len();
    let n_sweep = samples_v;
    let mut grid: Vec<Vec<Vec3>> = Vec::with_capacity(n_curve);
    for i in 0..n_curve {
        let mut row = Vec::with_capacity(n_sweep);
        for j in 0..n_sweep {
            let t = j as f32 / (n_sweep - 1).max(1) as f32;
            row.push(curve_pts[i] + direction * t);
        }
        grid.push(row);
    }
    Some(grid)
}

fn eval_revolution(
    params: &StepValue,
    entities: &EntityIndex,
    samples_u: usize,
    _samples_v: usize,
) -> Option<Vec<Vec<Vec3>>> {
    // SURFACE_OF_REVOLUTION: (name, #swept_curve, #axis_position)
    let curve_id = nth_ref(params, 1)?;
    let axis_placement_id = nth_ref(params, 2);

    let (origin, axis) = axis_placement_id
        .and_then(|id| topology::resolve_sweep_axis(id, entities))
        .unwrap_or((Vec3::ZERO, Vec3::Z));

    // Sample the generatrix curve
    let curve_pts = sample_curve(curve_id, entities, Vec3::ZERO, Vec3::ZERO, 0.1);
    if curve_pts.is_empty() {
        return None;
    }

    let n_curve = curve_pts.len();
    let n_angle = samples_u;
    let mut grid: Vec<Vec<Vec3>> = Vec::with_capacity(n_angle + 1);
    for i in 0..=n_angle {
        let angle = (i as f32) * 2.0 * std::f32::consts::PI / n_angle as f32;
        let mut row = Vec::with_capacity(n_curve);
        for pt in &curve_pts {
            // Rotate point around axis through origin
            let rel = *pt - origin;
            let rotated = rotate_around_axis(rel, axis, angle) + origin;
            row.push(rotated);
        }
        if i == n_angle {
            // Duplicate first row so grid_to_mesh can seam the last quad ring
            row = grid[0].clone();
        }
        grid.push(row);
    }
    Some(grid)
}

fn rotate_around_axis(v: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    v * cos_a + axis.cross(v) * sin_a + axis * axis.dot(v) * (1.0 - cos_a)
}

fn resolve_direction(dir_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    let record = entities.get(&dir_id)?;
    if record.name != "DIRECTION" && record.name != "VECTOR" {
        return None;
    }
    let coords: Vec<f64> = record.params.nth_param(1)
        .and_then(|v| v.as_list())
        .map(|list| list.iter().filter_map(|v| v.as_real()).collect())
        .unwrap_or_default();
    if coords.len() < 3 {
        return None;
    }
    Some(Vec3::new(coords[0] as f32, coords[1] as f32, coords[2] as f32))
}

fn sample_bspline(
    params: &StepValue,
    entities: &EntityIndex,
    _tolerance: f32,
) -> Vec<Vec3> {
    // B_SPLINE_CURVE_WITH_KNOTS args:
    // (name, degree, control_points, curve_form, closed, self_intersect,
    //  knot_multiplicities, knots, knot_spec)
    let degree = nth_int(params, 1).unwrap_or(3) as usize;

    // Control points are a list of references: (#pt1, #pt2, ...)
    // NOT nested coordinate lists as previously assumed.
    let ctrl_pts = resolve_bspline_ctrl_pts(params, 2, entities);
    let knots = nth_list_reals(params, 7);
    let multiplicities = nth_list_ints(params, 6);

    if ctrl_pts.is_empty() || knots.is_empty() {
        return vec![];
    }

    // Build the full knot vector from knot values and multiplicities
    // For PIECEWISE_BEZIER_KNOTS, all interior knots have multiplicity = degree
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

    // For PIECEWISE_BEZIER_KNOTS, we need enough knots
    // If knot_vec is too short, expand it properly
    if knot_vec.len() < ctrl_pts.len() + degree + 1 {
        // Expand knots for piecewise bezier: the knot values define segments
        // Each segment corresponds to one bezier curve
        let mut expanded: Vec<f32> = Vec::new();
        for (i, k) in knots.iter().enumerate() {
            let kval = k.as_real().unwrap_or(i as f64) as f32;
            let mult = if i < multiplicities.len() { multiplicities[i] as usize } else { degree };
            for _ in 0..mult {
                expanded.push(kval);
            }
        }
        // Ensure we have enough knots by interpolating
        if expanded.len() < ctrl_pts.len() + degree + 1 {
            let min_k = *expanded.first().unwrap_or(&0.0);
            let max_k = *expanded.last().unwrap_or(&1.0);
            let needed = ctrl_pts.len() + degree + 1 - expanded.len();
            for i in 0..needed {
                let t = (i + 1) as f32 / (needed + 1) as f32;
                expanded.push(min_k + (max_k - min_k) * t);
            }
        }
        knot_vec = expanded;
    }

    // Ensure knot vector is sorted (expansion may append interpolated values out of order).
    // Use stable sort to preserve multiplicity ordering for identical knot values.
    knot_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Extract rational weights — scan all params for a List of reals matching ctrl_pts count.
    // RATIONAL_B_SPLINE_CURVE entity has its weight list merged after the knot spec params.
    let weights: Vec<f32> = find_weights_list(params, ctrl_pts.len())
        .unwrap_or_else(|| vec![1.0; ctrl_pts.len()]);
    let pts: Vec<[f32; 4]> = ctrl_pts.iter().enumerate()
        .map(|(i, p)| [p.x, p.y, p.z, weights.get(i).copied().unwrap_or(1.0)])
        .collect();

    let n_pts = (pts.len() * 8).max(16);
    let mut result = Vec::with_capacity(n_pts);

    // Use valid knot range for evaluation
    let knot_min_idx = degree.min(knot_vec.len().saturating_sub(1));
    let knot_max_idx = (knot_vec.len() - degree - 1).min(knot_vec.len().saturating_sub(1));
    let actual_u_min = knot_vec.get(knot_min_idx).copied().unwrap_or(0.0);
    let actual_u_max = knot_vec.get(knot_max_idx).copied().unwrap_or(1.0);

    for i in 0..=n_pts {
        let t = actual_u_min + (actual_u_max - actual_u_min) * (i as f32) / (n_pts as f32);
        let pt = eval_bspline_curve(&pts, degree, &knot_vec, t);
        result.push(Vec3::new(pt[0], pt[1], pt[2]));
    }
    result
}

/// Resolve B-spline control points from a parameter that contains
/// a list of entity references (e.g. (#10, #20, #30, ...)).
pub fn resolve_bspline_ctrl_pts(
    params: &StepValue,
    index: usize,
    entities: &EntityIndex,
) -> Vec<Vec3> {
    let list = match params.nth_param(index).and_then(|v| v.as_list()) {
        Some(l) => l,
        None => return vec![],
    };
    list.iter()
        .filter_map(|v| v.as_ref_id())
        .filter_map(|id| topology::resolve_point(id, entities))
        .collect()
}

fn eval_bspline_curve(ctrl: &[[f32; 4]], degree: usize, knots: &[f32], t: f32) -> [f32; 4] {
    let n = ctrl.len();
    let span = find_span(degree, knots, t);
    let basis = bspline_bases(span, degree, t, knots);
    let mut pt = [0.0f32; 4];
    for (k, b) in &basis {
        let k = *k;
        if k < n {
            let w = ctrl[k][3];
            pt[0] += *b * ctrl[k][0] * w;
            pt[1] += *b * ctrl[k][1] * w;
            pt[2] += *b * ctrl[k][2] * w;
            pt[3] += b * w;
        }
    }
    if pt[3] != 0.0 {
        pt[0] /= pt[3];
        pt[1] /= pt[3];
        pt[2] /= pt[3];
    }
    pt
}

/// Scan all params for a List of real values matching the expected count.
/// Used to extract rational B-spline weights that are appended after knot_spec.
pub fn find_weights_list(params: &StepValue, expected_count: usize) -> Option<Vec<f32>> {
    let list = params.as_list()?;
    // Distinguish non-rational B-splines (only knots list at index 7)
    // from rational B-splines (knots at 7 + weights at 9).
    // Count ALL List-of-Reals in params. If there are 2+,
    // the last one is the weights (and it must match expected_count).
    // If only 1, it's the knots of a non-rational curve → return None.
    let real_lists: Vec<&Vec<StepValue>> = list.iter()
        .filter_map(|v| {
            if let StepValue::List(inner) = v {
                if inner.iter().all(|x| matches!(x, StepValue::Real(_))) {
                    return Some(inner);
                }
            }
            None
        })
        .collect();
    if real_lists.len() >= 2 {
        let candidate = real_lists.last().unwrap();
        if candidate.len() == expected_count {
            return Some(candidate.iter().map(|v| v.as_real().unwrap() as f32).collect());
        }
    }
    None
}

pub use crate::step::brep::geom::bspline::{bspline_bases, find_span};

// ── Helpers ─────────────────────────────────────────────────

pub fn nth_ref(params: &StepValue, index: usize) -> Option<u64> {
    params.nth_param(index)?.as_ref_id()
}

pub fn nth_list_refs(params: &StepValue, index: usize) -> Option<Vec<u64>> {
    params.nth_param(index)?.as_list()
        .map(|v| v.iter().filter_map(|p| p.as_ref_id()).collect())
}

pub fn nth_real(params: &StepValue, index: usize) -> Option<f64> {
    params.nth_param(index)?.as_real()
}

pub fn nth_int(params: &StepValue, index: usize) -> Option<i64> {
    match params.nth_param(index) {
        Some(StepValue::Integer(v)) => Some(*v),
        Some(StepValue::Real(v)) => Some(*v as i64),
        _ => None,
    }
}

pub fn nth_list_reals(params: &StepValue, index: usize) -> Vec<StepValue> {
    match params.nth_param(index) {
        Some(StepValue::List(v)) => v.clone(),
        _ => vec![],
    }
}

pub fn nth_list_ints(params: &StepValue, index: usize) -> Vec<i64> {
    match params.nth_param(index) {
        Some(StepValue::List(v)) => v.iter().filter_map(|p| p.as_int()).collect(),
        _ => vec![],
    }
}

#[allow(dead_code)]
// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser;

    fn make_entities(data_section: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data_section
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_polyline_returns_all_points() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (1.0, 2.0, 3.0));
#3 = CARTESIAN_POINT('', (4.0, 5.0, 6.0));
#10 = POLYLINE('', (#1, #2, #3));\
",
        );
        let pts = sample_curve(10, &entities, Vec3::ZERO, Vec3::ZERO, 0.1);
        assert_eq!(pts.len(), 3);
    }

    #[test]
    fn test_polyline_as_edge_curve() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#10 = POLYLINE('', (#1, #2));\
",
        );
        let pts = sample_curve(10, &entities, Vec3::ZERO, Vec3::ZERO, 0.1);
        assert_eq!(pts.len(), 2);
    }

    #[test]
    fn test_extrusion_surface() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (0.0, 5.0, 0.0));
#3 = DIRECTION('', (0.0, 0.0, 10.0));
#10 = LINE('', #1, #2);
#20 = SURFACE_OF_LINEAR_EXTRUSION('', #10, #3);\
",
        );
        let grid = evaluate_surface(20, &entities, 8, 8).unwrap();
        // grid[rows=curve_pts][cols=extrusion_steps]
        assert!(!grid.is_empty());
        assert!(!grid[0].is_empty());
    }

    #[test]
    fn test_revolution_surface() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (2.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#4 = DIRECTION('', (0.0, 0.0, 1.0));
#5 = DIRECTION('', (1.0, 0.0, 0.0));
#6 = AXIS2_PLACEMENT_3D('', #3, #4, #5);
#10 = LINE('', #1, #2);
#20 = SURFACE_OF_REVOLUTION('', #10, #6);\
",
        );
        let grid = evaluate_surface(20, &entities, 16, 8).unwrap();
        assert!(!grid.is_empty());
        assert!(!grid[0].is_empty());
        // Revolution of a radius-2 line around Z should produce cylindrical-like grid
        assert!(grid.len() >= 16);
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
        // TRIMMED_CURVE unwraps to the inner LINE — should produce start + direction
        let pts = sample_curve(20, &entities, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.1);
        assert_eq!(pts.len(), 2, "trimmed line should produce 2 points");
        assert!((pts[0] - Vec3::ZERO).length() < 1e-4);
        // LINE: pnt=(0,0,0) + dir=(1,0,0) = (1,0,0)
        assert!((pts[1] - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn test_composite_curve_segments() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (1.0, 0.0, 0.0));
#10 = LINE('', #1, #2);
#11 = LINE('', #1, #2);
#20 = COMPOSITE_CURVE_SEGMENT('', .CONTINUOUS., .T., #10);
#21 = COMPOSITE_CURVE_SEGMENT('', .CONTINUOUS., .T., #11);
#30 = COMPOSITE_CURVE('', (#20, #21), .F.);\
",
        );
        let pts = sample_curve(30, &entities, Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.1);
        assert!(pts.len() >= 3, "composite curve should sample points from both segments");
    }

    #[test]
    fn test_bspline_knot_vector_sorted() {
        // B-spline with only 2 distinct knots but degree=3 → needs expansion.
        // Verifies the expanded knot vector is sorted for correct de Boor evaluation.
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (2.0, 0.0, 0.0));
#4 = CARTESIAN_POINT('', (3.0, 0.0, 0.0));
#10 = B_SPLINE_CURVE_WITH_KNOTS('', 3, (#1, #2, #3, #4), .UNSPECIFIED., .F., .F., (1, 1), (0.0, 1.0), .UNSPECIFIED.);\
",
        );
        // Should not panic and should produce sorted evaluation
        let pts = sample_curve(10, &entities, Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0), 0.1);
        assert!(!pts.is_empty(), "should produce points even with expanded knot vector");
        // Points should be monotonically increasing in x (knots are sorted)
        for w in pts.windows(2) {
            assert!(w[1].x >= w[0].x - 1e-5, "points should be monotonic in x with sorted knots");
        }
    }

    #[test]
    fn test_rational_bspline_curve_weights() {
        // A degree-1 B-spline with control points (0,0,0) and (2,0,0),
        // multiplicities [2,2] → knot vector [0,0,1,1], rational weights [1.0, 0.5].
        // With all weights 1.0, midpoint = (1.0, 0, 0).
        // With weight 0.5 on pt2, the rational midpoint shifts toward pt1.
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (2.0, 0.0, 0.0));
#10 = RATIONAL_B_SPLINE_CURVE('', 1, (#1, #2), .UNSPECIFIED., .F., .F., (2, 2), (0.0, 1.0), .UNSPECIFIED., (1.0, 0.5));\
",
        );
        let pts = sample_curve(10, &entities, Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.1);
        // With rational weights [1.0, 0.5], midpoint should be closer to pt1.
        // At t=0.5: w1*B1 = 1.0*0.5=0.5, w2*B2 = 0.5*0.5=0.25
        // x = (0*0.5 + 2*0.25)/(0.5+0.25) = 0.5/0.75 ≈ 0.667
        if pts.len() >= 2 {
            let midpoint = &pts[pts.len() / 2];
            assert!((midpoint.x - 0.667).abs() < 0.1,
                "rational midpoint x={:.3} should be ≈0.667 (weighted toward pt1), not 1.0", midpoint.x);
        }
    }
}
