//! PCURVE extraction from SURFACE_CURVE entities for trimmed face tessellation.

use super::parser::EntityIndex;
use super::value::StepValue;
use super::topology;
use super::geom::{self, nth_int};

/// 2D UV-domain point.
#[derive(Debug, Clone, Copy)]
pub struct UVPoint {
    pub u: f32,
    pub v: f32,
}

/// Trim polygon for one edge loop in UV space.
#[derive(Debug, Clone, Default)]
pub struct TrimLoop {
    pub points: Vec<UVPoint>,
}

/// Per-face trim data: one TrimLoop per boundary loop.
#[derive(Debug, Clone, Default)]
pub struct FaceTrim {
    pub loops: Vec<TrimLoop>,
}

/// An exact 2D trim curve (kept as geometry, not sampled points).
#[derive(Debug, Clone)]
pub enum ExactTrimCurve2D {
    Line { start: UVPoint, end: UVPoint },
    Circle { center: UVPoint, radius: f32 },
    Ellipse { center: UVPoint, semi_u: f32, semi_v: f32 },
    BSpline { degree: usize, control_points: Vec<UVPoint>, knots: Vec<f32> },
}

impl ExactTrimCurve2D {
    /// Evaluate the curve at parameter t in [0, 1].
    pub fn evaluate(&self, t: f32) -> UVPoint {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Line { start, end } => UVPoint {
                u: start.u + t * (end.u - start.u),
                v: start.v + t * (end.v - start.v),
            },
            Self::Circle { center, radius } => {
                let angle = t * 2.0 * std::f32::consts::PI;
                UVPoint { u: center.u + radius * angle.cos(), v: center.v + radius * angle.sin() }
            }
            Self::Ellipse { center, semi_u, semi_v } => {
                let angle = t * 2.0 * std::f32::consts::PI;
                UVPoint { u: center.u + semi_u * angle.cos(), v: center.v + semi_v * angle.sin() }
            }
            Self::BSpline { degree, control_points, knots } => {
                eval_bspline_2d(t, *degree, control_points, knots)
            }
        }
    }
}

/// An exact trim loop composed of 2D geometric curves.
#[derive(Debug, Clone, Default)]
pub struct ExactFaceTrim {
    pub loops: Vec<ExactTrimLoop>,
}

#[derive(Debug, Clone)]
pub struct ExactTrimLoop {
    pub curves: Vec<ExactTrimCurve2D>,
}

/// Extract trim loops for a face by resolving SURFACE_CURVE → PCURVE chains.
pub fn extract_face_trim(
    face: &topology::StepFace,
    entities: &EntityIndex,
) -> Option<FaceTrim> {
    let surface_id = face.surface_id?;
    let mut trim = FaceTrim::default();
    for bloop in &face.bounds {
        let mut loop_pts = Vec::new();
        for edge in &bloop.edges {
            if let Some(mut uv_pts) = resolve_edge_pcurve(edge.curve_id, surface_id, entities) {
                // If edge is reversed in this loop, reverse the point order
                if edge.reversed {
                    uv_pts.reverse();
                }
                // Skip duplicate connection point (last of prev = first of next)
                if !loop_pts.is_empty() && !uv_pts.is_empty() {
                    let last: &UVPoint = loop_pts.last().unwrap();
                    let first: &UVPoint = uv_pts.first().unwrap();
                    if (last.u - first.u).abs() < 1e-6 && (last.v - first.v).abs() < 1e-6 {
                        uv_pts.remove(0);
                    }
                }
                loop_pts.extend(uv_pts);
            }
        }
        if loop_pts.len() >= 3 {
            // Ensure loop is closed
            let first = loop_pts[0];
            let last = *loop_pts.last().unwrap();
            if (first.u - last.u).abs() > 1e-6 || (first.v - last.v).abs() > 1e-6 {
                loop_pts.push(first);
            }
            trim.loops.push(TrimLoop { points: loop_pts });
        }
    }
    if trim.loops.is_empty() {
        None
    } else {
        Some(trim)
    }
}

/// Resolve an edge curve ID through SURFACE_CURVE → PCURVE to UV points.
/// `surface_id` is the face's surface, used to pick the correct pcurve
/// when multiple pcurves exist (one per adjacent surface).
fn resolve_edge_pcurve(edge_curve_id: u64, surface_id: u64, entities: &EntityIndex) -> Option<Vec<UVPoint>> {
    let record = entities.get(&edge_curve_id)?;
    match record.name.as_str() {
        "EDGE_CURVE" | "SURFACE_CURVE" | "SEAM_CURVE" => {}
        _ => return None,
    }

    // If it's an EDGE_CURVE, it references a curve geometry which may be SURFACE_CURVE
    let sc_record = if record.name == "EDGE_CURVE" {
        let curve_geom_id = geom::nth_ref(&record.params, 3)?;
        let cg_record = entities.get(&curve_geom_id)?;
        if cg_record.name != "SURFACE_CURVE" {
            return None;
        }
        cg_record
    } else {
        // SEAM_CURVE and SURFACE_CURVE have the same param structure:
        // (name, #curve_3d, pcurve_list, master_rep)
        record
    };

    // SURFACE_CURVE: params[2] = pcurve list, each PCURVE has basis_surface at params[1].
    // Select the pcurve whose basis_surface matches our face's surface_id.
    resolve_matching_pcurve(&sc_record.params, 2, surface_id, entities)
}

/// From a param list at `list_index`, find the pcurve whose basis_surface matches `surface_id`.
/// Handles both single ref and list-of-refs.
fn resolve_matching_pcurve(
    params: &StepValue,
    list_index: usize,
    surface_id: u64,
    entities: &EntityIndex,
) -> Option<Vec<UVPoint>> {
    // Try single ref
    if let Some(single_id) = geom::nth_ref(params, list_index) {
        if let Some(pts) = resolve_pcurve_for_surface(single_id, surface_id, entities) {
            return Some(pts);
        }
        return None;
    }
    // Try list of refs — pick the one matching the surface
    if let Some(pcurve_list) = params.nth_param(list_index).and_then(|v| v.as_list()) {
        // First pass: find exact surface match (OCCT behavior)
        for item in pcurve_list {
            if let Some(pid) = item.as_ref_id() {
                if let Some(pts) = resolve_pcurve_for_surface(pid, surface_id, entities) {
                    return Some(pts);
                }
            }
        }
        // Fallback: use first valid pcurve (when no exact match)
        for item in pcurve_list {
            if let Some(pid) = item.as_ref_id() {
                if let Some(pts) = resolve_pcurve(pid, entities) {
                    return Some(pts);
                }
            }
        }
    }
    None
}

/// Resolve a pcurve only if its basis_surface matches the expected surface_id.
fn resolve_pcurve_for_surface(
    pcurve_id: u64,
    surface_id: u64,
    entities: &EntityIndex,
) -> Option<Vec<UVPoint>> {
    let record = entities.get(&pcurve_id)?;
    if record.name != "PCURVE" && record.name != "DEFINITIONAL_REPRESENTATION" {
        return None;
    }
    // PCURVE: (name, #basis_surface, #reference_to_curve)
    let basis_surface = geom::nth_ref(&record.params, 1)?;
    if basis_surface == surface_id {
        resolve_pcurve(pcurve_id, entities)
    } else {
        None
    }
}

/// Resolve a PCURVE entity to UV points.
/// PCURVE: (name, #basis_surface, #reference_to_curve)
fn resolve_pcurve(pcurve_id: u64, entities: &EntityIndex) -> Option<Vec<UVPoint>> {
    let record = entities.get(&pcurve_id)?;
    if record.name != "PCURVE" && record.name != "DEFINITIONAL_REPRESENTATION" {
        return None;
    }

    // PCURVE: reference_to_curve is at params[2]
    let curve_ref = geom::nth_ref(&record.params, 2)?;
    resolve_2d_curve(curve_ref, entities)
}

/// Resolve a 2D curve (LINE, CIRCLE, ELLIPSE, B_SPLINE_CURVE in UV space) to UV points.
fn resolve_2d_curve(curve_id: u64, entities: &EntityIndex) -> Option<Vec<UVPoint>> {
    let record = entities.get(&curve_id)?;
    match record.name.as_str() {
        "DEFINITIONAL_REPRESENTATION" => {
            // DEFINITIONAL_REPRESENTATION('', (#item1, ...), #context)
            // Unwrap and recurse into the first item
            let items = record.params.nth_param(1)?.as_list()?;
            if let Some(first_id) = items.first().and_then(|v| v.as_ref_id()) {
                return resolve_2d_curve(first_id, entities);
            }
            None
        }
        "LINE" => {
            // 2D LINE: (name, #pnt, #dir) where pnt and dir are 2D
            let pt = resolve_cartesian_2d(geom::nth_ref(&record.params, 1)?, entities)?;
            let dir = resolve_vector_2d(geom::nth_ref(&record.params, 2)?, entities)?;
            // Sample at multiple points for better trim polygon quality
            let len = (dir.u * dir.u + dir.v * dir.v).sqrt();
            let n = (len / 0.05).max(2.0).min(32.0) as usize;
            let mut pts = Vec::with_capacity(n + 1);
            for i in 0..=n {
                let t = i as f32 / n as f32;
                pts.push(UVPoint {
                    u: pt.u + t * dir.u,
                    v: pt.v + t * dir.v,
                });
            }
            Some(pts)
        }
        "CIRCLE" => {
            // 2D CIRCLE: (name, #position, radius)
            let center = resolve_placement_2d(geom::nth_ref(&record.params, 1)?, entities)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let n = (radius.abs() * std::f32::consts::TAU / 0.05).max(12.0).min(128.0) as usize;
            let mut pts = Vec::with_capacity(n + 1);
            for i in 0..=n {
                let angle = (i as f32) * 2.0 * std::f32::consts::PI / (n as f32);
                pts.push(UVPoint {
                    u: center.u + radius * angle.cos(),
                    v: center.v + radius * angle.sin(),
                });
            }
            Some(pts)
        }
        "ELLIPSE" => {
            // 2D ELLIPSE: (name, #position, semi_axis_1, semi_axis_2)
            let center = resolve_placement_2d(geom::nth_ref(&record.params, 1)?, entities)?;
            let semi1 = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let semi2 = geom::nth_real(&record.params, 3).unwrap_or(1.0) as f32;
            let n = ((semi1.max(semi2)).abs() * std::f32::consts::TAU / 0.05).max(12.0).min(128.0) as usize;
            let mut pts = Vec::with_capacity(n + 1);
            for i in 0..=n {
                let angle = (i as f32) * 2.0 * std::f32::consts::PI / (n as f32);
                pts.push(UVPoint {
                    u: center.u + semi1 * angle.cos(),
                    v: center.v + semi2 * angle.sin(),
                });
            }
            Some(pts)
        }
        "B_SPLINE_CURVE" | "B_SPLINE_CURVE_WITH_KNOTS" | "RATIONAL_B_SPLINE_CURVE" => {
            resolve_bspline_curve(record, entities)
        }
        _ => None,
    }
}

fn resolve_cartesian_2d(pt_id: u64, entities: &EntityIndex) -> Option<UVPoint> {
    let record = entities.get(&pt_id)?;
    if record.name != "CARTESIAN_POINT" { return None; }
    let coords = nth_list_reals(&record.params, 1)?;
    if coords.len() < 2 { return None; }
    Some(UVPoint { u: coords[0] as f32, v: coords[1] as f32 })
}

fn resolve_vector_2d(vec_id: u64, entities: &EntityIndex) -> Option<UVPoint> {
    let record = entities.get(&vec_id)?;
    if record.name != "DIRECTION" && record.name != "VECTOR" { return None; }
    let coords = nth_list_reals(&record.params, 1)?;
    if coords.len() < 2 { return None; }
    Some(UVPoint { u: coords[0] as f32, v: coords[1] as f32 })
}

/// Resolve a B-spline curve in UV space.
fn resolve_bspline_curve(
    record: &super::parser::EntityRecord,
    entities: &EntityIndex,
) -> Option<Vec<UVPoint>> {
    // B_SPLINE_CURVE: (name, degree, control_points_list, curve_form, closed_curve, self_intersect)
    // B_SPLINE_CURVE_WITH_KNOTS: adds (knot_multiplicities, knots, knot_spec)
    let degree = nth_int(&record.params, 1).unwrap_or(2) as usize;
    let cp_list = geom::nth_list_refs(&record.params, 2)?;

    let control_points: Vec<UVPoint> = cp_list
        .iter()
        .filter_map(|&id| resolve_cartesian_2d(id, entities))
        .collect();

    if control_points.len() < degree + 1 {
        return None;
    }

    // Extract or build knot vector
    let knots = if record.name == "B_SPLINE_CURVE_WITH_KNOTS"
        || record.name == "RATIONAL_B_SPLINE_CURVE"
    {
        let mults = nth_list_ints(&record.params, 6)?;
        let knot_vals = nth_list_reals(&record.params, 7)?;
        let mut knots = Vec::new();
        for (&m, &k) in mults.iter().zip(knot_vals.iter()) {
            for _ in 0..m {
                knots.push(k as f32);
            }
        }
        knots
    } else {
        // Uniform knot vector
        let n = control_points.len();
        let mut knots = Vec::with_capacity(n + degree + 1);
        for _ in 0..=degree { knots.push(0.0f32); }
        for i in 1..(n - degree) {
            knots.push(i as f32 / (n - degree) as f32);
        }
        for _ in 0..=degree { knots.push(1.0f32); }
        knots
    };

    // Sample the B-spline curve
    let n_samples = (control_points.len() * 4).max(16).min(256);
    let t_min = knots[degree];
    let t_max = knots[knots.len() - degree - 1];

    let mut pts = Vec::with_capacity(n_samples + 1);
    for i in 0..=n_samples {
        let t = if n_samples > 0 {
            t_min + (t_max - t_min) * (i as f32 / n_samples as f32)
        } else {
            t_min
        };
        pts.push(eval_bspline_2d(t, degree, &control_points, &knots));
    }
    Some(pts)
}

/// Evaluate a 2D B-spline curve at parameter t.
fn eval_bspline_2d(
    t: f32,
    degree: usize,
    control_points: &[UVPoint],
    knots: &[f32],
) -> UVPoint {
    use super::geom::{bspline_bases, find_span};
    let span = find_span(degree, knots, t);
    let bases = bspline_bases(span, degree, t, knots);

    let mut u = 0.0f32;
    let mut v = 0.0f32;
    for &(i, w) in &bases {
        if i < control_points.len() {
            u += w * control_points[i].u;
            v += w * control_points[i].v;
        }
    }
    UVPoint { u, v }
}

fn resolve_placement_2d(place_id: u64, entities: &EntityIndex) -> Option<UVPoint> {
    let record = entities.get(&place_id)?;
    if record.name != "AXIS2_PLACEMENT_2D" && record.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let pt_id = geom::nth_ref(&record.params, 1)?;
    resolve_cartesian_2d(pt_id, entities)
}

/// 2D point-in-polygon test using ray casting algorithm.
/// Returns true if the point is inside or on the boundary.
pub fn point_in_trim_polygon(u: f32, v: f32, loops: &[TrimLoop]) -> bool {
    for (i, trim_loop) in loops.iter().enumerate() {
        if trim_loop.points.len() < 3 {
            continue;
        }
        let inside = point_in_polygon(u, v, &trim_loop.points);
        if i == 0 {
            // Outer loop: must be inside
            if !inside { return false; }
        } else {
            // Inner loop (hole): must be outside
            if inside { return false; }
        }
    }
    true
}

fn point_in_polygon(u: f32, v: f32, polygon: &[UVPoint]) -> bool {
    let n = polygon.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let yi = polygon[i].v;
        let yj = polygon[j].v;
        if (yi > v) != (yj > v) {
            let xi = polygon[i].u;
            let xj = polygon[j].u;
            let intersect = xi + (v - yi) / (yj - yi) * (xj - xi);
            if u < intersect {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Extract exact trim curves for a face (keeps geometry, not point samples).
pub fn extract_exact_face_trim(
    face: &topology::StepFace,
    entities: &EntityIndex,
) -> Option<ExactFaceTrim> {
    let surface_id = face.surface_id?;
    let mut trim = ExactFaceTrim::default();

    for bloop in &face.bounds {
        let mut loop_curves = Vec::new();
        for edge in &bloop.edges {
            if let Some(curve) = resolve_edge_to_exact_2d_curve(
                edge.curve_id, surface_id, entities,
            ) {
                // Handle edge reversal
                let curve = if edge.reversed {
                    reverse_exact_curve(&curve)
                } else {
                    curve
                };
                loop_curves.push(curve);
            }
        }
        if !loop_curves.is_empty() {
            trim.loops.push(ExactTrimLoop { curves: loop_curves });
        }
    }

    if trim.loops.is_empty() { None } else { Some(trim) }
}

fn reverse_exact_curve(curve: &ExactTrimCurve2D) -> ExactTrimCurve2D {
    match curve {
        ExactTrimCurve2D::Line { start, end } => {
            ExactTrimCurve2D::Line { start: *end, end: *start }
        }
        ExactTrimCurve2D::Circle { center, radius } => {
            ExactTrimCurve2D::Circle { center: *center, radius: *radius }
        }
        ExactTrimCurve2D::Ellipse { center, semi_u, semi_v } => {
            ExactTrimCurve2D::Ellipse { center: *center, semi_u: *semi_u, semi_v: *semi_v }
        }
        ExactTrimCurve2D::BSpline { degree, control_points, knots } => {
            let mut rev_cp = control_points.clone();
            rev_cp.reverse();
            let max_knot = knots.last().copied().unwrap_or(1.0);
            let min_knot = knots.first().copied().unwrap_or(0.0);
            let rev_knots: Vec<f32> = knots.iter()
                .rev()
                .map(|&k| max_knot - k + min_knot)
                .collect();
            ExactTrimCurve2D::BSpline {
                degree: *degree,
                control_points: rev_cp,
                knots: rev_knots,
            }
        }
    }
}

fn resolve_edge_to_exact_2d_curve(
    edge_curve_id: u64, surface_id: u64, entities: &EntityIndex,
) -> Option<ExactTrimCurve2D> {
    let edge_rec = entities.get(&edge_curve_id)?;
    let curve_geom_id = if edge_rec.name == "EDGE_CURVE" {
        geom::nth_ref(&edge_rec.params, 3)?
    } else {
        edge_curve_id
    };

    let sc_rec = entities.get(&curve_geom_id)?;
    // SURFACE_CURVE/SEAM_CURVE/INTERSECTION_CURVE: params[2] = pcurve list
    let pcurve_list = sc_rec.params.nth_param(2).and_then(|v| v.as_list())?;
    for item in pcurve_list {
        let pcurve_id = item.as_ref_id()?;
        // Match by basis_surface
        let prec = entities.get(&pcurve_id)?;
        if prec.name == "PCURVE" || prec.name == "DEFINITIONAL_REPRESENTATION" {
            let basis_surface = geom::nth_ref(&prec.params, 1)?;
            if basis_surface == surface_id {
                let curve_ref = geom::nth_ref(&prec.params, 2)?;
                if let Some(curve) = resolve_exact_2d_curve(curve_ref, entities) {
                    return Some(curve);
                }
            }
        }
    }
    None
}

fn resolve_exact_2d_curve(
    curve_id: u64, entities: &EntityIndex,
) -> Option<ExactTrimCurve2D> {
    let record = entities.get(&curve_id)?;
    match record.name.as_str() {
        "LINE" => {
            let start = resolve_cartesian_2d(geom::nth_ref(&record.params, 1)?, entities)?;
            let dir = resolve_vector_2d(geom::nth_ref(&record.params, 2)?, entities)?;
            Some(ExactTrimCurve2D::Line {
                start,
                end: UVPoint { u: start.u + dir.u, v: start.v + dir.v },
            })
        }
        "CIRCLE" => {
            let center = resolve_placement_2d(geom::nth_ref(&record.params, 1)?, entities)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            Some(ExactTrimCurve2D::Circle { center, radius })
        }
        "ELLIPSE" => {
            let center = resolve_placement_2d(geom::nth_ref(&record.params, 1)?, entities)?;
            let semi_u = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let semi_v = geom::nth_real(&record.params, 3).unwrap_or(1.0) as f32;
            Some(ExactTrimCurve2D::Ellipse { center, semi_u, semi_v })
        }
        "B_SPLINE_CURVE" | "B_SPLINE_CURVE_WITH_KNOTS" | "RATIONAL_B_SPLINE_CURVE" => {
            let degree = geom::nth_int(&record.params, 1).unwrap_or(2) as usize;
            let cp_list = geom::nth_list_refs(&record.params, 2)?;
            let control_points: Vec<UVPoint> = cp_list.iter()
                .filter_map(|&id| resolve_cartesian_2d(id, entities))
                .collect();
            if control_points.len() < degree + 1 { return None; }
            let knots = build_knots(degree, control_points.len(), &record, entities);
            Some(ExactTrimCurve2D::BSpline { degree, control_points, knots })
        }
        "DEFINITIONAL_REPRESENTATION" => {
            let items = record.params.nth_param(1)?.as_list()?;
            let first_id = items.first()?.as_ref_id()?;
            resolve_exact_2d_curve(first_id, entities)
        }
        _ => None,
    }
}

fn build_knots(
    degree: usize, cp_count: usize,
    record: &super::parser::EntityRecord,
    _entities: &EntityIndex,
) -> Vec<f32> {
    let mults = geom::nth_list_ints(&record.params, 6);
    let knot_vals: Vec<f32> = geom::nth_list_reals(&record.params, 7)
        .iter().filter_map(|v| v.as_real()).map(|r| r as f32).collect();
    let mut knots = Vec::new();
    if !mults.is_empty() && !knot_vals.is_empty() {
        for (i, &m) in mults.iter().enumerate() {
            let k = knot_vals.get(i).copied().unwrap_or(0.0);
            for _ in 0..m.max(1) { knots.push(k); }
        }
    } else {
        for _ in 0..=degree { knots.push(0.0); }
        for i in 1..(cp_count - degree) {
            knots.push(i as f32 / (cp_count - degree) as f32);
        }
        for _ in 0..=degree { knots.push(1.0); }
    }
    knots
}

// ── Helpers ─────────────────────────────────────────────────

fn nth_list_reals(params: &StepValue, index: usize) -> Option<Vec<f64>> {
    params.nth_param(index)?.as_list()
        .map(|list| list.iter().filter_map(|v| v.as_real()).collect())
}

fn nth_list_ints(params: &StepValue, index: usize) -> Option<Vec<i64>> {
    params.nth_param(index)?.as_list()
        .map(|list| list.iter().filter_map(|v| v.as_int()).collect())
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_inside_rectangle() {
        let rect = TrimLoop { points: vec![
            UVPoint { u: 0.0, v: 0.0 },
            UVPoint { u: 0.0, v: 1.0 },
            UVPoint { u: 1.0, v: 1.0 },
            UVPoint { u: 1.0, v: 0.0 },
        ]};
        let loops = vec![rect];
        assert!(point_in_trim_polygon(0.5, 0.5, &loops));
    }

    #[test]
    fn test_point_outside_rectangle() {
        let rect = TrimLoop { points: vec![
            UVPoint { u: 0.0, v: 0.0 },
            UVPoint { u: 0.0, v: 1.0 },
            UVPoint { u: 1.0, v: 1.0 },
            UVPoint { u: 1.0, v: 0.0 },
        ]};
        let loops = vec![rect];
        assert!(!point_in_trim_polygon(1.5, 0.5, &loops));
    }

    #[test]
    fn test_point_inside_with_hole() {
        let outer = TrimLoop { points: vec![
            UVPoint { u: 0.0, v: 0.0 },
            UVPoint { u: 0.0, v: 4.0 },
            UVPoint { u: 4.0, v: 4.0 },
            UVPoint { u: 4.0, v: 0.0 },
        ]};
        let hole = TrimLoop { points: vec![
            UVPoint { u: 1.0, v: 1.0 },
            UVPoint { u: 1.0, v: 2.0 },
            UVPoint { u: 2.0, v: 2.0 },
            UVPoint { u: 2.0, v: 1.0 },
        ]};
        let loops = vec![outer, hole];
        assert!(point_in_trim_polygon(0.5, 2.0, &loops));  // outside hole, inside outer
        assert!(!point_in_trim_polygon(1.5, 1.5, &loops));  // inside hole
        assert!(!point_in_trim_polygon(5.0, 2.0, &loops));  // outside outer
    }

    #[test]
    fn test_empty_trim_accepted() {
        let loops: Vec<TrimLoop> = vec![];
        // Empty trim data → point accepted (no constraint)
        assert!(point_in_trim_polygon(0.5, 0.5, &loops));
    }

    #[test]
    fn test_exact_line_trim_evaluate() {
        let curve = ExactTrimCurve2D::Line {
            start: UVPoint { u: 0.0, v: 0.0 },
            end: UVPoint { u: 10.0, v: 0.0 },
        };
        let mid = curve.evaluate(0.5);
        assert!((mid.u - 5.0).abs() < 1e-6);
        assert_eq!(mid.v, 0.0);
    }

    #[test]
    fn test_reverse_line_curve() {
        let curve = ExactTrimCurve2D::Line {
            start: UVPoint { u: 0.0, v: 0.0 },
            end: UVPoint { u: 10.0, v: 0.0 },
        };
        let rev = reverse_exact_curve(&curve);
        let pt = rev.evaluate(0.0);
        assert!((pt.u - 10.0).abs() < 1e-6);
    }
}