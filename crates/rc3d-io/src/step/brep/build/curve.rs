use rc3d_core::math::Real;
use super::*;
use super::surface::resolve_vector_magnitude;

// ── Curve building ────────────────────────────────────────────────

/// Build a 3D CurveGeom from a STEP curve entity.
pub fn build_curve(curve_id: u64, entities: &EntityIndex) -> Option<CurveGeom> {
    let record = entities.get(&curve_id)?;
    match record.name.as_str() {
        "LINE" => {
            let pnt_id = geom::nth_ref(&record.params, 1)?;
            let dir_id = geom::nth_ref(&record.params, 2)?;
            let origin = topology::resolve_point(pnt_id, entities)?;
            // Resolve the full VECTOR (direction × magnitude), not just the unit direction.
            let direction = resolve_vector_magnitude(dir_id, entities).unwrap_or(PVec3::X);
            Some(CurveGeom::Line { origin, direction })
        }
        "CIRCLE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as Real;
            let (center, _, axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((PVec3::ZERO, PVec3::X, PVec3::Z));
            Some(CurveGeom::circle(center, axis, radius))
        }
        "ELLIPSE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let semi_major = geom::nth_real(&record.params, 2).unwrap_or(1.0) as Real;
            let semi_minor = geom::nth_real(&record.params, 3).unwrap_or(0.5) as Real;
            let (center, _, axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((PVec3::ZERO, PVec3::X, PVec3::Z));
            Some(CurveGeom::ellipse(center, axis, semi_major, semi_minor))
        }
        "HYPERBOLA" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let semi_major = geom::nth_real(&record.params, 2).unwrap_or(1.0) as Real;
            let semi_minor = geom::nth_real(&record.params, 3).unwrap_or(1.0) as Real;
            let (center, _, axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((PVec3::ZERO, PVec3::X, PVec3::Z));
            Some(CurveGeom::hyperbola(center, axis, semi_major, semi_minor))
        }
        "PARABOLA" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let focal_dist = geom::nth_real(&record.params, 2).unwrap_or(1.0) as Real;
            let (center, _, axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((PVec3::ZERO, PVec3::X, PVec3::Z));
            Some(CurveGeom::parabola(center, axis, focal_dist))
        }
        "B_SPLINE_CURVE" | "B_SPLINE_CURVE_WITH_KNOTS" | "RATIONAL_B_SPLINE_CURVE" => {
            build_bspline_3d(record, entities)
        }
        "TRIMMED_CURVE" => {
            // TRIMMED_CURVE('', #basis_curve, (trim1), (trim2), sense, master_rep)
            let basis_id = geom::nth_ref(&record.params, 1)?;
            let basis = build_curve(basis_id, entities)?;
            let t_min = parse_trim_bound(&record.params, 2).unwrap_or(0.0);
            let t_max = parse_trim_bound(&record.params, 3).unwrap_or(1.0);
            Some(CurveGeom::Trimmed {
                basis: Box::new(basis),
                t_min: t_min.min(t_max),
                t_max: t_min.max(t_max),
            })
        }
        "POLYLINE" => {
            let pt_ids = geom::nth_list_refs(&record.params, 1).unwrap_or_default();
            let points: Vec<PVec3> = pt_ids.iter()
                .filter_map(|&id| topology::resolve_point(id, entities))
                .collect();
            if points.len() < 2 { None } else { Some(CurveGeom::Polyline { points }) }
        }
        "COMPOSITE_CURVE" => {
            let seg_ids = geom::nth_list_refs(&record.params, 1).unwrap_or_default();
            let segments: Vec<(CurveGeom, bool)> = seg_ids.iter()
                .filter_map(|&seg_id| {
                    let seg_rec = entities.get(&seg_id)?;
                    // COMPOSITE_CURVE_SEGMENT: (name, transition, same_sense, #parent_curve)
                    let parent_id = geom::nth_ref(&seg_rec.params, 3)?;
                    // same_sense is at params[2]; TRANSITION code at params[1], parents at [3]
                    let same_sense = match seg_rec.params.nth_param(2) {
                        Some(StepValue::Enum(s)) => s == ".T.",
                        _ => true,
                    };
                    build_curve(parent_id, entities).map(|c| (c, same_sense))
                })
                .collect();
            if segments.is_empty() {
                None
            } else {
                // Precompute segment lengths once at construction to avoid
                // O(N×32) recomputation on every parameter evaluation.
                let cached_lengths: Vec<Real> = segments.iter()
                    .map(|(seg, _)| approx_chordal_length(seg).max(1e-10))
                    .collect();
                Some(CurveGeom::Composite { segments, cached_lengths: Some(cached_lengths) })
            }
        }
        // SURFACE_CURVE/SEAM_CURVE: unwrap to the 3D curve
        "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE" => {
            let curve_3d_id = geom::nth_ref(&record.params, 1)?;
            build_curve(curve_3d_id, entities)
        }
        "OFFSET_CURVE_3D" => {
            let basis_id = geom::nth_ref(&record.params, 1)?;
            let basis = build_curve(basis_id, entities)?;
            let dir_id = geom::nth_ref(&record.params, 2)?;
            let offset_dir = topology::resolve_direction(dir_id, entities)?;
            let distance = geom::nth_real(&record.params, 3).unwrap_or(0.0) as Real;
            Some(CurveGeom::Offset {
                basis: Box::new(basis),
                offset_dir,
                distance,
            })
        }
        "BOUNDED_CURVE" => {
            // Unwrap to the underlying curve
            let inner_id = geom::nth_ref(&record.params, 1)?;
            build_curve(inner_id, entities)
        }
        _ => None,
    }
}

/// Build a 3D B-spline CurveGeom from a B_SPLINE_CURVE* entity.
fn build_bspline_3d(
    record: &EntityRecord,
    entities: &EntityIndex,
) -> Option<CurveGeom> {
    // Subsuper entities (merged BOUNDED_CURVE+B_SPLINE_CURVE+...) have no entity name
    // as params[0], so all indices are offset by 1 vs normal entities.
    let off: usize = if geom::nth_int(&record.params, 0).is_some() { 0 } else { 1 };
    let degree = geom::nth_int(&record.params, off).unwrap_or(2) as usize;
    let cp_ids = geom::nth_list_refs(&record.params, off + 1)?;

    let control_points: Vec<PVec3> = cp_ids.iter()
        .filter_map(|&id| topology::resolve_point(id, entities))
        .collect();

    if control_points.len() < degree + 1 {
        return None;
    }

    let cp_count = control_points.len();

    // Extract or build knot vector
    let knots = if record.name == "B_SPLINE_CURVE_WITH_KNOTS"
        || record.name == "RATIONAL_B_SPLINE_CURVE"
    {
        // Normal layout (after name): [off+0]=degree, [off+1]=ctrl_pts,
        // [off+5]=knot_multiplicities, [off+6]=knots
        // RATIONAL_B_SPLINE_CURVE appends weights after knot_spec
        let mults = geom::nth_list_ints(&record.params, off + 5);
        let knot_vals = geom::nth_list_reals(&record.params, off + 6);
        let mut knots = Vec::new();
        if !mults.is_empty() && !knot_vals.is_empty() {
            for (i, &m) in mults.iter().enumerate() {
                let k = knot_vals.get(i)
                    .and_then(|v| v.as_real())
                    .unwrap_or(0.0) as Real;
                for _ in 0..m.max(1) {
                    knots.push(k);
                }
            }
        }
        knots
    } else {
        Vec::new()
    };

    let knots = if knots.len() > cp_count + degree {
        knots
    } else {
        // Build uniform knot vector
        let mut k = Vec::with_capacity(cp_count + degree + 1);
        for _ in 0..=degree { k.push(0.0_f64); }
        for i in 1..(cp_count - degree) {
            k.push(i as Real / (cp_count - degree) as Real);
        }
        for _ in 0..=degree { k.push(1.0_f64); }
        k
    };

    // Extract rational weights if present
    let weights = if record.name == "RATIONAL_B_SPLINE_CURVE" {
        // Weights are the last param in RATIONAL_B_SPLINE_CURVE
        let list = record.params.as_list()?;
        let weight_data = list.last()?;
        // Try treating it as a list of reals
        find_curve_weights(weight_data, cp_count)
    } else {
        None
    };

    Some(CurveGeom::BSpline {
        degree,
        control_points,
        knots,
        weights,
    })
}

/// Build a 2D B-spline PCurve in native surface UV space.
pub(crate) fn build_bspline_2d(
    record: &EntityRecord,
    entities: &EntityIndex,
) -> Option<CurveGeom> {
    let off: usize = if geom::nth_int(&record.params, 0).is_some() { 0 } else { 1 };
    let degree = geom::nth_int(&record.params, off).unwrap_or(2) as usize;
    let cp_ids = geom::nth_list_refs(&record.params, off + 1)?;

    let control_points: Vec<PVec3> = cp_ids
        .iter()
        .filter_map(|&id| {
            resolve_cartesian_2d(id, entities).map(|(u, v)| PVec3::new(u, v, 0.0))
        })
        .collect();

    if control_points.len() < degree + 1 {
        return None;
    }

    let cp_count = control_points.len();

    let knots = if record.name == "B_SPLINE_CURVE_WITH_KNOTS"
        || record.name == "RATIONAL_B_SPLINE_CURVE"
    {
        let mults = geom::nth_list_ints(&record.params, off + 5);
        let knot_vals = geom::nth_list_reals(&record.params, off + 6);
        let mut knots = Vec::new();
        if !mults.is_empty() && !knot_vals.is_empty() {
            for (i, &m) in mults.iter().enumerate() {
                let k = knot_vals
                    .get(i)
                    .and_then(|v| v.as_real())
                    .unwrap_or(0.0) as Real;
                for _ in 0..m.max(1) {
                    knots.push(k);
                }
            }
        }
        knots
    } else {
        Vec::new()
    };

    let knots = if knots.len() > cp_count + degree {
        knots
    } else {
        let mut k = Vec::with_capacity(cp_count + degree + 1);
        for _ in 0..=degree {
            k.push(0.0_f64);
        }
        for i in 1..(cp_count - degree) {
            k.push(i as Real / (cp_count - degree) as Real);
        }
        for _ in 0..=degree {
            k.push(1.0_f64);
        }
        k
    };

    let weights = if record.name == "RATIONAL_B_SPLINE_CURVE" {
        let list = record.params.as_list()?;
        let weight_data = list.last()?;
        find_curve_weights(weight_data, cp_count)
    } else {
        None
    };

    Some(CurveGeom::BSpline {
        degree,
        control_points,
        knots,
        weights,
    })
}

pub(crate) fn parse_trim_bound(params: &StepValue, index: usize) -> Option<Real> {
    params
        .nth_param(index)?
        .as_list()?
        .first()?
        .as_real()
        .map(|v| v as Real)
}

pub(crate) fn resolve_cartesian_2d(pt_id: u64, entities: &EntityIndex) -> Option<(Real, Real)> {
    let record = entities.get(&pt_id)?;
    if record.name != "CARTESIAN_POINT" {
        return None;
    }
    let coords = geom::nth_list_reals(&record.params, 1);
    if coords.len() < 2 {
        return None;
    }
    Some((
        coords[0].as_real()? as Real,
        coords[1].as_real()? as Real,
    ))
}

/// Extract weights from a RATIONAL_B_SPLINE_CURVE weight param.
fn find_curve_weights(weight_val: &StepValue, expected_count: usize) -> Option<Vec<Real>> {
    if let StepValue::List(items) = weight_val {
        let weights: Vec<Real> = items.iter()
            .filter_map(|v| v.as_real())
            .map(|r| r as Real)
            .collect();
        if weights.len() == expected_count {
            return Some(weights);
        }
    }
    // Try Typed wrapping
    if let StepValue::Typed(_, inner) = weight_val {
        return find_curve_weights(inner, expected_count);
    }
    None
}

