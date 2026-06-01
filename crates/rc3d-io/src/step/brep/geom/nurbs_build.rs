//! Shared NURBS surface construction from STEP B_SPLINE_SURFACE entities.
//!
//! Extracted from `brep/build/surface.rs` to be reusable by other modules
//! (e.g. legacy `surface_tess.rs` before its removal).

use crate::step::parser::{EntityIndex, EntityRecord};
use crate::step::value::StepValue;
use crate::step::{geom, topology};
use rc3d_shape::nurbs::NurbsSurface;

/// Build a NurbsSurface from a B_SPLINE_SURFACE* entity.
pub fn build_nurbs_surface(
    record: &EntityRecord,
    entities: &EntityIndex,
) -> Option<NurbsSurface> {
    let params = &record.params;

    // Off=0 when params[0] is the degree (integer). Off=1 when params[0]
    // is an omitted value (from subsuper merge) and degree starts at [1].
    // Param count alone is unreliable — extra params from trailing supertypes
    // (e.g. REPRESENTATION_ITEM, RATIONAL weights) can make count > 12.
    let off: usize = if geom::nth_int(params, 0).is_some() { 0 } else { 1 };

    let degree_u = geom::nth_int(params, off)? as usize;
    let degree_v = geom::nth_int(params, off + 1)? as usize;

    let cp_list = params.nth_param(off + 2)?.as_list()?;
    let mut control_points = Vec::with_capacity(cp_list.len());
    for row_val in cp_list {
        let row_refs = row_val.as_list()?;
        let mut row_pts = Vec::with_capacity(row_refs.len());
        for pt_val in row_refs {
            let pt_id = pt_val.as_ref_id()?;
            let pt = topology::resolve_point(pt_id, entities)?;
            row_pts.push(pt);
        }
        control_points.push(row_pts);
    }

    if control_points.is_empty() || control_points[0].is_empty() {
        return None;
    }

    // Extract knot vectors from multiplicities (hardcoded positions for
    // standalone WITH_KNOTS format). Falls back to type-based scanning for
    // subsuper-merged params where the layout differs.
    let mult_base = off + 7;
    let u_mults = geom::nth_list_ints(params, mult_base);
    let v_mults = geom::nth_list_ints(params, mult_base + 1);
    let u_knot_vals = geom::nth_list_reals(params, mult_base + 2);
    let v_knot_vals = geom::nth_list_reals(params, mult_base + 3);

    let (u_mults, v_mults, u_knot_vals, v_knot_vals) =
        if u_mults.is_empty() || v_mults.is_empty() {
            if let Some((um, vm, uk, vk)) = scan_knot_data(params) {
                (um, vm, uk, vk)
            } else {
                (u_mults, v_mults, u_knot_vals, v_knot_vals)
            }
        } else {
            (u_mults, v_mults, u_knot_vals, v_knot_vals)
        };

    let knots_u = build_surface_knots(
        &u_mults, &u_knot_vals, degree_u, control_points.len(),
    );
    let knots_v = build_surface_knots(
        &v_mults, &v_knot_vals, degree_v, control_points[0].len(),
    );

    // Extract rational weights if present
    let weights = find_surface_weights(params, control_points.len(), control_points[0].len())
        .unwrap_or_else(|| vec![vec![1.0f32; control_points[0].len()]; control_points.len()]);

    Some(NurbsSurface {
        degree_u,
        degree_v,
        control_points,
        weights,
        knots_u,
        knots_v,
    })
}

/// Build a full knot vector from multiplicities and distinct knot values.
pub fn build_surface_knots(
    multiplicities: &[i64],
    knot_values: &[StepValue],
    degree: usize,
    cp_count: usize,
) -> Vec<f32> {
    let mut knots = Vec::new();
    if !multiplicities.is_empty() && !knot_values.is_empty() {
        for (i, &mult) in multiplicities.iter().enumerate() {
            if i < knot_values.len() {
                let k = knot_values[i].as_real().unwrap_or(0.0) as f32;
                for _ in 0..mult.max(1) {
                    knots.push(k);
                }
            }
        }
    }
    let needed = cp_count + degree + 1;
    if knots.len() < needed {
        let min_k = *knots.first().unwrap_or(&0.0);
        let max_k = *knots.last().unwrap_or(&1.0);
        let extra = needed - knots.len();
        for i in 0..extra {
            let t = (i + 1) as f32 / (extra + 1) as f32;
            knots.push(min_k + (max_k - min_k) * t);
        }
    }
    // Ensure knot vector is monotonically non-decreasing (STEP files may provide
    // knot values in descending order for some axis, or multiplicities/knot_values
    // may not align perfectly after param merge).
    knots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    knots
}

/// Scan merged params for knot multiplicity/value lists by type inspection.
/// Used when hardcoded mult_base positions fail (CompatMerge layout differs
/// from standalone B_SPLINE_SURFACE_WITH_KNOTS format).
fn scan_knot_data(
    params: &StepValue,
) -> Option<(Vec<i64>, Vec<i64>, Vec<StepValue>, Vec<StepValue>)> {
    let list = params.as_list()?;
    let mut past_cps = false;
    let mut int_lists: Vec<&[StepValue]> = Vec::new();
    let mut real_lists: Vec<&[StepValue]> = Vec::new();

    for val in list {
        match val {
            StepValue::List(inner) if !inner.is_empty() => {
                // Detect the CP list: List of Lists of Refs
                if inner.iter().any(|v| matches!(v, StepValue::List(_))) {
                    past_cps = true;
                    continue;
                }
                if !past_cps {
                    continue;
                }
                if inner.iter().all(|v| matches!(v, StepValue::Integer(_))) {
                    int_lists.push(inner.as_slice());
                } else if inner.iter().all(|v| matches!(v, StepValue::Real(_))) {
                    real_lists.push(inner.as_slice());
                }
            }
            _ => {}
        }
    }

    if int_lists.len() < 2 || real_lists.len() < 2 {
        return None;
    }

    Some((
        int_lists[0].iter().filter_map(|v| {
            if let StepValue::Integer(n) = v { Some(*n) } else { None }
        }).collect(),
        int_lists[1].iter().filter_map(|v| {
            if let StepValue::Integer(n) = v { Some(*n) } else { None }
        }).collect(),
        real_lists[0].to_vec(),
        real_lists[1].to_vec(),
    ))
}

/// Scan params for a 2D weights list matching NURBS control point dimensions.
pub fn find_surface_weights(params: &StepValue, rows: usize, cols: usize) -> Option<Vec<Vec<f32>>> {
    let list = params.as_list()?;
    for val in list.iter().rev() {
        if let StepValue::List(inner) = val {
            if inner.len() == rows
                && inner.iter().all(|v| {
                    v.as_list().map_or(false, |l| {
                        l.len() == cols && l.iter().all(|r| matches!(r, StepValue::Real(_)))
                    })
                })
            {
                return Some(
                    inner.iter()
                        .map(|v| v.as_list().unwrap().iter()
                            .map(|r| r.as_real().unwrap() as f32)
                            .collect())
                        .collect(),
                );
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_knot_data_from_merged_params() {
        // Simulate CompatMerge output for B_SPLINE_SURFACE + B_SPLINE_SURFACE_WITH_KNOTS
        let params = StepValue::List(vec![
            StepValue::Integer(2),  // degree_u
            StepValue::Integer(3),  // degree_v
            StepValue::List(vec![  // control_points
                StepValue::List(vec![StepValue::Ref(10), StepValue::Ref(11)]),
                StepValue::List(vec![StepValue::Ref(12), StepValue::Ref(13)]),
            ]),
            StepValue::Enum(".UNSPECIFIED.".to_string()),
            StepValue::Enum(".F.".to_string()),
            StepValue::Enum(".F.".to_string()),
            StepValue::Enum(".F.".to_string()),
            StepValue::List(vec![StepValue::Integer(2), StepValue::Integer(2)]), // u_mults
            StepValue::List(vec![StepValue::Integer(2), StepValue::Integer(2)]), // v_mults
            StepValue::List(vec![StepValue::Real(0.0), StepValue::Real(1.0)]),   // u_knots
            StepValue::List(vec![StepValue::Real(0.0), StepValue::Real(1.0)]),   // v_knots
        ]);
        let result = scan_knot_data(&params);
        assert!(result.is_some(), "should find knot data in merged params");
        let (um, vm, _uk, _vk) = result.unwrap();
        assert_eq!(um, vec![2, 2]);
        assert_eq!(vm, vec![2, 2]);
    }

    #[test]
    fn test_scan_knot_data_empty_params() {
        let params = StepValue::List(vec![
            StepValue::Integer(2),
            StepValue::Integer(3),
        ]);
        assert!(scan_knot_data(&params).is_none());
    }
}
