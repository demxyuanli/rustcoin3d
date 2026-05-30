use super::*;
pub fn build_surface(surface_id: u64, entities: &EntityIndex) -> Option<SurfaceGeom> {
    let record = entities.get(&surface_id)?;
    match record.name.as_str() {
        "PLANE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let (origin, x_axis, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(SurfaceGeom::Plane {
                origin,
                normal: z_axis.normalize(),
                u_dir: x_axis,
            })
        }
        "CYLINDRICAL_SURFACE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let (origin, _, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(SurfaceGeom::Cylinder {
                origin,
                axis: z_axis.normalize(),
                radius,
            })
        }
        "CONICAL_SURFACE" => {
            // CONICAL_SURFACE('', #placement, radius, semi_angle)
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(0.0) as f32;
            let semi_angle = geom::nth_real(&record.params, 3).unwrap_or(0.7854) as f32;
            let (origin, _, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(SurfaceGeom::Cone {
                apex: origin,
                axis: z_axis.normalize(),
                semi_angle,
                radius_at_apex: radius,
            })
        }
        "SPHERICAL_SURFACE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let (origin, _, _) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(SurfaceGeom::Sphere { center: origin, radius })
        }
        "TOROIDAL_SURFACE" => {
            // TOROIDAL_SURFACE('', #placement, major_r, minor_r)
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let major_r = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            let minor_r = geom::nth_real(&record.params, 3).unwrap_or(0.5) as f32;
            let (origin, _, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((Vec3::ZERO, Vec3::X, Vec3::Z));
            Some(SurfaceGeom::Torus {
                center: origin,
                axis: z_axis.normalize(),
                major_r,
                minor_r,
            })
        }
        "B_SPLINE_SURFACE" | "B_SPLINE_SURFACE_WITH_KNOTS" | "RATIONAL_B_SPLINE_SURFACE" => {
            let nurbs = build_nurbs_surface(record, entities)?;
            Some(SurfaceGeom::BSpline(nurbs))
        }
        "SURFACE_OF_LINEAR_EXTRUSION" => {
            // SURFACE_OF_LINEAR_EXTRUSION('', #curve, #direction)
            let curve_id = geom::nth_ref(&record.params, 1)?;
            let dir_id = geom::nth_ref(&record.params, 2)?;
            let generatrix = build_curve(curve_id, entities)?;
            let direction = topology::resolve_direction(dir_id, entities).unwrap_or(Vec3::Z);
            Some(SurfaceGeom::Extrusion {
                generatrix: Box::new(generatrix),
                direction,
            })
        }
        "SURFACE_OF_REVOLUTION" => {
            // SURFACE_OF_REVOLUTION('', #curve, #axis_placement)
            let curve_id = geom::nth_ref(&record.params, 1)?;
            let axis_id = geom::nth_ref(&record.params, 2);
            let generatrix = build_curve(curve_id, entities)?;
            let (axis_origin, axis_dir) = axis_id
                .and_then(|id| topology::resolve_sweep_axis(id, entities))
                .unwrap_or((Vec3::ZERO, Vec3::Z));
            Some(SurfaceGeom::Revolution {
                generatrix: Box::new(generatrix),
                axis_origin,
                axis_dir,
            })
        }
        "OFFSET_SURFACE" => {
            let basis_id = geom::nth_ref(&record.params, 1)?;
            let distance = geom::nth_real(&record.params, 2).unwrap_or(0.0) as f32;
            let basis = build_surface(basis_id, entities)?;
            Some(SurfaceGeom::Offset { basis: Box::new(basis), distance })
        }
        "BOUNDED_SURFACE" | "CURVE_BOUNDED_SURFACE" => {
            // Unwrap: BOUNDED_SURFACE('', #basis_surface, ...)
            let basis_id = geom::nth_ref(&record.params, 1)?;
            build_surface(basis_id, entities)
        }
        "RECTANGULAR_TRIMMED_SURFACE" => {
            let basis_id = geom::nth_ref(&record.params, 1)?;
            build_surface(basis_id, entities)
        }
        // AP242 supertypes — unwrap to underlying surface when referenced directly
        "ELEMENTARY_SURFACE" | "SWEPT_SURFACE" => {
            if let Some(basis_id) = geom::nth_ref(&record.params, 1) {
                build_surface(basis_id, entities)
            } else {
                log::warn!(
                    "[BRep] build_surface: supertype '{}' for #{} has no basis surface",
                    record.name, surface_id
                );
                None
            }
        }
        _ => None,
    }
}

/// Build a NurbsSurface from a B_SPLINE_SURFACE* entity.
fn build_nurbs_surface(
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

    // Extract knot vectors from multiplicities
    let mult_base = off + 7;
    let u_mults = geom::nth_list_ints(params, mult_base);
    let v_mults = geom::nth_list_ints(params, mult_base + 1);
    let u_knot_vals = geom::nth_list_reals(params, mult_base + 2);
    let v_knot_vals = geom::nth_list_reals(params, mult_base + 3);

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
fn build_surface_knots(
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
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    knots
}

/// Scan params for a 2D weights list matching NURBS control point dimensions.
fn find_surface_weights(params: &StepValue, rows: usize, cols: usize) -> Option<Vec<Vec<f32>>> {
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

/// Resolve a VECTOR entity to its full 3D vector (direction × magnitude).
pub(crate) fn resolve_vector_magnitude(vec_id: u64, entities: &EntityIndex) -> Option<Vec3> {
    let record = entities.get(&vec_id)?;
    match record.name.as_str() {
        "VECTOR" => {
            let dir_id = geom::nth_ref(&record.params, 1)?;
            let dir = topology::resolve_direction(dir_id, entities)?;
            let mag = geom::nth_real(&record.params, 2).unwrap_or(1.0) as f32;
            Some(dir * mag)
        }
        "DIRECTION" => {
            topology::resolve_direction(vec_id, entities)
        }
        _ => topology::resolve_direction(vec_id, entities),
    }
}

