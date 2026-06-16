use rc3d_core::math::Real;
use super::*;
use crate::step::brep::geom::nurbs_build::build_nurbs_surface;

/// Extract the rectangular trim range if this is a RECTANGULAR_TRIMMED_SURFACE.
/// Returns (u_min, u_max, v_min, v_max) or None.
pub fn build_surface_trim_range(surface_id: u64, entities: &EntityIndex) -> Option<(Real, Real, Real, Real)> {
    let record = entities.get(&surface_id)?;
    if record.name == "RECTANGULAR_TRIMMED_SURFACE" {
        let u1 = geom::nth_real(&record.params, 2).unwrap_or(0.0) as Real;
        let u2 = geom::nth_real(&record.params, 3).unwrap_or(1.0) as Real;
        let v1 = geom::nth_real(&record.params, 4).unwrap_or(0.0) as Real;
        let v2 = geom::nth_real(&record.params, 5).unwrap_or(1.0) as Real;
        Some((u1.min(u2), u1.max(u2), v1.min(v2), v1.max(v2)))
    } else {
        None
    }
}

pub fn build_surface(surface_id: u64, entities: &EntityIndex) -> Option<SurfaceGeom> {
    let record = entities.get(&surface_id)?;
    match record.name.as_str() {
        "PLANE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let (origin, x_axis, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((PVec3::ZERO, PVec3::X, PVec3::Z));
            Some(SurfaceGeom::Plane {
                origin,
                normal: z_axis.normalize(),
                u_dir: x_axis,
            })
        }
        "CYLINDRICAL_SURFACE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as Real;
            let (origin, _, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((PVec3::ZERO, PVec3::X, PVec3::Z));
            Some(SurfaceGeom::cylinder(origin, z_axis.normalize(), radius))
        }
        "CONICAL_SURFACE" => {
            // CONICAL_SURFACE('', #placement, radius, semi_angle)
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(0.0) as Real;
            let semi_angle = geom::nth_real(&record.params, 3).unwrap_or(0.7854) as Real;
            let (origin, _, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((PVec3::ZERO, PVec3::X, PVec3::Z));
            Some(SurfaceGeom::cone(origin, z_axis.normalize(), semi_angle, radius))
        }
        "SPHERICAL_SURFACE" => {
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let radius = geom::nth_real(&record.params, 2).unwrap_or(1.0) as Real;
            let (origin, _, _) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((PVec3::ZERO, PVec3::X, PVec3::Z));
            Some(SurfaceGeom::Sphere { center: origin, radius })
        }
        "TOROIDAL_SURFACE" => {
            // TOROIDAL_SURFACE('', #placement, major_r, minor_r)
            let placement_id = geom::nth_ref(&record.params, 1)?;
            let major_r = geom::nth_real(&record.params, 2).unwrap_or(1.0) as Real;
            let minor_r = geom::nth_real(&record.params, 3).unwrap_or(0.5) as Real;
            let (origin, _, z_axis) = topology::resolve_placement(placement_id, entities)
                .unwrap_or((PVec3::ZERO, PVec3::X, PVec3::Z));
            Some(SurfaceGeom::torus(origin, z_axis.normalize(), major_r, minor_r))
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
            let direction = topology::resolve_direction(dir_id, entities).unwrap_or(PVec3::Z);
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
                .unwrap_or((PVec3::ZERO, PVec3::Z));
            Some(SurfaceGeom::Revolution {
                generatrix: Box::new(generatrix),
                axis_origin,
                axis_dir,
            })
        }
        "OFFSET_SURFACE" => {
            let basis_id = geom::nth_ref(&record.params, 1)?;
            let distance = geom::nth_real(&record.params, 2).unwrap_or(0.0) as Real;
            let basis = build_surface(basis_id, entities)?;
            Some(SurfaceGeom::Offset { basis: Box::new(basis), distance })
        }
        "BOUNDED_SURFACE" | "CURVE_BOUNDED_SURFACE" => {
            let basis_id = geom::nth_ref(&record.params, 1)?;
            // CURVE_BOUNDED_SURFACE has boundary curves in params[2]
            if record.name == "CURVE_BOUNDED_SURFACE" {
                if let Some(boundary_ids) = geom::nth_list_refs(&record.params, 2) {
                    log::debug!(
                        "[BRep] {:?}: surface #{} has {} boundary curve(s)",
                        surface_id, basis_id, boundary_ids.len()
                    );
                }
            }
            build_surface(basis_id, entities)
        }
        "RECTANGULAR_TRIMMED_SURFACE" => {
            let basis_id = geom::nth_ref(&record.params, 1)?;
            let u1 = geom::nth_real(&record.params, 2).unwrap_or(0.0) as Real;
            let u2 = geom::nth_real(&record.params, 3).unwrap_or(1.0) as Real;
            let v1 = geom::nth_real(&record.params, 4).unwrap_or(0.0) as Real;
            let v2 = geom::nth_real(&record.params, 5).unwrap_or(1.0) as Real;
            log::debug!(
                "[BRep] {:?}: RECTANGULAR_TRIMMED_SURFACE trim range: u=[{}, {}], v=[{}, {}]",
                surface_id, u1, u2, v1, v2
            );
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


/// Resolve a VECTOR entity to its full 3D vector (direction × magnitude).
pub(crate) fn resolve_vector_magnitude(vec_id: u64, entities: &EntityIndex) -> Option<PVec3> {
    let record = entities.get(&vec_id)?;
    match record.name.as_str() {
        "VECTOR" => {
            let dir_id = geom::nth_ref(&record.params, 1)?;
            let dir = topology::resolve_direction(dir_id, entities)?;
            let mag = geom::nth_real(&record.params, 2).unwrap_or(1.0) as Real;
            Some(dir * mag)
        }
        "DIRECTION" => {
            topology::resolve_direction(vec_id, entities)
        }
        _ => topology::resolve_direction(vec_id, entities),
    }
}

