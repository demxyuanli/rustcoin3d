use super::*;
use crate::step::brep::geom::nurbs_build::build_nurbs_surface;

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

