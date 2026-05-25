//! Curved surface tessellation via UV sampling.

use std::collections::HashMap;
use rc3d_core::math::Vec3;
use super::parser::EntityIndex;
use super::entity_types::EntityType;
use super::topology::{self, StepFace};
use super::pcurve::FaceTrim;
use super::tessellate::MeshResult;
use super::geom;
use super::nurbs::NurbsSurface;
use super::value::StepValue;
use super::refine;
use super::pcurve;
use earcutr;

const DEFAULT_SAMPLES: usize = 48;

/// Extracted surface geometry info including placement and params.
#[allow(dead_code)]
struct SurfaceInfo {
    /// Placement frame: (origin, x_axis, y_axis, z_axis)
    origin: Vec3,
    x_axis: Vec3,
    y_axis: Vec3,
    z_axis: Vec3,
    /// Geometry-specific numeric parameters (kept for diagnostics)
    params: Vec<f32>,
}

/// Extract surface geometry from a STEP surface entity.
/// Properly resolves placement references and extracts per-type numeric params.
fn extract_surface_info(
    entity: &super::parser::EntityRecord,
    entities: &EntityIndex,
) -> Option<SurfaceInfo> {
    // All analytic surfaces: (name, #position, ...numeric_params...)
    let placement_id = entity.params.nth_param(1)?.as_ref_id()?;
    let (origin, x_axis, z_axis) = topology::resolve_placement(placement_id, entities)?;
    let y_axis = z_axis.cross(x_axis).normalize();

    let params = match entity.entity_type {
        EntityType::CylindricalSurface => {
            // CYLINDRICAL_SURFACE(name, #position, radius)
            let radius = entity.params.nth_param(2)
                .and_then(|v: &super::value::StepValue| v.as_real()).unwrap_or(1.0) as f32;
            vec![radius]
        }
        EntityType::ConicalSurface => {
            // CONICAL_SURFACE(name, #position, radius, semi_angle)
            let radius = entity.params.nth_param(2)
                .and_then(|v: &super::value::StepValue| v.as_real()).unwrap_or(1.0) as f32;
            let semi_angle = entity.params.nth_param(3)
                .and_then(|v: &super::value::StepValue| v.as_real()).unwrap_or(0.7854) as f32;
            vec![radius, semi_angle]
        }
        EntityType::SphericalSurface => {
            // SPHERICAL_SURFACE(name, #position, radius)
            let radius = entity.params.nth_param(2)
                .and_then(|v: &super::value::StepValue| v.as_real()).unwrap_or(1.0) as f32;
            vec![radius]
        }
        EntityType::ToroidalSurface => {
            // TOROIDAL_SURFACE(name, #position, major_radius, minor_radius)
            let major = entity.params.nth_param(2)
                .and_then(|v: &super::value::StepValue| v.as_real()).unwrap_or(1.0) as f32;
            let minor = entity.params.nth_param(3)
                .and_then(|v: &super::value::StepValue| v.as_real()).unwrap_or(0.1) as f32;
            vec![major, minor]
        }
        EntityType::Plane => vec![],
        _ => return None,
    };

    Some(SurfaceInfo { origin, x_axis, y_axis, z_axis, params })
}

/// Build a NurbsSurface from a STEP B_SPLINE_SURFACE or B_SPLINE_SURFACE_WITH_KNOTS entity.
fn build_nurbs_from_bspline_surface(
    entity: &super::parser::EntityRecord,
    entities: &EntityIndex,
) -> Option<NurbsSurface> {
    let params = &entity.params;

    // Handle two param layouts from merge_subsuper_params:
    // Simple (12 params): [0]=deg_u(int), [1]=deg_v(int), [2]=ctrl_pts(list),
    //   [3-6]=form/enums, [7]=u_mult, [8]=v_mult, [9]=u_knots, [10]=v_knots, [11]=knot_spec
    // With leading omitted (13 params): [0]=omitted, [1]=deg_u, [2]=deg_v,
    //   [3]=ctrl_pts, [4-7]=form/enums, [8]=u_mult, [9]=v_mult,
    //   [10]=u_knots, [11]=v_knots, [12]=knot_spec
    let off: usize = if geom::nth_int(params, 0).is_some() { 0 } else { 1 };
    let degree_u = geom::nth_int(params, off)? as usize;
    let degree_v = geom::nth_int(params, off + 1)? as usize;

    // Resolve control points from nested list of references
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

    let mult_base = off + 7;
    let u_multiplicities = geom::nth_list_ints(params, mult_base);
    let v_multiplicities = geom::nth_list_ints(params, mult_base + 1);
    let u_knot_vals = geom::nth_list_reals(params, mult_base + 2);
    let v_knot_vals = geom::nth_list_reals(params, mult_base + 3);

    let knots_u = build_knot_vector_from_multiplicities(
        &u_multiplicities, &u_knot_vals, degree_u, control_points.len(),
    );
    let knots_v = build_knot_vector_from_multiplicities(
        &v_multiplicities, &v_knot_vals, degree_v, control_points[0].len(),
    );

    // Extract rational weights — scan for a 2D list of reals matching control point dimensions.
    // RATIONAL_B_SPLINE_SURFACE appends weight lists after knot_spec in the merged params.
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
fn build_knot_vector_from_multiplicities(
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
    // Ensure sufficient knots: cp_count + degree + 1
    let needed = cp_count + degree + 1;
    if knots.len() < needed {
        let min_k = *knots.first().unwrap_or(&0.0);
        let max_k = *knots.last().unwrap_or(&1.0);
        let extra = needed - knots.len();
        for i in 0..extra {
            let t = (i + 1) as f32 / (extra + 1) as f32;
            knots.push(min_k + (max_k - min_k) * t);
        }
        // Sort after expansion — interpolated values may be out of order
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    knots
}

/// Scan params for a 2D weights list matching NURBS control point dimensions.
/// Used to extract rational B-spline surface weights appended after knot_spec.
fn find_surface_weights(params: &StepValue, rows: usize, cols: usize) -> Option<Vec<Vec<f32>>> {
    let list = params.as_list()?;
    // Scan from end — weights are always the last meaningful 2D param.
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

/// Build a NurbsSurface from a STEP analytic surface entity.
fn build_nurbs_from_surface(
    entity: &super::parser::EntityRecord,
    u_min: f32,
    u_max: f32,
    v_min: f32,
    v_max: f32,
) -> Option<NurbsSurface> {
    match entity.entity_type {
        EntityType::Plane => {
            // Build plane NURBS matching the estimated UV bounds from edges/trim
            Some(NurbsSurface::plane(u_min, u_max, v_min, v_max))
        }
        EntityType::CylindricalSurface => {
            let radius = entity.params.nth_param(2)
                .and_then(|v| v.as_real()).unwrap_or(1.0) as f32;
            Some(NurbsSurface::cylinder(radius, v_min, v_max))
        }
        EntityType::ConicalSurface => {
            let radius = entity.params.nth_param(2)
                .and_then(|v| v.as_real()).unwrap_or(1.0) as f32;
            let semi_angle = entity.params.nth_param(3)
                .and_then(|v| v.as_real()).unwrap_or(0.7854) as f32;
            Some(NurbsSurface::cone(radius, semi_angle, v_min, v_max))
        }
        EntityType::SphericalSurface => {
            let radius = entity.params.nth_param(2)
                .and_then(|v| v.as_real()).unwrap_or(1.0) as f32;
            Some(NurbsSurface::sphere(radius))
        }
        EntityType::ToroidalSurface => {
            let major = entity.params.nth_param(2)
                .and_then(|v| v.as_real()).unwrap_or(1.0) as f32;
            let minor = entity.params.nth_param(3)
                .and_then(|v| v.as_real()).unwrap_or(0.1) as f32;
            Some(NurbsSurface::torus(major, minor))
        }
        _ => None,
    }
}

/// Map external UV parameters (in surface natural domain) to NurbsSurface [0,1] domain.
pub fn map_uv_to_nurbs(entity_type: EntityType, u: f32, v: f32) -> (f32, f32) {
    let two_pi = 2.0f32 * std::f32::consts::PI;
    match entity_type {
        EntityType::CylindricalSurface | EntityType::ConicalSurface => {
            (u / two_pi, v)
        }
        EntityType::ToroidalSurface => {
            (u / two_pi, v / two_pi)
        }
        EntityType::SphericalSurface => {
            (u / two_pi, v / std::f32::consts::PI)
        }
        _ => (u, v),
    }
}


/// Transform a point from local surface coordinates to world coordinates.
fn local_to_world(local_pt: Vec3, info: &SurfaceInfo) -> Vec3 {
    info.origin
        + info.x_axis * local_pt.x
        + info.y_axis * local_pt.y
        + info.z_axis * local_pt.z
}

/// Compute UV bounds from trim loops.
/// Returns (u_min, u_max, v_min, v_max) or None if no points.
fn compute_uv_bounds(trim: &FaceTrim) -> Option<(f32, f32, f32, f32)> {
    let mut u_min = f32::INFINITY;
    let mut u_max = f32::NEG_INFINITY;
    let mut v_min = f32::INFINITY;
    let mut v_max = f32::NEG_INFINITY;
    let mut has_points = false;

    for trim_loop in &trim.loops {
        for pt in &trim_loop.points {
            u_min = u_min.min(pt.u);
            u_max = u_max.max(pt.u);
            v_min = v_min.min(pt.v);
            v_max = v_max.max(pt.v);
            has_points = true;
        }
    }

    if has_points {
        Some((u_min, u_max, v_min, v_max))
    } else {
        None
    }
}

/// Estimate UV bounds from face edge loop vertices when PCURVE trim is unavailable.
fn estimate_uv_bounds_from_edges(
    face: &StepFace,
    entities: &EntityIndex,
    entity_type: EntityType,
) -> Option<(f32, f32, f32, f32)> {
    let surface_id = face.surface_id?;
    let surface = entities.get(&surface_id)?;
    let placement_id = surface.params.nth_param(1)?.as_ref_id()?;
    let (origin, x_axis, z_axis) = topology::resolve_placement(placement_id, entities)?;
    let y_axis = z_axis.cross(x_axis).normalize();

    // Collect edge vertices: sample curves for better coverage, not just endpoints.
    // This is essential for circular edges where start == end.
    let mut pts = Vec::new();
    for bloop in &face.bounds {
        for edge in &bloop.edges {
            // Sample the curve between start and end to get interior points
            let sampled = geom::sample_curve(edge.curve_id, entities, edge.start, edge.end, 0.5);
            pts.extend(sampled);
        }
    }
    if pts.is_empty() { return None; }

    match entity_type {
        EntityType::Plane => {
            let mut min_u = f32::INFINITY;
            let mut max_u = f32::NEG_INFINITY;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for pt in pts {
                let rel = pt - origin;
                let u = rel.dot(x_axis);
                let v = rel.dot(y_axis);
                min_u = min_u.min(u);
                max_u = max_u.max(u);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
            let du = (max_u - min_u).max(1e-3);
            let dv = (max_v - min_v).max(1e-3);
            Some((min_u - du * 0.05, max_u + du * 0.05, min_v - dv * 0.05, max_v + dv * 0.05))
        }
        EntityType::CylindricalSurface | EntityType::ConicalSurface => {
            let mut min_z = f32::INFINITY;
            let mut max_z = f32::NEG_INFINITY;
            let mut has_circle = false;
            for pt in pts {
                let rel = pt - origin;
                let z = rel.dot(z_axis);
                min_z = min_z.min(z);
                max_z = max_z.max(z);
                let proj = rel - z_axis * z;
                if proj.length() > 1e-3 {
                    has_circle = true;
                }
            }
            if !has_circle { return None; }
            let dz = (max_z - min_z).max(1e-3);
            Some((0.0, 2.0 * std::f32::consts::PI, min_z - dz * 0.05, max_z + dz * 0.05))
        }
        EntityType::SphericalSurface => {
            let mut min_z = f32::INFINITY;
            let mut max_z = f32::NEG_INFINITY;
            let mut has_radial = false;
            for pt in pts {
                let rel = pt - origin;
                let z = rel.dot(z_axis);
                min_z = min_z.min(z);
                max_z = max_z.max(z);
                let proj = rel - z_axis * z;
                if proj.length() > 1e-3 {
                    has_radial = true;
                }
            }
            if !has_radial { return None; }
            Some((0.0, 2.0 * std::f32::consts::PI, 0.0, std::f32::consts::PI))
        }
        EntityType::ToroidalSurface => {
            Some((0.0, 2.0 * std::f32::consts::PI, 0.0, 2.0 * std::f32::consts::PI))
        }
        _ => None,
    }
}

/// Compute UV bounds from trim, or use defaults for the surface type.
fn compute_uv_bounds_from_trim(
    trim: Option<&FaceTrim>,
    entity_type: EntityType,
) -> (f32, f32, f32, f32) {
    if let Some(tr) = trim {
        if let Some(bounds) = compute_uv_bounds(tr) {
            return bounds;
        }
    }

    // Default domains by surface type
    // For infinite surfaces (cylinder/cone), use a reasonable symmetric range.
    // Ideally the trim should always provide the bounds.
    match entity_type {
        EntityType::CylindricalSurface => (0.0, 2.0 * std::f32::consts::PI, -50.0, 50.0),
        EntityType::ConicalSurface => (0.0, 2.0 * std::f32::consts::PI, -50.0, 50.0),
        EntityType::SphericalSurface => (0.0, 2.0 * std::f32::consts::PI, 0.0, std::f32::consts::PI),
        EntityType::ToroidalSurface => (0.0, 2.0 * std::f32::consts::PI, 0.0, 2.0 * std::f32::consts::PI),
        EntityType::Plane => (-50.0, 50.0, -50.0, 50.0),
        _ => (0.0, 1.0, 0.0, 1.0),
    }
}

/// Apply mesh refinement for non-planar surfaces after initial tessellation.
fn maybe_refine_nurbs(
    mesh: MeshResult,
    nurbs: &NurbsSurface,
    entity_type: EntityType,
    u_min: f32, u_max: f32, v_min: f32, v_max: f32,
) -> MeshResult {
    if entity_type == EntityType::Plane {
        return mesh;
    }
    let config = refine::RefineConfig::default();
    let (refined_v, refined_i, refined_n) = refine::refine_mesh(
        &mesh.vertices, &mesh.indices, &mesh.normals,
        nurbs, entity_type, u_min, u_max, v_min, v_max, &config,
    );
    MeshResult { vertices: refined_v, indices: refined_i, normals: refined_n }
}

/// Tessellate a curved face using UV sampling.
/// Returns None if the surface type is unsupported or has no surface_id.
pub fn tessellate_curved_face(
    face: &StepFace,
    entities: &EntityIndex,
    trim: Option<&FaceTrim>,
) -> Option<MeshResult> {
    let surface_id = face.surface_id?;
    let surface = entities.get(&surface_id)?;

    // For analytic surfaces, use unified NURBS representation with analytic normals.
    match surface.entity_type {
        EntityType::Plane
        | EntityType::CylindricalSurface
        | EntityType::ConicalSurface
        | EntityType::SphericalSurface
        | EntityType::ToroidalSurface => {
            let (u_min, u_max, v_min, v_max) = if let Some(tr) = trim {
                compute_uv_bounds_from_trim(Some(tr), surface.entity_type)
            } else {
                estimate_uv_bounds_from_edges(face, entities, surface.entity_type)
                    .unwrap_or_else(|| compute_uv_bounds_from_trim(None, surface.entity_type))
            };
            let nurbs = build_nurbs_from_surface(surface, u_min, u_max, v_min, v_max)?;
            // Prefer exact trim curves; fall back to polygon trim
            let exact_trims = pcurve::extract_exact_face_trim(face, entities);
            let use_exact = exact_trims.is_some() && trim.is_some();
            if let Some(tr) = trim {
                let mesh = if use_exact {
                    tessellate_trimmed_via_exact(exact_trims.as_ref().unwrap(), &nurbs, surface_id, entities, surface.entity_type, face.same_sense, u_min, u_max, v_min, v_max)
                        .or_else(|| tessellate_trimmed_via_uv(tr, &nurbs, surface_id, entities, surface.entity_type, face.same_sense, u_min, u_max, v_min, v_max))
                } else {
                    tessellate_trimmed_via_uv(tr, &nurbs, surface_id, entities, surface.entity_type, face.same_sense, u_min, u_max, v_min, v_max)
                };
                mesh.map(|m| maybe_refine_nurbs(m, &nurbs, surface.entity_type, u_min, u_max, v_min, v_max))
            } else {
                let n = estimate_sample_count(&nurbs, surface.entity_type, u_min, u_max, v_min, v_max);
                let mesh = sample_grid_nurbs(n, surface_id, trim, &nurbs, entities, surface.entity_type, face.same_sense, u_min, u_max, v_min, v_max);
                Some(maybe_refine_nurbs(mesh, &nurbs, surface.entity_type, u_min, u_max, v_min, v_max))
            }
        }
        EntityType::SurfaceOfLinearExtrusion | EntityType::SurfaceOfRevolution => {
            let n = estimate_extrusion_revolution_samples(surface, entities, surface.entity_type);
            geom::evaluate_surface(surface_id, entities, n, n)
                .map(|grid| {
                    let mut mesh = grid_to_mesh(&grid);
                    // Apply same_sense: if false, reverse normals
                    if !face.same_sense {
                        for n in &mut mesh.normals {
                            *n = -*n;
                        }
                    }
                    mesh
                })
        }
        EntityType::BSplineSurface | EntityType::BSplineSurfaceWithKnots => {
            let nurbs = build_nurbs_from_bspline_surface(surface, entities)?;
            let nurbs_u_min = nurbs.knots_u[nurbs.degree_u];
            let nurbs_u_max = nurbs.knots_u[nurbs.knots_u.len().saturating_sub(nurbs.degree_u + 1)];
            let nurbs_v_min = nurbs.knots_v[nurbs.degree_v];
            let nurbs_v_max = nurbs.knots_v[nurbs.knots_v.len().saturating_sub(nurbs.degree_v + 1)];
            // When trim is available, restrict sampling to the trim polygon
            // bounding box to avoid wasting grid points far outside the
            // trimmed region (e.g. NURBS domain 81×81 but trim only covers 15×1).
            let (u_min, u_max, v_min, v_max) = if let Some(tr) = trim {
                if let Some(bounds) = compute_uv_bounds(tr) {
                    (bounds.0.max(nurbs_u_min).min(nurbs_u_max),
                     bounds.1.max(nurbs_u_min).min(nurbs_u_max),
                     bounds.2.max(nurbs_v_min).min(nurbs_v_max),
                     bounds.3.max(nurbs_v_min).min(nurbs_v_max))
                } else {
                    (nurbs_u_min, nurbs_u_max, nurbs_v_min, nurbs_v_max)
                }
            } else {
                (nurbs_u_min, nurbs_u_max, nurbs_v_min, nurbs_v_max)
            };
            let n = estimate_sample_count(&nurbs, surface.entity_type, u_min, u_max, v_min, v_max);
            let mesh = sample_grid_nurbs(n, surface_id, trim, &nurbs, entities, surface.entity_type, face.same_sense, u_min, u_max, v_min, v_max);
            Some(maybe_refine_nurbs(mesh, &nurbs, surface.entity_type, u_min, u_max, v_min, v_max))
        }
        EntityType::OffsetSurface => {
            // OFFSET_SURFACE('', #base_surface, offset_distance, same_sense)
            let base_id = geom::nth_ref(&surface.params, 1)?;
            let offset_dist = geom::nth_real(&surface.params, 2)? as f32;
            // Create a temporary face with the base surface for tessellation
            let base_surface = entities.get(&base_id)?;
            let base_face = StepFace {
                bounds: face.bounds.clone(),
                surface_id: Some(base_id),
                same_sense: face.same_sense,
            };
            // Tessellate the base surface
            let base_trim = trim.cloned();
            let mesh = tessellate_curved_face(&base_face, entities, base_trim.as_ref())?;
            // Apply offset to all vertices along the normal direction
            if offset_dist.abs() > 1e-6 {
                let mut result_mesh = mesh;
                for v in &mut result_mesh.vertices {
                    if let Some(normal) = face_normal_at_point(&base_face, base_surface, *v, entities) {
                        *v = *v + normal * offset_dist;
                    }
                }
                Some(result_mesh)
            } else {
                Some(mesh)
            }
        }
        EntityType::BoundedSurface => {
            // BOUNDED_SURFACE('', (B_SPLINE_SURFACE_WITH_KNOTS or other bounded surface data))
            // The actual surface data is nested in the params list
            // Look for B_SPLINE_SURFACE or B_SPLINE_SURFACE_WITH_KNOTS in the nested params
            let inner_surface_id = find_bounded_surface_inner(face.surface_id?, entities)?;
            let inner_face = StepFace {
                bounds: face.bounds.clone(),
                surface_id: Some(inner_surface_id),
                same_sense: face.same_sense,
            };
            tessellate_curved_face(&inner_face, entities, trim)
        }
        EntityType::CurveBoundedSurface => {
            // CURVE_BOUNDED_SURFACE('', #basis_surface, #boundary, .F.)
            // Unwrap to the underlying basis surface
            let base_id = geom::nth_ref(&surface.params, 1)?;
            let base_face = StepFace {
                bounds: face.bounds.clone(),
                surface_id: Some(base_id),
                same_sense: face.same_sense,
            };
            tessellate_curved_face(&base_face, entities, trim)
        }
        EntityType::RectangularTrimmedSurface => {
            // RECTANGULAR_TRIMMED_SURFACE('', #basis_surface, u1, u2, v1, v2)
            let base_id = geom::nth_ref(&surface.params, 1)?;
            let u1 = geom::nth_real(&surface.params, 2).unwrap_or(0.0) as f32;
            let u2 = geom::nth_real(&surface.params, 3).unwrap_or(1.0) as f32;
            let v1 = geom::nth_real(&surface.params, 4).unwrap_or(0.0) as f32;
            let v2 = geom::nth_real(&surface.params, 5).unwrap_or(1.0) as f32;
            let base_face = StepFace {
                bounds: face.bounds.clone(),
                surface_id: Some(base_id),
                same_sense: face.same_sense,
            };
            // Constrain tessellation to the trimmed UV domain
            let base_surface = entities.get(&base_id)?;
            match base_surface.entity_type {
                EntityType::Plane
                | EntityType::CylindricalSurface
                | EntityType::ConicalSurface
                | EntityType::SphericalSurface
                | EntityType::ToroidalSurface => {
                    let nurbs = build_nurbs_from_surface(base_surface, u1, u2, v1, v2)?;
                    let n = estimate_sample_count(&nurbs, base_surface.entity_type, u1, u2, v1, v2);
                    Some(sample_grid_nurbs(n, base_id, trim, &nurbs, entities, base_surface.entity_type, face.same_sense, u1, u2, v1, v2))
                }
                _ => {
                    // For other surface types, fall back to the standard path with UV bounds
                    tessellate_curved_face(&base_face, entities, trim)
                }
            }
        }
        _ => None,
    }
}

/// Estimate adaptive sample count for extrusion or revolution surfaces
/// based on the generatrix curve arc length and the sweep extent.
fn estimate_extrusion_revolution_samples(
    surface: &super::parser::EntityRecord,
    entities: &EntityIndex,
    entity_type: EntityType,
) -> usize {
    let curve_id = match geom::nth_ref(&surface.params, 1) {
        Some(id) => id,
        None => return DEFAULT_SAMPLES,
    };

    // Sample the generatrix curve to estimate arc length (use same tolerance
    // as eval_extrusion/eval_revolution for consistent grid sizing)
    const ESTIMATION_TOL: f32 = 0.1;
    let curve_pts = geom::sample_curve(curve_id, entities, Vec3::ZERO, Vec3::ZERO, ESTIMATION_TOL);
    let arc_length: f32 = curve_pts.windows(2)
        .map(|w| (w[1] - w[0]).length())
        .sum();

    // For revolution, the circumference at average radius determines angular samples.
    // For extrusion, the direction magnitude determines sweep samples.
    let sweep_samples = if entity_type == EntityType::SurfaceOfRevolution {
        // Resolve the actual rotation axis from AXIS2_PLACEMENT_3D / AXIS1_PLACEMENT
        let axis_id = geom::nth_ref(&surface.params, 2);
        let (axis_origin, axis_dir) = axis_id
            .and_then(|id| topology::resolve_placement(id, entities))
            .map(|(origin, _x, z)| (origin, z))
            .unwrap_or((Vec3::ZERO, Vec3::Z));
        let axis = axis_dir.normalize();
        // Compute perpendicular distance from each curve point to the rotation axis
        let avg_r: f32 = curve_pts.iter()
            .map(|p| {
                let rel = *p - axis_origin;
                (rel - axis * rel.dot(axis)).length()
            })
            .sum::<f32>() / curve_pts.len().max(1) as f32;
        let circumference = 2.0 * std::f32::consts::PI * avg_r.max(0.1);
        (circumference / 2.0).clamp(16.0, 96.0) as usize
    } else {
        // Extrusion: look at the direction magnitude
        let dir_id = geom::nth_ref(&surface.params, 2);
        if let Some(did) = dir_id {
            if let Some(dir_rec) = entities.get(&did) {
                let mag = dir_rec.params.nth_param(1)
                    .and_then(|v| v.as_list())
                    .map(|coords| {
                        let x = coords.get(0).and_then(|v| v.as_real()).unwrap_or(0.0) as f32;
                        let y = coords.get(1).and_then(|v| v.as_real()).unwrap_or(0.0) as f32;
                        let z = coords.get(2).and_then(|v| v.as_real()).unwrap_or(0.0) as f32;
                        (x * x + y * y + z * z).sqrt()
                    })
                    .unwrap_or(10.0);
                (mag / 0.5).clamp(8.0, 64.0) as usize
            } else {
                DEFAULT_SAMPLES / 2
            }
        } else {
            DEFAULT_SAMPLES / 2
        }
    };

    // Curve direction samples: proportional to arc length at estimation tolerance
    let curve_samples = ((arc_length / ESTIMATION_TOL).clamp(8.0, 64.0) as usize).max(4);

    // Use the max of both directions, clamped
    curve_samples.max(sweep_samples).min(96)
}

/// Estimate sample count from surface curvature using a coarse probe grid.
/// Combines normal variation (curvature proxy) with chordal deflection
/// (geometric deviation) for robust adaptive sampling.
fn estimate_sample_count(
    nurbs: &NurbsSurface,
    entity_type: EntityType,
    u_min: f32,
    u_max: f32,
    v_min: f32,
    v_max: f32,
) -> usize {
    if entity_type == EntityType::Plane {
        return 8;
    }

    const PROBE: usize = 8;
    let mut max_normal_var = 0.0f32;
    let mut max_deflection = 0.0f32;
    let du = (u_max - u_min) / PROBE as f32;
    let dv = (v_max - v_min) / PROBE as f32;

    for i in 0..=PROBE {
        let u = u_min + i as f32 * du;
        for j in 0..=PROBE {
            let v = v_min + j as f32 * dv;
            let (u_nurbs, v_nurbs) = map_uv_to_nurbs(entity_type, u, v);
            let p = nurbs.evaluate(u_nurbs, v_nurbs);
            let n = nurbs.normal(u_nurbs, v_nurbs);

            // Normal variation with immediate neighbours (curvature proxy)
            if i < PROBE {
                let u2 = u_min + (i + 1) as f32 * du;
                let (un2, vn2) = map_uv_to_nurbs(entity_type, u2, v);
                let n2 = nurbs.normal(un2, vn2);
                max_normal_var = max_normal_var.max((n - n2).length());
            }
            if j < PROBE {
                let v2 = v_min + (j + 1) as f32 * dv;
                let (un2, vn2) = map_uv_to_nurbs(entity_type, u, v2);
                let n2 = nurbs.normal(un2, vn2);
                max_normal_var = max_normal_var.max((n - n2).length());
            }

            // Chordal deflection: linear midpoint prediction vs actual surface
            if i < PROBE {
                let u2 = u_min + (i + 1) as f32 * du;
                let (un2, vn2) = map_uv_to_nurbs(entity_type, u2, v);
                let p2 = nurbs.evaluate(un2, vn2);
                let u_mid = u + du * 0.5;
                let (um, vm) = map_uv_to_nurbs(entity_type, u_mid, v);
                let p_mid = nurbs.evaluate(um, vm);
                max_deflection = max_deflection.max((p_mid - (p + p2) * 0.5).length());
            }
            if j < PROBE {
                let v2 = v_min + (j + 1) as f32 * dv;
                let (un2, vn2) = map_uv_to_nurbs(entity_type, u, v2);
                let p2 = nurbs.evaluate(un2, vn2);
                let v_mid = v + dv * 0.5;
                let (um, vm) = map_uv_to_nurbs(entity_type, u, v_mid);
                let p_mid = nurbs.evaluate(um, vm);
                max_deflection = max_deflection.max((p_mid - (p + p2) * 0.5).length());
            }
        }
    }

    // Map combined metrics → [8, 64] samples
    let t_n = (max_normal_var * 4.0).min(1.0);
    let t_d = (max_deflection * 8.0).min(1.0);
    let t = t_n.max(t_d);
    (8.0 + t * 56.0) as usize
}

fn sample_grid_nurbs(
    n: usize,
    surface_id: u64,
    trim: Option<&FaceTrim>,
    nurbs: &NurbsSurface,
    entities: &EntityIndex,
    entity_type: EntityType,
    same_sense: bool,
    u_min: f32,
    u_max: f32,
    v_min: f32,
    v_max: f32,
) -> MeshResult {
    let surface = entities.get(&surface_id).unwrap();
    let info = extract_surface_info(surface, entities);

    let du = if (u_max - u_min).abs() > 1e-4 {
        (u_max - u_min) / n as f32
    } else {
        1.0 / n as f32
    };
    let dv = if (v_max - v_min).abs() > 1e-4 {
        (v_max - v_min) / n as f32
    } else {
        1.0 / n as f32
    };

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut pos_map: HashMap<[u32; 3], i32> = HashMap::new();
    let mut grid_idx: Vec<Vec<i32>> = Vec::with_capacity(n + 1);

    // Pre-extract cone semi-angle for geometric normal
    let cone_tan_a = if entity_type == EntityType::ConicalSurface {
        (surface.params.nth_param(3)
            .and_then(|v| v.as_real())
            .unwrap_or(0.7854) as f32)
            .tan()
    } else {
        0.0
    };

    for iu in 0..=n {
        let u = u_min + iu as f32 * du;
        let mut row = Vec::with_capacity(n + 1);
        for iv in 0..=n {
            let v = v_min + iv as f32 * dv;

            if let Some(tr) = trim {
                if !super::pcurve::point_in_trim_polygon(u, v, &tr.loops) {
                    row.push(-1);
                    continue;
                }
            }

            let (u_nurbs, v_nurbs) = map_uv_to_nurbs(entity_type, u, v);

            let mut pt = nurbs.evaluate(u_nurbs, v_nurbs);

            // For analytic surfaces, compute the geometric normal directly
            // instead of relying on NURBS parametric derivatives which
            // degenerate at knot-multiplicity points.
            let mut n = match entity_type {
                EntityType::Plane => Vec3::Z,
                EntityType::CylindricalSurface => {
                    let r = (pt.x * pt.x + pt.y * pt.y).sqrt();
                    if r > 1e-6 {
                        Vec3::new(pt.x / r, pt.y / r, 0.0)
                    } else {
                        Vec3::Z
                    }
                }
                EntityType::ConicalSurface => {
                    let r = (pt.x * pt.x + pt.y * pt.y).sqrt();
                    if r > 1e-6 {
                        let denom = (1.0 + cone_tan_a * cone_tan_a).sqrt();
                        Vec3::new(pt.x / r / denom, pt.y / r / denom, -cone_tan_a / denom)
                    } else {
                        Vec3::Z
                    }
                }
                EntityType::SphericalSurface => {
                    let r = pt.length();
                    if r > 1e-6 { pt / r } else { Vec3::Z }
                }
                EntityType::ToroidalSurface => {
                    // Torus normal: point from tube center to surface point.
                    // Tube center is at (pt.x, pt.y, 0) projected onto major circle.
                    let r_xy = (pt.x * pt.x + pt.y * pt.y).sqrt();
                    if r_xy > 1e-6 {
                        let major_r = surface.params.nth_param(2)
                            .and_then(|v| v.as_real()).unwrap_or(1.0) as f32;
                        let tube_center_xy = major_r / r_xy;
                        Vec3::new(
                            pt.x - tube_center_xy * pt.x,
                            pt.y - tube_center_xy * pt.y,
                            pt.z,
                        ).normalize()
                    } else {
                        Vec3::Z
                    }
                }
                _ => nurbs.normal(u_nurbs, v_nurbs),
            };

            if !same_sense {
                n = -n;
            }

            if let Some(ref surf_info) = info {
                pt = local_to_world(pt, surf_info);
                n = (surf_info.x_axis * n.x + surf_info.y_axis * n.y + surf_info.z_axis * n.z)
                    .normalize();
            }

            let hash = rc3d_core::utils::hash::f32x3_quantized_bits([pt.x, pt.y, pt.z]);
            let idx = *pos_map.entry(hash).or_insert_with(|| {
                let i = vertices.len() as i32;
                vertices.push(pt);
                normals.push(n);
                i
            });
            row.push(idx);
        }
        grid_idx.push(row);
    }

    // Triangulate grid quads, skipping trimmed-away vertices
    let mut indices = Vec::new();
    for iu in 0..n {
        for iv in 0..n {
            let a = grid_idx[iu][iv];
            let b = grid_idx[iu + 1][iv];
            let c = grid_idx[iu + 1][iv + 1];
            let d = grid_idx[iu][iv + 1];
            if a < 0 || b < 0 || c < 0 || d < 0 {
                continue;
            }
            indices.extend_from_slice(&[a, b, c, -1, a, c, d, -1]);
        }
    }

    MeshResult { vertices, indices, normals }
}

/// Tessellate a trimmed face using UV-space earcut triangulation.
/// Instead of uniform grid + point-in-polygon, this meshes the exact
/// trim boundary in UV space using ear clipping, then maps UV → 3D.
/// For curved surfaces, interior subdivision refines the approximation.
fn tessellate_trimmed_via_uv(
    trim: &FaceTrim,
    nurbs: &NurbsSurface,
    surface_id: u64,
    entities: &EntityIndex,
    entity_type: EntityType,
    same_sense: bool,
    u_min: f32,
    u_max: f32,
    v_min: f32,
    v_max: f32,
) -> Option<MeshResult> {
    let surface = entities.get(&surface_id)?;
    let info = extract_surface_info(surface, entities);

    // --- Step 1: Collect all UV points from trim loops into a flat array ---
    let total_pts: usize = trim.loops.iter().map(|l| l.points.len()).sum();
    let mut flat_uv: Vec<f64> = Vec::with_capacity(total_pts * 2);
    let mut hole_indices: Vec<usize> = Vec::new();
    let mut uv_vertex_index: Vec<(f32, f32)> = Vec::with_capacity(total_pts);

    for (li, trim_loop) in trim.loops.iter().enumerate() {
        if li > 0 {
            // Record hole start index (in vertex count, not flat index)
            hole_indices.push(uv_vertex_index.len());
        }
        for uv_pt in &trim_loop.points {
            flat_uv.push(uv_pt.u as f64);
            flat_uv.push(uv_pt.v as f64);
            uv_vertex_index.push((uv_pt.u, uv_pt.v));
        }
    }

    if total_pts < 3 {
        return None;
    }

    // --- Step 2: Earcut triangulation in UV space ---
    let tri_indices = match earcutr::earcut(&flat_uv, &hole_indices, 2) {
        Ok(indices) => indices,
        Err(_) => {
            // Fallback: use grid sampling
            return None;
        }
    };

    // --- Step 3: Add interior refinement points for curved surfaces ---
    // For B-spline surfaces, subdivide each triangle to better capture curvature
    let subdivisions: usize = match entity_type {
        EntityType::Plane => 0,
        EntityType::BSplineSurface | EntityType::BSplineSurfaceWithKnots => 1,
        _ => 0,
    };

    let mut all_uv_pts: Vec<(f32, f32)> = uv_vertex_index.clone();
    let mut all_triangles: Vec<[usize; 3]> = Vec::new();

    if subdivisions == 0 {
        // Planes: direct triangle output
        for chunk in tri_indices.chunks(3) {
            if chunk.len() == 3 {
                all_triangles.push([chunk[0], chunk[1], chunk[2]]);
            }
        }
    } else {
        // Curved surfaces: subdivide each triangle midpoint
        for chunk in tri_indices.chunks(3) {
            if chunk.len() != 3 { continue; }
            let (i0, i1, i2) = (chunk[0], chunk[1], chunk[2]);
            let (u0, v0) = uv_vertex_index[i0];
            let (u1, v1) = uv_vertex_index[i1];
            let (u2, v2) = uv_vertex_index[i2];

            // Center point
            let uc = (u0 + u1 + u2) / 3.0;
            let vc = (v0 + v1 + v2) / 3.0;
            let ic = all_uv_pts.len();
            all_uv_pts.push((uc, vc));

            // Three sub-triangles
            all_triangles.push([i0, i1, ic]);
            all_triangles.push([i1, i2, ic]);
            all_triangles.push([i2, i0, ic]);
        }
    }

    // --- Step 4: Evaluate NURBS at each UV point → 3D ---
    let mut vertices: Vec<Vec3> = Vec::with_capacity(all_uv_pts.len());
    let mut normals: Vec<Vec3> = Vec::with_capacity(all_uv_pts.len());
    let mut pos_map: HashMap<[u32; 3], i32> = HashMap::new();
    let mut remap: Vec<i32> = vec![-1; all_uv_pts.len()];

    for (i, &(u, v)) in all_uv_pts.iter().enumerate() {
        // Clamp UV to valid NURBS parameter range to prevent extrapolation artifacts
        let u_clamped = u.clamp(u_min, u_max);
        let v_clamped = v.clamp(v_min, v_max);
        let (u_nurbs, v_nurbs) = map_uv_to_nurbs(entity_type, u_clamped, v_clamped);
        let mut pt = nurbs.evaluate(u_nurbs, v_nurbs);
        let mut n = match entity_type {
            EntityType::Plane => Vec3::Z,
            _ => nurbs.normal(u_nurbs, v_nurbs),
        };

        if !same_sense {
            n = -n;
        }

        if let Some(ref surf_info) = info {
            pt = surf_info.origin + surf_info.x_axis * pt.x
                + surf_info.y_axis * pt.y + surf_info.z_axis * pt.z;
            n = (surf_info.x_axis * n.x + surf_info.y_axis * n.y
                + surf_info.z_axis * n.z).normalize();
        }

        let hash = rc3d_core::utils::hash::f32x3_quantized_bits([pt.x, pt.y, pt.z]);
        let idx = *pos_map.entry(hash).or_insert_with(|| {
            let i = vertices.len() as i32;
            vertices.push(pt);
            normals.push(n);
            i
        });
        remap[i] = idx;
    }

    // --- Step 5: Emit triangle indices ---
    let mut indices = Vec::new();
    for tri in &all_triangles {
        let a = remap[tri[0]];
        let b = remap[tri[1]];
        let c = remap[tri[2]];
        if a >= 0 && b >= 0 && c >= 0 {
            indices.extend_from_slice(&[a, b, c, -1]);
        }
    }

    Some(MeshResult { vertices, indices, normals })
}

/// Tessellate using exact trim curves (2D geometry → adaptive UV sampling → earcut).
fn tessellate_trimmed_via_exact(
    trim: &pcurve::ExactFaceTrim,
    nurbs: &NurbsSurface,
    surface_id: u64,
    entities: &EntityIndex,
    entity_type: EntityType,
    same_sense: bool,
    u_min: f32,
    u_max: f32,
    v_min: f32,
    v_max: f32,
) -> Option<MeshResult> {
    let surface = entities.get(&surface_id)?;
    let info = extract_surface_info(surface, entities);

    // Step 1: Adaptively sample exact trim curves into UV polygon
    let mut flat_uv: Vec<f64> = Vec::new();
    let mut hole_indices: Vec<usize> = Vec::new();
    let mut all_uv_pts: Vec<(f32, f32)> = Vec::new();

    for (li, trim_loop) in trim.loops.iter().enumerate() {
        if li > 0 {
            hole_indices.push(all_uv_pts.len());
        }
        for curve in &trim_loop.curves {
            let n = match curve {
                pcurve::ExactTrimCurve2D::Line { .. } => 2,
                pcurve::ExactTrimCurve2D::Circle { radius, .. } => {
                    (radius.abs() * 6.28 / 0.1).max(12.0).min(128.0) as usize
                }
                pcurve::ExactTrimCurve2D::Ellipse { .. } => 32,
                pcurve::ExactTrimCurve2D::BSpline { control_points, .. } => {
                    (control_points.len() * 4).max(16).min(256)
                }
            };
            for j in 0..=n {
                let t = j as f32 / n.max(1) as f32;
                let uv = curve.evaluate(t);
                flat_uv.push(uv.u as f64);
                flat_uv.push(uv.v as f64);
                all_uv_pts.push((uv.u, uv.v));
            }
        }
    }

    if all_uv_pts.len() < 3 {
        return None;
    }

    // Step 2: Earcut triangulation in UV space
    let tri_indices = match earcutr::earcut(&flat_uv, &hole_indices, 2) {
        Ok(indices) => indices,
        Err(_) => return None,
    };

    // Step 3: Optional interior refinement for curved surfaces
    let subdivisions: usize = match entity_type {
        EntityType::Plane => 0,
        EntityType::BSplineSurface | EntityType::BSplineSurfaceWithKnots => 1,
        _ => 0,
    };

    let mut final_uv_pts = all_uv_pts.clone();
    let mut all_triangles: Vec<[usize; 3]> = Vec::new();

    if subdivisions == 0 {
        for chunk in tri_indices.chunks(3) {
            if chunk.len() == 3 {
                all_triangles.push([chunk[0], chunk[1], chunk[2]]);
            }
        }
    } else {
        for chunk in tri_indices.chunks(3) {
            if chunk.len() != 3 { continue; }
            let (i0, i1, i2) = (chunk[0], chunk[1], chunk[2]);
            let (u0, v0) = all_uv_pts[i0];
            let (u1, v1) = all_uv_pts[i1];
            let (u2, v2) = all_uv_pts[i2];
            let ic = final_uv_pts.len();
            final_uv_pts.push(((u0 + u1 + u2) / 3.0, (v0 + v1 + v2) / 3.0));
            all_triangles.push([i0, i1, ic]);
            all_triangles.push([i1, i2, ic]);
            all_triangles.push([i2, i0, ic]);
        }
    }

    // Step 4: Evaluate NURBS at each UV point → 3D
    let mut vertices: Vec<Vec3> = Vec::with_capacity(final_uv_pts.len());
    let mut normals: Vec<Vec3> = Vec::with_capacity(final_uv_pts.len());
    let mut pos_map: HashMap<[u32; 3], i32> = HashMap::new();
    let mut remap: Vec<i32> = vec![-1; final_uv_pts.len()];

    let cone_tan_a = if entity_type == EntityType::ConicalSurface {
        surface.params.nth_param(3).and_then(|v| v.as_real()).unwrap_or(0.7854) as f32
    } else { 0.0 };

    for (i, &(u, v)) in final_uv_pts.iter().enumerate() {
        let uc = u.clamp(u_min, u_max);
        let vc = v.clamp(v_min, v_max);
        let (un, vn) = map_uv_to_nurbs(entity_type, uc, vc);
        let mut pt = nurbs.evaluate(un, vn);
        let mut n = match entity_type {
            EntityType::Plane => Vec3::Z,
            EntityType::CylindricalSurface => {
                let r = (pt.x * pt.x + pt.y * pt.y).sqrt();
                if r > 1e-6 { Vec3::new(pt.x / r, pt.y / r, 0.0) } else { Vec3::Z }
            }
            EntityType::ConicalSurface => {
                let r = (pt.x * pt.x + pt.y * pt.y).sqrt();
                if r > 1e-6 {
                    let d = (1.0 + cone_tan_a * cone_tan_a).sqrt();
                    Vec3::new(pt.x / r / d, pt.y / r / d, -cone_tan_a / d)
                } else { Vec3::Z }
            }
            EntityType::SphericalSurface => {
                let r = pt.length();
                if r > 1e-6 { pt / r } else { Vec3::Z }
            }
            EntityType::ToroidalSurface => {
                let r_xy = (pt.x * pt.x + pt.y * pt.y).sqrt();
                if r_xy > 1e-6 {
                    let major_r = surface.params.nth_param(2).and_then(|v| v.as_real()).unwrap_or(1.0) as f32;
                    Vec3::new(pt.x - (major_r / r_xy) * pt.x, pt.y - (major_r / r_xy) * pt.y, pt.z).normalize()
                } else { Vec3::Z }
            }
            _ => nurbs.normal(un, vn),
        };

        if !same_sense { n = -n; }
        if let Some(ref si) = info {
            pt = si.origin + si.x_axis * pt.x + si.y_axis * pt.y + si.z_axis * pt.z;
            n = (si.x_axis * n.x + si.y_axis * n.y + si.z_axis * n.z).normalize();
        }

        let hash = rc3d_core::utils::hash::f32x3_quantized_bits([pt.x, pt.y, pt.z]);
        let idx = *pos_map.entry(hash).or_insert_with(|| {
            let i = vertices.len() as i32;
            vertices.push(pt);
            normals.push(n);
            i
        });
        remap[i] = idx;
    }

    let mut indices = Vec::new();
    for tri in &all_triangles {
        let a = remap[tri[0]];
        let b = remap[tri[1]];
        let c = remap[tri[2]];
        if a >= 0 && b >= 0 && c >= 0 && a != b && b != c && a != c {
            indices.extend_from_slice(&[a, b, c, -1]);
        }
    }

    Some(MeshResult { vertices, indices, normals })
}

fn grid_to_mesh(grid: &[Vec<Vec3>]) -> MeshResult {
    if grid.is_empty() || grid[0].is_empty() {
        return MeshResult::default();
    }
    let rows = grid.len();
    let cols = grid[0].len();
    let mut vertices = Vec::with_capacity(rows * cols);
    let mut grid_idx: Vec<Vec<i32>> = Vec::with_capacity(rows);
    let mut pos_map: HashMap<[u32; 3], i32> = HashMap::new();

    for row in grid {
        let mut idx_row = Vec::with_capacity(cols);
        for pt in row {
            let hash = rc3d_core::utils::hash::f32x3_quantized_bits([pt.x, pt.y, pt.z]);
            let idx = *pos_map.entry(hash).or_insert_with(|| {
                let i = vertices.len() as i32;
                vertices.push(*pt);
                i
            });
            idx_row.push(idx);
        }
        grid_idx.push(idx_row);
    }

    let mut indices = Vec::new();
    for i in 0..rows - 1 {
        for j in 0..cols - 1 {
            let a = grid_idx[i][j];
            let b = grid_idx[i + 1][j];
            let c = grid_idx[i + 1][j + 1];
            let d = grid_idx[i][j + 1];
            indices.extend_from_slice(&[a, b, c, -1, a, c, d, -1]);
        }
    }

    let mut result = MeshResult { vertices, indices, normals: Vec::new() };
    result.compute_normals();
    result
}

/// Compute approximate surface normal at a given point on a face.
/// Used for OFFSET_SURFACE to determine offset direction.
fn face_normal_at_point(
    _face: &StepFace,
    surface: &super::parser::EntityRecord,
    point: Vec3,
    entities: &EntityIndex,
) -> Option<Vec3> {
    use super::entity_types::EntityType;
    use rc3d_core::math::Vec3;

    match surface.entity_type {
        EntityType::Plane => {
            // For plane, normal is the z-axis of the placement
            let placement_id = geom::nth_ref(&surface.params, 1)?;
            let (_, _, z_axis) = topology::resolve_placement(placement_id, entities)?;
            Some(z_axis.normalize())
        }
        EntityType::CylindricalSurface => {
            // Normal is radial direction from axis to point
            let placement_id = geom::nth_ref(&surface.params, 1)?;
            let (origin, _, z_axis) = topology::resolve_placement(placement_id, entities)?;
            let axis_point = origin + z_axis * (point - origin).dot(z_axis);
            let radial = point - axis_point;
            let len = radial.length();
            if len > 1e-6 { Some(radial / len) } else { Some(Vec3::Z) }
        }
        EntityType::SurfaceOfRevolution => {
            // Normal is perpendicular to revolution axis and radial direction
            let _profile_id = geom::nth_ref(&surface.params, 1)?;
            let axis_entity_id = geom::nth_ref(&surface.params, 2)?;

            // Get axis direction - handle both AXIS1_PLACEMENT and AXIS2_PLACEMENT_3D
            let z_axis = if let Some(record) = entities.get(&axis_entity_id) {
                if record.name == "AXIS1_PLACEMENT" {
                    // AXIS1_PLACEMENT('', location, direction)
                    let direction_id = geom::nth_ref(&record.params, 2)?;
                    topology::resolve_direction(direction_id, entities)?
                } else {
                    // AXIS2_PLACEMENT_3D - use resolve_placement
                    let (_, _, z) = topology::resolve_placement(axis_entity_id, entities)?;
                    z
                }
            } else {
                return None;
            };

            let axis_point = point - z_axis * (point - Vec3::ZERO).dot(z_axis);
            let radial = point - axis_point;
            let len = radial.length();
            if len > 1e-6 { Some(radial / len) } else { Some(z_axis) }
        }
        EntityType::BSplineSurface | EntityType::BSplineSurfaceWithKnots => {
            // For B-spline, compute normal via surface tangent cross product
            let nurbs = build_nurbs_from_bspline_surface(surface, entities)?;
            let u_min = nurbs.knots_u[nurbs.degree_u];
            let u_max = nurbs.knots_u[nurbs.knots_u.len().saturating_sub(nurbs.degree_u + 1)];
            let v_min = nurbs.knots_v[nurbs.degree_v];
            let v_max = nurbs.knots_v[nurbs.knots_v.len().saturating_sub(nurbs.degree_v + 1)];

            // Find closest (u,v) parameter to the point by searching
            let mut best_u = u_min;
            let mut best_v = v_min;
            let mut best_dist = f32::MAX;
            let steps = 20;
            for i in 0..=steps {
                let u = u_min + (u_max - u_min) * i as f32 / steps as f32;
                for j in 0..=steps {
                    let v = v_min + (v_max - v_min) * j as f32 / steps as f32;
                    let p = nurbs.evaluate(u, v);
                    let d = (p - point).length();
                    if d < best_dist {
                        best_dist = d;
                        best_u = u;
                        best_v = v;
                    }
                }
            }

            // Compute tangent vectors
            let eps = 1e-4;
            let du = if best_u + eps <= u_max { eps } else { -eps };
            let dv = if best_v + eps <= v_max { eps } else { -eps };
            let p1 = nurbs.evaluate(best_u + du, best_v);
            let p2 = nurbs.evaluate(best_u, best_v + dv);
            let tangent_u = (p1 - point) / du;
            let tangent_v = (p2 - point) / dv;
            let normal = tangent_u.cross(tangent_v).normalize();
            if normal.length() > 1e-6 { Some(normal) } else { Some(Vec3::Z) }
        }
        _ => Some(Vec3::Z),
    }
}

/// Find the inner surface ID from a BOUNDED_SURFACE entity.
/// The actual surface is embedded in the params as a nested subtype.
fn find_bounded_surface_inner(bounded_id: u64, entities: &EntityIndex) -> Option<u64> {
    let record = entities.get(&bounded_id)?;
    // BOUNDED_SURFACE params contain a list of nested subtype parameters.
    // We look for references to other surface entities within the params.
    // For the specific case: (BOUNDED_SURFACE() B_SPLINE_SURFACE(...) ...)
    // the inner surface data is directly in the params as a list.
    let params_list = record.params.nth_param(1)?.as_list()?;
    // Look for the first REFINER entity id (starts with #) in the nested structure
    let wrapper = StepValue::List(params_list.to_vec());
    find_first_ref_id(&wrapper)
}

/// Recursively search a StepValue for the first entity reference ID.
fn find_first_ref_id(value: &StepValue) -> Option<u64> {
    match value {
        StepValue::Ref(id) => Some(*id),
        StepValue::List(items) => {
            for item in items {
                if let Some(id) = find_first_ref_id(item) {
                    return Some(id);
                }
            }
            None
        }
        _ => None,
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser;
    use super::super::topology;

    fn make_entities(data: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
            data
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_plane_tessellation_position_and_normal() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#5 = PLANE('', #4);
#6 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#7 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#8 = CARTESIAN_POINT('', (1.0, 1.0, 0.0));
#9 = CARTESIAN_POINT('', (0.0, 1.0, 0.0));
#10 = EDGE_CURVE('', #6, #7, #20, .T.);
#11 = EDGE_CURVE('', #7, #8, #20, .T.);
#12 = EDGE_CURVE('', #8, #9, #20, .T.);
#13 = EDGE_CURVE('', #9, #6, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = ADVANCED_FACE('', (#15), #5, .T.);
#17 = CLOSED_SHELL('', (#16));
#20 = LINE('', #6, #7);
",
        );
        let faces = topology::collect_shell_faces(&entities);
        assert_eq!(faces.len(), 1);
        let face = &faces[0];
        let mesh = tessellate_curved_face(face, &entities, None).unwrap();
        assert!(!mesh.vertices.is_empty(), "should produce vertices");
        assert!(!mesh.indices.is_empty(), "should produce indices");
        assert!(!mesh.normals.is_empty(), "should produce normals");

        // All vertices of a plane should lie on Z=0 (in this placement)
        for v in &mesh.vertices {
            assert!(v.z.abs() < 1e-3, "plane vertex {:?} should have z≈0", v);
        }

        // Normals should point roughly in +Z direction
        for n in &mesh.normals {
            assert!(n.z > 0.9, "plane normal {:?} should point +Z", n);
        }
    }

    #[test]
    fn test_cylinder_tessellation_radius_and_normal() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#5 = CYLINDRICAL_SURFACE('', #4, 2.0);
#6 = CARTESIAN_POINT('', (2.0, 0.0, 0.0));
#7 = CARTESIAN_POINT('', (2.0, 0.0, 1.0));
#8 = CARTESIAN_POINT('', (-2.0, 0.0, 1.0));
#9 = CARTESIAN_POINT('', (-2.0, 0.0, 0.0));
#10 = EDGE_CURVE('', #6, #7, #20, .T.);
#11 = EDGE_CURVE('', #7, #8, #20, .T.);
#12 = EDGE_CURVE('', #8, #9, #20, .T.);
#13 = EDGE_CURVE('', #9, #6, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = ADVANCED_FACE('', (#15), #5, .T.);
#17 = CLOSED_SHELL('', (#16));
#20 = LINE('', #6, #7);
",
        );
        let faces = topology::collect_shell_faces(&entities);
        assert_eq!(faces.len(), 1);
        let face = &faces[0];
        let mesh = tessellate_curved_face(face, &entities, None).unwrap();
        assert!(!mesh.vertices.is_empty(), "should produce vertices");
        assert!(!mesh.indices.is_empty(), "should produce indices");
        assert!(!mesh.normals.is_empty(), "should produce normals");

        // Cylinder vertices should have radius ≈ 2.0 around Z axis
        for v in &mesh.vertices {
            let r = (v.x * v.x + v.y * v.y).sqrt();
            assert!((r - 2.0).abs() < 0.5, "cylinder vertex {:?} should have radius≈2, got {}", v, r);
        }

        // Normals should point outward (horizontal)
        for n in &mesh.normals {
            assert!(n.z.abs() < 0.5, "cylinder normal {:?} should be mostly horizontal", n);
        }
    }

    #[test]
    fn test_cone_tessellation_radius_increases() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#5 = CONICAL_SURFACE('', #4, 1.0, 0.261799);
#6 = CARTESIAN_POINT('', (1.0, 0.0, 0.0));
#7 = CARTESIAN_POINT('', (1.2, 0.0, 1.0));
#8 = CARTESIAN_POINT('', (-1.2, 0.0, 1.0));
#9 = CARTESIAN_POINT('', (-1.0, 0.0, 0.0));
#10 = EDGE_CURVE('', #6, #7, #20, .T.);
#11 = EDGE_CURVE('', #7, #8, #20, .T.);
#12 = EDGE_CURVE('', #8, #9, #20, .T.);
#13 = EDGE_CURVE('', #9, #6, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = ADVANCED_FACE('', (#15), #5, .T.);
#17 = CLOSED_SHELL('', (#16));
#20 = LINE('', #6, #7);
",
        );
        let faces = topology::collect_shell_faces(&entities);
        assert_eq!(faces.len(), 1);
        let face = &faces[0];
        let mesh = tessellate_curved_face(face, &entities, None).unwrap();
        assert!(!mesh.vertices.is_empty(), "should produce vertices");
        assert!(!mesh.indices.is_empty(), "should produce indices");
        assert!(!mesh.normals.is_empty(), "should produce normals");

        // Cone base radius ≈ 1.0, top radius ≈ 1.2
        let mut min_r = f32::INFINITY;
        let mut max_r = 0.0f32;
        for v in &mesh.vertices {
            let r = (v.x * v.x + v.y * v.y).sqrt();
            min_r = min_r.min(r);
            max_r = max_r.max(r);
        }
        // Default cone range includes negative v where radius shrinks toward axis
        assert!(min_r < 1.0, "cone min radius should be < 1.0 (narrower at bottom), got {}", min_r);
        assert!(max_r > 1.0, "cone max radius should be > 1.0 (wider at top), got {}", max_r);

        // Normals should be roughly horizontal
        for n in &mesh.normals {
            assert!(n.z.abs() < 0.6, "cone normal {:?} should be mostly horizontal", n);
        }
    }

    #[test]
    fn test_torus_tessellation_major_minor_radius() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#5 = TOROIDAL_SURFACE('', #4, 3.0, 1.0);
#6 = CARTESIAN_POINT('', (4.0, 0.0, 0.0));
#7 = CARTESIAN_POINT('', (4.0, 0.0, 1.0));
#8 = CARTESIAN_POINT('', (2.0, 0.0, 1.0));
#9 = CARTESIAN_POINT('', (2.0, 0.0, 0.0));
#10 = EDGE_CURVE('', #6, #7, #20, .T.);
#11 = EDGE_CURVE('', #7, #8, #20, .T.);
#12 = EDGE_CURVE('', #8, #9, #20, .T.);
#13 = EDGE_CURVE('', #9, #6, #20, .T.);
#14 = EDGE_LOOP('', (#10, #11, #12, #13));
#15 = FACE_OUTER_BOUND('', #14, .T.);
#16 = ADVANCED_FACE('', (#15), #5, .T.);
#17 = CLOSED_SHELL('', (#16));
#20 = LINE('', #6, #7);
",
        );
        let faces = topology::collect_shell_faces(&entities);
        assert_eq!(faces.len(), 1);
        let face = &faces[0];
        let mesh = tessellate_curved_face(face, &entities, None).unwrap();
        assert!(!mesh.vertices.is_empty(), "should produce vertices");
        assert!(!mesh.indices.is_empty(), "should produce indices");
        assert!(!mesh.normals.is_empty(), "should produce normals");

        // Torus vertices should have radius between major-minor=2 and major+minor=4
        for v in &mesh.vertices {
            let r_xy = (v.x * v.x + v.y * v.y).sqrt();
            assert!(r_xy >= 1.5 && r_xy <= 4.5,
                "torus vertex {:?} should have XY radius in [2,4], got {}", v, r_xy);
            assert!(v.z.abs() <= 1.5,
                "torus vertex {:?} should have |z| <= 1, got {}", v, v.z);
        }

        // Normals can be vertical at top/bottom of minor circle — no strict directional check
    }

    #[test]
    fn test_rectangular_trimmed_surface_unwrap() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#5 = PLANE('', #4);
#6 = RECTANGULAR_TRIMMED_SURFACE('', #5, 0.0, 10.0, 0.0, 10.0);
#7 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#8 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#9 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#10 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#11 = EDGE_CURVE('', #7, #8, #20, .T.);
#12 = EDGE_CURVE('', #8, #9, #20, .T.);
#13 = EDGE_CURVE('', #9, #10, #20, .T.);
#14 = EDGE_CURVE('', #10, #7, #20, .T.);
#15 = EDGE_LOOP('', (#11, #12, #13, #14));
#16 = FACE_OUTER_BOUND('', #15, .T.);
#17 = ADVANCED_FACE('', (#16), #6, .T.);
#18 = CLOSED_SHELL('', (#17));
#20 = LINE('', #7, #8);
",
        );
        let faces = topology::collect_shell_faces(&entities);
        assert_eq!(faces.len(), 1);
        let face = &faces[0];
        let mesh = tessellate_curved_face(face, &entities, None).unwrap();
        assert!(!mesh.vertices.is_empty(), "RECTANGULAR_TRIMMED_SURFACE wrapping PLANE should produce vertices");
        assert!(!mesh.indices.is_empty(), "should produce indices");
        // Vertices should stay on Z=0 plane
        for v in &mesh.vertices {
            assert!(v.z.abs() < 1e-3, "vertex {:?} should lie on Z=0 plane", v);
        }
    }
}
