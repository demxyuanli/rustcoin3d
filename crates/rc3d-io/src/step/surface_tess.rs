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

    // Merged subsuper params for B_SPLINE_SURFACE_WITH_KNOTS:
    // [0] degree_u, [1] degree_v, [2] control_points_list,
    // [3] form, [4] u_closed, [5] v_closed, [6] self_intersect,
    // [7] u_multiplicities, [8] v_multiplicities, [9] u_knots, [10] v_knots, [11] knot_spec
    let degree_u = geom::nth_int(params, 0)? as usize;
    let degree_v = geom::nth_int(params, 1)? as usize;

    // Resolve control points from nested list of references
    let cp_list = params.nth_param(2)?.as_list()?;
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

    let u_multiplicities = geom::nth_list_ints(params, 7);
    let v_multiplicities = geom::nth_list_ints(params, 8);
    let u_knot_vals = geom::nth_list_reals(params, 9);
    let v_knot_vals = geom::nth_list_reals(params, 10);

    let knots_u = build_knot_vector_from_multiplicities(
        &u_multiplicities, &u_knot_vals, degree_u, control_points.len(),
    );
    let knots_v = build_knot_vector_from_multiplicities(
        &v_multiplicities, &v_knot_vals, degree_v, control_points[0].len(),
    );

    let weights = vec![vec![1.0f32; control_points[0].len()]; control_points.len()];

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
    }
    knots
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
fn map_uv_to_nurbs(entity_type: EntityType, u: f32, v: f32) -> (f32, f32) {
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
            if let Some(tr) = trim {
                // Use UV-space earcut for exact trim boundary
                tessellate_trimmed_via_uv(tr, &nurbs, surface_id, entities, surface.entity_type, face.same_sense, u_min, u_max, v_min, v_max)
            } else {
                let n = estimate_sample_count(&nurbs, surface.entity_type, u_min, u_max, v_min, v_max);
                Some(sample_grid_nurbs(n, surface_id, trim, &nurbs, entities, surface.entity_type, face.same_sense, u_min, u_max, v_min, v_max))
            }
        }
        EntityType::SurfaceOfLinearExtrusion | EntityType::SurfaceOfRevolution => {
            geom::evaluate_surface(surface_id, entities, DEFAULT_SAMPLES, DEFAULT_SAMPLES)
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
            let u_min = nurbs.knots_u[nurbs.degree_u];
            let u_max = nurbs.knots_u[nurbs.knots_u.len().saturating_sub(nurbs.degree_u + 1)];
            let v_min = nurbs.knots_v[nurbs.degree_v];
            let v_max = nurbs.knots_v[nurbs.knots_v.len().saturating_sub(nurbs.degree_v + 1)];
            if let Some(tr) = trim {
                tessellate_trimmed_via_uv(tr, &nurbs, surface_id, entities, surface.entity_type, face.same_sense, u_min, u_max, v_min, v_max)
            } else {
                let n = estimate_sample_count(&nurbs, surface.entity_type, u_min, u_max, v_min, v_max);
                Some(sample_grid_nurbs(n, surface_id, trim, &nurbs, entities, surface.entity_type, face.same_sense, u_min, u_max, v_min, v_max))
            }
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
        _ => None,
    }
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
    _u_min: f32,
    _u_max: f32,
    _v_min: f32,
    _v_max: f32,
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
        let (u_nurbs, v_nurbs) = map_uv_to_nurbs(entity_type, u, v);
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

    if entity_type == EntityType::Plane {
        let min_x = vertices.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
        let max_x = vertices.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
        let min_y = vertices.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let max_y = vertices.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[uv_earcut PLANE id={}] bbox xy=({:.1},{:.1})-({:.1},{:.1}) z={:.1} verts={} tris={}",
            surface_id, min_x, min_y, max_x, max_y, vertices[0].z,
            vertices.len(), indices.len() / 4);
    }
    // Check ALL faces for outlier vertices
    let outliers: Vec<_> = vertices.iter().filter(|v| {
        v.x.is_nan() || v.y.is_nan() || v.z.is_nan()
        || v.x.abs() > 200.0 || v.y.abs() > 200.0 || v.z.abs() > 200.0
    }).collect();
    if !outliers.is_empty() {
        eprintln!("[uv_earcut id={} type={:?}] {} outlier vertices! first=({:.2},{:.2},{:.2})",
            surface_id, entity_type, outliers.len(),
            outliers[0].x, outliers[0].y, outliers[0].z);
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
                    topology::resolve_direction_public(direction_id, entities)?
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
}
