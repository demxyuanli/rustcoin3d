//! Face splitting along intersection curves (topology-aware).
//!
//! Two layers:
//! 1. Legacy topology-based splitting (StepShell/StepFace) for the old pipeline.
//! 2. B-Rep registry-based splitting (BRepRegistry/FaceKey) for the Phase 3 boolean pipeline.
//!
//! For each intersection curve between two faces, we:
//! 1. Project intersection points onto the UV domain of each face
//! 2. Split the 2D trim loop polygon along the projected curve
//! 3. Create new polygon loops for each resulting face region

use rc3d_core::math::Vec3;
use super::super::parser::EntityIndex;
use super::super::topology::{StepShell, StepFace, StepEdge, StepLoop};
use super::super::entity_types::EntityType;
use super::super::geom;
use super::intersect::{FaceIntersection, IntersectionCurve};

// ── B-Rep registry-based splitting (Phase 3) ──────────────────────────────

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{FaceKey, ShellKey, BRepFace, BRepWire};

/// A curve where two faces intersect, parameterized on both surfaces.
#[derive(Debug, Clone)]
pub struct BRepIntersectionCurve {
    /// 3D polyline approximation of the intersection.
    pub points_3d: Vec<Vec3>,
    /// Parameter pairs (u,v) on face A (native surface parameters).
    pub params_a: Vec<(f32, f32)>,
    /// Parameter pairs (u,v) on face B (native surface parameters).
    pub params_b: Vec<(f32, f32)>,
    /// Face keys involved.
    pub face_a: FaceKey,
    pub face_b: FaceKey,
}

/// Result of splitting a face along intersection curves.
#[derive(Debug, Clone)]
pub struct SplitFaceRegion {
    pub original_face: FaceKey,
    pub sub_faces: Vec<SubFaceRegion>,
}

/// A sub-region of a split face with UV boundary and interior sample point.
#[derive(Debug, Clone)]
pub struct SubFaceRegion {
    /// Boundary loop in UV space (outer boundary + holes).
    pub uv_boundary: Vec<Vec<(f32, f32)>>,
    /// Interior sample point for classification (native UV).
    pub interior_point: (f32, f32),
    /// 3D point corresponding to interior_point.
    pub interior_point_3d: Vec3,
    /// The original face key this sub-region belongs to.
    pub original_face: FaceKey,
}

/// Compute B-Rep intersection curves from `FaceIntersectionResult`.
/// Samples 3D points along the analytic intersection and projects to UV on both faces.
pub fn compute_brep_intersection_curves(
    intersections: &[super::intersect::FaceIntersectionResult],
    reg: &BRepRegistry,
) -> Vec<BRepIntersectionCurve> {
    let mut out = Vec::new();
    for fi in intersections {
        let face_a = match reg.faces.get(fi.face_a) { Some(f) => f, None => continue };
        let face_b = match reg.faces.get(fi.face_b) { Some(f) => f, None => continue };

        for curve in &fi.curves_3d {
            let samples = sample_intersection_curve(curve, 32);
            if samples.is_empty() { continue; }

            let mut pts_3d = Vec::with_capacity(samples.len());
            let mut params_a = Vec::with_capacity(samples.len());
            let mut params_b = Vec::with_capacity(samples.len());

            for &pt in &samples {
                if let Some(uv_a) = face_a.surface.project(pt) {
                    if let Some(uv_b) = face_b.surface.project(pt) {
                        pts_3d.push(pt);
                        params_a.push(uv_a);
                        params_b.push(uv_b);
                    }
                }
            }

            if pts_3d.len() >= 2 {
                out.push(BRepIntersectionCurve {
                    points_3d: pts_3d,
                    params_a,
                    params_b,
                    face_a: fi.face_a,
                    face_b: fi.face_b,
                });
            }
        }
    }
    out
}

/// Sample points along an analytic intersection curve.
fn sample_intersection_curve(curve: &crate::step::brep::geom::CurveGeom, n: usize) -> Vec<Vec3> {
    use crate::step::brep::geom::CurveGeom;
    match curve {
        CurveGeom::Line { origin, direction } => {
            // Sample a finite segment of the line
            let ext = 100.0; // generous extent for bounded faces
            (0..n).map(|i| {
                let t = -ext + 2.0 * ext * i as f32 / (n - 1) as f32;
                *origin + *direction * t
            }).collect()
        }
        CurveGeom::Circle { center, x_dir, y_dir, radius, .. } => {
            (0..n).map(|i| {
                let theta = std::f32::consts::TAU * i as f32 / n as f32;
                *center + *x_dir * (*radius * theta.cos()) + *y_dir * (*radius * theta.sin())
            }).collect()
        }
        _ => Vec::new(),
    }
}

/// Split all faces of given shells along the intersection curves.
pub fn split_all_faces_brep(
    shells: &[ShellKey],
    curves: &[BRepIntersectionCurve],
    reg: &BRepRegistry,
) -> Vec<SplitFaceRegion> {
    let mut results = Vec::new();
    for &sk in shells {
        let shell = match reg.shells.get(sk) { Some(s) => s, None => continue };
        for &(face_key, _) in &shell.faces {
            let face_curves: Vec<&BRepIntersectionCurve> = curves.iter()
                .filter(|c| c.face_a == face_key || c.face_b == face_key)
                .collect();
            let regions = split_face_along_curves(face_key, &face_curves, reg);
            results.push(SplitFaceRegion {
                original_face: face_key,
                sub_faces: regions,
            });
        }
    }
    results
}

/// Split a B-Rep face along intersection curves in UV space.
/// The intersection curves become new boundary edges.
pub fn split_face_along_curves(
    face_key: FaceKey,
    curves: &[&BRepIntersectionCurve],
    reg: &BRepRegistry,
) -> Vec<SubFaceRegion> {
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return vec![],
    };

    if curves.is_empty() {
        // No split needed — entire face is one region
        return vec![whole_face_region(face_key, face)];
    }

    // Phase 3 MVP: support single intersection curve splitting face into 2 regions
    split_face_single_curve(face_key, face, curves[0], reg)
}

/// Create a SubFaceRegion for the entire unsplit face.
fn whole_face_region(face_key: FaceKey, face: &BRepFace) -> SubFaceRegion {
    let range = face.surface.param_range();
    let mid_u = (range.u_min + range.u_max) * 0.5;
    let mid_v = (range.v_min + range.v_max) * 0.5;
    let (un, vn) = face.surface.native_uv_to_d0(mid_u, mid_v);
    let interior_3d = face.surface.d0(un, vn);

    SubFaceRegion {
        uv_boundary: vec![], // empty = use original face boundary
        interior_point: (mid_u, mid_v),
        interior_point_3d: interior_3d,
        original_face: face_key,
    }
}

/// Split a face with a single intersection curve that traverses the face.
/// Creates two sub-regions separated by the curve.
fn split_face_single_curve(
    face_key: FaceKey,
    face: &BRepFace,
    curve: &BRepIntersectionCurve,
    _reg: &BRepRegistry,
) -> Vec<SubFaceRegion> {
    // Get UV params on this face from the intersection curve
    let params = if curve.face_a == face_key {
        &curve.params_a
    } else {
        &curve.params_b
    };

    if params.len() < 2 {
        return vec![whole_face_region(face_key, face)];
    }

    // Compute two interior sample points on opposite sides of the splitting curve.
    // Use the curve midpoint and offset perpendicular to the curve direction in UV.
    let mid_idx = params.len() / 2;
    let (cu, cv) = params[mid_idx];

    // Compute curve tangent direction in UV at midpoint
    let tangent = if mid_idx > 0 && mid_idx + 1 < params.len() {
        let (u0, v0) = params[mid_idx - 1];
        let (u1, v1) = params[mid_idx + 1];
        (u1 - u0, v1 - v0)
    } else if mid_idx > 0 {
        let (u0, v0) = params[mid_idx - 1];
        (cu - u0, cv - v0)
    } else {
        let (u1, v1) = params[1];
        (u1 - cu, v1 - cv)
    };

    let t_len = (tangent.0 * tangent.0 + tangent.1 * tangent.1).sqrt();
    if t_len < 1e-12 {
        return vec![whole_face_region(face_key, face)];
    }

    // Perpendicular in UV (rotate 90°)
    let perp = (-tangent.1 / t_len, tangent.0 / t_len);
    let offset = 0.01; // small offset in UV space

    let range = face.surface.param_range();
    let u_span = range.u_span();
    let v_span = range.v_span();
    let du = perp.0 * offset * u_span;
    let dv = perp.1 * offset * v_span;

    // Two sample points on opposite sides
    let side_a_uv = (cu + du, cv + dv);
    let side_b_uv = (cu - du, cv - dv);

    let (un_a, vn_a) = face.surface.native_uv_to_d0(side_a_uv.0, side_a_uv.1);
    let (un_b, vn_b) = face.surface.native_uv_to_d0(side_b_uv.0, side_b_uv.1);
    let pt_a = face.surface.d0(un_a, vn_a);
    let pt_b = face.surface.d0(un_b, vn_b);

    // Build UV boundary polygons for each sub-region using the curve + face bounds
    let uv_poly_a: Vec<(f32, f32)> = params.iter().copied().collect();
    let uv_poly_b: Vec<(f32, f32)> = params.iter().rev().copied().collect();

    vec![
        SubFaceRegion {
            uv_boundary: vec![uv_poly_a],
            interior_point: side_a_uv,
            interior_point_3d: pt_a,
            original_face: face_key,
        },
        SubFaceRegion {
            uv_boundary: vec![uv_poly_b],
            interior_point: side_b_uv,
            interior_point_3d: pt_b,
            original_face: face_key,
        },
    ]
}

#[cfg(test)]
mod brep_split_tests {
    use super::*;
    use crate::step::brep::topo::*;
    use crate::step::brep::geom::SurfaceGeom;

    /// Build a simple plane face in the registry for testing.
    fn make_plane_face(
        reg: &mut BRepRegistry,
        origin: Vec3, normal: Vec3, u_dir: Vec3,
    ) -> FaceKey {
        let surface = SurfaceGeom::Plane { origin, normal, u_dir };
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        reg.faces.insert(BRepFace {
            surface,
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        })
    }

    #[test]
    fn test_whole_face_region_no_curves() {
        let mut reg = BRepRegistry::new();
        let fk = make_plane_face(
            &mut reg,
            Vec3::ZERO, Vec3::Z, Vec3::X,
        );
        let result = split_face_along_curves(fk, &[], &reg);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].original_face, fk);
    }

    #[test]
    fn test_split_two_planes() {
        let mut reg = BRepRegistry::new();
        let face_a = make_plane_face(
            &mut reg,
            Vec3::ZERO, Vec3::Z, Vec3::X,
        );
        let _face_b = make_plane_face(
            &mut reg,
            Vec3::ZERO, Vec3::X, Vec3::Z,
        );

        // Simulate an intersection curve along Y axis (where Z=0 plane meets X=0 plane)
        let curve = BRepIntersectionCurve {
            points_3d: vec![
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            params_a: vec![(0.0, -1.0), (0.0, 0.0), (0.0, 1.0)],
            params_b: vec![(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
            face_a,
            face_b: _face_b,
        };

        let curves_ref = vec![&curve];
        let result = split_face_along_curves(face_a, &curves_ref, &reg);
        assert_eq!(result.len(), 2, "Single curve should split plane into 2 regions");
        // Interior points should be on opposite sides
        let a = result[0].interior_point;
        let b = result[1].interior_point;
        assert!((a.0 - b.0).abs() > 1e-6 || (a.1 - b.1).abs() > 1e-6,
            "Interior points should differ: ({}, {}) vs ({}, {})", a.0, a.1, b.0, b.1);
    }
}

// ── Legacy topology-based splitting ────────────────────────────────────────

/// Split faces along intersection curves.
pub fn split_faces(
    a_shells: &[StepShell],
    b_shells: &[StepShell],
    entities_a: &EntityIndex,
    _entities_b: &EntityIndex,
    intersections: &[FaceIntersection],
) -> (Vec<StepFace>, Vec<StepFace>) {
    let faces_a: Vec<StepFace> = a_shells.iter().flat_map(|s| s.faces.iter().cloned()).collect();
    let faces_b: Vec<StepFace> = b_shells.iter().flat_map(|s| s.faces.iter().cloned()).collect();

    if intersections.is_empty() {
        return (faces_a, faces_b);
    }

    let mut result_a: Vec<StepFace> = faces_a.clone();
    let result_b: Vec<StepFace> = faces_b.clone();

    for inter in intersections {
        if inter.face_a < faces_a.len() && inter.face_b < faces_b.len() {
            // For now: try to split face A along the intersection curve
            if let Some(split_regions) = split_face_by_curve(
                &faces_a[inter.face_a], &inter.curves, entities_a,
            ) {
                // Replace the original face with split regions
                let mut new_faces_a = Vec::new();
                for (i, face) in faces_a.iter().enumerate() {
                    if i == inter.face_a {
                        new_faces_a.extend(split_regions.clone());
                    } else {
                        new_faces_a.push(face.clone());
                    }
                }
                result_a = new_faces_a;
            }
        }
    }

    (result_a, result_b)
}

/// Project a 3D point onto a surface's UV domain for analytic surface types.
pub fn project_point_to_surface_uv(
    point: Vec3, surface: &super::super::parser::EntityRecord, entities: &EntityIndex,
) -> Option<(f32, f32)> {
    match surface.entity_type {
        EntityType::Plane => {
            let placement_id = geom::nth_ref(&surface.params, 1)?;
            let (origin, x_axis, z_axis) =
                super::super::topology::resolve_placement(placement_id, entities)?;
            let y_axis = z_axis.cross(x_axis).normalize();
            let rel = point - origin;
            Some((rel.dot(x_axis), rel.dot(y_axis)))
        }
        EntityType::CylindricalSurface => {
            let placement_id = geom::nth_ref(&surface.params, 1)?;
            let (origin, _x, z_axis) =
                super::super::topology::resolve_placement(placement_id, entities)?;
            let axis = z_axis.normalize();
            let rel = point - origin;
            let v = rel.dot(axis);
            let radial = rel - axis * v;
            let r = radial.length();
            if r < 1e-10 { return None; }
            let mut u = radial.y.atan2(radial.x);
            if u < 0.0 { u += 2.0 * std::f32::consts::PI; }
            Some((u, v))
        }
        EntityType::SphericalSurface => {
            let placement_id = geom::nth_ref(&surface.params, 1)?;
            let (origin, _x, _z_axis) =
                super::super::topology::resolve_placement(placement_id, entities)?;
            let rel = point - origin;
            let r = rel.length();
            if r < 1e-10 { return None; }
            let v = (rel.z / r).acos();
            let mut u = rel.y.atan2(rel.x);
            if u < 0.0 { u += 2.0 * std::f32::consts::PI; }
            Some((u, v))
        }
        EntityType::ConicalSurface => {
            let placement_id = geom::nth_ref(&surface.params, 1)?;
            let (origin, _x, z_axis) =
                super::super::topology::resolve_placement(placement_id, entities)?;
            let axis = z_axis.normalize();
            let rel = point - origin;
            let v = rel.dot(axis);
            let radial = rel - axis * v;
            let r = radial.length();
            if r < 1e-10 { return None; }
            let mut u = radial.y.atan2(radial.x);
            if u < 0.0 { u += 2.0 * std::f32::consts::PI; }
            Some((u, v))
        }
        _ => None,
    }
}

/// Split a single face by projecting 3D intersection curve points
/// into the face's UV domain.
fn split_face_by_curve(
    face: &StepFace,
    curves: &[IntersectionCurve],
    entities: &EntityIndex,
) -> Option<Vec<StepFace>> {
    if curves.is_empty() { return None; }

    let surface_id = face.surface_id?;
    let surface = entities.get(&surface_id)?;

    // Project intersection points to UV domain
    let uv_curves: Vec<Vec<(f32, f32)>> = curves.iter().map(|curve| {
        let pts: Vec<(f32, f32)> = curve.points.iter().filter_map(|pt| {
            project_point_to_surface_uv(*pt, surface, entities)
        }).collect();
        pts
    }).filter(|v| v.len() >= 2).collect();

    if uv_curves.is_empty() { return None; }

    // For now, return the original face marked with additional edges from the
    // intersection curve (approximation — full clipping needs polygon boolean ops)
    let mut new_edges = Vec::new();
    for uv_curve in &uv_curves {
        for pts in uv_curve.windows(2) {
            new_edges.push(StepEdge {
                start: Vec3::new(pts[0].0, pts[0].1, 0.0),
                end: Vec3::new(pts[1].0, pts[1].1, 0.0),
                curve_id: 0,
                curve_type: "LINE".into(),
                reversed: false,
                tolerance: 1e-4,
            });
        }
    }

    let mut new_face = face.clone();
    if !new_edges.is_empty() {
        new_face.bounds.push(StepLoop {
            edges: new_edges,
            vertex_loop_point: None,
        });
    }

    Some(vec![new_face])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::parser;

    fn make_entities(data: &str) -> EntityIndex {
        let input = format!(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n", data
        );
        parser::parse_exchange(&input).unwrap().entities
    }

    #[test]
    fn test_project_plane_point_to_uv() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#5 = PLANE('', #4);\
");
        let surface = entities.get(&5).unwrap();
        let pt = Vec3::new(3.0, 4.0, 0.0);
        let (u, v) = project_point_to_surface_uv(pt, surface, &entities).unwrap();
        assert!((u - 3.0).abs() < 1e-4, "u should be 3.0, got {u}");
        assert!((v - 4.0).abs() < 1e-4, "v should be 4.0, got {v}");
    }

    #[test]
    fn test_project_cylinder_point_to_uv() {
        let entities = make_entities(
            "\
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = DIRECTION('', (1.0, 0.0, 0.0));
#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);
#5 = CYLINDRICAL_SURFACE('', #4, 2.0);\
");
        let surface = entities.get(&5).unwrap();
        let pt = Vec3::new(2.0, 0.0, 5.0);
        let (u, v) = project_point_to_surface_uv(pt, surface, &entities).unwrap();
        assert!((v - 5.0).abs() < 1e-4, "v should be 5.0 at height 5, got {v}");
    }
}
