//! Face splitting along intersection curves (topology-aware).
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
        new_face.bounds.push(StepLoop { edges: new_edges });
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
