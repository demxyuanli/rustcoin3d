//! Face selection per boolean operation type.
//!
//! After classification, faces are selected based on the operation:
//! - Union: faces outside the other solid + boundary
//! - Intersection: faces inside the other solid + boundary
//! - Difference (A-B): faces of A outside B + boundary of B inside A (inverted)

use super::super::topology::StepFace;
use super::classify::{ClassifiedFace, RegionClass};
use super::BoolOp;

// ── B-Rep registry-based face selection (Phase 3) ─────────────────────────

use crate::step::brep::topo::FaceKey;
use super::classify::PointClassification;
use super::split::SplitFaceRegion;

/// Select B-Rep faces for the boolean result based on operation type.
pub fn select_brep_faces(
    regions_a: &[(usize, PointClassification)],
    regions_b: &[(usize, PointClassification)],
    split_a: &[SplitFaceRegion],
    split_b: &[SplitFaceRegion],
    op: BoolOp,
) -> Vec<FaceKey> {
    let mut selected = Vec::new();

    // Select from A
    for &(i, class) in regions_a {
        let keep = match op {
            BoolOp::Union => class == PointClassification::Outside,
            BoolOp::Intersection => class == PointClassification::Inside,
            BoolOp::Difference => class == PointClassification::Outside,
        };
        if keep && i < split_a.len() {
            for sub in &split_a[i].sub_faces {
                selected.push(sub.original_face);
            }
        }
    }

    // Select from B
    for &(i, class) in regions_b {
        let keep = match op {
            BoolOp::Union => class == PointClassification::Outside,
            BoolOp::Intersection => class == PointClassification::Inside,
            BoolOp::Difference => class == PointClassification::Inside, // inverted
        };
        if keep && i < split_b.len() {
            for sub in &split_b[i].sub_faces {
                selected.push(sub.original_face);
            }
        }
    }

    selected
}

// ── Legacy topology-based face selection ────────────────────────────────────

/// Select faces for the output solid based on the boolean operation.
pub fn select_faces(
    classified_a: &[ClassifiedFace],
    classified_b: &[ClassifiedFace],
    op: BoolOp,
) -> Vec<StepFace> {
    match op {
        BoolOp::Union => {
            // Keep faces outside the other solid + boundary faces
            let mut result: Vec<StepFace> = classified_a.iter()
                .filter(|cf| cf.class == RegionClass::Outside || cf.class == RegionClass::OnBoundary)
                .map(|cf| cf.face.clone())
                .collect();
            result.extend(
                classified_b.iter()
                    .filter(|cf| cf.class == RegionClass::Outside || cf.class == RegionClass::OnBoundary)
                    .map(|cf| cf.face.clone())
            );
            result
        }
        BoolOp::Intersection => {
            // Keep faces inside the other solid + boundary faces
            let mut result: Vec<StepFace> = classified_a.iter()
                .filter(|cf| cf.class == RegionClass::Inside || cf.class == RegionClass::OnBoundary)
                .map(|cf| cf.face.clone())
                .collect();
            result.extend(
                classified_b.iter()
                    .filter(|cf| cf.class == RegionClass::Inside || cf.class == RegionClass::OnBoundary)
                    .map(|cf| cf.face.clone())
            );
            result
        }
        BoolOp::Difference => {
            // A - B: faces of A outside B + faces of B inside A (inverted/reversed)
            let mut result: Vec<StepFace> = classified_a.iter()
                .filter(|cf| cf.class == RegionClass::Outside || cf.class == RegionClass::OnBoundary)
                .map(|cf| cf.face.clone())
                .collect();
            // Faces of B that are inside A need to be inverted
            for cf in classified_b.iter() {
                if cf.class == RegionClass::Inside || cf.class == RegionClass::OnBoundary {
                    let mut inverted = cf.face.clone();
                    inverted.same_sense = !inverted.same_sense;
                    result.push(inverted);
                }
            }
            result
        }
    }
}

#[cfg(test)]
mod brep_select_tests {
    use super::*;
    use crate::step::brep::topo::FaceKey;
    use crate::step::brep::registry::BRepRegistry;
    use crate::step::brep::topo::BRepShell;

    fn make_dummy_sub_face(reg: &mut BRepRegistry) -> FaceKey {
        use crate::step::brep::topo::{BRepFace, BRepWire};
        use crate::step::brep::geom::SurfaceGeom;
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane { origin: rc3d_core::math::Vec3::ZERO, normal: rc3d_core::math::Vec3::Z, u_dir: rc3d_core::math::Vec3::X },
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
    fn test_brep_union_selects_outside() {
        let mut reg = BRepRegistry::new();
        let fk_a = make_dummy_sub_face(&mut reg);
        let fk_b = make_dummy_sub_face(&mut reg);

        let split_a = vec![SplitFaceRegion {
            original_face: fk_a,
            sub_faces: vec![super::super::split::SubFaceRegion {
                uv_boundary: vec![],
                interior_point: (0.5, 0.5),
                interior_point_3d: rc3d_core::math::Vec3::ZERO,
                original_face: fk_a,
            }],
        }];
        let split_b = vec![SplitFaceRegion {
            original_face: fk_b,
            sub_faces: vec![super::super::split::SubFaceRegion {
                uv_boundary: vec![],
                interior_point: (0.5, 0.5),
                interior_point_3d: rc3d_core::math::Vec3::ZERO,
                original_face: fk_b,
            }],
        }];

        let regions_a = vec![(0, PointClassification::Outside)];
        let regions_b = vec![(0, PointClassification::Inside)]; // B inside A

        let selected = select_brep_faces(&regions_a, &regions_b, &split_a, &split_b, BoolOp::Union);
        assert_eq!(selected.len(), 1, "Union: only A outside should be selected");
        assert_eq!(selected[0], fk_a);
    }

    #[test]
    fn test_brep_intersection_selects_inside() {
        let mut reg = BRepRegistry::new();
        let fk_a = make_dummy_sub_face(&mut reg);
        let fk_b = make_dummy_sub_face(&mut reg);

        let split_a = vec![SplitFaceRegion {
            original_face: fk_a,
            sub_faces: vec![super::super::split::SubFaceRegion {
                uv_boundary: vec![],
                interior_point: (0.5, 0.5),
                interior_point_3d: rc3d_core::math::Vec3::ZERO,
                original_face: fk_a,
            }],
        }];
        let split_b = vec![SplitFaceRegion {
            original_face: fk_b,
            sub_faces: vec![super::super::split::SubFaceRegion {
                uv_boundary: vec![],
                interior_point: (0.5, 0.5),
                interior_point_3d: rc3d_core::math::Vec3::ZERO,
                original_face: fk_b,
            }],
        }];

        let regions_a = vec![(0, PointClassification::Inside)];
        let regions_b = vec![(0, PointClassification::Inside)];

        let selected = select_brep_faces(&regions_a, &regions_b, &split_a, &split_b, BoolOp::Intersection);
        assert_eq!(selected.len(), 2, "Intersection: both inside should be selected");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::topology::StepFace;

    fn make_face() -> StepFace {
        StepFace {
            bounds: vec![],
            surface_id: None,
            same_sense: true,
            face_id: None,
        }
    }

    #[test]
    fn test_union_selects_outside() {
        let a = ClassifiedFace { face: make_face(), class: RegionClass::Outside };
        let b = ClassifiedFace { face: make_face(), class: RegionClass::Inside };
        let result = select_faces(&[a], &[b], BoolOp::Union);
        assert_eq!(result.len(), 1, "union should select outside (1 face) and reject inside");
    }

    #[test]
    fn test_intersection_selects_inside() {
        let a = ClassifiedFace { face: make_face(), class: RegionClass::Outside };
        let b = ClassifiedFace { face: make_face(), class: RegionClass::Inside };
        let result = select_faces(&[a], &[b], BoolOp::Intersection);
        assert_eq!(result.len(), 1, "intersection should select inside face");
    }

    #[test]
    fn test_difference_inverts_b_faces() {
        let a = ClassifiedFace { face: make_face(), class: RegionClass::Outside };
        let b = ClassifiedFace { face: make_face(), class: RegionClass::Inside };
        let result = select_faces(&[a], &[b], BoolOp::Difference);
        assert_eq!(result.len(), 2);
        // B face should be inverted
        assert!(!result[1].same_sense, "B face inside A should be inverted");
    }
}
