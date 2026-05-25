//! Face selection per boolean operation type.
//!
//! After classification, faces are selected based on the operation:
//! - Union: faces outside the other solid + boundary
//! - Intersection: faces inside the other solid + boundary
//! - Difference (A-B): faces of A outside B + boundary of B inside A (inverted)

use super::super::topology::StepFace;
use super::classify::{ClassifiedFace, RegionClass};
use super::BoolOp;

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
mod tests {
    use super::*;
    use super::super::super::topology::StepFace;

    fn make_face() -> StepFace {
        StepFace {
            bounds: vec![],
            surface_id: None,
            same_sense: true,
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
