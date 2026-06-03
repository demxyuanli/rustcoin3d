//! Face selection per boolean operation type (B-Rep native).

use crate::store::BRepStore;
use crate::topo::FaceKey;
use super::classify::PointClassification;
use super::split::SplitFaceRegion;
use super::BoolOp;

/// Select B-Rep faces for the boolean result based on operation type.
/// Each sub-face is individually classified; new B-Rep faces are created
/// for sub-faces that have a non-empty UV boundary.
pub fn select_brep_faces(
    regions_a: &[(usize, Vec<PointClassification>)],
    regions_b: &[(usize, Vec<PointClassification>)],
    split_a: &[SplitFaceRegion],
    split_b: &[SplitFaceRegion],
    op: BoolOp,
    reg: &mut BRepStore,
) -> Vec<FaceKey> {
    let mut selected = Vec::new();

    for (i, classes) in regions_a {
        let i = *i;
        if i >= split_a.len() { continue; }
        for (j, sub) in split_a[i].sub_faces.iter().enumerate() {
            let class = classes.get(j).copied().unwrap_or(PointClassification::Outside);
            let keep = match op {
                BoolOp::Union => class == PointClassification::Outside,
                BoolOp::Intersection => class == PointClassification::Inside,
                BoolOp::Difference => class == PointClassification::Outside,
            };
            if keep {
                push_sub_face(sub, reg, &mut selected);
            }
        }
    }

    for (i, classes) in regions_b {
        let i = *i;
        if i >= split_b.len() { continue; }
        for (j, sub) in split_b[i].sub_faces.iter().enumerate() {
            let class = classes.get(j).copied().unwrap_or(PointClassification::Outside);
            let keep = match op {
                BoolOp::Union => class == PointClassification::Outside,
                BoolOp::Intersection => class == PointClassification::Inside,
                BoolOp::Difference => class == PointClassification::Inside,
            };
            if keep {
                push_sub_face(sub, reg, &mut selected);
            }
        }
    }

    selected
}

fn push_sub_face(
    sub: &super::split::SubFaceRegion,
    reg: &mut BRepStore,
    selected: &mut Vec<FaceKey>,
) {
    if !sub.uv_boundary.is_empty() {
        if let Some(new_fk) = super::split::create_sub_face(
            sub.original_face,
            &sub.uv_boundary[0],
            reg,
        ) {
            selected.push(new_fk);
        }
    } else {
        // No boundary = whole face region, keep original
        selected.push(sub.original_face);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::BRepStore;
    use crate::topo::BRepWire;
    use crate::geom::SurfaceGeom;
    use crate::topo::BRepFace;
    use super::super::split::SubFaceRegion;

    fn make_dummy_sub_face(reg: &mut BRepStore) -> FaceKey {
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
        let mut reg = BRepStore::new();
        let fk_a = make_dummy_sub_face(&mut reg);
        let fk_b = make_dummy_sub_face(&mut reg);

        let split_a = vec![SplitFaceRegion {
            original_face: fk_a,
            sub_faces: vec![SubFaceRegion {
                uv_boundary: vec![],
                interior_point: (0.5, 0.5),
                interior_point_3d: rc3d_core::math::Vec3::ZERO,
                original_face: fk_a,
            }],
        }];
        let split_b = vec![SplitFaceRegion {
            original_face: fk_b,
            sub_faces: vec![SubFaceRegion {
                uv_boundary: vec![],
                interior_point: (0.5, 0.5),
                interior_point_3d: rc3d_core::math::Vec3::ZERO,
                original_face: fk_b,
            }],
        }];

        let regions_a = vec![(0, vec![PointClassification::Outside])];
        let regions_b = vec![(0, vec![PointClassification::Inside])];
        let selected = select_brep_faces(&regions_a, &regions_b, &split_a, &split_b, BoolOp::Union, &mut reg);
        assert_eq!(selected.len(), 1, "Union: only A outside should be selected");
        assert_eq!(selected[0], fk_a);
    }

    #[test]
    fn test_brep_intersection_selects_inside() {
        let mut reg = BRepStore::new();
        let fk_a = make_dummy_sub_face(&mut reg);
        let fk_b = make_dummy_sub_face(&mut reg);

        let split_a = vec![SplitFaceRegion {
            original_face: fk_a,
            sub_faces: vec![SubFaceRegion {
                uv_boundary: vec![],
                interior_point: (0.5, 0.5),
                interior_point_3d: rc3d_core::math::Vec3::ZERO,
                original_face: fk_a,
            }],
        }];
        let split_b = vec![SplitFaceRegion {
            original_face: fk_b,
            sub_faces: vec![SubFaceRegion {
                uv_boundary: vec![],
                interior_point: (0.5, 0.5),
                interior_point_3d: rc3d_core::math::Vec3::ZERO,
                original_face: fk_b,
            }],
        }];

        let regions_a = vec![(0, vec![PointClassification::Inside])];
        let regions_b = vec![(0, vec![PointClassification::Inside])];
        let selected = select_brep_faces(&regions_a, &regions_b, &split_a, &split_b, BoolOp::Intersection, &mut reg);
        assert_eq!(selected.len(), 2, "Intersection: both inside should be selected");
    }
}
