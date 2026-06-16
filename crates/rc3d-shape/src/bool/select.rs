//! Face selection per boolean operation type.
//!
//! OCC alignment: BOPAlgo_BOP — classifies each split face region and
//! selects faces based on the boolean operation type.
//!
//! Uses the new BuilderFace path to create BRep faces from UV regions.

use crate::store::BRepStore;
use crate::topo::FaceKey;
use super::classify::PointClassification;
use super::split::SplitFaceRegion;
use super::builder_face;
use super::BoolOp;

/// Select B-Rep faces for the boolean result based on operation type.
///
/// Each sub-face is individually classified; kept faces are converted
/// to BRep via BuilderFace (OCC BOPAlgo_BuilderFace path).
pub fn select_brep_faces(
    regions_a: &[(usize, Vec<PointClassification>)],
    regions_b: &[(usize, Vec<PointClassification>)],
    split_a: &[SplitFaceRegion],
    split_b: &[SplitFaceRegion],
    op: BoolOp,
    reg: &mut BRepStore,
) -> Vec<FaceKey> {
    let mut selected = Vec::new();

    // Process A faces
    for (i, classes) in regions_a {
        let i = *i;
        if i >= split_a.len() { continue; }
        let sfr = &split_a[i];
        for (j, sub) in sfr.sub_faces.iter().enumerate() {
            let class = classes.get(j).copied().unwrap_or(PointClassification::Outside);
            let keep = match op {
                BoolOp::Union => class == PointClassification::Outside,
                BoolOp::Intersection => class == PointClassification::Inside,
                BoolOp::Difference => class == PointClassification::Outside,
            };
            if keep {
                push_kept_face(sub, reg, &mut selected, &[]);
            }
        }
    }

    // Process B faces
    for (i, classes) in regions_b {
        let i = *i;
        if i >= split_b.len() { continue; }
        let sfr = &split_b[i];
        for (j, sub) in sfr.sub_faces.iter().enumerate() {
            let class = classes.get(j).copied().unwrap_or(PointClassification::Outside);
            let keep = match op {
                BoolOp::Union => class == PointClassification::Outside,
                BoolOp::Intersection => class == PointClassification::Inside,
                BoolOp::Difference => class == PointClassification::Inside,
            };
            if keep {
                push_kept_face(sub, reg, &mut selected, &[]);
            }
        }
    }

    selected
}

/// Convert a kept sub-face region to actual BRep faces.
///
/// Uses BuilderFace to create proper BRep topology from the UV region,
/// falling back to the original face if no UV boundary is available.
fn push_kept_face(
    sub: &super::split::SubFaceRegion,
    reg: &mut BRepStore,
    selected: &mut Vec<FaceKey>,
    curves: &[super::split::BRepIntersectionCurve],
) {
    if !sub.uv_boundary.is_empty() {
        // Use BuilderFace to create proper BRep face from UV boundary
        let bf_result = builder_face::build_faces_from_split(
            sub.original_face,
            std::slice::from_ref(sub),
            curves,
            reg,
            None,
        );
        let has_new = !bf_result.new_faces.is_empty();
        selected.extend(bf_result.new_faces);
        if !has_new {
            selected.push(sub.original_face);
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
            surface: SurfaceGeom::Plane { origin: rc3d_core::math::PVec3::ZERO, normal: rc3d_core::math::PVec3::Z, u_dir: rc3d_core::math::PVec3::X },
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
                interior_point_3d: rc3d_core::math::PVec3::ZERO,
                original_face: fk_a,
            }],
        }];
        let split_b = vec![SplitFaceRegion {
            original_face: fk_b,
            sub_faces: vec![SubFaceRegion {
                uv_boundary: vec![],
                interior_point: (0.5, 0.5),
                interior_point_3d: rc3d_core::math::PVec3::ZERO,
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
                interior_point_3d: rc3d_core::math::PVec3::ZERO,
                original_face: fk_a,
            }],
        }];
        let split_b = vec![SplitFaceRegion {
            original_face: fk_b,
            sub_faces: vec![SubFaceRegion {
                uv_boundary: vec![],
                interior_point: (0.5, 0.5),
                interior_point_3d: rc3d_core::math::PVec3::ZERO,
                original_face: fk_b,
            }],
        }];

        let regions_a = vec![(0, vec![PointClassification::Inside])];
        let regions_b = vec![(0, vec![PointClassification::Inside])];
        let selected = select_brep_faces(&regions_a, &regions_b, &split_a, &split_b, BoolOp::Intersection, &mut reg);
        assert_eq!(selected.len(), 2, "Intersection: both inside should be selected");
    }
}
