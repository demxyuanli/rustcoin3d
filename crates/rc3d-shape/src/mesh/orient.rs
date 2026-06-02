//! Orientation propagation from shell face to triangle winding.
//!
//! Replaces the ad-hoc same_sense checks scattered across 12 mesh algorithms
//! with a single authoritative computation.

use crate::topo::{BRepFace, Orientation};

/// Compute the effective sense for a face given its shell-level orientation.
///
/// OCC semantics: face normal = surface normal * same_sense.
/// Shell normal = face normal * orientation.
/// effective_sense = same_sense XOR (orientation == Reversed).
#[inline]
pub fn effective_sense(face: &BRepFace, shell_orient: Orientation) -> bool {
    face.same_sense ^ (shell_orient == Orientation::Reversed)
}

/// Compute effective sense from raw same_sense bool. For use when the full BRepFace
/// is not available (e.g., during shell loop iteration before face lookup).
#[inline]
pub fn effective_sense_raw(same_sense: bool, shell_orient: Orientation) -> bool {
    same_sense ^ (shell_orient == Orientation::Reversed)
}

/// Apply effective_sense to a normal vector.
/// When same_sense is false, the normal is negated.
#[inline]
pub fn orient_normal(mut normal: rc3d_core::math::Vec3, sense: bool) -> rc3d_core::math::Vec3 {
    if !sense {
        normal = -normal;
    }
    normal
}

/// Fix triangle winding to match surface normal at centroid.
/// Swaps indices[1] and indices[2] if the cross-product normal
/// disagrees with the surface normal.
pub fn fix_tri_winding_sense(
    tri: &mut [i32; 3],
    vertices: &[rc3d_core::math::Vec3],
    surface_normal: rc3d_core::math::Vec3,
    sense: bool,
) {
    let expected = orient_normal(surface_normal, sense);
    let v0 = vertices[tri[0] as usize];
    let v1 = vertices[tri[1] as usize];
    let v2 = vertices[tri[2] as usize];
    let tri_normal = (v1 - v0).cross(v2 - v0);
    if tri_normal.dot(expected) < 0.0 {
        tri.swap(1, 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::SurfaceGeom;
    use crate::topo::BRepFace;

    fn make_face(same_sense: bool) -> BRepFace {
        BRepFace {
            surface: SurfaceGeom::Plane {
                origin: rc3d_core::math::Vec3::ZERO,
                normal: rc3d_core::math::Vec3::Z,
                u_dir: rc3d_core::math::Vec3::X,
            },
            outer_wire: crate::topo::WireKey::default(),
            inner_wires: vec![],
            same_sense,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        }
    }

    #[test]
    fn forward_both_true() {
        let face = make_face(true);
        assert!(effective_sense(&face, Orientation::Forward));
    }

    #[test]
    fn forward_false_same_sense() {
        let face = make_face(false);
        assert!(!effective_sense(&face, Orientation::Forward));
    }

    #[test]
    fn reversed_inverts_sense() {
        let face = make_face(true);
        assert!(!effective_sense(&face, Orientation::Reversed));
    }

    #[test]
    fn reversed_false_double_negation() {
        let face = make_face(false);
        assert!(effective_sense(&face, Orientation::Reversed));
    }

    #[test]
    fn orient_normal_preserves_when_true() {
        let n = rc3d_core::math::Vec3::Z;
        assert_eq!(orient_normal(n, true), n);
    }

    #[test]
    fn orient_normal_flips_when_false() {
        let n = rc3d_core::math::Vec3::Z;
        assert_eq!(orient_normal(n, false), -n);
    }
}
