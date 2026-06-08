//! UV coordinate sources for mesh triangulation.
//!
//! Strategies: Pcurve (trusted), NativeUV (cylinder/cone/sphere/torus),
//! Projected (generic), PlaneLocal (plane faces).

use crate::geom::SurfaceGeom;
use crate::topo::BRepFace;

/// Available UV source strategies (ordered by quality).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum UvSourceKind {
    Pcurve,
    NativeUv,
    Projected,
    PlaneLocal,
}

/// Determine the best UV source for a face based on surface type.
#[allow(dead_code)]
pub fn select_uv_source(face: &BRepFace) -> UvSourceKind {
    match &face.surface {
        SurfaceGeom::Cylinder { .. }
        | SurfaceGeom::Cone { .. }
        | SurfaceGeom::Sphere { .. }
        | SurfaceGeom::Torus { .. }
        | SurfaceGeom::Revolution { .. } => UvSourceKind::NativeUv,
        SurfaceGeom::BSpline(_) | SurfaceGeom::Offset { .. } => UvSourceKind::Pcurve,
        SurfaceGeom::Plane { .. } => UvSourceKind::PlaneLocal,
        _ => UvSourceKind::Projected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::BRepFace;

    fn test_face(surface: SurfaceGeom) -> BRepFace {
        BRepFace {
            surface,
            outer_wire: crate::topo::WireKey::default(),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-6,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        }
    }

    #[test]
    fn cylinder_gets_native_uv() {
        let face = test_face(SurfaceGeom::cylinder(rc3d_core::math::Vec3::ZERO, rc3d_core::math::Vec3::Z, 1.0));
        assert_eq!(select_uv_source(&face), UvSourceKind::NativeUv);
    }

    #[test]
    fn cone_gets_native_uv() {
        let face = test_face(SurfaceGeom::cone(rc3d_core::math::Vec3::ZERO, rc3d_core::math::Vec3::Z, 0.5, 0.0));
        assert_eq!(select_uv_source(&face), UvSourceKind::NativeUv);
    }

    #[test]
    fn sphere_gets_native_uv() {
        let face = test_face(SurfaceGeom::Sphere { center: rc3d_core::math::Vec3::ZERO, radius: 1.0 });
        assert_eq!(select_uv_source(&face), UvSourceKind::NativeUv);
    }

    #[test]
    fn torus_gets_native_uv() {
        let face = test_face(SurfaceGeom::torus(rc3d_core::math::Vec3::ZERO, rc3d_core::math::Vec3::Z, 2.0, 0.5));
        assert_eq!(select_uv_source(&face), UvSourceKind::NativeUv);
    }

    #[test]
    fn plane_gets_plane_local() {
        let face = test_face(SurfaceGeom::Plane { origin: rc3d_core::math::Vec3::ZERO, normal: rc3d_core::math::Vec3::Z, u_dir: rc3d_core::math::Vec3::X });
        assert_eq!(select_uv_source(&face), UvSourceKind::PlaneLocal);
    }
}
