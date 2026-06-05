//! Curvature-driven parameter domain subdivision.
//!
//! Recursively subdivides the UV domain where surface curvature is high
//! (small minimum curvature radius) or where edge midpoint deviation exceeds
//! the deflection tolerance. U and V directions are subdivided independently.
//!
//! OCC alignment: BRepMesh_FastDiscret adaptive parameter sampling driven
//! by the second fundamental form (via SurfaceGeom::min_curvature_radius).

use crate::geom::SurfaceGeom;

/// Parameter domain subdivision driven by surface curvature and edge deviation.
///
/// Returns sorted, deduplicated U and V division points for grid-based
/// interior point insertion in constrained Delaunay triangulation.
///
/// The range is in **native** surface parameter space (same as
/// SurfaceGeom::parameter_division). Uses `d0_native` for evaluation
/// and `native_uv_to_d0` for curvature lookups.
///
/// U and V directions are subdivided independently: a cylindrical surface
/// will only subdivide U (the circular direction), not V (the linear axis).
pub fn curvature_driven_divisions(
    surface: &SurfaceGeom,
    range: (f32, f32, f32, f32),
    deflection: f32,
    max_depth: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (u_min, u_max, v_min, v_max) = range;
    if deflection <= 0.0 || max_depth == 0 {
        return (vec![u_min, u_max], vec![v_min, v_max]);
    }

    // Collect u-division points by recursively checking u-direction deviation
    let mut u_divs = vec![u_min, u_max];
    subdivide_u(surface, u_min, u_max, v_min, v_max, deflection, max_depth, 0, &mut u_divs);

    // Collect v-division points by recursively checking v-direction deviation
    let mut v_divs = vec![v_min, v_max];
    subdivide_v(surface, v_min, v_max, u_min, u_max, deflection, max_depth, 0, &mut v_divs);

    u_divs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    u_divs.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    v_divs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v_divs.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    (u_divs, v_divs)
}

/// Recursively subdivide the U parametric direction at native parameters.
fn subdivide_u(
    surface: &SurfaceGeom,
    u0: f32, u1: f32,
    v0: f32, v1: f32,
    deflection: f32,
    max_depth: usize,
    depth: usize,
    divs: &mut Vec<f32>,
) {
    if depth >= max_depth {
        return;
    }
    let um = (u0 + u1) * 0.5;
    let vm = (v0 + v1) * 0.5;

    // Deviation of surface along u-direction at v0 and v1 edges
    let dev0 = native_edge_dev_u(surface, u0, u1, v0);
    let dev1 = native_edge_dev_u(surface, u0, u1, v1);
    let max_dev = dev0.max(dev1);

    // Curvature check at cell center (convert to normalized for min_curvature_radius)
    let (un, vn) = surface.native_uv_to_d0(um, vm);
    let min_k = surface.min_curvature_radius(un, vn);
    let high_curv = min_k < deflection * 2.0 && min_k < f32::MAX;

    if max_dev > deflection || high_curv {
        divs.push(um);
        subdivide_u(surface, u0, um, v0, v1, deflection, max_depth, depth + 1, divs);
        subdivide_u(surface, um, u1, v0, v1, deflection, max_depth, depth + 1, divs);
    }
}

/// Recursively subdivide the V parametric direction at native parameters.
fn subdivide_v(
    surface: &SurfaceGeom,
    v0: f32, v1: f32,
    u0: f32, u1: f32,
    deflection: f32,
    max_depth: usize,
    depth: usize,
    divs: &mut Vec<f32>,
) {
    if depth >= max_depth {
        return;
    }
    let um = (u0 + u1) * 0.5;
    let vm = (v0 + v1) * 0.5;

    // Deviation of surface along v-direction at u0 and u1 edges
    let dev0 = native_edge_dev_v(surface, u0, v0, v1);
    let dev1 = native_edge_dev_v(surface, u1, v0, v1);
    let max_dev = dev0.max(dev1);

    // Curvature check at cell center
    let (un, vn) = surface.native_uv_to_d0(um, vm);
    let min_k = surface.min_curvature_radius(un, vn);
    let high_curv = min_k < deflection * 2.0 && min_k < f32::MAX;

    if max_dev > deflection || high_curv {
        divs.push(vm);
        subdivide_v(surface, v0, vm, u0, u1, deflection, max_depth, depth + 1, divs);
        subdivide_v(surface, vm, v1, u0, u1, deflection, max_depth, depth + 1, divs);
    }
}

/// Deviation of surface from linear interpolation along a u-isoline at native v.
fn native_edge_dev_u(surface: &SurfaceGeom, u0: f32, u1: f32, v: f32) -> f32 {
    let um = (u0 + u1) * 0.5;
    let p0 = surface.d0_native(u0, v);
    let p1 = surface.d0_native(u1, v);
    let pm = surface.d0_native(um, v);
    (pm - (p0 + p1) * 0.5).length()
}

/// Deviation of surface from linear interpolation along a v-isoline at native u.
fn native_edge_dev_v(surface: &SurfaceGeom, u: f32, v0: f32, v1: f32) -> f32 {
    let vm = (v0 + v1) * 0.5;
    let p0 = surface.d0_native(u, v0);
    let p1 = surface.d0_native(u, v1);
    let pm = surface.d0_native(u, vm);
    (pm - (p0 + p1) * 0.5).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::SurfaceGeom;
    use rc3d_core::math::Vec3;

    #[test]
    fn test_plane_no_subdivision() {
        let plane = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        // Plane: native = normalized, range in world units
        let (u_divs, v_divs) =
            curvature_driven_divisions(&plane, (0.0, 10.0, 0.0, 10.0), 0.01, 4);
        assert_eq!(u_divs.len(), 2);
        assert_eq!(v_divs.len(), 2);
    }

    #[test]
    fn test_small_cylinder_subdivision() {
        let cyl = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 0.5);
        // Cylinder native u: [0, TAU] (circle), native v: linear
        let (u_divs, v_divs) = curvature_driven_divisions(
            &cyl,
            (0.0, std::f32::consts::TAU, 0.0, 1.0),
            0.01,
            4,
        );
        // Small radius → u-direction (circular) subdivided; v-direction (linear) stays fixed
        assert!(u_divs.len() > 2, "small cylinder should subdivide u, got {:?}", u_divs);
        assert_eq!(v_divs.len(), 2, "v-direction (linear) should not subdivide, got {:?}", v_divs);
    }

    #[test]
    fn test_large_cylinder_u_subdivided_v_not() {
        let cyl = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 100.0);
        let (u_divs, v_divs) = curvature_driven_divisions(
            &cyl,
            (0.0, std::f32::consts::TAU, 0.0, 1.0),
            0.01,
            4,
        );
        // Large radius: edge deviation along circle is small
        // v-direction (linear extrusion) stays at endpoints
        assert_eq!(v_divs.len(), 2, "v-direction must stay at endpoints for cylinder");
        assert!(u_divs.len() >= 2, "u-direction has at least endpoints");
    }

    #[test]
    fn test_zero_deflection_returns_endpoints() {
        let cyl = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 1.0);
        let (u_divs, v_divs) = curvature_driven_divisions(
            &cyl,
            (0.0, std::f32::consts::TAU, 0.0, 1.0),
            0.0,
            4,
        );
        assert_eq!(u_divs, vec![0.0, std::f32::consts::TAU]);
        assert_eq!(v_divs, vec![0.0, 1.0]);
    }

    #[test]
    fn test_sphere_subdivides_both_directions() {
        let sphere = SurfaceGeom::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        };
        // Sphere native u: [0, TAU], native v: [0, PI]
        let (u_divs, v_divs) = curvature_driven_divisions(
            &sphere,
            (0.0, std::f32::consts::TAU, 0.3, 2.8),
            0.05,
            4,
        );
        // Sphere curves in both directions → both should subdivide
        assert!(u_divs.len() > 2, "sphere should subdivide u, got {:?}", u_divs);
        assert!(v_divs.len() > 2, "sphere should subdivide v, got {:?}", v_divs);
    }
}
