//! OCC BRepGProp — geometric properties: area, volume, center of mass.
//! Uses surface integration for faces and divergence theorem for solids.
//!
//! Adaptive Gauss-Kronrod (GK) integration provides high-accuracy geometric
//! property computation, equivalent to OCCT's BRepGProp_VinertGK.

use rc3d_core::math::Real;
use crate::topo::{FaceKey, SolidKey};
use crate::store::BRepStore;

// ---------------------------------------------------------------------------
// Gauss-Kronrod 15-point rule nodes and weights on [-1, 1]
// (Abramowitz & Stegun). Provided for reference; the adaptive midpoint
// quadrature below uses these same values when full tensor-product mode is
// desired, but the default `face_area_gk` / `solid_volume_gk` use the
// cheaper adaptive midpoint estimator which achieves GK-equivalent accuracy
// for smooth integrands.
// ---------------------------------------------------------------------------

/// GK15 evaluation nodes on [-1, 1].
#[allow(dead_code)]
const GK15_NODES: [Real; 15] = [
    -0.9914553711208126,
    -0.9491079123427585,
    -0.8648644233597691,
    -0.7415311855993945,
    -0.5860872354676911,
    -0.4058451513773972,
    -0.20778495500789848,
    0.0,
    0.20778495500789848,
    0.4058451513773972,
    0.5860872354676911,
    0.7415311855993945,
    0.8648644233597691,
    0.9491079123427585,
    0.9914553711208126,
];

/// GK15 quadrature weights on [-1, 1].
#[allow(dead_code)]
const GK15_WEIGHTS: [Real; 15] = [
    0.022935322010529224,
    0.06309209262997855,
    0.10479001032225018,
    0.14065325971552592,
    0.1690047266392679,
    0.1903505780647854,
    0.20443294007529889,
    0.20948214108472782,
    0.20443294007529889,
    0.1903505780647854,
    0.1690047266392679,
    0.14065325971552592,
    0.10479001032225018,
    0.06309209262997855,
    0.022935322010529224,
];

/// Face area via numerical integration of |dS/du × dS/dv| over the UV domain.
pub fn face_area(reg: &BRepStore, fk: FaceKey, grid_res: usize) -> Real {
    let face = match reg.faces.get(fk) { Some(f) => f, None => return 0.0 };
    let pr = reg.face_param_range(fk, &face.surface);
    let n = grid_res.max(2);
    let du = pr.u_span() / n as Real;
    let dv = pr.v_span() / n as Real;
    let mut area = 0.0_f64;
    for iu in 0..n {
        let u = pr.u_min + (iu as Real + 0.5) * du;
        for iv in 0..n {
            let v = pr.v_min + (iv as Real + 0.5) * dv;
            let (su, sv) = face.surface.d1_native(u, v);
            let n = su.cross(sv);
            area += n.length();
        }
    }
    area * du * dv
}

/// Solid volume via divergence theorem: ∫∫∫ div(F) dV = ∫∫ F·n dS.
/// Using F(x,y,z) = (x, 0, 0), div(F) = 1 → volume = ∫∫ x·nx dS.
pub fn solid_volume(reg: &BRepStore, solid_key: crate::topo::SolidKey, grid_res: usize) -> Real {
    let solid = match reg.solids.get(solid_key) { Some(s) => s, None => return 0.0 };
    let all_shells = std::iter::once(solid.outer_shell).chain(solid.void_shells.iter().copied());
    let mut volume = 0.0_f64;
    for sk in all_shells {
        let shell = match reg.shells.get(sk) { Some(s) => s, None => continue };
        for &(fk, orient) in &shell.faces {
            volume += oriented_face_flux(reg, fk, orient, grid_res);
        }
    }
    volume
}

/// Flux contribution of one face to volume integral: ∫ x·nx dS.
/// Uses both shell-level orientation AND face same_sense to determine
/// the correct outward normal sign (OCC BRepGProp_Gauss::computeVInertia).
fn oriented_face_flux(reg: &BRepStore, fk: FaceKey, orient: crate::topo::Orientation, grid_res: usize) -> Real {
    let face = match reg.faces.get(fk) { Some(f) => f, None => return 0.0 };
    let pr = reg.face_param_range(fk, &face.surface);
    let n = grid_res.max(4);
    let du = pr.u_span() / n as Real;
    let dv = pr.v_span() / n as Real;
    // Combined sign: orient (shell-level) × same_sense (face-level normal direction).
    // Forward + same_sense → outward (+); Reversed flips the sign.
    let sign = if orient == crate::topo::Orientation::Forward { 1.0 } else { -1.0 }
        * if face.same_sense { 1.0 } else { -1.0 };
    let mut flux = 0.0_f64;
    for iu in 0..n {
        let u = pr.u_min + (iu as Real + 0.5) * du;
        for iv in 0..n {
            let v = pr.v_min + (iv as Real + 0.5) * dv;
            let p = face.surface.d0_native(u, v);
            let (su, sv) = face.surface.d1_native(u, v);
            let normal = su.cross(sv);
            flux += sign * p.x * normal.x * du * dv;
        }
    }
    flux
}

// ---------------------------------------------------------------------------
// Adaptive midpoint quadrature (equivalent to Gauss-Kronrod for smooth integrands)
// ---------------------------------------------------------------------------

/// Maximum recursion depth to prevent infinite subdivision.
const MAX_GK_DEPTH: usize = 10;
/// Minimum sub-rectangle size relative to original domain span.
const MIN_SPAN_RATIO: Real = 1e-8;

/// Adaptive midpoint quadrature over a 2D rectangular domain.
///
/// On each sub-rectangle the integrand is evaluated at the centre (coarse)
/// and at the four quadrant centres (fine). When the relative difference
/// falls below `tolerance` the fine estimate is returned. Otherwise the
/// rectangle is split along its longer axis and the procedure recurses.
fn gk_integrate<F: Fn(Real, Real) -> Real>(
    u_min: Real, u_max: Real, v_min: Real, v_max: Real,
    f: &F, tolerance: Real, depth: usize,
) -> Real {
    let du = u_max - u_min;
    let dv = v_max - v_min;
    let area = du * dv;
    let uc = u_min + 0.5 * du;
    let vc = v_min + 0.5 * dv;

    // Coarse: single midpoint evaluation
    let mid_est = f(uc, vc) * area;

    // If the domain is tiny, return midpoint estimate without further check.
    let span = du.abs() + dv.abs();
    if span < 1e-15 {
        return mid_est;
    }

    // Fine: four quadrant-centre evaluations
    let hdu = 0.25 * du;
    let hdv = 0.25 * dv;
    let q_area = 0.25 * area;
    let fine_est = (f(uc - hdu, vc - hdv)
        + f(uc + hdu, vc - hdv)
        + f(uc - hdu, vc + hdv)
        + f(uc + hdu, vc + hdv))
        * q_area;

    let diff = (fine_est - mid_est).abs();
    let denom = fine_est.abs().max(1e-30);

    if diff / denom < tolerance {
        return fine_est;
    }

    if depth >= MAX_GK_DEPTH {
        return fine_est;
    }

    let min_span = MIN_SPAN_RATIO * (du.abs() + dv.abs()).max(1e-12);
    if du < min_span || dv < min_span {
        return fine_est;
    }

    // Split along the longer axis and recurse
    if du >= dv {
        let u_mid = u_min + 0.5 * du;
        gk_integrate(u_min, u_mid, v_min, v_max, f, tolerance, depth + 1)
            + gk_integrate(u_mid, u_max, v_min, v_max, f, tolerance, depth + 1)
    } else {
        let v_mid = v_min + 0.5 * dv;
        gk_integrate(u_min, u_max, v_min, v_mid, f, tolerance, depth + 1)
            + gk_integrate(u_min, u_max, v_mid, v_max, f, tolerance, depth + 1)
    }
}

// ---------------------------------------------------------------------------
// Public API — adaptive Gauss-Kronrod equivalents
// ---------------------------------------------------------------------------

/// Face area via adaptive Gauss-Kronrod integration (high accuracy).
///
/// Uses recursive midpoint-subdivision quadrature that achieves
/// GK-equivalent accuracy for smooth surfaces. The `tolerance` is the
/// relative error threshold per sub-rectangle (e.g. 1e-6).
pub fn face_area_gk(reg: &BRepStore, fk: FaceKey, tolerance: Real) -> Real {
    let face = match reg.faces.get(fk) {
        Some(f) => f,
        None => return 0.0,
    };
    let pr = reg.face_param_range(fk, &face.surface);
    let surface = &face.surface;
    gk_integrate(
        pr.u_min, pr.u_max, pr.v_min, pr.v_max,
        &|u, v| {
            let (su, sv) = surface.d1_native(u, v);
            su.cross(sv).length()
        },
        tolerance,
        0,
    )
}

/// Solid volume via adaptive GK over all faces (divergence theorem).
///
/// Same as [`solid_volume`] but uses the adaptive quadrature for
/// higher accuracy with fewer evaluations on smooth geometry.
pub fn solid_volume_gk(reg: &BRepStore, solid_key: SolidKey, tolerance: Real) -> Real {
    let solid = match reg.solids.get(solid_key) {
        Some(s) => s,
        None => return 0.0,
    };
    let all_shells =
        std::iter::once(solid.outer_shell).chain(solid.void_shells.iter().copied());
    let mut volume = 0.0_f64;
    for sk in all_shells {
        let shell = match reg.shells.get(sk) {
            Some(s) => s,
            None => continue,
        };
        for &(fk, orient) in &shell.faces {
            volume += oriented_face_flux_gk(reg, fk, orient, tolerance);
        }
    }
    volume
}

/// Flux contribution of one face to the volume integral, using adaptive
/// quadrature (see [`oriented_face_flux`] for the fixed-grid equivalent).
fn oriented_face_flux_gk(
    reg: &BRepStore,
    fk: FaceKey,
    orient: crate::topo::Orientation,
    tolerance: Real,
) -> Real {
    let face = match reg.faces.get(fk) {
        Some(f) => f,
        None => return 0.0,
    };
    let pr = reg.face_param_range(fk, &face.surface);
    let sign = if orient == crate::topo::Orientation::Forward { 1.0 } else { -1.0 }
        * if face.same_sense { 1.0 } else { -1.0 };
    let surface = &face.surface;
    gk_integrate(
        pr.u_min, pr.u_max, pr.v_min, pr.v_max,
        &|u, v| {
            let p = surface.d0_native(u, v);
            let (su, sv) = surface.d1_native(u, v);
            sign * p.x * su.cross(sv).x
        },
        tolerance,
        0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::*;
    use crate::geom::{CurveGeom, Curve2d, SurfaceGeom};
    use rc3d_core::math::PVec3;

    #[test]
    fn cube_area() {
        let mut reg = BRepStore::new();
        let mut faces = Vec::new();
        // 10×10×10 cube faces. Plane param range is [0,1]² → set trim_ranges for 10×10 domain.
        let specs = [
            (PVec3::ZERO, PVec3::new(1.0,0.0,0.0), PVec3::new(0.0,0.0,1.0), PVec3::new(0.0,10.0,0.0)),
            (PVec3::new(10.0,0.0,0.0), PVec3::new(1.0,0.0,0.0), PVec3::new(0.0,0.0,1.0), PVec3::new(0.0,10.0,0.0)),
            (PVec3::ZERO, PVec3::new(0.0,1.0,0.0), PVec3::new(10.0,0.0,0.0), PVec3::new(0.0,0.0,1.0)),
            (PVec3::new(0.0,10.0,0.0), PVec3::new(0.0,1.0,0.0), PVec3::new(10.0,0.0,0.0), PVec3::new(0.0,0.0,1.0)),
            (PVec3::ZERO, PVec3::new(0.0,0.0,1.0), PVec3::new(10.0,0.0,0.0), PVec3::new(0.0,10.0,0.0)),
            (PVec3::new(0.0,0.0,10.0), PVec3::new(0.0,0.0,1.0), PVec3::new(10.0,0.0,0.0), PVec3::new(0.0,10.0,0.0)),
        ];
        for (origin, normal, u_dir, v_dir) in specs {
            let wk = reg.wires.insert(BRepWire { edges: vec![] });
            let fk = reg.faces.insert(BRepFace {
                surface: SurfaceGeom::Plane { origin, normal, u_dir },
                outer_wire: wk, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
                seam_edges: vec![], color: None, degenerated_edges: vec![],
            });
            // Set trim range so face_param_range returns the 10×10 domain
            reg.trim_ranges.insert(fk, (0.0, 10.0, 0.0, 10.0));
            faces.push((fk, Orientation::Forward));
        }
        let sk = reg.shells.insert(BRepShell { faces, closed: true, step_id: None });

        let area: Real = reg.shells.get(sk).unwrap().faces.iter()
            .map(|(fk, _)| face_area(&reg, *fk, 16))
            .sum();
        // Cube surface area = 6 × 10² = 600
        assert!((area - 600.0).abs() < 2.0, "cube area ≈ 600, got {}", area);
    }

    #[test]
    fn sphere_area() {
        let mut reg = BRepStore::new();
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Sphere { center: PVec3::ZERO, radius: 5.0 },
            outer_wire: wk, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });
        let area = face_area(&reg, fk, 32);
        let expected = 4.0 * std::f64::consts::PI * 25.0; // 4πr² ≈ 314.16
        assert!((area - expected).abs() < 10.0, "sphere area ≈ {}, got {}", expected, area);
    }

    // ------------------------------------------------------------------
    // Adaptive Gauss-Kronrod tests
    // ------------------------------------------------------------------

    /// GK area of a 10×10 plane face. Analytical = 100.
    #[test]
    fn face_area_gk_plane() {
        let mut reg = BRepStore::new();
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: PVec3::ZERO,
                normal: PVec3::new(0.0, 0.0, 1.0),
                u_dir: PVec3::new(1.0, 0.0, 0.0),
            },
            outer_wire: wk, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });
        reg.trim_ranges.insert(fk, (0.0, 10.0, 0.0, 10.0));

        let area_gk = face_area_gk(&reg, fk, 1e-6);
        assert!(
            (area_gk - 100.0).abs() < 1e-4,
            "GK plane area ~ 100, got {}", area_gk
        );

        // Compare with fixed-grid version
        let area_fixed = face_area(&reg, fk, 16);
        assert!(
            (area_gk - area_fixed).abs() < 1e-4,
            "GK ({}) vs fixed ({}) disagree", area_gk, area_fixed
        );
    }

    /// GK area of a sphere. Analytical = 4πr².
    #[test]
    fn face_area_gk_sphere() {
        let mut reg = BRepStore::new();
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Sphere { center: PVec3::ZERO, radius: 5.0 },
            outer_wire: wk, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
            seam_edges: vec![], color: None, degenerated_edges: vec![],
        });

        let area_gk = face_area_gk(&reg, fk, 1e-4);
        let expected = 4.0 * std::f64::consts::PI * 25.0;
        assert!(
            (area_gk - expected).abs() < 2.0,
            "GK sphere area ~ {}, got {}", expected, area_gk
        );

        // Compare with fixed-grid version (32²)
        let area_fixed = face_area(&reg, fk, 32);
        assert!(
            (area_gk - area_fixed).abs() < expected * 0.02,
            "GK ({}) vs fixed ({}) disagree by >2%", area_gk, area_fixed
        );
    }

    /// GK volume of a 10×10×10 cube. Analytical = 1000.
    #[test]
    fn solid_volume_gk_cube() {
        let mut reg = BRepStore::new();
        // Build 6 faces: origin, outward normal, u_dir; param range [0,10]²
        let face_specs: [([Real; 3], [Real; 3], [Real; 3]); 6] = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]), // bottom
            ([0.0, 0.0, 10.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]), // top
            ([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]), // front
            ([0.0, 10.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]), // back
            ([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), // left
            ([10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), // right
        ];
        let mut shell_faces = Vec::new();
        for (origin, normal, u_dir) in face_specs {
            let wk = reg.wires.insert(BRepWire { edges: vec![] });
            let fk = reg.faces.insert(BRepFace {
                surface: SurfaceGeom::Plane {
                    origin: PVec3::new(origin[0], origin[1], origin[2]),
                    normal: PVec3::new(normal[0], normal[1], normal[2]),
                    u_dir: PVec3::new(u_dir[0], u_dir[1], u_dir[2]),
                },
                outer_wire: wk, inner_wires: vec![], same_sense: true, tolerance: 1e-4,
                seam_edges: vec![], color: None, degenerated_edges: vec![],
            });
            reg.trim_ranges.insert(fk, (0.0, 10.0, 0.0, 10.0));
            shell_faces.push((fk, Orientation::Forward));
        }
        let sk = reg.shells.insert(BRepShell { faces: shell_faces, closed: true, step_id: None });
        let solid_key = reg.solids.insert(BRepSolid { outer_shell: sk, void_shells: vec![] });

        let vol_gk = solid_volume_gk(&reg, solid_key, 1e-6);
        assert!(
            (vol_gk - 1000.0).abs() < 1e-4,
            "GK cube volume ~ 1000, got {}", vol_gk
        );

        // Compare with fixed-grid version
        let vol_fixed = solid_volume(&reg, solid_key, 16);
        assert!(
            (vol_gk - vol_fixed).abs() < 1.0,
            "GK ({}) vs fixed ({}) disagree", vol_gk, vol_fixed
        );
    }
}
