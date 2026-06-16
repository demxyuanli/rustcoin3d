//! OCC BRepGProp — geometric properties: area, volume, center of mass.
//! Uses surface integration for faces and divergence theorem for solids.

use rc3d_core::math::{Real, PVec3};
use crate::topo::FaceKey;
use crate::store::BRepStore;

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

/// Flux contribution of one face to volume integral: ∫ x·nx dS
fn oriented_face_flux(reg: &BRepStore, fk: FaceKey, _orient: crate::topo::Orientation, grid_res: usize) -> Real {
    let face = match reg.faces.get(fk) { Some(f) => f, None => return 0.0 };
    let pr = reg.face_param_range(fk, &face.surface);
    let n = grid_res.max(4);
    let du = pr.u_span() / n as Real;
    let dv = pr.v_span() / n as Real;
    let mut flux = 0.0_f64;
    for iu in 0..n {
        let u = pr.u_min + (iu as Real + 0.5) * du;
        for iv in 0..n {
            let v = pr.v_min + (iv as Real + 0.5) * dv;
            let p = face.surface.d0_native(u, v);
            let (su, sv) = face.surface.d1_native(u, v);
            let normal = su.cross(sv);
            if !face.same_sense { flux -= p.x * normal.x * du * dv; }
            else { flux += p.x * normal.x * du * dv; }
        }
    }
    flux
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::*;
    use crate::geom::{CurveGeom, Curve2d, SurfaceGeom};

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
}
