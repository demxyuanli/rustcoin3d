//! Periodic surface pole degeneracy (OCC ShapeFix_Face::FixPeriodicDegenerated).
//! Reconstructs degenerated edges at poles of periodic surfaces before seam fixing.

use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::FaceKey;
use rc3d_core::math::Vec3;

#[derive(Debug, Default)]
pub struct PeriodicDegeneratedReport {
    pub degeneracies_reconstructed: usize,
    pub pole_edges_created: usize,
}

/// For a wire that wraps a full parameter period on a sphere, reconstruct
/// degenerated edges at the poles. Must run BEFORE FixMissingSeam.
pub fn fix_periodic_degenerated(
    face_key: FaceKey,
    reg: &mut BRepRegistry,
) -> PeriodicDegeneratedReport {
    let mut report = PeriodicDegeneratedReport::default();

    let (surface, tolerance, outer_wire, existing_degen) = {
        let Some(face) = reg.faces.get(face_key) else { return report; };
        (
            face.surface.clone(),
            face.tolerance,
            face.outer_wire,
            face.degenerated_edges.clone(),
        )
    };

    // Only applies to sphere surfaces with a single wire wrapping the full U range
    let sphere = match &surface {
        SurfaceGeom::Sphere { center, radius } => (*center, *radius),
        _ => return report,
    };

    // Check if degeneracies are already present
    if !existing_degen.is_empty() {
        return report;
    }

    let wire = match reg.wires.get(outer_wire) {
        Some(w) => w,
        None => return report,
    };

    // Check if the wire wraps full U range (all 2*PI)
    let mut u_min = f32::MAX;
    let mut u_max = f32::MIN;
    for &(ek, _) in &wire.edges {
        let Some(edge) = reg.edges.get(ek) else { continue; };
        let Some(pc) = edge.pcurves.get(&face_key) else { continue; };
        for t in [0.0, 0.5, 1.0] {
            let uv = pc.d0(t);
            u_min = u_min.min(uv.x);
            u_max = u_max.max(uv.x);
        }
    }

    let u_span = u_max - u_min;
    if u_span < std::f32::consts::TAU * 0.9 {
        return report; // Doesn't wrap full period
    }

    // Wire wraps full U → need pole degeneracies
    let poles = [
        (sphere.0 + Vec3::new(0.0, 0.0, sphere.1), (0.0, std::f32::consts::FRAC_PI_2)),
        (sphere.0 - Vec3::new(0.0, 0.0, sphere.1), (0.0, -std::f32::consts::FRAC_PI_2)),
    ];

    let mut new_degen = Vec::new();
    for &(pole_3d, pole_uv) in &poles {
        let vk = reg.find_or_add_vertex(pole_3d, tolerance);
        let degen_curve = crate::step::brep::geom::CurveGeom::Line {
            origin: pole_3d,
            direction: Vec3::ZERO,
        };
        let degen_pc = crate::step::brep::geom::CurveGeom::Line {
            origin: Vec3::new(pole_uv.0, pole_uv.1, 0.0),
            direction: Vec3::ZERO,
        };
        let dek = reg.add_seam_edge(vk, vk, degen_curve, tolerance, face_key, degen_pc);
        new_degen.push(dek);
        report.pole_edges_created += 1;
    }

    report.degeneracies_reconstructed = poles.len();

    if let Some(face) = reg.faces.get_mut(face_key) {
        for dek in new_degen {
            if !face.degenerated_edges.contains(&dek) {
                face.degenerated_edges.push(dek);
            }
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::CurveGeom;
    use crate::step::brep::topo::{BRepWire, Orientation};
    use rc3d_core::math::Vec3;

    #[test]
    fn test_sphere_wrap_detected() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 1.0 };
        // Create a wire with edges spanning full U range
        let v_np = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v_sp = reg.find_or_add_vertex(Vec3::new(-1.0, 0.0, 0.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        // Create edges that span the full U range
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        // PCurve wrapping 0 to 2*PI in U, V near equator
        let pc = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(std::f32::consts::TAU, 0.0, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v_np, v_sp, line, 1e-4, fk, pc);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];

        let report = fix_periodic_degenerated(fk, &mut reg);
        assert!(report.pole_edges_created > 0, "sphere wrap should create pole edges");
        let face = reg.faces.get(fk).unwrap();
        assert!(!face.degenerated_edges.is_empty(), "degenerated edges should be added");
    }

    #[test]
    fn test_no_wrap_no_degeneracy() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 1.0 };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let report = fix_periodic_degenerated(fk, &mut reg);
        assert_eq!(report.pole_edges_created, 0, "no wire edges -> no degeneracies");
    }
}
