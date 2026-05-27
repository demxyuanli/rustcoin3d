//! Degenerated edge detection at surface singularities (OCC ShapeFix_Face::FixDegenerated).
//! Phase 2 handles detection and entity creation. Phase 3 handles CDT integration.

use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::FaceKey;
use rc3d_core::math::Vec3;

#[derive(Debug, Default)]
pub struct DegeneratedReport {
    pub degeneracies_found: usize,
    pub degenerate_edges_created: usize,
}

/// Detect surface singularities and create degenerated edges for a face.
/// Adds edges to face.degenerated_edges list.
pub fn fix_degenerated_edges(
    face_key: FaceKey,
    reg: &mut BRepRegistry,
) -> DegeneratedReport {
    let mut report = DegeneratedReport::default();

    let (surface, tolerance) = {
        let Some(face) = reg.faces.get(face_key) else { return report; };
        (face.surface.clone(), face.tolerance)
    };

    let singularities = find_singularities(&surface);
    if singularities.is_empty() {
        return report;
    }
    report.degeneracies_found = singularities.len();

    let outer_wire = {
        let Some(face) = reg.faces.get(face_key) else { return report; };
        face.outer_wire
    };

    let wire_edges = {
        let Some(wire) = reg.wires.get(outer_wire) else { return report; };
        wire.edges.clone()
    };

    let mut degen_edges = Vec::new();

    for singularity in &singularities {
        // For each wire edge that passes near the singularity in UV space,
        // check if we should create a degenerated edge
        for &(ek, _) in &wire_edges {
            let edge = match reg.edges.get(ek) {
                Some(e) => e,
                None => continue,
            };
            let pc = match edge.pcurves.get(&face_key) {
                Some(p) => p,
                None => continue,
            };

            for t in [0.0, 0.5, 1.0] {
                let uv = pc.d0(t);
                let dist_uv = ((uv.x - singularity.uv.0).powi(2) + (uv.y - singularity.uv.1).powi(2)).sqrt();
                if dist_uv < 1e-4 {
                    // Create degenerated edge at this singularity
                    let vk = reg.find_or_add_vertex(singularity.point_3d, tolerance);
                    let degen_curve = crate::step::brep::geom::CurveGeom::Line {
                        origin: singularity.point_3d,
                        direction: Vec3::ZERO,
                    };
                    let degen_pc = crate::step::brep::geom::CurveGeom::Line {
                        origin: Vec3::new(singularity.uv.0, singularity.uv.1, 0.0),
                        direction: Vec3::ZERO,
                    };
                    let dek = reg.add_seam_edge(vk, vk, degen_curve, tolerance, face_key, degen_pc);
                    degen_edges.push(dek);
                    report.degenerate_edges_created += 1;
                    break; // one degenerated edge per wire edge per singularity
                }
            }
        }
    }

    if let Some(face) = reg.faces.get_mut(face_key) {
        face.degenerated_edges.extend(degen_edges);
    }

    report
}

struct SingularityInfo {
    point_3d: Vec3,
    uv: (f32, f32),
}

fn find_singularities(surface: &SurfaceGeom) -> Vec<SingularityInfo> {
    match surface {
        SurfaceGeom::Sphere { center, radius } => {
            vec![
                SingularityInfo { point_3d: *center + Vec3::new(0.0, 0.0, *radius), uv: (0.0, std::f32::consts::FRAC_PI_2) },
                SingularityInfo { point_3d: *center - Vec3::new(0.0, 0.0, *radius), uv: (0.0, -std::f32::consts::FRAC_PI_2) },
            ]
        }
        SurfaceGeom::Cone { apex, .. } => {
            vec![SingularityInfo { point_3d: *apex, uv: (0.0, 0.0) }]
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::CurveGeom;
    use crate::step::brep::topo::BRepWire;
    use rc3d_core::math::Vec3;

    #[test]
    fn test_detect_sphere_pole_degeneracy() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 1.0 };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let report = fix_degenerated_edges(fk, &mut reg);
        assert!(report.degeneracies_found > 0, "sphere should have pole singularities");
    }

    #[test]
    fn test_detect_cone_apex_degeneracy() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Cone { apex: Vec3::ZERO, axis: Vec3::Z, semi_angle: 0.5, radius_at_apex: 0.0 };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let report = fix_degenerated_edges(fk, &mut reg);
        assert!(report.degeneracies_found > 0, "cone should have apex singularity");
    }

    #[test]
    fn test_no_false_degeneracy_plane() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let report = fix_degenerated_edges(fk, &mut reg);
        assert_eq!(report.degeneracies_found, 0);
        assert_eq!(report.degenerate_edges_created, 0);
    }
}
