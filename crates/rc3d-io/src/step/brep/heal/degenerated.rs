//! Degenerated edge detection at surface singularities (OCC ShapeFix_Face::FixDegenerated).

use super::curve_trim::{add_degenerated_edge_at_pole, split_edge_at_params};
use crate::step::brep::geom::SurfaceGeom;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{FaceKey, Orientation};
use rc3d_core::math::Vec3;

/// Descriptive info about a degenerated edge (OCC equivalent).
#[derive(Debug, Clone)]
pub struct DegeneratedEdgeInfo {
    pub edge_key: crate::step::brep::topo::EdgeKey,
    pub singularity_3d: Vec3,
    pub singularity_uv: (f32, f32),
    pub regular_vertex: crate::step::brep::topo::VertexKey,
}

impl DegeneratedEdgeInfo {
    pub fn new(ek: crate::step::brep::topo::EdgeKey, sing_3d: Vec3, sing_uv: (f32, f32), reg_vk: crate::step::brep::topo::VertexKey) -> Self {
        Self { edge_key: ek, singularity_3d: sing_3d, singularity_uv: sing_uv, regular_vertex: reg_vk }
    }
}

#[derive(Debug, Default)]
pub struct DegeneratedReport {
    pub degeneracies_found: usize,
    pub degenerate_edges_created: usize,
    pub degenerated_edge_infos: Vec<DegeneratedEdgeInfo>,
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
        for &(ek, orient) in &wire_edges {
            let (tolerance, pc) = {
                let edge = match reg.edges.get(ek) {
                    Some(e) => e,
                    None => continue,
                };
                let pc = match edge.pcurves.get(&face_key) {
                    Some(p) => p.clone(),
                    None => continue,
                };
                (edge.tolerance, pc)
            };

            let mut best_t = None;
            let mut best_dist = f32::MAX;
            for s in 0..=16 {
                let t = s as f32 / 16.0;
                let uv = pc.d0(t);
                let dist = ((uv.x - singularity.uv.0).powi(2)
                    + (uv.y - singularity.uv.1).powi(2))
                .sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_t = Some(t);
                }
            }

            let Some(t_sing) = best_t else { continue };
            if best_dist > 1e-3 {
                continue;
            }

            let pole_vk = reg.find_or_add_vertex(singularity.point_3d, tolerance);
            let uv_sing = pc.d0(t_sing);
            let uv_other = pc.d0(if t_sing < 0.5 { 1.0 } else { 0.0 });

            let split_parts = if t_sing > 1e-4 && t_sing < 1.0 - 1e-4 {
                split_edge_at_params(ek, face_key, orient, &[t_sing], reg)
            } else {
                vec![(ek, orient)]
            };

            let dek = add_degenerated_edge_at_pole(
                pole_vk,
                (uv_other.x, uv_other.y),
                (uv_sing.x, uv_sing.y),
                singularity.point_3d,
                tolerance,
                face_key,
                reg,
            );
            degen_edges.push(dek);
            report.degenerate_edges_created += 1;
            report.degenerated_edge_infos.push(DegeneratedEdgeInfo::new(
                dek,
                singularity.point_3d,
                singularity.uv,
                pole_vk,
            ));

            if split_parts.len() > 1 {
                if let Some(wire) = reg.wires.get_mut(outer_wire) {
                    let mut rebuilt = Vec::new();
                    for &(wek, worient) in &wire.edges {
                        if wek == ek {
                            rebuilt.extend(split_parts.clone());
                            rebuilt.push((dek, Orientation::Forward));
                        } else {
                            rebuilt.push((wek, worient));
                        }
                    }
                    wire.edges = rebuilt;
                }
            } else if let Some(wire) = reg.wires.get_mut(outer_wire) {
                let idx = wire
                    .edges
                    .iter()
                    .position(|&(wek, _)| wek == ek)
                    .unwrap_or(wire.edges.len());
                wire.edges.insert(idx + 1, (dek, Orientation::Forward));
            }

            break;
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
        SurfaceGeom::BSpline(nurbs) => find_bspline_singularities(nurbs),
        SurfaceGeom::Revolution { generatrix, axis_origin, axis_dir } => {
            find_revolution_singularities(generatrix, *axis_origin, *axis_dir)
        }
        _ => vec![],
    }
}

fn find_bspline_singularities(nurbs: &crate::step::nurbs::NurbsSurface) -> Vec<SingularityInfo> {
    // Sample the derivative grid at the parameter boundaries.
    // A singularity exists where |dS/du × dS/dv| ≈ 0.
    let mut result = Vec::new();
    let samples = 8;
    for iu in 0..=samples {
        let u = nurbs.knots_u[0] + (nurbs.knots_u[nurbs.knots_u.len() - 1] - nurbs.knots_u[0]) * iu as f32 / samples as f32;
        for iv in 0..=samples {
            let v = nurbs.knots_v[0] + (nurbs.knots_v[nurbs.knots_v.len() - 1] - nurbs.knots_v[0]) * iv as f32 / samples as f32;
            let d1 = nurbs.derivative(u, v);
            let cross = d1.0.cross(d1.1).length();
            if cross < 1e-6 {
                let pt = nurbs.evaluate(u, v);
                result.push(SingularityInfo { point_3d: pt, uv: (u, v) });
            }
        }
    }
    result
}

fn find_revolution_singularities(generatrix: &crate::step::brep::geom::CurveGeom, axis_origin: Vec3, axis_dir: Vec3) -> Vec<SingularityInfo> {
    // A revolution surface has a singularity where the generatrix touches the axis.
    // Check the generatrix endpoints: if either is on the axis, that's a pole.
    let axis_n = axis_dir.normalize();
    let mut result = Vec::new();
    for t in [0.0, 1.0] {
        let pt = generatrix.d0(t);
        let rel = pt - axis_origin;
        let proj = rel - axis_n * rel.dot(axis_n);
        if proj.length() < 1e-4 {
            result.push(SingularityInfo { point_3d: pt, uv: (0.0, t) });
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::CurveGeom;
    use crate::step::brep::topo::{BRepWire, Orientation};
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
        let surface = SurfaceGeom::cone(Vec3::ZERO, Vec3::Z, 0.5, 0.0);
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
    fn test_create_degenerated_edge_with_uv_extent() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 1.0 };
        // Create a wire edge whose PCurve passes near the north pole
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        // Vertex near equator, PCurve goes from equator to near north pole
        let v_eq = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let v_near_pole = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 1.0), 1e-4);
        let line = CurveGeom::Line { origin: Vec3::new(1.0, 0.0, 0.0), direction: Vec3::new(-1.0, 0.0, 1.0) };
        // PCurve: UV from (0,0) at the equator to near the north pole at (0, PI/2 - 1e-6)
        let pole_v = std::f32::consts::FRAC_PI_2 - 1e-6;
        let pc = CurveGeom::Line {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(0.0, pole_v, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v_eq, v_near_pole, line, 1e-4, fk, pc);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];

        let report = fix_degenerated_edges(fk, &mut reg);
        assert!(report.degenerate_edges_created > 0, "should create degenerated edges");
        // Verify degenerated edge has real UV extent (non-zero PCurve direction)
        let face = reg.faces.get(fk).unwrap();
        for &dek in &face.degenerated_edges {
            let edge = reg.edges.get(dek).unwrap();
            assert_eq!(
                edge.v_low, edge.v_high,
                "OCC degenerated edge should have v_low == v_high"
            );
            let pc = edge.pcurves.values().next().unwrap();
            let uv0 = pc.d0(0.0);
            let uv1 = pc.d0(1.0);
            let uv_len = ((uv0.x - uv1.x).powi(2) + (uv0.y - uv1.y).powi(2)).sqrt();
            assert!(uv_len > 1e-6, "degenerated edge PCurve should have real UV extent, got {}", uv_len);
        }
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

    #[test]
    fn test_degenerated_edges_on_face() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 1.0 };
        let v_np = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 1.0), 1e-4);
        let v_eq = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let line = CurveGeom::Line { origin: Vec3::new(0.0, 0.0, 1.0), direction: Vec3::new(1.0, 0.0, -1.0) };
        let pc = CurveGeom::Line {
            origin: Vec3::new(0.0, std::f32::consts::FRAC_PI_2 - 1e-6, 0.0),
            direction: Vec3::new(0.1, -std::f32::consts::FRAC_PI_2 + 1e-6, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v_np, v_eq, line, 1e-4, fk, pc);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let report = fix_degenerated_edges(fk, &mut reg);
        assert!(report.degenerate_edges_created > 0);
        let face = reg.faces.get(fk).unwrap();
        assert!(!face.degenerated_edges.is_empty());
        for &dek in &face.degenerated_edges {
            let edge = reg.edges.get(dek).unwrap();
            // Degenerated edge should have a PCurve with real UV extent
            let pc = edge.pcurves.values().next().unwrap();
            let uv0 = pc.d0(0.0);
            let uv1 = pc.d0(1.0);
            let uv_len = ((uv0.x - uv1.x).powi(2) + (uv0.y - uv1.y).powi(2)).sqrt();
            assert!(uv_len > 1e-6, "degenerated edge PCurve should have real UV extent, got {}", uv_len);
        }
    }
}
