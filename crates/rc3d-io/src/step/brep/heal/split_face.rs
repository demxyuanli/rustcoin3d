//! Face splitting for multiple outer wires (OCC ShapeFix_Face::FixSplitFace).
//! When a face has inner wires that are actually separate outer regions,
//! split them into distinct faces.

use std::collections::HashMap;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{EdgeKey, FaceKey, Orientation, ShellKey, WireKey};
use crate::step::brep::topo::BRepFace;

#[derive(Debug, Default)]
pub struct SplitFaceReport {
    pub faces_created: usize,
    pub wires_reassigned: usize,
}

/// Detect and split faces that have more than one outer wire.
/// After FixOrientation, inner wires should have opposite signed area from
/// the outer wire. Wires with the same orientation sign are actually outer
/// wires for separate face regions and should be split into new faces.
pub fn fix_split_face(
    shell_key: ShellKey,
    reg: &mut BRepRegistry,
) -> SplitFaceReport {
    let mut report = SplitFaceReport::default();

    // Collect existing faces and their wires
    let shell_faces: Vec<(FaceKey, Orientation)> = {
        let Some(shell) = reg.shells.get(shell_key) else { return report; };
        shell.faces.clone()
    };

    let mut new_faces: Vec<(FaceKey, Orientation)> = Vec::new();

    for &(face_key, shell_orient) in &shell_faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        if face.inner_wires.is_empty() {
            new_faces.push((face_key, shell_orient));
            continue;
        }

        let surface = face.surface.clone();
        let tolerance = face.tolerance;
        let same_sense = face.same_sense;
        let seam_edges = face.seam_edges.clone();
        let color = face.color;

        // Compute signed area of the outer wire to determine its orientation
        let outer_area = match wire_signed_area_uv(face.outer_wire, face_key, reg) {
            Some(a) => a,
            None => {
                new_faces.push((face_key, shell_orient));
                continue;
            }
        };

        let outer_positive = outer_area > 0.0;
        let mut keep_inner = Vec::new();
        let mut split_wires = Vec::new();

        for &inner_wk in &face.inner_wires {
            let inner_area = match wire_signed_area_uv(inner_wk, face_key, reg) {
                Some(a) => a,
                None => {
                    keep_inner.push(inner_wk);
                    continue;
                }
            };
            let inner_positive = inner_area > 0.0;

            // If inner wire has the SAME sign as the outer wire, it's actually
            // an outer wire for a separate face region — split it out
            if inner_positive == outer_positive {
                split_wires.push(inner_wk);
            } else {
                keep_inner.push(inner_wk);
            }
        }

        // Update the original face to remove split wires
        if split_wires.is_empty() {
            new_faces.push((face_key, shell_orient));
            continue;
        }

        if let Some(face_mut) = reg.faces.get_mut(face_key) {
            face_mut.inner_wires = keep_inner;
        }
        new_faces.push((face_key, shell_orient));
        report.wires_reassigned += split_wires.len();

        // Create new faces for each split wire
        for wk in split_wires {
            let new_face_key = reg.faces.insert(BRepFace {
                surface: surface.clone(),
                outer_wire: wk,
                inner_wires: vec![],
                same_sense,
                tolerance,
                seam_edges: seam_edges.clone(),
                color,
                degenerated_edges: vec![],
            });
            new_faces.push((new_face_key, shell_orient));
            report.faces_created += 1;
            log::debug!(
                "[BRep heal] FixSplitFace: created new face {:?} from inner wire {:?} of face {:?}",
                new_face_key, wk, face_key
            );
        }
    }

    // Update shell with (possibly) additional faces
    if report.faces_created > 0 {
        if let Some(shell) = reg.shells.get_mut(shell_key) {
            shell.faces = new_faces;
        }
    }

    report
}

/// Compute signed area of a wire in UV space using PCurve endpoints.
fn wire_signed_area_uv(wk: WireKey, fk: FaceKey, reg: &BRepRegistry) -> Option<f32> {
    let wire = reg.wires.get(wk)?;
    let mut pts: Vec<(f32, f32)> = Vec::new();
    for &(ek, _) in &wire.edges {
        let edge = reg.edges.get(ek)?;
        let pc = edge.pcurves.get(&fk)?;
        let uv = pc.d0(0.0);
        pts.push((uv.x, uv.y));
    }
    if pts.len() < 3 {
        return None;
    }
    // Close the polygon
    if let Some(&(last_ek, _)) = wire.edges.last() {
        if let Some(edge) = reg.edges.get(last_ek) {
            if let Some(pc) = edge.pcurves.get(&fk) {
                let uv = pc.d0(1.0);
                pts.push((uv.x, uv.y));
            }
        }
    }
    let n = pts.len();
    if n < 3 {
        return None;
    }
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].0 * pts[j].1;
        area -= pts[j].0 * pts[i].1;
    }
    Some(area * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::topo::BRepWire;
    use rc3d_core::math::Vec3;

    #[test]
    fn test_no_split_single_outer() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(2.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(2.0, 2.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 2.0, 0.0), 1e-4);

        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let pc = CurveGeom::Line { origin: Vec3::new(0.0, 0.0, 0.0), direction: Vec3::new(2.0, 0.0, 0.0) };
        let wk_outer = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface, outer_wire: wk_outer, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone());
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc.clone());
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc.clone());
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc);
        reg.wires.get_mut(wk_outer).unwrap().edges = vec![
            (e1, Orientation::Forward), (e2, Orientation::Forward),
            (e3, Orientation::Forward), (e4, Orientation::Forward),
        ];
        let sk = reg.shells.insert(crate::step::brep::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None,
        });
        let report = fix_split_face(sk, &mut reg);
        assert_eq!(report.faces_created, 0, "single outer wire should not split");
    }

    #[test]
    fn test_split_face_with_same_sign_inner() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };

        // Outer wire: CCW square (positive area)
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(3.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(3.0, 3.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(Vec3::new(0.0, 3.0, 0.0), 1e-4);
        let wk_outer = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface, outer_wire: wk_outer, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let pc = CurveGeom::Line { origin: Vec3::new(0.0, 0.0, 0.0), direction: Vec3::new(3.0, 0.0, 0.0) };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone());
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc.clone());
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc.clone());
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc);
        reg.wires.get_mut(wk_outer).unwrap().edges = vec![
            (e1, Orientation::Forward), (e2, Orientation::Forward),
            (e3, Orientation::Forward), (e4, Orientation::Forward),
        ];

        // "Inner" wire that's also CCW (same sign as outer) — should be split out
        let v4 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let v5 = reg.find_or_add_vertex(Vec3::new(2.0, 1.0, 0.0), 1e-4);
        let v6 = reg.find_or_add_vertex(Vec3::new(2.0, 2.0, 0.0), 1e-4);
        let v7 = reg.find_or_add_vertex(Vec3::new(1.0, 2.0, 0.0), 1e-4);
        let pc_inner = CurveGeom::Line { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };
        let ei1 = reg.add_edge_with_pcurve(v4, v5, line.clone(), 1e-4, fk, pc_inner.clone());
        let ei2 = reg.add_edge_with_pcurve(v5, v6, line.clone(), 1e-4, fk, pc_inner.clone());
        let ei3 = reg.add_edge_with_pcurve(v6, v7, line.clone(), 1e-4, fk, pc_inner.clone());
        let ei4 = reg.add_edge_with_pcurve(v7, v4, line.clone(), 1e-4, fk, pc_inner);
        let wk_inner = reg.wires.insert(BRepWire {
            edges: vec![
                (ei1, Orientation::Forward), (ei2, Orientation::Forward),
                (ei3, Orientation::Forward), (ei4, Orientation::Forward),
            ],
        });
        if let Some(face) = reg.faces.get_mut(fk) {
            face.inner_wires = vec![wk_inner];
        }
        let sk = reg.shells.insert(crate::step::brep::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None,
        });
        let report = fix_split_face(sk, &mut reg);
        assert_eq!(report.faces_created, 1, "same-sign inner wire should be split into new face");
        let shell = reg.shells.get(sk).unwrap();
        assert_eq!(shell.faces.len(), 2, "shell should have 2 faces after split");
    }
}
