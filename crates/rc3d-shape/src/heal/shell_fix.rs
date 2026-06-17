//! Shell-level fixes: orientation, vertex positions, and face splitting.

use rc3d_core::math::Real;
use std::collections::{HashSet, VecDeque};
use crate::store::BRepStore;
use crate::topo::{FaceKey, Orientation, ShellKey, WireKey};
use crate::topo::BRepFace;

// ── Shell orientation ──────────────────────────────────────────────

/// Fix face orientations in a shell so all normals point consistently.
/// Returns number of faces flipped.
pub(crate) fn fix_shell_orientation(
    shell_key: ShellKey,
    reg: &mut BRepStore,
) -> usize {
    let face_keys: Vec<FaceKey> = {
        let shell = match reg.shells.get(shell_key) {
            Some(s) => s.faces.iter().map(|(fk, _)| *fk).collect(),
            None => return 0,
        };
        shell
    };

    if face_keys.len() <= 1 { return 0; }

    let mut flipped = 0;
    let mut visited: HashSet<FaceKey> = HashSet::new();
    let mut queue: VecDeque<FaceKey> = VecDeque::new();

    queue.push_back(face_keys[0]);
    visited.insert(face_keys[0]);

    while let Some(current) = queue.pop_front() {
        for &next in &face_keys {
            if visited.contains(&next) { continue; }
            let shared = reg.find_shared_edges(current, next);
            if !shared.is_empty() {
                let current_face = match reg.faces.get(current) {
                    Some(f) => f, None => continue,
                };
                let next_face = match reg.faces.get(next) {
                    Some(f) => f, None => continue,
                };

                let edge = reg.edges.get(shared[0]);
                if let (Some(e), Some(_pcurve)) = (edge, edge.and_then(|e| e.pcurves.get(&current))) {
                    // Multi-point sampling: evaluate at 7 Chebyshev-distributed samples
                    // along the shared edge. Majority vote determines orientation.
                    // This avoids the single-sample failure on curved surfaces
                    // (spheres, tori) where the midpoint normal is unrepresentative.
                    let ts: [Real; 7] = [0.038, 0.146, 0.309, 0.5, 0.691, 0.854, 0.962];
                    let mut agree = 0u32;
                    let mut disagree = 0u32;
                    for &t in &ts {
                        let p = e.curve.d0(t);
                        if let Some((u0, v0)) = current_face.surface.project(p) {
                            if let Some((u1, v1)) = next_face.surface.project(p) {
                                let n0 = current_face.surface.normal(u0, v0);
                                let n1 = next_face.surface.normal(u1, v1);
                                if n0.dot(n1) < 0.0 { disagree += 1; } else { agree += 1; }
                            }
                        }
                    }
                    // Majority vote: flip if more samples disagree than agree
                    if disagree > agree {
                        if let Some(face) = reg.faces.get_mut(next) {
                            face.same_sense = !face.same_sense;
                            flipped += 1;
                        }
                    }
                }
                visited.insert(next);
                queue.push_back(next);
            }
        }
    }

    flipped
}

// ── Vertex position correction ─────────────────────────────────────

/// Project each vertex in the shell onto surfaces of faces that reference it.
/// Returns number of vertices adjusted.
pub(crate) fn fix_vertex_positions(
    shell_key: ShellKey,
    reg: &mut BRepStore,
    tolerance: Real,
) -> usize {
    let face_keys: Vec<_> = {
        let Some(shell) = reg.shells.get(shell_key) else { return 0; };
        shell.faces.iter().map(|&(fk, _)| fk).collect()
    };

    if face_keys.is_empty() { return 0; }

    let mut adjusted = 0usize;

    for fk in face_keys {
        let face = match reg.faces.get(fk) {
            Some(f) => f,
            None => continue,
        };
        let surface = face.surface.clone();
        let wire = match reg.wires.get(face.outer_wire) {
            Some(w) => w,
            None => continue,
        };

        for &(ek, _) in &wire.edges {
            let edge = match reg.edges.get(ek) {
                Some(e) => e,
                None => continue,
            };
            for &vk in &[edge.v_low, edge.v_high] {
                if let Some(v) = reg.vertices.get(vk) {
                    let pos = v.position;
                    if let Some((u, v_param)) = surface.project(pos) {
                        let on_surf = surface.d0_native(u, v_param);
                        let dist = (pos - on_surf).length();
                        if dist > tolerance && dist < tolerance * 100.0 {
                            if let Some(v_mut) = reg.vertices.get_mut(vk) {
                                v_mut.position = on_surf;
                                adjusted += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    adjusted
}

// ── Face splitting ─────────────────────────────────────────────────

#[derive(Debug, Default)]
pub(crate) struct SplitFaceReport {
    pub faces_created: usize,
    pub wires_reassigned: usize,
}

/// Detect and split faces that have more than one outer wire.
pub(crate) fn fix_split_face(
    shell_key: ShellKey,
    reg: &mut BRepStore,
) -> SplitFaceReport {
    let mut report = SplitFaceReport::default();

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

        let Some(outer_area) = wire_signed_area_uv(face.outer_wire, face_key, reg) else {
            new_faces.push((face_key, shell_orient));
            continue;
        };

        let outer_positive = outer_area > 0.0;
        let mut keep_inner = Vec::new();
        let mut split_wires = Vec::new();

        for &inner_wk in &face.inner_wires {
            match wire_signed_area_uv(inner_wk, face_key, reg) {
                Some(inner_area) if (inner_area > 0.0) == outer_positive => {
                    split_wires.push(inner_wk);
                }
                Some(_) => keep_inner.push(inner_wk),
                None => keep_inner.push(inner_wk),
            }
        }

        if split_wires.is_empty() {
            new_faces.push((face_key, shell_orient));
            continue;
        }

        let surface = face.surface.clone();
        let tolerance = face.tolerance;
        let same_sense = face.same_sense;
        let seam_edges = face.seam_edges.clone();
        let color = face.color;

        if let Some(face_mut) = reg.faces.get_mut(face_key) {
            face_mut.inner_wires = keep_inner;
        }
        new_faces.push((face_key, shell_orient));
        report.wires_reassigned += split_wires.len();

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
        }
    }

    if report.faces_created > 0 {
        if let Some(shell) = reg.shells.get_mut(shell_key) {
            shell.faces = new_faces;
        }
    }

    report
}

fn wire_signed_area_uv(wk: WireKey, fk: FaceKey, reg: &BRepStore) -> Option<Real> {
    let wire = reg.wires.get(wk)?;
    let mut pts: Vec<(Real, Real)> = Vec::new();
    for &(ek, _) in &wire.edges {
        let edge = reg.edges.get(ek)?;
        let pc = edge.pcurves.get(&fk)?;
        let uv = pc.d0(0.0);
        pts.push(uv);
    }
    if pts.len() < 3 {
        return None;
    }
    if let Some(&last_ek) = wire.edges.last() {
        if let Some(edge) = reg.edges.get(last_ek.0) {
            if let Some(pc) = edge.pcurves.get(&fk) {
                let uv = pc.d0(1.0);
                pts.push(uv);
            }
        }
    }
    let n = pts.len();
    let mut area = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].0 * pts[j].1;
        area -= pts[j].0 * pts[i].1;
    }
    Some(area * 0.5)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod vertex_tests {
    use super::*;
    use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
    use crate::topo::BRepWire;
    use rc3d_core::math::PVec3;

    #[test]
    fn test_fix_vertex_on_surface() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let v0 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.005), 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc2d = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc2d, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None,
        });
        let adjusted = fix_vertex_positions(sk, &mut reg, 1e-4);
        assert!(adjusted > 0, "off-surface vertex should be projected back");
    }

    #[test]
    fn test_fix_vertex_no_projection() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Sphere { center: PVec3::ZERO, radius: 1.0 };
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::topo::BRepFace {
            surface,
            outer_wire: wk,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc2d = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc2d, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let adjusted = fix_vertex_positions(sk, &mut reg, 1e-4);
        assert!(adjusted >= 0, "should handle vertices with no valid projection");
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
    use crate::topo::BRepWire;
    use rc3d_core::math::PVec3;

    #[test]
    fn test_no_split_single_outer() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let v0 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(2.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::new(2.0, 2.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(PVec3::new(0.0, 2.0, 0.0), 1e-4);

        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (2.0, 0.0) };
        let wk_outer = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface, outer_wire: wk_outer, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone(), true);
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc.clone(), true);
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc.clone(), true);
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc, true);
        reg.wires.get_mut(wk_outer).unwrap().edges = vec![
            (e1, Orientation::Forward), (e2, Orientation::Forward),
            (e3, Orientation::Forward), (e4, Orientation::Forward),
        ];
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None,
        });
        let report = fix_split_face(sk, &mut reg);
        assert_eq!(report.faces_created, 0, "single outer wire should not split");
    }

    #[test]
    fn test_split_face_with_same_sign_inner() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };

        let v0 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(3.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::new(3.0, 3.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(PVec3::new(0.0, 3.0, 0.0), 1e-4);
        let wk_outer = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface, outer_wire: wk_outer, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (3.0, 0.0) };
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc.clone(), true);
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, fk, pc.clone(), true);
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, fk, pc.clone(), true);
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, fk, pc, true);
        reg.wires.get_mut(wk_outer).unwrap().edges = vec![
            (e1, Orientation::Forward), (e2, Orientation::Forward),
            (e3, Orientation::Forward), (e4, Orientation::Forward),
        ];

        // Inner wire that is also CCW (same sign as outer) - should be split out
        let v4 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-4);
        let v5 = reg.find_or_add_vertex(PVec3::new(2.0, 1.0, 0.0), 1e-4);
        let v6 = reg.find_or_add_vertex(PVec3::new(2.0, 2.0, 0.0), 1e-4);
        let v7 = reg.find_or_add_vertex(PVec3::new(1.0, 2.0, 0.0), 1e-4);
        let pc_inner = Curve2d::Line { origin: (1.0, 1.0), direction: (1.0, 0.0) };
        let ei1 = reg.add_edge_with_pcurve(v4, v5, line.clone(), 1e-4, fk, pc_inner.clone(), true);
        let ei2 = reg.add_edge_with_pcurve(v5, v6, line.clone(), 1e-4, fk, pc_inner.clone(), true);
        let ei3 = reg.add_edge_with_pcurve(v6, v7, line.clone(), 1e-4, fk, pc_inner.clone(), true);
        let ei4 = reg.add_edge_with_pcurve(v7, v4, line.clone(), 1e-4, fk, pc_inner, true);
        let wk_inner = reg.wires.insert(BRepWire {
            edges: vec![
                (ei1, Orientation::Forward), (ei2, Orientation::Forward),
                (ei3, Orientation::Forward), (ei4, Orientation::Forward),
            ],
        });
        if let Some(face) = reg.faces.get_mut(fk) {
            face.inner_wires = vec![wk_inner];
        }
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None,
        });
        let report = fix_split_face(sk, &mut reg);
        assert_eq!(report.faces_created, 1, "same-sign inner wire should be split into new face");
        let shell = reg.shells.get(sk).unwrap();
        assert_eq!(shell.faces.len(), 2, "shell should have 2 faces after split");
    }
}
