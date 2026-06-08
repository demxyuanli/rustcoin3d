//! Geometric continuity check along shared edges (OCC ShapeAnalysis).
//! Detects G0 (positional gap) and G1 (tangential mismatch) defects.

use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, ShellKey};

#[derive(Debug, Clone)]
pub struct ContinuityDefect {
    pub edge_key: EdgeKey,
    pub face_a: FaceKey,
    pub face_b: FaceKey,
    pub kind: ContinuityKind,
    pub max_deviation: f32,
}

#[derive(Debug, Clone)]
pub enum ContinuityKind {
    G0 { max_gap: f32 },
    G1 { max_angle_deg: f32 },
}

/// Check G0/G1 continuity along all shared edges in a shell.
pub fn check_shell_continuity(
    shell_key: ShellKey,
    reg: &BRepStore,
    g0_tolerance: f32,
    g1_angle_tolerance_deg: f32,
) -> Vec<ContinuityDefect> {
    let mut defects = Vec::new();

    let face_keys: Vec<FaceKey> = {
        let Some(shell) = reg.shells.get(shell_key) else {
            return defects;
        };
        shell.faces.iter().map(|&(fk, _)| fk).collect()
    };

    for i in 0..face_keys.len() {
        for j in (i + 1)..face_keys.len() {
            let shared = reg.find_shared_edges(face_keys[i], face_keys[j]);
            for ek in shared {
                let defect = check_edge_continuity(
                    ek,
                    face_keys[i],
                    face_keys[j],
                    reg,
                    g0_tolerance,
                    g1_angle_tolerance_deg,
                );
                if let Some(d) = defect {
                    defects.push(d);
                }
            }
        }
    }

    defects
}

fn check_edge_continuity(
    ek: EdgeKey,
    fa: FaceKey,
    fb: FaceKey,
    reg: &BRepStore,
    g0_tol: f32,
    g1_tol_deg: f32,
) -> Option<ContinuityDefect> {
    let edge = reg.edges.get(ek)?;
    let face_a = reg.faces.get(fa)?;
    let face_b = reg.faces.get(fb)?;

    let samples = 4;
    let mut max_gap = 0.0f32;
    let mut max_angle = 0.0f32;

    for s in 0..=samples {
        let t = s as f32 / samples as f32;
        let p3d = edge.curve.d0(t);

        // Project onto each face surface using native UV
        let (ua, va) = face_a.surface.project(p3d)?;
        let (ub, vb) = face_b.surface.project(p3d)?;

        let pa = face_a.surface.d0_native(ua, va);
        let pb = face_b.surface.d0_native(ub, vb);

        let gap = (pa - pb).length();
        max_gap = max_gap.max(gap);

        let na = face_a.surface.normal_native(ua, va);
        let nb = face_b.surface.normal_native(ub, vb);
        let dot = na.normalize().dot(nb.normalize()).max(-1.0).min(1.0);
        let angle = dot.acos().to_degrees();
        max_angle = max_angle.max(angle);
    }

    if max_gap > g0_tol {
        Some(ContinuityDefect {
            edge_key: ek,
            face_a: fa,
            face_b: fb,
            kind: ContinuityKind::G0 { max_gap },
            max_deviation: max_gap,
        })
    } else if max_angle > g1_tol_deg {
        Some(ContinuityDefect {
            edge_key: ek,
            face_a: fa,
            face_b: fb,
            kind: ContinuityKind::G1 {
                max_angle_deg: max_angle,
            },
            max_deviation: max_angle,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepWire, BRepShell, Orientation};
    use rc3d_core::math::Vec3;

    fn make_plane_face(reg: &mut BRepStore, surface: SurfaceGeom) -> FaceKey {
        reg.add_face(surface, 1e-4)
    }

    #[test]
    fn test_g0_continuous_shell() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(Vec3::new(1.0, 1.0, 0.0), 1e-4);

        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let f1 = make_plane_face(&mut reg, surface.clone());
        let f2 = make_plane_face(&mut reg, surface);
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, pc.clone());
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, f1, pc.clone());
        reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f2, pc);

        reg.wires
            .get_mut(
                reg.faces
                    .get(f1)
                    .unwrap()
                    .outer_wire,
            )
            .unwrap()
            .edges = vec![(e1, Orientation::Forward), (e2, Orientation::Forward)];
        reg.wires
            .get_mut(
                reg.faces
                    .get(f2)
                    .unwrap()
                    .outer_wire,
            )
            .unwrap()
            .edges = vec![(e1, Orientation::Forward)];

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(f1, Orientation::Forward), (f2, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let defects = check_shell_continuity(sk, &reg, 0.01, 1.0);
        assert!(
            defects.is_empty(),
            "coplanar faces should have no G0 defects, got {:?}",
            defects
        );
    }

    #[test]
    fn test_g1_discontinuity_at_seam() {
        let mut reg = BRepStore::new();
        // Two faces meeting at 90 degrees along a shared edge
        let surface1 = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        let surface2 = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Y, u_dir: Vec3::X };
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let wk1 = reg.wires.insert(BRepWire { edges: vec![] });
        let f1 = reg.faces.insert(crate::topo::BRepFace {
            surface: surface1, outer_wire: wk1, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let wk2 = reg.wires.insert(BRepWire { edges: vec![] });
        let f2 = reg.faces.insert(crate::topo::BRepFace {
            surface: surface2, outer_wire: wk2, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, pc.clone());
        reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f2, pc);
        reg.wires.get_mut(wk1).unwrap().edges = vec![(ek, Orientation::Forward)];
        reg.wires.get_mut(wk2).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(f1, Orientation::Forward), (f2, Orientation::Forward)],
            closed: false, step_id: None,
        });
        // With strict G1 tolerance, 90-degree faces should produce a G1 defect
        let defects = check_shell_continuity(sk, &reg, 0.01, 1.0);
        // 90 degrees > 1 degree, so G1 defect expected
        assert!(!defects.is_empty(), "90-degree faces should produce G1 defect");
        assert!(
            defects
                .iter()
                .any(|d| matches!(d.kind, ContinuityKind::G1 { .. })),
            "expected G1 defect kind"
        );
    }

    #[test]
    fn test_g1_only_no_g0() {
        let mut reg = BRepStore::new();
        let surface1 = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Z,
            u_dir: Vec3::X,
        };
        let surface2 = SurfaceGeom::Plane {
            origin: Vec3::ZERO,
            normal: Vec3::Y,
            u_dir: Vec3::X,
        };
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let wk1 = reg.wires.insert(BRepWire { edges: vec![] });
        let f1 = reg.faces.insert(crate::topo::BRepFace {
            surface: surface1,
            outer_wire: wk1,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let wk2 = reg.wires.insert(BRepWire { edges: vec![] });
        let f2 = reg.faces.insert(crate::topo::BRepFace {
            surface: surface2,
            outer_wire: wk2,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, pc.clone());
        reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f2, pc);
        reg.wires.get_mut(wk1).unwrap().edges = vec![(ek, Orientation::Forward)];
        reg.wires.get_mut(wk2).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(crate::topo::BRepShell {
            faces: vec![(f1, Orientation::Forward), (f2, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let defects = check_shell_continuity(sk, &reg, 1e-6, 1.0);
        assert!(
            defects.iter().all(|d| !matches!(d.kind, ContinuityKind::G0 { .. })),
            "coplanar junction should not report G0 when gap is zero"
        );
    }
}
