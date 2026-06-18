//! BRepCheck_Shell — shell closure, non-manifold, Euler-Poincare, and unorientable detection.

use std::collections::{HashMap, HashSet};

use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation, ShellKey, VertexKey, WireKey};
use crate::topo_iter;

use super::CheckStatus;

// ── ShellClosedReport ──────────────────────────────────────────────────────

/// Result of shell closure analysis.
#[derive(Debug, Clone, Default)]
pub struct ShellClosedReport {
    /// Edges shared by exactly 2 faces (correct for closed shell).
    pub closed_edges: usize,
    /// Total unique edges referenced by the shell's faces.
    pub total_edges: usize,
    /// Edges referenced by 0 or 1 face (open / dangling boundary).
    pub open_edges: Vec<EdgeKey>,
    /// Edges referenced by >2 faces (non-manifold topology).
    pub non_manifold_edges: Vec<(EdgeKey, usize)>,
    /// True when every edge is shared by exactly 2 faces and total_edges > 0.
    pub is_closed: bool,
}

/// Check shell closure: every edge must be shared by exactly 2 faces.
///
/// A closed (watertight) shell has each edge referenced exactly twice --
/// once per adjacent face. Open edges (count=1) indicate a boundary;
/// non-manifold edges (count>2) indicate topology errors.
///
/// OCC alignment: BRepCheck_Shell -- counts face references per edge.
pub fn check_shell_closed(shell_key: ShellKey, reg: &BRepStore) -> ShellClosedReport {
    let mut report = ShellClosedReport::default();
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return report,
    };

    let mut edge_refs: HashMap<EdgeKey, u32> = HashMap::new();

    for &(face_key, _) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wire_keys: Vec<WireKey> = std::iter::once(face.outer_wire)
            .chain(face.inner_wires.iter().copied())
            .collect();
        for wk in wire_keys {
            let wire = match reg.wires.get(wk) {
                Some(w) => w,
                None => continue,
            };
            for &(ek, _) in &wire.edges {
                *edge_refs.entry(ek).or_default() += 1;
            }
        }
    }

    report.total_edges = edge_refs.len();
    for (ek, count) in &edge_refs {
        match count {
            2 => report.closed_edges += 1,
            0 | 1 => report.open_edges.push(*ek),
            n => report.non_manifold_edges.push((*ek, *n as usize)),
        }
    }
    report.is_closed = report.open_edges.is_empty()
        && report.non_manifold_edges.is_empty()
        && report.total_edges > 0;
    report
}

// ── Euler-Poincare ─────────────────────────────────────────────────────────

/// Euler-Poincare formula validation for shells.
/// For a closed manifold shell: V - E + F = 2(1 - genus)
/// genus=0 (sphere-like) -> V - E + F = 2
/// Returns None if shell has no faces.
pub fn check_euler_poincare(shell_key: ShellKey, reg: &BRepStore) -> Option<i32> {
    let shell = reg.shells.get(shell_key)?;
    if shell.faces.is_empty() {
        return None;
    }

    let mut edge_set: HashSet<EdgeKey> = HashSet::new();
    let mut vertex_set: HashSet<VertexKey> = HashSet::new();

    for &(fk, _) in &shell.faces {
        let _ = reg.faces.get(fk)?;
        for ek in topo_iter::iter_edges_of_face(fk, reg) {
            edge_set.insert(ek);
            if let Some((vl, vh)) = topo_iter::iter_vertices_of_edge(ek, reg) {
                vertex_set.insert(vl);
                vertex_set.insert(vh);
            }
        }
    }

    let v = vertex_set.len() as i32;
    let e = edge_set.len() as i32;
    let f = shell.faces.len() as i32;

    Some(v - e + f)
}

// ── Non-manifold ───────────────────────────────────────────────────────────

/// Check for non-manifold edges (shared by more than 2 faces).
pub fn check_non_manifold(
    face_keys: &[(FaceKey, Orientation)],
    reg: &BRepStore,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let mut edge_face_count: HashMap<EdgeKey, usize> = HashMap::new();
    for &(face_key, _) in face_keys {
        for (ek, _) in topo_iter::iter_edge_orientations_of_face(face_key, reg) {
            *edge_face_count.entry(ek).or_default() += 1;
        }
    }
    for (ek, count) in edge_face_count {
        if count > 2 {
            warnings.push(format!(
                "edge {:?} is non-manifold (shared by {} faces)",
                ek, count
            ));
        }
    }
    warnings
}

// ── Unorientable shape detection ───────────────────────────────────────────

/// Detect potentially unorientable topology (Mobius-band-type).
///
/// A non-orientable shell has at least one edge where both adjacent faces
/// traverse the edge in the SAME direction (both Forward or both Reversed),
/// which means the normal orientation flips across that edge.
///
/// This is a basic heuristic that checks for orientation mismatch across
/// shared edges. A more thorough check would trace face normals.
pub fn check_unorientable(shell_key: ShellKey, reg: &BRepStore) -> Vec<CheckStatus> {
    let mut statuses = Vec::new();
    let shell = match reg.shells.get(shell_key) {
        Some(s) => s,
        None => return statuses,
    };

    // For each edge, collect the wire orientations used by faces
    let mut edge_usages: HashMap<EdgeKey, Vec<Orientation>> = HashMap::new();

    for &(face_key, _) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wire_keys: Vec<WireKey> = std::iter::once(face.outer_wire)
            .chain(face.inner_wires.iter().copied())
            .collect();
        for wk in wire_keys {
            let wire = match reg.wires.get(wk) {
                Some(w) => w,
                None => continue,
            };
            for &(ek, orient) in &wire.edges {
                edge_usages.entry(ek).or_default().push(orient);
            }
        }
    }

    // Check if any edge has both faces traversing it the same way
    for (_ek, orients) in &edge_usages {
        if orients.len() == 2 {
            // Different faces traversing the same edge with same orientation
            // is the key indicator of a non-orientable topology.
            // For orientable shells, adjacent faces should traverse shared
            // edges in opposite directions.
            if orients[0] == orients[1] {
                statuses.push(CheckStatus::UnorientableShape);
                break;
            }
        }
    }

    statuses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepEdge, BRepFace, BRepShell, BRepWire, Orientation};
    use rc3d_core::math::PVec3;

    fn closed_cube_shell(reg: &mut BRepStore) -> ShellKey {
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: PVec3::ZERO,
                normal: PVec3::Z,
                u_dir: PVec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let edges_data = [
            (
                PVec3::ZERO,
                PVec3::new(10.0, 0.0, 0.0),
                (0.0, 0.0),
                (10.0, 0.0),
            ),
            (
                PVec3::new(10.0, 0.0, 0.0),
                PVec3::new(10.0, 10.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
            ),
            (
                PVec3::new(10.0, 10.0, 0.0),
                PVec3::new(0.0, 10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
            ),
            (
                PVec3::new(0.0, 10.0, 0.0),
                PVec3::ZERO,
                (0.0, 10.0),
                (0.0, 0.0),
            ),
        ];
        let mut wire_edges = Vec::new();
        for (a, b, u0, u1) in edges_data {
            let v0 = reg.find_or_add_vertex(a, 1e-4);
            let v1 = reg.find_or_add_vertex(b, 1e-4);
            let curve_3d = CurveGeom::Line {
                origin: a,
                direction: b - a,
            };
            let pcurve = Curve2d::Line {
                origin: (u0.0, u0.1),
                direction: (u1.0 - u0.0, u1.1 - u0.1),
            };
            let ek =
                reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve, true);
            let edge = reg.edges.get(ek).unwrap();
            let orient = if edge.v_low == v0 {
                Orientation::Forward
            } else {
                Orientation::Reversed
            };
            wire_edges.push((ek, orient));
        }
        let outer_wire = reg.wires.insert(BRepWire {
            edges: wire_edges,
        });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer_wire;
        }
        reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: true,
            step_id: None,
        })
    }

    #[test]
    fn test_check_shell_closed_plane_not_closed() {
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        let report = check_shell_closed(sk, &reg);
        assert!(!report.is_closed, "single-face shell should not be closed");
        assert_eq!(report.total_edges, 0, "no edges in face wire");
    }

    #[test]
    fn test_euler_poincare_single_face() {
        let mut reg = BRepStore::new();
        let sk = closed_cube_shell(&mut reg);
        let chi = check_euler_poincare(sk, &reg);
        assert_eq!(chi, Some(1), "single-face shell should have chi=1");
    }

    #[test]
    fn test_euler_poincare_empty_shell() {
        let mut reg = BRepStore::new();
        let sk = reg.shells.insert(BRepShell {
            faces: vec![],
            closed: false,
            step_id: None,
        });
        let chi = check_euler_poincare(sk, &reg);
        assert_eq!(chi, None, "empty shell should return None");
    }

    #[test]
    fn test_non_manifold_edge_yields_warning() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let curve = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let ek = reg.edges.insert(BRepEdge {
            v_low: v0,
            v_high: v1,
            curve: curve.clone(),
            tolerance: 1e-4,
            t_min: 0.0,
            t_max: 1.0,
            cached_deflection: None,
            pcurves: HashMap::new(),
        });
        let mut face_keys = Vec::new();
        for _ in 0..3 {
            let wire = reg.wires.insert(BRepWire {
                edges: vec![(ek, Orientation::Forward)],
            });
            face_keys.push(reg.faces.insert(BRepFace {
                surface: SurfaceGeom::Plane {
                    origin: PVec3::ZERO,
                    normal: PVec3::Z,
                    u_dir: PVec3::X,
                },
                outer_wire: wire,
                inner_wires: vec![],
                same_sense: true,
                tolerance: 1e-4,
                seam_edges: vec![],
                color: None,
                degenerated_edges: vec![],
            }));
        }
        let faces: Vec<_> = face_keys
            .iter()
            .map(|&fk| (fk, Orientation::Forward))
            .collect();
        let warnings = check_non_manifold(&faces, &reg);
        assert!(
            !warnings.is_empty(),
            "edge shared by 3 faces should be flagged non-manifold"
        );
    }

    #[test]
    fn test_unorientable_no_flag_on_single_face() {
        let mut reg = BRepStore::new();
        let sk = closed_cube_shell(&mut reg);
        let statuses = check_unorientable(sk, &reg);
        // Single face shell: the four edges each appear only once (in one wire),
        // so there are no edge-sharing orientation mismatches.
        assert!(
            !statuses.contains(&CheckStatus::UnorientableShape),
            "single face should not be unorientable"
        );
    }
}
