//! BRepCheck_Solid — solid orientation and enclosed region validation.

use rc3d_core::math::Real;

use crate::store::BRepStore;
use crate::topo::{Orientation, SolidKey};
use super::CheckStatus;

/// Verify that the solid's shells have outward-pointing normals.
///
/// Computes a signed volume approximation from the outer shell's faces.
/// The signed volume of a closed tessellated surface is computed as
/// the sum of signed tetrahedron volumes from each triangle.
///
/// A negative signed volume means the normals point inward (bad orientation).
///
/// OCC alignment: BRepCheck_Solid — checks shell orientation.
pub fn check_solid_orientation(sk: SolidKey, reg: &BRepStore) -> Vec<CheckStatus> {
    let mut statuses = Vec::new();
    let solid = match reg.solids.get(sk) {
        Some(s) => s,
        None => return statuses,
    };

    let shell = match reg.shells.get(solid.outer_shell) {
        Some(s) => s,
        None => return statuses,
    };

    // Compute approximate signed volume from face polygons.
    // For each face, compute a centroid and area-weighted normal contribution.
    // This is an approximation; a full tessellation-based volume would be
    // more accurate but much more expensive.
    let mut signed_volume: Real = 0.0;

    for &(face_key, face_orient) in &shell.faces {
        let face = match reg.faces.get(face_key) {
            Some(f) => f,
            None => continue,
        };
        let wire = match reg.wires.get(face.outer_wire) {
            Some(w) => w,
            None => continue,
        };

        // Collect polygon vertices in 3D
        let mut points: Vec<rc3d_core::math::PVec3> = Vec::new();
        for &(ek, orient) in &wire.edges {
            let edge = match reg.edges.get(ek) {
                Some(e) => e,
                None => continue,
            };
            let vk = if orient == Orientation::Forward {
                edge.v_low
            } else {
                edge.v_high
            };
            if let Some(v) = reg.vertices.get(vk) {
                points.push(v.position);
            }
        }

        if points.len() < 3 {
            continue;
        }

        // Compute face centroid
        let centroid: rc3d_core::math::PVec3 =
            points.iter().fold(rc3d_core::math::PVec3::ZERO, |acc, p| acc + *p)
                / (points.len() as Real);

        // Compute area-weighted normal via Newell's method
        let mut normal = rc3d_core::math::PVec3::ZERO;
        let n = points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let pi = points[i];
            let pj = points[j];
            normal.x += (pi.y - pj.y) * (pi.z + pj.z);
            normal.y += (pi.z - pj.z) * (pi.x + pj.x);
            normal.z += (pi.x - pj.x) * (pi.y + pj.y);
        }

        // Apply face orientation relative to shell
        let face_sign = match face_orient {
            Orientation::Forward => 1.0,
            Orientation::Reversed => -1.0,
            _ => 1.0,
        };

        // Tetrahedral volume contribution: (centroid dot normal) / 6
        let face_contrib = centroid.dot(normal) * face_sign;
        signed_volume += face_contrib;
    }

    // The volume should be positive for outward normals
    // (using right-hand rule with CCW faces)
    if signed_volume < 0.0 && signed_volume.abs() > 1e-12 {
        statuses.push(CheckStatus::SolidBadOrientation);
    }

    statuses
}

/// Verify that the solid forms a watertight enclosed region.
///
/// An enclosed region requires all outer shells to be closed (every edge
/// shared by exactly 2 faces) and all void shells to be similarly closed.
///
/// OCC alignment: BRepCheck_Solid — enclosed region validation.
pub fn check_enclosed_region(sk: SolidKey, reg: &BRepStore) -> Vec<CheckStatus> {
    let mut statuses = Vec::new();
    let solid = match reg.solids.get(sk) {
        Some(s) => s,
        None => return statuses,
    };

    // Check outer shell closure
    let outer_closed = super::shell::check_shell_closed(solid.outer_shell, reg);
    if !outer_closed.is_closed {
        statuses.push(CheckStatus::EnclosedRegionViolation);
    }

    // Check void shells closure
    for &vk in &solid.void_shells {
        let void_closed = super::shell::check_shell_closed(vk, reg);
        if !void_closed.is_closed {
            statuses.push(CheckStatus::EnclosedRegionViolation);
            break;
        }
    }

    statuses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{
        BRepFace, BRepShell, BRepSolid, BRepWire, FaceKey, Orientation, ShellKey,
    };
    use rc3d_core::math::PVec3;

    fn closed_2face_shell(reg: &mut BRepStore) -> (ShellKey, FaceKey, FaceKey) {
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };

        // Face 0: square in XY plane
        let p0 = PVec3::ZERO;
        let p1 = PVec3::new(1.0, 0.0, 0.0);
        let p2 = PVec3::new(1.0, 1.0, 0.0);
        let p3 = PVec3::new(0.0, 1.0, 0.0);
        let v0 = reg.find_or_add_vertex(p0, 1e-4);
        let v1 = reg.find_or_add_vertex(p1, 1e-4);
        let v2 = reg.find_or_add_vertex(p2, 1e-4);
        let v3 = reg.find_or_add_vertex(p3, 1e-4);

        let wk0 = reg.wires.insert(BRepWire { edges: vec![] });
        let fk0 = reg.faces.insert(BRepFace {
            surface: surface.clone(),
            outer_wire: wk0,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let line = |a: PVec3, b: PVec3| CurveGeom::Line {
            origin: a,
            direction: b - a,
        };
        let pc = |a: PVec3, b: PVec3| Curve2d::Line {
            origin: (a.x, a.y),
            direction: (b.x - a.x, b.y - a.y),
        };
        let e01 = reg.add_edge_with_pcurve(v0, v1, line(p0, p1), 1e-4, fk0, pc(p0, p1), true);
        let e12 = reg.add_edge_with_pcurve(v1, v2, line(p1, p2), 1e-4, fk0, pc(p1, p2), true);
        let e23 = reg.add_edge_with_pcurve(v2, v3, line(p2, p3), 1e-4, fk0, pc(p2, p3), true);
        let e30 = reg.add_edge_with_pcurve(v3, v0, line(p3, p0), 1e-4, fk0, pc(p3, p0), true);
        reg.wires.get_mut(wk0).unwrap().edges = vec![
            (e01, Orientation::Forward),
            (e12, Orientation::Forward),
            (e23, Orientation::Forward),
            (e30, Orientation::Forward),
        ];

        // Face 1: same square but with reversed edges (sharing same edges)
        let wk1 = reg.wires.insert(BRepWire { edges: vec![] });
        let fk1 = reg.faces.insert(BRepFace {
            surface: surface.clone(),
            outer_wire: wk1,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        // Add the same edges as pcurves for face 1 (share same edges)
        reg.add_edge_with_pcurve(v0, v1, line(p0, p1), 1e-4, fk1, pc(p0, p1), true);
        reg.add_edge_with_pcurve(v1, v2, line(p1, p2), 1e-4, fk1, pc(p1, p2), true);
        reg.add_edge_with_pcurve(v2, v3, line(p2, p3), 1e-4, fk1, pc(p2, p3), true);
        reg.add_edge_with_pcurve(v3, v0, line(p3, p0), 1e-4, fk1, pc(p3, p0), true);
        reg.wires.get_mut(wk1).unwrap().edges = vec![
            (e01, Orientation::Reversed),
            (e30, Orientation::Reversed),
            (e23, Orientation::Reversed),
            (e12, Orientation::Reversed),
        ];

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fk0, Orientation::Forward), (fk1, Orientation::Forward)],
            closed: true,
            step_id: None,
        });
        (sk, fk0, fk1)
    }

    #[test]
    fn test_solid_orientation_runs() {
        let mut reg = BRepStore::new();
        let (sk, _fk0, _fk1) = closed_2face_shell(&mut reg);
        let solid_key = reg.solids.insert(BRepSolid {
            outer_shell: sk,
            void_shells: vec![],
        });
        let statuses = check_solid_orientation(solid_key, &reg);
        // Two coplanar faces produce zero (or near-zero) volume -- that's OK,
        // the test just verifies the function runs without panic.
        // We don't assert on orientation since a flat 2-face shell has ~zero volume.
        let _ = statuses;
    }

    #[test]
    fn test_enclosed_region_closed() {
        let mut reg = BRepStore::new();
        let (sk, _fk0, _fk1) = closed_2face_shell(&mut reg);
        let solid_key = reg.solids.insert(BRepSolid {
            outer_shell: sk,
            void_shells: vec![],
        });
        let statuses = check_enclosed_region(solid_key, &reg);
        // 2-face shell has each edge shared by both faces -> closed
        assert!(
            !statuses.contains(&CheckStatus::EnclosedRegionViolation),
            "2-face closed shell should pass enclosed region check"
        );
    }

    #[test]
    fn test_enclosed_region_open_shell_fails() {
        let mut reg = BRepStore::new();
        // Single-face shell is not closed
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
        let solid_key = reg.solids.insert(BRepSolid {
            outer_shell: sk,
            void_shells: vec![],
        });
        let statuses = check_enclosed_region(solid_key, &reg);
        assert!(
            statuses.contains(&CheckStatus::EnclosedRegionViolation),
            "open shell should fail enclosed region check"
        );
    }
}
