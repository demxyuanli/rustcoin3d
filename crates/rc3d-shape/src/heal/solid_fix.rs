//! Fix small or degenerate solids.
//!
//! Removes solids with volume below a threshold and shells with zero faces.
//!
//! ## OCC alignment
//! Corresponds to `ShapeFix_Solid` / `ShapeFix_FixSmallSolid`.
//!
//! ## Key functions
//! - `fix_small_solids()` — remove solids with `|volume| < min_volume`
//! - `remove_empty_shells()` — remove shells with zero faces

use crate::store::BRepStore;
use crate::topo::{ShellKey, SolidKey};
use rc3d_core::math::Real;
use std::collections::HashSet;

/// Remove solids with volume below threshold.
/// Returns set of removed SolidKeys.
pub fn fix_small_solids(
    solid_keys: &[SolidKey],
    reg: &mut BRepStore,
    min_volume: Real,
) -> HashSet<SolidKey> {
    let mut removed = HashSet::new();
    for &sk in solid_keys {
        let vol = crate::geom::properties::solid_volume(reg, sk, 16);
        if vol.abs() < min_volume {
            log::info!(
                "[heal] FixSmallSolid: removing solid {:?} (volume={:.6})",
                sk,
                vol
            );
            removed.insert(sk);
        }
    }
    for &sk in &removed {
        reg.solids.remove(sk);
    }
    removed
}

/// Remove empty shells (no faces) from the store.
pub fn remove_empty_shells(reg: &mut BRepStore) -> usize {
    let empty: Vec<ShellKey> = reg
        .shells
        .iter()
        .filter(|(_, s)| s.faces.is_empty())
        .map(|(k, _)| k)
        .collect();
    let count = empty.len();
    for sk in empty {
        reg.shells.remove(sk);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::topo::{BRepFace, BRepShell, BRepSolid, BRepWire, Orientation};
    use rc3d_core::math::PVec3;

    /// Build a unit cube (volume ≈ 1.0) solid in the store.
    fn build_unit_cube_solid(reg: &mut BRepStore) -> SolidKey {
        // Bottom face (z=0)
        let v0 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), 1e-6);
        let v1 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-6);
        let v2 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-6);
        let v3 = reg.find_or_add_vertex(PVec3::new(0.0, 1.0, 0.0), 1e-6);
        // Top face (z=1)
        let v4 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 1.0), 1e-6);
        let v5 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 1.0), 1e-6);
        let v6 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 1.0), 1e-6);
        let v7 = reg.find_or_add_vertex(PVec3::new(0.0, 1.0, 1.0), 1e-6);

        let make_pc =
            |a: PVec3, b: PVec3, ax: usize, ay: usize| Curve2d::Line {
                origin: (a.as_ref()[ax], a.as_ref()[ay]),
                direction: (
                    b.as_ref()[ax] - a.as_ref()[ax],
                    b.as_ref()[ay] - a.as_ref()[ay],
                ),
            };
        let make_3d = |a: PVec3, b: PVec3| CurveGeom::Line {
            origin: a,
            direction: b - a,
        };

        let mut shell_faces = Vec::new();

        // Helper to build a quadrilateral face
        let mut build_quad = |corners: [(
            crate::topo::VertexKey,
            crate::topo::VertexKey,
            PVec3,
            PVec3,
        ); 4],
                              origin: PVec3,
                              normal: PVec3,
                              u_dir: PVec3| {
            let surf = SurfaceGeom::Plane {
                origin,
                normal,
                u_dir,
            };
            let wk = reg.wires.insert(BRepWire { edges: vec![] });
            let fk = reg.faces.insert(BRepFace {
                surface: surf,
                outer_wire: wk,
                inner_wires: vec![],
                same_sense: true,
                tolerance: 1e-6,
                seam_edges: vec![],
                color: None,
                degenerated_edges: vec![],
            });

            let mut edges = Vec::new();
            for &(a, b, pa, pb) in &corners {
                let e3d = make_3d(pa, pb);
                let pc = make_pc(pa, pb, 0, 1);
                let ek = reg.add_edge_with_pcurve(a, b, e3d, 1e-6, fk, pc, true);
                let orient = if reg.edges.get(ek).unwrap().v_low == a {
                    Orientation::Forward
                } else {
                    Orientation::Reversed
                };
                edges.push((ek, orient));
            }
            reg.wires.get_mut(wk).unwrap().edges = edges;
            (fk, Orientation::Forward)
        };

        // Bottom face (z=0, normal -Z)
        shell_faces.push(build_quad(
            [
                (v0, v1, PVec3::new(0.0, 0.0, 0.0), PVec3::new(1.0, 0.0, 0.0)),
                (v1, v2, PVec3::new(1.0, 0.0, 0.0), PVec3::new(1.0, 1.0, 0.0)),
                (v2, v3, PVec3::new(1.0, 1.0, 0.0), PVec3::new(0.0, 1.0, 0.0)),
                (v3, v0, PVec3::new(0.0, 1.0, 0.0), PVec3::new(0.0, 0.0, 0.0)),
            ],
            PVec3::ZERO,
            PVec3::NEG_Z,
            PVec3::X,
        ));
        // Top face (z=1, normal +Z)
        shell_faces.push(build_quad(
            [
                (v4, v5, PVec3::new(0.0, 0.0, 1.0), PVec3::new(1.0, 0.0, 1.0)),
                (v5, v6, PVec3::new(1.0, 0.0, 1.0), PVec3::new(1.0, 1.0, 1.0)),
                (v6, v7, PVec3::new(1.0, 1.0, 1.0), PVec3::new(0.0, 1.0, 1.0)),
                (v7, v4, PVec3::new(0.0, 1.0, 1.0), PVec3::new(0.0, 0.0, 1.0)),
            ],
            PVec3::new(0.0, 0.0, 1.0),
            PVec3::Z,
            PVec3::X,
        ));
        // Front face (y=0, normal -Y)
        shell_faces.push(build_quad(
            [
                (v0, v1, PVec3::new(0.0, 0.0, 0.0), PVec3::new(1.0, 0.0, 0.0)),
                (v1, v5, PVec3::new(1.0, 0.0, 0.0), PVec3::new(1.0, 0.0, 1.0)),
                (v5, v4, PVec3::new(1.0, 0.0, 1.0), PVec3::new(0.0, 0.0, 1.0)),
                (v4, v0, PVec3::new(0.0, 0.0, 1.0), PVec3::new(0.0, 0.0, 0.0)),
            ],
            PVec3::new(0.0, 0.0, 0.0),
            PVec3::NEG_Y,
            PVec3::X,
        ));
        // Back face (y=1, normal +Y)
        shell_faces.push(build_quad(
            [
                (v3, v2, PVec3::new(0.0, 1.0, 0.0), PVec3::new(1.0, 1.0, 0.0)),
                (v2, v6, PVec3::new(1.0, 1.0, 0.0), PVec3::new(1.0, 1.0, 1.0)),
                (v6, v7, PVec3::new(1.0, 1.0, 1.0), PVec3::new(0.0, 1.0, 1.0)),
                (v7, v3, PVec3::new(0.0, 1.0, 1.0), PVec3::new(0.0, 1.0, 0.0)),
            ],
            PVec3::new(0.0, 1.0, 0.0),
            PVec3::Y,
            PVec3::X,
        ));
        // Left face (x=0, normal -X)
        shell_faces.push(build_quad(
            [
                (v0, v3, PVec3::new(0.0, 0.0, 0.0), PVec3::new(0.0, 1.0, 0.0)),
                (v3, v7, PVec3::new(0.0, 1.0, 0.0), PVec3::new(0.0, 1.0, 1.0)),
                (v7, v4, PVec3::new(0.0, 1.0, 1.0), PVec3::new(0.0, 0.0, 1.0)),
                (v4, v0, PVec3::new(0.0, 0.0, 1.0), PVec3::new(0.0, 0.0, 0.0)),
            ],
            PVec3::ZERO,
            PVec3::NEG_X,
            PVec3::Y,
        ));
        // Right face (x=1, normal +X)
        shell_faces.push(build_quad(
            [
                (v1, v2, PVec3::new(1.0, 0.0, 0.0), PVec3::new(1.0, 1.0, 0.0)),
                (v2, v6, PVec3::new(1.0, 1.0, 0.0), PVec3::new(1.0, 1.0, 1.0)),
                (v6, v5, PVec3::new(1.0, 1.0, 1.0), PVec3::new(1.0, 0.0, 1.0)),
                (v5, v1, PVec3::new(1.0, 0.0, 1.0), PVec3::new(1.0, 0.0, 0.0)),
            ],
            PVec3::new(1.0, 0.0, 0.0),
            PVec3::X,
            PVec3::Y,
        ));

        let shell = reg.shells.insert(BRepShell {
            faces: shell_faces,
            closed: true,
            step_id: None,
        });
        reg.solids.insert(BRepSolid {
            outer_shell: shell,
            void_shells: vec![],
        })
    }

    #[test]
    fn test_solid_volume_near_one() {
        let mut reg = BRepStore::new();
        let sk = build_unit_cube_solid(&mut reg);
        let vol = crate::geom::properties::solid_volume(&reg, sk, 16);
        // Unit cube should have volume ≈ 1.0
        assert!(
            (vol - 1.0).abs() < 0.1,
            "unit cube volume should be ~1.0, got {}",
            vol
        );
    }

    #[test]
    fn test_fix_small_solids_removes_tiny_solid() {
        let mut reg = BRepStore::new();
        let sk = build_unit_cube_solid(&mut reg);
        // Unit cube volume ≈ 1.0, threshold 0.001 → NOT removed
        let removed = fix_small_solids(&[sk], &mut reg, 0.001);
        assert!(removed.is_empty(), "unit cube should not be removed");
        assert!(reg.solids.contains_key(sk), "solid should still exist");
    }

    #[test]
    fn test_fix_small_solids_keeps_large_solid() {
        let mut reg = BRepStore::new();
        let sk = build_unit_cube_solid(&mut reg);
        // Threshold 100.0 → volume ~1.0 < 100.0 → removed
        let removed = fix_small_solids(&[sk], &mut reg, 100.0);
        assert_eq!(removed.len(), 1);
        assert!(removed.contains(&sk));
        assert!(!reg.solids.contains_key(sk), "solid should be removed from store");
    }

    #[test]
    fn test_remove_empty_shells() {
        let mut reg = BRepStore::new();
        let sk = reg.shells.insert(BRepShell {
            faces: vec![],
            closed: false,
            step_id: None,
        });
        let count = remove_empty_shells(&mut reg);
        assert_eq!(count, 1);
        assert!(!reg.shells.contains_key(sk));
    }

    #[test]
    fn test_remove_empty_shells_preserves_nonempty() {
        let mut reg = BRepStore::new();
        let _sk = build_unit_cube_solid(&mut reg);
        let nonempty_count = reg.shells.iter().filter(|(_, s)| !s.faces.is_empty()).count();
        let count = remove_empty_shells(&mut reg);
        assert_eq!(count, 0, "no empty shells in a cube solid");
        assert_eq!(
            reg.shells.iter().filter(|(_, s)| !s.faces.is_empty()).count(),
            nonempty_count,
            "non-empty shells preserved"
        );
    }
}
