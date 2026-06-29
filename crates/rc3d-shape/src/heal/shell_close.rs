//! Shell closure: detect and fill missing planar faces for watertight solids.
//!
//! Scans each shell for free edges (used by fewer than 2 faces), then
//! creates planar closure faces where circular free edges indicate an
//! unclosed opening.

use rc3d_core::math::{Real, PVec3};
use crate::store::BRepStore;
use crate::topo::*;
use crate::geom::{CurveGeom, SurfaceGeom, plane_tangent_basis};
use std::collections::HashMap;

/// Scan all solids and add missing planar closure faces for free circular
/// edges. Returns the number of faces added.
pub fn close_open_shells(reg: &mut BRepStore) -> usize {
    let solid_keys: Vec<SolidKey> = reg.solids.keys().collect();
    let mut added = 0usize;

    for sk in solid_keys {
        let shell_key = match reg.solids.get(sk) {
            Some(s) => s.outer_shell,
            None => continue,
        };
        added += close_shell(reg, shell_key);
    }

    added
}

fn close_shell(reg: &mut BRepStore, shell_key: ShellKey) -> usize {
    // ── Build edge → face count + internal-seam detection ──────────
    let mut edge_face_count: HashMap<EdgeKey, Vec<FaceKey>> = HashMap::new();
    let mut internal_seam: HashMap<EdgeKey, bool> = HashMap::new();

    let shell_faces: Vec<(FaceKey, Orientation)> = match reg.shells.get(shell_key) {
        Some(sh) => sh.faces.clone(),
        None => return 0,
    };

    for &(fk, _) in &shell_faces {
        let face = match reg.faces.get(fk) {
            Some(f) => f,
            None => continue,
        };
        // Track F/R orientation per edge within this face
        let mut face_orients: HashMap<EdgeKey, (bool, bool)> = HashMap::new();
        for wire_key in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
            let wire = match reg.wires.get(*wire_key) {
                Some(w) => w,
                None => continue,
            };
            for &(ek, orient) in &wire.edges {
                let faces = edge_face_count.entry(ek).or_default();
                if !faces.contains(&fk) {
                    faces.push(fk);
                }
                let (fwd, rev) = face_orients.entry(ek).or_default();
                match orient {
                    Orientation::Forward => *fwd = true,
                    Orientation::Reversed => *rev = true,
                    _ => *fwd = true,
                }
            }
        }
        // Edges that appear both F and R in the same face are internal seams
        for (ek, (fwd, rev)) in face_orients {
            if fwd && rev {
                internal_seam.insert(ek, true);
            }
        }
        // Also count seam and degenerated edges
        for &ek in &face.seam_edges {
            let faces = edge_face_count.entry(ek).or_default();
            if !faces.contains(&fk) { faces.push(fk); }
        }
        for &ek in &face.degenerated_edges {
            let faces = edge_face_count.entry(ek).or_default();
            if !faces.contains(&fk) { faces.push(fk); }
        }
    }

    // ── Find free circular edges (skip internal seams) ────────────
    let mut free_circles: Vec<(EdgeKey, PVec3, PVec3, Real)> = Vec::new();

    for (&ek, face_keys) in &edge_face_count {
        if face_keys.len() >= 2 { continue; }
        if internal_seam.contains_key(&ek) { continue; } // F+R in same wire = not a boundary

        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };

        if !is_closed_circle(&edge.curve, edge.t_min, edge.t_max) { continue; }

        let c = match &edge.curve {
            CurveGeom::Trimmed { basis, .. } => basis.as_ref(),
            other => other,
        };
        if let CurveGeom::Circle { center, axis, radius, .. } = c {
            free_circles.push((ek, *center, *axis, *radius));
        }
    }

    let mut added = 0usize;
    for (ek, center, axis, _radius) in free_circles {
        if let Some(face_key) = build_closure_face(reg, ek, center, axis) {
            if let Some(shell) = reg.shells.get_mut(shell_key) {
                shell.faces.push((face_key, Orientation::Forward));
            }
            added += 1;
        }
    }

    added
}

/// Check if a curve is a closed circle (full 2π revolution).
fn is_closed_circle(curve: &CurveGeom, t_min: Real, t_max: Real) -> bool {
    let c = match curve {
        CurveGeom::Trimmed { basis, .. } => basis.as_ref(),
        other => other,
    };
    matches!(c, CurveGeom::Circle { .. }) && (t_max - t_min - std::f64::consts::TAU).abs() < 1e-6
}

/// Build a planar closure face whose outer wire is the given circular edge.
fn build_closure_face(
    reg: &mut BRepStore,
    ek: EdgeKey,
    center: PVec3,
    axis: PVec3,
) -> Option<FaceKey> {
    let edge = reg.edges.get(ek)?;
    let tol = edge.tolerance;

    // Build a planar surface at the circle's position
    let normal = axis.normalize();
    let (u_dir, _v_dir) = plane_tangent_basis(normal, PVec3::X);
    let plane = SurfaceGeom::Plane { origin: center, normal, u_dir };

    // Create wire with the circle edge
    let wire_key = reg.wires.insert(BRepWire {
        edges: vec![(ek, Orientation::Forward)],
    });

    // Create the new face
    let face_key = reg.faces.insert(BRepFace {
        surface: plane,
        outer_wire: wire_key,
        inner_wires: vec![],
        same_sense: true,
        tolerance: tol,
        seam_edges: vec![],
        color: None,
        degenerated_edges: vec![],
    });

    // Register the face in the edge's PCurve and edge_to_faces
    // ponytail: the closure face shares the same 3D edge; PCurve should
    // be a circle in the plane's UV coords.

    Some(face_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::CurveGeom;
    use crate::geom::curve2d::Curve2d;

    #[test]
    fn close_open_cylinder_shell_adds_closure_face() {
        let mut reg = BRepStore::new();
        let tol = 1e-4;

        // Two vertices on the circle
        let v0 = reg.vertices.insert(BRepVertex {
            position: PVec3::new(5.0, 0.0, 0.0),
            tolerance: tol,
        });

        // A circular edge (think: open end of a cylinder)
        let circle = CurveGeom::Circle {
            center: PVec3::new(0.0, 0.0, 0.0),
            axis: PVec3::Z,
            radius: 5.0,
            x_dir: PVec3::X,
            y_dir: PVec3::Y,
        };
        let face_key = reg.add_face(
            SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 5.0),
            tol,
        );
        let ek = reg.add_edge_with_pcurve(
            v0, v0, circle, tol, face_key,
            Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) },
            true,
        );

        // Put the edge in the face's outer wire
        let wire_key = reg.wires.insert(BRepWire { edges: vec![(ek, Orientation::Forward)] });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = wire_key;
        }

        // Build a shell with one face (the cylinder) — missing closure faces
        let sh_key = reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: false,
            step_id: None,
        });
        reg.solids.insert(BRepSolid {
            outer_shell: sh_key,
            void_shells: vec![],
        });

        let n = close_open_shells(&mut reg);
        assert_eq!(n, 1, "should add one closure face for the free circle edge");

        // Verify the shell now has 2 faces
        let sh = reg.shells.get(sh_key).unwrap();
        assert_eq!(sh.faces.len(), 2, "shell should have 2 faces after closure");
    }
}
