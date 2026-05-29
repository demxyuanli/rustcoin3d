//! BRepLib::SameParameter subset — propagate 3D/PCurve deviation into edge tolerance (OCC-aligned).

use std::collections::HashSet;

use super::registry::BRepRegistry;
use super::topo::{EdgeKey, ShellKey};

const SAMPLE_COUNT: usize = 32;

/// Recompute edge tolerances from PCurve vs 3D curve deviation for every edge in a shell.
pub fn same_parameter_shell(reg: &mut BRepRegistry, shell_key: ShellKey, base_tol: f32) -> usize {
    let edges = shell_edge_keys(reg, shell_key);
    let mut updated = 0usize;
    for ek in edges {
        if update_edge_tolerance(reg, ek, base_tol) {
            updated += 1;
        }
    }
    updated
}

fn shell_edge_keys(reg: &BRepRegistry, shell_key: ShellKey) -> Vec<EdgeKey> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let Some(shell) = reg.shells.get(shell_key) else {
        return out;
    };
    for &(fk, _) in &shell.faces {
        let Some(face) = reg.faces.get(fk) else {
            continue;
        };
        let wires = std::iter::once(face.outer_wire)
            .chain(face.inner_wires.iter().copied());
        for wk in wires {
            let Some(wire) = reg.wires.get(wk) else {
                continue;
            };
            for &(ek, _) in &wire.edges {
                if seen.insert(ek) {
                    out.push(ek);
                }
            }
        }
        for &ek in &face.seam_edges {
            if seen.insert(ek) {
                out.push(ek);
            }
        }
        for &ek in &face.degenerated_edges {
            if seen.insert(ek) {
                out.push(ek);
            }
        }
    }
    out
}

fn update_edge_tolerance(reg: &mut BRepRegistry, ek: EdgeKey, base_tol: f32) -> bool {
    let edge = match reg.edges.get(ek) {
        Some(e) => e,
        None => return false,
    };
    if edge.pcurves.is_empty() || edge.v_low == edge.v_high {
        return false;
    }

    let mut max_dev = 0.0f32;
    for i in 0..=SAMPLE_COUNT {
        let t = i as f32 / SAMPLE_COUNT as f32;
        let p3d = edge.curve.d0(t);
        for (&face_key, pcurve) in &edge.pcurves {
            let face = match reg.faces.get(face_key) {
                Some(f) => f,
                None => continue,
            };
            let uv = pcurve.d0(t);
            let on_surf = face.surface.d0_native(uv.x, uv.y);
            max_dev = max_dev.max((on_surf - p3d).length());
        }
    }

    let new_tol = max_dev.max(base_tol);
    let prev = edge.tolerance;
    if (new_tol - prev).abs() > base_tol * 0.01 {
        if let Some(e) = reg.edges.get_mut(ek) {
            e.tolerance = new_tol;
        }
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::Vec3;
    use std::collections::HashMap;

    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::topo::{BRepEdge, BRepFace, BRepShell, BRepWire, Orientation};

    #[test]
    fn misaligned_pcurve_raises_edge_tolerance() {
        let mut reg = BRepRegistry::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-6);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-6);
        let mut pcurves = HashMap::new();
        pcurves.insert(
            face_key,
            CurveGeom::Line {
                origin: Vec3::ZERO,
                direction: Vec3::new(1.0, 0.0, 0.0),
            },
        );
        let ek = reg.edges.insert(BRepEdge {
            v_low: v0,
            v_high: v1,
            curve: CurveGeom::Line {
                origin: Vec3::new(0.0, 0.0, 0.1),
                direction: Vec3::new(1.0, 0.0, 0.0),
            },
            tolerance: 1e-6,
            pcurves,
        });
        reg.wires.get_mut(wire).unwrap().edges.push((ek, Orientation::Forward));
        let shell_key = reg.shells.insert(BRepShell {
            faces: vec![(face_key, Orientation::Forward)],
            closed: false,
            step_id: None,
        });

        assert!(same_parameter_shell(&mut reg, shell_key, 1e-6) > 0);
        assert!(reg.edges.get(ek).unwrap().tolerance >= 0.09);
    }
}
