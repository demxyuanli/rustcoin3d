//! Vertex position correction (OCC ShapeFix_Shape::FixVertexPositionMode).

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::ShellKey;

/// Project each vertex in the shell onto surfaces of faces that reference it.
/// Returns number of vertices adjusted.
pub fn fix_vertex_positions(
    shell_key: ShellKey,
    reg: &mut BRepRegistry,
    tolerance: f32,
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
                        let on_surf = surface.d0(u, v_param);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::topo::{BRepWire, Orientation};
    use rc3d_core::math::Vec3;

    #[test]
    fn test_fix_vertex_on_surface() {
        let mut reg = BRepRegistry::new();
        let surface = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
        // Create a vertex slightly off the plane (~50x tolerance)
        let v0 = reg.find_or_add_vertex(Vec3::new(0.0, 0.0, 0.005), 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(1.0, 0.0, 0.0), 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(crate::step::brep::topo::BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![], color: None,
            degenerated_edges: vec![],
        });
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, line.clone());
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(crate::step::brep::topo::BRepShell {
            faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None,
        });
        let adjusted = fix_vertex_positions(sk, &mut reg, 1e-4);
        assert!(adjusted > 0, "off-surface vertex should be projected back");
    }
}
