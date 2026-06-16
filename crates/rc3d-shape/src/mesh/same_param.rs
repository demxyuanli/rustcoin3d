//! SameParameter snap — align edge 3D samples with PCurve-on-surface (OCC ShapeFix_Edge subset).

use std::collections::HashMap;

use super::edge_disc::EdgePolygon;
use crate::store::BRepStore;
use crate::topo::EdgeKey;

/// Snap discretized edge 3D points onto their PCurve-evaluated surface positions.
pub fn apply_same_parameter(
    edge_polygons: &mut HashMap<EdgeKey, EdgePolygon>,
    reg: &BRepStore,
    base_tol: f32,
) {
    for (&ek, poly) in edge_polygons.iter_mut() {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        if edge.pcurves.is_empty() {
            continue;
        }

        let mut best_tol = base_tol;
        for &face_key in edge.pcurves.keys() {
            if let Some(face) = reg.faces.get(face_key) {
                best_tol = best_tol.max(face.tolerance.max(base_tol));
            }
        }

        let mut snapped = poly.params_3d.clone();
        for (&face_key, pcurve) in &edge.pcurves {
            let face = match reg.faces.get(face_key) {
                Some(f) => f,
                None => continue,
            };
            for (i, &(t, p)) in poly.params_3d.iter().enumerate() {
                let uv = pcurve.d0(t);
                let (nu, nv) = reg.face_native_uv(face_key, uv.0, uv.1);
                let on_surf = face.surface.d0_native(nu, nv);
                if (on_surf - p).length() > best_tol {
                    snapped[i] = (t, on_surf);
                }
            }
        }
        poly.params_3d = snapped;

        for (&face_key, pcurve) in &edge.pcurves {
            if reg.faces.get(face_key).is_some() {
                let pts_2d: Vec<(f32, (f32, f32))> = poly
                    .params_3d
                    .iter()
                    .map(|&(t, _)| {
                        let uv = pcurve.d0(t);
                        (t, (uv.0, uv.1))
                    })
                    .collect();
                poly.params_2d.insert(face_key, pts_2d);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::Vec3;
    use crate::geom::{CurveGeom, Curve2d, SurfaceGeom};
    use crate::topo::{BRepEdge, BRepFace};

    #[test]
    fn offset_3d_point_snaps_to_surface() {
        let mut reg = BRepStore::new();
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: reg.wires.insert(crate::topo::BRepWire { edges: vec![] }),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::new(5.0, 0.0, 0.0), 1e-4);
        let ek = reg.edges.insert(BRepEdge {
            v_low: v0,
            v_high: v1,
            curve: CurveGeom::Line {
                origin: Vec3::ZERO,
                direction: Vec3::new(5.0, 0.0, 0.0),
            },
            tolerance: 1e-4,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: HashMap::new(),
        });

        let pcurve = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (5.0, 0.0),
        };
        if let Some(edge) = reg.edges.get_mut(ek) {
            edge.pcurves.insert(face_key, pcurve);
        }

        let offset = Vec3::new(0.0, 0.0, 0.5);
        let mut polys = HashMap::new();
        polys.insert(
            ek,
            EdgePolygon {
                params_3d: vec![
                    (0.0, Vec3::ZERO + offset),
                    (0.5, Vec3::new(2.5, 0.0, 0.0) + offset),
                    (1.0, Vec3::new(5.0, 0.0, 0.0) + offset),
                ],
                params_2d: HashMap::new(),
            },
        );

        apply_same_parameter(&mut polys, &reg, 1e-4);
        let poly = polys.get(&ek).unwrap();
        for &(_, p) in &poly.params_3d {
            assert!(
                p.z.abs() < 1e-3,
                "snapped point should lie on z=0 plane, got {:?}",
                p
            );
        }
    }
}
