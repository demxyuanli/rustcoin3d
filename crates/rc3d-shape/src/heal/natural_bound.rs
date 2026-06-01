//! FixAddNaturalBound — rectangular parametric boundary when a face has no wire (OCC subset).

use rc3d_core::math::Vec3;

use crate::geom::{CurveGeom, SurfaceGeom};
use crate::store::BRepRegistry;
use crate::topo::{FaceKey, Orientation};

/// Add a natural UV rectangle wire on analytic surfaces with an empty outer wire.
pub fn fix_add_natural_bound(reg: &mut BRepRegistry, face_key: FaceKey) -> bool {
    let (surface, tolerance, outer_wire) = {
        let Some(face) = reg.faces.get(face_key) else {
            return false;
        };
        (
            face.surface.clone(),
            face.tolerance,
            face.outer_wire,
        )
    };

    let wire_empty = reg
        .wires
        .get(outer_wire)
        .map(|w| w.edges.is_empty())
        .unwrap_or(true);
    if !wire_empty {
        return false;
    }

    match surface {
        SurfaceGeom::Plane { .. }
        | SurfaceGeom::Cylinder { .. }
        | SurfaceGeom::Cone { .. } => {}
        _ => return false,
    }

    let pr = surface.param_range();
    let corners = [
        (pr.u_min, pr.v_min),
        (pr.u_max, pr.v_min),
        (pr.u_max, pr.v_max),
        (pr.u_min, pr.v_max),
    ];

    let mut edges = Vec::with_capacity(4);
    for i in 0..4 {
        let (u0, v0) = corners[i];
        let (u1, v1) = corners[(i + 1) % 4];
        let p0 = surface.d0_native(u0, v0);
        let p1 = surface.d0_native(u1, v1);
        let v0k = reg.find_or_add_vertex(p0, tolerance);
        let v1k = reg.find_or_add_vertex(p1, tolerance);
        let dir3 = p1 - p0;
        let curve_3d = CurveGeom::Line {
            origin: p0,
            direction: dir3,
        };
        let pcurve = CurveGeom::Line {
            origin: Vec3::new(u0, v0, 0.0),
            direction: Vec3::new(u1 - u0, v1 - v0, 0.0),
        };
        let ek = reg.add_edge_with_pcurve(v0k, v1k, curve_3d, tolerance, face_key, pcurve);
        edges.push((ek, Orientation::Forward));
    }

    if let Some(w) = reg.wires.get_mut(outer_wire) {
        w.edges = edges;
        true
    } else {
        false
    }
}
