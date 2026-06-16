//! Face-level fixes: natural boundary + reversed 2d.

use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
use crate::store::BRepStore;
use crate::topo::{FaceKey, Orientation, WireKey};

// ── Add natural boundary ───────────────────────────────────────────

/// Add a natural UV rectangle wire on analytic surfaces with an empty outer wire.
pub(crate) fn fix_add_natural_bound(reg: &mut BRepStore, face_key: FaceKey) -> bool {
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
        let pcurve = Curve2d::Line {
            origin: (u0, v0),
            direction: (u1 - u0, v1 - v0),
        };
        let ek = reg.add_edge_with_pcurve(v0k, v1k, curve_3d, tolerance, face_key, pcurve, true);
        edges.push((ek, Orientation::Forward));
    }

    if let Some(w) = reg.wires.get_mut(outer_wire) {
        w.edges = edges;
        true
    } else {
        false
    }
}

// ── Fix reversed 2d ────────────────────────────────────────────────

/// Flip outer wire orientation when its UV projection winds clockwise.
pub(crate) fn fix_reversed_2d(reg: &mut BRepStore, face_key: FaceKey) -> bool {
    let outer_wire = match reg.faces.get(face_key) {
        Some(f) => f.outer_wire,
        None => return false,
    };
    let area = signed_uv_wire_area(reg, face_key, outer_wire);
    if area >= 0.0 {
        return false;
    }
    let Some(w) = reg.wires.get_mut(outer_wire) else {
        return false;
    };
    for (_, orient) in &mut w.edges {
        *orient = match *orient {
            Orientation::Forward => Orientation::Reversed,
            Orientation::Reversed => Orientation::Forward,
            other => other,
        };
    }
    true
}

fn signed_uv_wire_area(reg: &BRepStore, face_key: FaceKey, wire_key: WireKey) -> f32 {
    if reg.faces.get(face_key).is_none() {
        return 0.0;
    }
    let wire = match reg.wires.get(wire_key) {
        Some(w) => w,
        None => return 0.0,
    };
    let mut poly: Vec<(f32, f32)> = Vec::new();
    for &(ek, orient) in &wire.edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        let pcurve = match edge.pcurves.get(&face_key) {
            Some(pc) => pc,
            None => continue,
        };
        let n = 8usize;
        let mut pts: Vec<(f32, f32)> = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let uv = pcurve.d0(t);
            pts.push((uv.0, uv.1));
        }
        if orient == Orientation::Reversed {
            pts.reverse();
        }
        if let Some(last) = poly.last() {
            if let Some(first) = pts.first() {
                if (last.0 - first.0).abs() < 1e-6 && (last.1 - first.1).abs() < 1e-6 {
                    pts.remove(0);
                }
            }
        }
        poly.extend(pts);
    }
    if poly.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..poly.len() {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % poly.len()];
        area += x0 * y1 - x1 * y0;
    }
    area * 0.5
}
