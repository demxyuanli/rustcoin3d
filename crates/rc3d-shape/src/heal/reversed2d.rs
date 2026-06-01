//! FixReversed2d — ensure outer wire has positive signed UV area (OCC ShapeFix_Face subset).

use crate::store::BRepRegistry;
use crate::topo::{FaceKey, Orientation, WireKey};

/// Flip outer wire orientation when its UV projection winds clockwise.
pub fn fix_reversed_2d(reg: &mut BRepRegistry, face_key: FaceKey) -> bool {
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

fn signed_uv_wire_area(reg: &BRepRegistry, face_key: FaceKey, wire_key: WireKey) -> f32 {
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
            pts.push((uv.x, uv.y));
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
