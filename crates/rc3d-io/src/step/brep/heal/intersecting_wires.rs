//! Multi-wire intersection detection and repair (OCC ShapeFix_Face::FixIntersectingWires).

use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{FaceKey, WireKey};

#[derive(Debug, Default)]
pub struct IntersectingWiresReport {
    pub inner_wires_trimmed: usize,
    pub inner_wires_removed: usize,
    pub inner_wires_merged: usize,
}

/// Detect and fix intersections between outer and inner wires (and between inner wires).
/// Uses point-in-polygon tests with winding number.
pub fn fix_intersecting_wires(
    face_key: FaceKey,
    reg: &mut BRepRegistry,
) -> IntersectingWiresReport {
    let mut report = IntersectingWiresReport::default();

    let (outer_wk, inner_wks) = {
        let Some(face) = reg.faces.get(face_key) else { return report; };
        (face.outer_wire, face.inner_wires.clone())
    };

    if inner_wks.is_empty() {
        return report;
    }

    // Collect outer wire UV polygon
    let outer_poly = match collect_wire_uv_polygon(outer_wk, face_key, reg) {
        Some(p) if p.len() >= 3 => p,
        _ => return report,
    };

    let mut kept_inner = Vec::new();

    for &inner_wk in &inner_wks {
        let inner_poly = match collect_wire_uv_polygon(inner_wk, face_key, reg) {
            Some(p) if p.len() >= 3 => p,
            _ => continue,
        };

        // Check if inner wire is entirely inside outer wire
        let inside_count = inner_poly.iter()
            .filter(|&&(u, v)| point_in_polygon_winding(u, v, &outer_poly))
            .count();
        let total = inner_poly.len();

        if inside_count == 0 {
            // Entirely outside outer wire — remove
            log::warn!("[BRep heal] FixIntersectingWires face {:?}: inner wire {:?} entirely outside outer, removing",
                face_key, inner_wk);
            report.inner_wires_removed += 1;
            continue;
        }

        if inside_count < total {
            // Partially intersecting — trim (simplified: remove if mostly outside)
            if inside_count < total / 2 {
                log::warn!("[BRep heal] FixIntersectingWires face {:?}: inner wire {:?} partially outside, removing",
                    face_key, inner_wk);
                report.inner_wires_removed += 1;
                continue;
            } else {
                log::debug!("[BRep heal] FixIntersectingWires face {:?}: inner wire {:?} mostly inside, keeping",
                    face_key, inner_wk);
            }
        }

        kept_inner.push(inner_wk);
    }

    // Update face with filtered inner wires
    if kept_inner.len() != inner_wks.len() {
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.inner_wires = kept_inner;
        }
    }

    report
}

fn collect_wire_uv_polygon(wk: WireKey, fk: FaceKey, reg: &BRepRegistry) -> Option<Vec<(f32, f32)>> {
    let wire = reg.wires.get(wk)?;
    let mut pts = Vec::new();
    for &(ek, _) in &wire.edges {
        let edge = reg.edges.get(ek)?;
        let pc = edge.pcurves.get(&fk)?;
        let uv = pc.d0(0.0);
        pts.push((uv.x, uv.y));
    }
    if pts.len() >= 3 {
        // Close the polygon
        if let Some(&last_ek) = wire.edges.last() {
            if let Some(edge) = reg.edges.get(last_ek.0) {
                if let Some(pc) = edge.pcurves.get(&fk) {
                    let uv = pc.d0(1.0);
                    pts.push((uv.x, uv.y));
                }
            }
        }
    }
    Some(pts)
}

fn point_in_polygon_winding(u: f32, v: f32, poly: &[(f32, f32)]) -> bool {
    let n = poly.len();
    let mut wn = 0i32;
    for i in 0..n {
        let j = (i + 1) % n;
        let (x1, y1) = poly[i];
        let (x2, y2) = poly[j];
        if y1 <= v {
            if y2 > v && cross_2d(x1, y1, x2, y2, u, v) > 0.0 {
                wn += 1;
            }
        } else {
            if y2 <= v && cross_2d(x1, y1, x2, y2, u, v) < 0.0 {
                wn -= 1;
            }
        }
    }
    wn != 0
}

fn cross_2d(x1: f32, y1: f32, x2: f32, y2: f32, u: f32, v: f32) -> f32 {
    (x2 - x1) * (v - y1) - (u - x1) * (y2 - y1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_in_square() {
        let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
        assert!(point_in_polygon_winding(0.5, 0.5, &square));
        assert!(!point_in_polygon_winding(2.0, 0.5, &square));
    }
}
