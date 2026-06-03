use crate::store::BRepStore;
use crate::topo::{BRepWire, FaceKey};

pub fn collect_wire_uv_polygon(
    wire: &BRepWire,
    face_key: FaceKey,
    reg: &BRepStore,
) -> Vec<(f32, f32)> {
    let mut points = Vec::new();
    for &(ek, _) in &wire.edges {
        if let Some(edge) = reg.edges.get(ek) {
            if let Some(pc) = edge.pcurves.get(&face_key) {
                let uv = pc.d0(0.0);
                points.push((uv.x, uv.y));
            }
        }
    }
    if points.len() >= 2 {
        if let Some(last_edge) = wire.edges.last().and_then(|&(ek, _)| {
            reg.edges.get(ek).and_then(|e| e.pcurves.get(&face_key)).map(|pc| {
                let uv = pc.d0(1.0);
                (uv.x, uv.y)
            })
        }) {
            points.push(last_edge);
        }
    }
    points
}

pub fn point_in_polygon_winding(u: f32, v: f32, poly: &[(f32, f32)]) -> bool {
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
        } else if y2 <= v && cross_2d(x1, y1, x2, y2, u, v) < 0.0 {
            wn -= 1;
        }
    }
    wn != 0
}

pub fn segment_intersection_strict(
    a0: (f32, f32),
    a1: (f32, f32),
    b0: (f32, f32),
    b1: (f32, f32),
) -> Option<(f32, f32)> {
    let da = (a1.0 - a0.0, a1.1 - a0.1);
    let db = (b1.0 - b0.0, b1.1 - b0.1);
    let det = da.0 * db.1 - da.1 * db.0;
    if det.abs() < 1e-12 {
        return None;
    }
    let d0 = (b0.0 - a0.0, b0.1 - a0.1);
    let t = (d0.0 * db.1 - d0.1 * db.0) / det;
    let u = (d0.0 * da.1 - d0.1 * da.0) / det;
    if t > 0.0 && t < 1.0 && u > 0.0 && u < 1.0 {
        Some((t, u))
    } else {
        None
    }
}

fn cross_2d(x1: f32, y1: f32, x2: f32, y2: f32, u: f32, v: f32) -> f32 {
    (x2 - x1) * (v - y1) - (u - x1) * (y2 - y1)
}
