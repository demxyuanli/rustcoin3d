//! Trim and split edge curves / PCurves at parameter values (OCC split support).

use crate::geom::{Curve2d, CurveGeom, SurfaceGeom};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation, VertexKey, WireKey};
use rc3d_core::math::{Real, PVec3};

/// Trim a 3D edge curve to parameter range [t0, t1] in edge parameter space [0, 1].
pub fn trim_edge_curve(curve: &CurveGeom, t0: Real, t1: Real) -> CurveGeom {
    if (t0 - 0.0).abs() < 1e-8 && (t1 - 1.0).abs() < 1e-8 {
        return curve.clone();
    }
    match curve {
        CurveGeom::Line { origin, direction } => {
            let o = *origin + *direction * t0;
            let d = *direction * (t1 - t0);
            CurveGeom::Line {
                origin: o,
                direction: d,
            }
        }
        CurveGeom::Polyline { points } => trim_polyline(points, t0, t1),
        CurveGeom::Trimmed { basis, t_min, t_max } => {
            let span = (*t_max - *t_min).max(1e-12);
            let new_min = *t_min + t0 * span;
            let new_max = *t_min + t1 * span;
            CurveGeom::Trimmed {
                basis: basis.clone(),
                t_min: new_min,
                t_max: new_max,
            }
        }
        other => CurveGeom::Trimmed {
            basis: Box::new(other.clone()),
            t_min: t0,
            t_max: t1,
        },
    }
}

fn trim_polyline(points: &[PVec3], t0: Real, t1: Real) -> CurveGeom {
    if points.len() < 2 {
        return CurveGeom::Polyline {
            points: points.to_vec(),
        };
    }
    let mut seg_lens = Vec::with_capacity(points.len() - 1);
    let mut total = 0.0_f64;
    for w in points.windows(2) {
        let l = (w[1] - w[0]).length();
        seg_lens.push(l);
        total += l;
    }
    if total < 1e-12 {
        return CurveGeom::Polyline {
            points: vec![points[0], points[0]],
        };
    }
    let sample = |t: Real| -> PVec3 {
        let target = t.clamp(0.0, 1.0) * total;
        let mut acc = 0.0_f64;
        for (i, &l) in seg_lens.iter().enumerate() {
            if acc + l >= target - 1e-8 || i + 1 == seg_lens.len() {
                let local = if l > 1e-12 {
                    (target - acc) / l
                } else {
                    0.0
                };
                return points[i] + (points[i + 1] - points[i]) * local.clamp(0.0, 1.0);
            }
            acc += l;
        }
        *points.last().unwrap()
    };
    CurveGeom::Polyline {
        points: vec![sample(t0), sample(t1)],
    }
}

/// Trim a PCurve to [t0, t1] in edge parameter space.
pub fn trim_pcurve(pc: &Curve2d, t0: Real, t1: Real) -> Curve2d {
    if (t0 - 0.0).abs() < 1e-8 && (t1 - 1.0).abs() < 1e-8 {
        return pc.clone();
    }
    Curve2d::Trimmed {
        basis: Box::new(pc.clone()),
        t_min: t0,
        t_max: t1,
    }
}

/// Split an edge at interior parameters; returns new edge keys in order.
pub fn split_edge_at_params(
    ek: EdgeKey,
    face_key: FaceKey,
    orient: Orientation,
    splits: &[Real],
    reg: &mut BRepStore,
) -> Vec<(EdgeKey, Orientation)> {
    let mut params: Vec<Real> = splits
        .iter()
        .copied()
        .filter(|t| *t > 1e-6 && *t < 1.0 - 1e-6)
        .collect();
    params.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    params.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    let (curve, tolerance, pcurve, _v_low, _v_high, surface) = {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => return vec![(ek, orient)],
        };
        let surface = reg
            .faces
            .get(face_key)
            .map(|f| f.surface.clone());
        (
            edge.curve.clone(),
            edge.tolerance,
            edge.pcurves.get(&face_key).cloned(),
            edge.v_low,
            edge.v_high,
            surface,
        )
    };
    let Some(pcurve) = pcurve else {
        return vec![(ek, orient)];
    };

    let mut breakpoints = vec![0.0_f64];
    breakpoints.extend(params);
    breakpoints.push(1.0);

    let mut out = Vec::new();
    for w in breakpoints.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        if t1 - t0 < 1e-6 {
            continue;
        }
        let seg_curve = trim_edge_curve(&curve, t0, t1);
        let seg_pc = trim_pcurve(&pcurve, t0, t1);
        let p_start = pcurve_point_3d(&seg_pc, surface.as_ref(), &seg_curve, 0.0);
        let p_end = pcurve_point_3d(&seg_pc, surface.as_ref(), &seg_curve, 1.0);
        let vk0 = reg.find_or_add_vertex(p_start, tolerance);
        let vk1 = reg.find_or_add_vertex(p_end, tolerance);
        let (v_lo, v_hi) = if orient == Orientation::Forward {
            (vk0, vk1)
        } else {
            (vk1, vk0)
        };
        let nek = reg.add_edge_with_pcurve(v_lo, v_hi, seg_curve, tolerance, face_key, seg_pc, true);
        out.push((nek, orient));
    }

    if out.is_empty() {
        vec![(ek, orient)]
    } else {
        out
    }
}

/// Replace one edge in a wire with a sequence of split edges.
pub fn replace_wire_edge_with_splits(
    wire_key: WireKey,
    old_ek: EdgeKey,
    new_edges: &[(EdgeKey, Orientation)],
    reg: &mut BRepStore,
) {
    let Some(wire) = reg.wires.get_mut(wire_key) else {
        return;
    };
    let mut rebuilt = Vec::new();
    for &(ek, orient) in &wire.edges {
        if ek == old_ek {
            rebuilt.extend_from_slice(new_edges);
        } else {
            rebuilt.push((ek, orient));
        }
    }
    wire.edges = rebuilt;
}

/// Create a degenerated edge at a pole: v_low == v_high, UV extent preserved.
pub fn add_degenerated_edge_at_pole(
    pole_vk: VertexKey,
    uv_start: (Real, Real),
    uv_end: (Real, Real),
    pole_3d: PVec3,
    tolerance: Real,
    face_key: FaceKey,
    reg: &mut BRepStore,
) -> EdgeKey {
    let zero_curve = CurveGeom::Line {
        origin: pole_3d,
        direction: PVec3::ZERO,
    };
    let degen_pc = Curve2d::Line {
        origin: (uv_start.0, uv_start.1),
        direction: (uv_end.0 - uv_start.0, uv_end.1 - uv_start.1),
    };
    reg.add_seam_edge(pole_vk, pole_vk, zero_curve, tolerance, face_key, degen_pc, true)
}

fn pcurve_point_3d(
    pc: &Curve2d,
    surface: Option<&SurfaceGeom>,
    fallback_curve: &CurveGeom,
    t: Real,
) -> PVec3 {
    if let Some(surf) = surface {
        let uv = pc.d0(t);
        surf.d0_native(uv.0, uv.1)
    } else {
        fallback_curve.d0(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::geom::SurfaceGeom;
    use crate::topo::{BRepFace, BRepWire, Orientation};

    #[test]
    fn trim_line_endpoints_match() {
        let line = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let trimmed = trim_edge_curve(&line, 0.25, 0.75);
        assert!((trimmed.d0(0.0) - PVec3::new(0.25, 0.0, 0.0)).length() < 1e-5);
        assert!((trimmed.d0(1.0) - PVec3::new(0.75, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn split_edge_produces_trimmed_segments() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let line = CurveGeom::Line {
            origin: PVec3::ZERO,
            direction: PVec3::X,
        };
        let pc = Curve2d::Line {
            origin: (0.0, 0.0),
            direction: (1.0, 0.0),
        };
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let fk = reg.faces.insert(BRepFace {
            surface,
            outer_wire: reg.wires.insert(BRepWire { edges: vec![] }),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, fk, pc, true);
        let parts = split_edge_at_params(ek, fk, Orientation::Forward, &[0.5], &mut reg);
        assert_eq!(parts.len(), 2);
        let e0 = reg.edges.get(parts[0].0).unwrap();
        assert!((e0.curve.d0(0.0) - PVec3::ZERO).length() < 1e-4);
        assert!((e0.curve.d0(1.0) - PVec3::new(0.5, 0.0, 0.0)).length() < 1e-4);
    }
}
