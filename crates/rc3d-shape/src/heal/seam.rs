//! Missing seam edges on closed parametric faces (OCCT ShapeFix_Face::FixMissingSeamMode).

use rc3d_core::math::{Real, PVec3};

use crate::geom::{Curve2d, CurveGeom, SurfaceGeom, SurfaceParamRange};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, Orientation};

/// Sampling resolution for isoparametric seam curves on closed surfaces.
const CLOSED_SURFACE_SEGS: u32 = 48;

/// Add parametric seam edges for closed faces and trimmed periodic faces.
pub fn fix_missing_seams(reg: &mut BRepStore, face_key: FaceKey) -> usize {
    let (surface, tolerance, _wire_empty) = {
        let Some(face) = reg.faces.get(face_key) else {
            return 0;
        };
        let wire_empty = reg
            .wires
            .get(face.outer_wire)
            .map(|w| w.edges.is_empty())
            .unwrap_or(false);
        (face.surface.clone(), face.tolerance, wire_empty)
    };

    let mut added = 0usize;

    if wire_needs_vertex_loop_seam(reg, face_key) {
        if !wire_has_parametric_seam(reg, face_key) {
            added += add_vertex_loop_seams(reg, face_key, &surface, tolerance);
        }
    } else {
        added += fix_trimmed_periodic_seam(reg, face_key, &surface, tolerance);
    }

    added
}

fn add_vertex_loop_seams(
    reg: &mut BRepStore,
    face_key: FaceKey,
    surface: &SurfaceGeom,
    tolerance: Real,
) -> usize {
    let mut added = Vec::new();
    match surface {
        SurfaceGeom::Sphere { .. } => {
            if let Some(ek) = build_u_isoparam_seam(reg, face_key, surface, seam_u(surface, 0.0), tolerance) {
                added.push(ek);
            }
        }
        SurfaceGeom::Torus { .. } => {
            if let Some(ek) = build_u_isoparam_seam(reg, face_key, surface, seam_u(surface, 0.0), tolerance) {
                added.push(ek);
            }
            if let Some(ek) = build_v_isoparam_seam(reg, face_key, surface, seam_v(surface, 0.0), tolerance) {
                added.push(ek);
            }
        }
        SurfaceGeom::Cylinder { .. } | SurfaceGeom::Cone { .. } => {
            if let Some(ek) = build_u_isoparam_seam(reg, face_key, surface, seam_u(surface, 0.0), tolerance) {
                added.push(ek);
            }
        }
        _ => {}
    }

    let count = added.len();
    if count > 0 {
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.seam_edges.extend(added.iter().copied());
            let wire_key = face.outer_wire;
            if let Some(w) = reg.wires.get_mut(wire_key) {
                for &ek in &added {
                    if !w.edges.iter().any(|(e, _)| *e == ek) {
                        w.edges.push((ek, Orientation::Forward));
                    }
                }
            }
        }
    }
    count
}

/// Empty wire, degenerate-only wire, or degenerate + already-inserted seam (OCCT VERTEX_LOOP path).
fn wire_needs_vertex_loop_seam(reg: &BRepStore, face_key: FaceKey) -> bool {
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return false,
    };
    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return false,
    };
    if wire.edges.is_empty() {
        return true;
    }
    wire.edges.iter().all(|&(ek, _)| {
        let Some(e) = reg.edges.get(ek) else {
            return false;
        };
        e.v_low == e.v_high || face.seam_edges.contains(&ek)
    })
}

fn wire_has_parametric_seam(reg: &BRepStore, face_key: FaceKey) -> bool {
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return false,
    };
    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return false,
    };
    wire.edges.iter().any(|&(ek, _)| {
        if face.seam_edges.contains(&ek) {
            return reg
                .edges
                .get(ek)
                .map(|e| e.v_low != e.v_high)
                .unwrap_or(false);
        }
        reg.edges
            .get(ek)
            .map(|e| e.v_low != e.v_high)
            .unwrap_or(false)
    })
}

/// Insert u=period seam when a trimmed periodic face wire does not touch the seam.
fn fix_trimmed_periodic_seam(
    reg: &mut BRepStore,
    face_key: FaceKey,
    surface: &SurfaceGeom,
    tol: Real,
) -> usize {
    let u_period = surface.native_u_period();
    if u_period.is_none() {
        return 0;
    }
    let u_period = u_period.unwrap();
    let pr = surface.param_range();

    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return 0,
    };
    if face.seam_edges.iter().any(|&ek| {
        reg.edges
            .get(ek)
            .map(|e| e.v_low == e.v_high)
            .unwrap_or(false)
    }) {
        return 0;
    }

    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return 0,
    };

    let mut min_u = f64::MAX;
    let mut max_u = f64::MIN;
    let mut has_uv = false;

    for &(ek, _) in &wire.edges {
        let edge = match reg.edges.get(ek) {
            Some(e) => e,
            None => continue,
        };
        let pcurve = match edge.pcurves.get(&face_key) {
            Some(c) => c,
            None => continue,
        };
        for i in 0..=32 {
            let t = i as Real / 32.0;
            let uv = pcurve.d0(t);
            min_u = min_u.min(uv.0);
            max_u = max_u.max(uv.0);
            has_uv = true;
        }
    }

    let seam_tol = tol.max(1e-3);
    let (touches_low, touches_high) = if has_uv {
        let tl = min_u <= pr.u_min + seam_tol;
        let th = max_u >= pr.u_max - seam_tol;
        (tl, th)
    } else {
        // No PCurves available — conservatively assume it doesn't touch both bounds
        (false, false)
    };
    if touches_low && touches_high {
        return 0;
    }

    let u_seam = if !touches_low { pr.u_min } else { pr.u_max - u_period * 0.001 };

    if let Some(ek) = build_u_isoparam_seam(reg, face_key, surface, u_seam, tol) {
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.seam_edges.push(ek);
        }
        1
    } else {
        0
    }
}

fn face_native_uv_bounds(reg: &BRepStore, face_key: FaceKey) -> Option<SurfaceParamRange> {
    let face = reg.faces.get(face_key)?;
    let mut u_min = f64::MAX;
    let mut u_max = f64::MIN;
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
    let mut has_uv = false;

    let mut scan_wire = |wire_key| {
        let wire = match reg.wires.get(wire_key) {
            Some(w) => w,
            None => return,
        };
        for &(ek, _) in &wire.edges {
            let edge = match reg.edges.get(ek) {
                Some(e) => e,
                None => continue,
            };
            let pcurve = match edge.pcurves.get(&face_key) {
                Some(c) => c,
                None => continue,
            };
            for i in 0..=32 {
                let t = i as Real / 32.0;
                let uv = pcurve.d0(t);
                u_min = u_min.min(uv.0);
                u_max = u_max.max(uv.0);
                v_min = v_min.min(uv.1);
                v_max = v_max.max(uv.1);
                has_uv = true;
            }
        }
    };

    scan_wire(face.outer_wire);
    for &iw in &face.inner_wires {
        scan_wire(iw);
    }

    if !has_uv {
        return None;
    }

    let pad = face.tolerance.max(1e-4);
    Some(SurfaceParamRange {
        u_min: u_min - pad,
        u_max: u_max + pad,
        v_min: v_min - pad,
        v_max: v_max + pad,
    })
}

/// Effective UV span for seam sampling: face trim bounds when available, else surface `param_range`.
fn seam_sample_range(
    reg: &BRepStore,
    face_key: FaceKey,
    surface: &SurfaceGeom,
) -> SurfaceParamRange {
    if let Some(face_bounds) = face_native_uv_bounds(reg, face_key) {
        return face_bounds;
    }
    surface.param_range()
}

fn seam_u(surface: &SurfaceGeom, normalized: Real) -> Real {
    let pr = surface.param_range();
    pr.u_min + normalized * (pr.u_max - pr.u_min)
}

fn seam_v(surface: &SurfaceGeom, normalized: Real) -> Real {
    let pr = surface.param_range();
    pr.v_min + normalized * (pr.v_max - pr.v_min)
}

/// Seam along fixed u, v spans native param range.
fn build_u_isoparam_seam(
    reg: &mut BRepStore,
    face_key: FaceKey,
    surface: &SurfaceGeom,
    u: Real,
    tol: Real,
) -> Option<EdgeKey> {
    let segs = CLOSED_SURFACE_SEGS;
    let pr = seam_sample_range(reg, face_key, surface);
    if matches!(surface, SurfaceGeom::Cylinder { .. } | SurfaceGeom::Cone { .. })
        && pr.v_max - pr.v_min > 1e4
    {
        return None;
    }
    let p_lo = surface.d0_native(u, pr.v_min);
    let p_hi = surface.d0_native(u, pr.v_max);
    let closed = (p_lo - p_hi).length() < tol.max(1e-4);

    let (mut pts_3d, mut pts_uv) = sample_isoparam(surface, u, true, segs, &pr);
    if pts_3d.len() < 2 {
        return None;
    }
    // Only reject if the seam polyline is definitively outside the face trim region
    // AND the face has non-seam boundary edges that define that region.
    let has_boundary = reg.wires.get(reg.faces.get(face_key)?.outer_wire)
        .map(|w| w.edges.iter().any(|&(ek, _)| {
            !reg.faces.get(face_key)
                .map(|f| f.seam_edges.contains(&ek) || f.degenerated_edges.contains(&ek))
                .unwrap_or(false)
        }))
        .unwrap_or(false);
    if has_boundary && !seam_polyline_within_face(reg, face_key, &pts_3d, tol) {
        return None;
    }

    if closed {
        pts_3d.pop();
        pts_uv.pop();
    }

    let curve_3d = CurveGeom::Polyline { points: pts_3d.clone() };
    let pcurve = Curve2d::Polyline { points: pts_uv };

    if closed {
        let vk = reg.find_or_add_vertex(p_lo, tol);
        Some(reg.add_seam_edge(vk, vk, curve_3d, tol, face_key, pcurve, true))
    } else {
        let v0 = reg.find_or_add_vertex(p_lo, tol);
        let v1 = reg.find_or_add_vertex(p_hi, tol);
        Some(reg.add_seam_edge(v0, v1, curve_3d, tol, face_key, pcurve, true))
    }
}

/// Seam along fixed v, u spans native param range.
fn build_v_isoparam_seam(
    reg: &mut BRepStore,
    face_key: FaceKey,
    surface: &SurfaceGeom,
    v: Real,
    tol: Real,
) -> Option<EdgeKey> {
    let segs = CLOSED_SURFACE_SEGS;
    let pr = seam_sample_range(reg, face_key, surface);
    let p_lo = surface.d0_native(pr.u_min, v);
    let p_hi = surface.d0_native(pr.u_max, v);
    let closed = (p_lo - p_hi).length() < tol.max(1e-4);

    let (mut pts_3d, mut pts_uv) = sample_isoparam(surface, v, false, segs, &pr);
    if pts_3d.len() < 2 {
        return None;
    }
    // Only reject if the seam polyline is definitively outside the face trim region
    // AND the face has non-seam boundary edges that define that region.
    let has_boundary = reg.wires.get(reg.faces.get(face_key)?.outer_wire)
        .map(|w| w.edges.iter().any(|&(ek, _)| {
            !reg.faces.get(face_key)
                .map(|f| f.seam_edges.contains(&ek) || f.degenerated_edges.contains(&ek))
                .unwrap_or(false)
        }))
        .unwrap_or(false);
    if has_boundary && !seam_polyline_within_face(reg, face_key, &pts_3d, tol) {
        return None;
    }

    if closed {
        pts_3d.pop();
        pts_uv.pop();
    }

    let curve_3d = CurveGeom::Polyline { points: pts_3d.clone() };
    let pcurve = Curve2d::Polyline { points: pts_uv };

    if closed {
        let vk = reg.find_or_add_vertex(p_lo, tol);
        Some(reg.add_seam_edge(vk, vk, curve_3d, tol, face_key, pcurve, true))
    } else {
        let v0 = reg.find_or_add_vertex(p_lo, tol);
        let v1 = reg.find_or_add_vertex(p_hi, tol);
        Some(reg.add_seam_edge(v0, v1, curve_3d, tol, face_key, pcurve, true))
    }
}

/// Reject seam polylines that extend far outside the face wire 3D bbox (prevents tearing spikes).
fn seam_polyline_within_face(
    reg: &BRepStore,
    face_key: FaceKey,
    pts: &[PVec3],
    tol: Real,
) -> bool {
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return false,
    };
    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return true,
    };
    if wire.edges.is_empty() {
        return true;
    }

    let mut mn = PVec3::splat(f64::MAX);
    let mut mx = PVec3::splat(f64::MIN);
    let mut has_pts = false;
    for &(ek, _) in &wire.edges {
        if face.seam_edges.contains(&ek) {
            continue;
        }
        let Some(edge) = reg.edges.get(ek) else {
            continue;
        };
        if let Some(v0) = reg.vertices.get(edge.v_low) {
            mn = mn.min(v0.position);
            mx = mx.max(v0.position);
            has_pts = true;
        }
        if let Some(v1) = reg.vertices.get(edge.v_high) {
            mn = mn.min(v1.position);
            mx = mx.max(v1.position);
            has_pts = true;
        }
        if let Some(pcurve) = edge.pcurves.get(&face_key) {
            for i in 0..=16 {
                let t = i as Real / 16.0;
                let uv = pcurve.d0(t);
                let p = face.surface.d0_native(uv.0, uv.1);
                mn = mn.min(p);
                mx = mx.max(p);
                has_pts = true;
            }
        }
    }
    if !has_pts {
        return true;
    }

    let pad = (mx - mn).length().max(tol * 10.0) * 0.25;
    mn -= PVec3::splat(pad);
    mx += PVec3::splat(pad);

    pts.iter().all(|p| p.x >= mn.x && p.x <= mx.x && p.y >= mn.y && p.y <= mx.y && p.z >= mn.z && p.z <= mx.z)
}

fn sample_isoparam(
    surface: &SurfaceGeom,
    param: Real,
    hold_u: bool,
    segs: u32,
    pr: &crate::geom::SurfaceParamRange,
) -> (Vec<PVec3>, Vec<(Real, Real)>) {
    let mut pts_3d = Vec::with_capacity(segs as usize + 1);
    let mut pts_uv = Vec::with_capacity(segs as usize + 1);
    for i in 0..=segs {
        let t = i as Real / segs as Real;
        let (u, v) = if hold_u {
            (param, pr.v_min + t * (pr.v_max - pr.v_min))
        } else {
            (pr.u_min + t * (pr.u_max - pr.u_min), param)
        };
        pts_3d.push(surface.d0_native(u, v));
        pts_uv.push((u, v));
    }
    (pts_3d, pts_uv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::{BRepFace, BRepWire, Orientation};

    #[test]
    fn sphere_vertex_loop_gets_seam_edge() {
        let mut reg = BRepStore::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Sphere {
                center: PVec3::ZERO,
                radius: 5.0,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let n = fix_missing_seams(&mut reg, face_key);
        assert_eq!(n, 1);
        let n2 = fix_missing_seams(&mut reg, face_key);
        assert_eq!(n2, 0, "second heal must be idempotent");
        let face = reg.faces.get(face_key).unwrap();
        assert_eq!(face.seam_edges.len(), 1);
        let ek = face.seam_edges[0];
        let edge = reg.edges.get(ek).unwrap();
        assert!(edge.pcurves.contains_key(&face_key));
    }

    #[test]
    fn cylinder_trimmed_face_gets_periodic_seam() {
        let mut reg = BRepStore::new();
        let wire_key = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 5.0),
            outer_wire: wire_key,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let z0 = 0.0_f64;
        let z1 = 10.0_f64;
        let u_mid = std::f64::consts::PI;
        let pts = [
            (u_mid, z0),
            (u_mid + 0.5, z0),
            (u_mid + 1.0, z1),
            (u_mid, z1),
        ];
        let mut wire_edges = Vec::new();
        for w in pts.windows(2) {
            let (ua, va) = w[0];
            let (ub, vb) = w[1];
            let pa = PVec3::new(5.0 * ua.cos(), 5.0 * ua.sin(), va);
            let pb = PVec3::new(5.0 * ub.cos(), 5.0 * ub.sin(), vb);
            let v0 = reg.find_or_add_vertex(pa, 1e-4);
            let v1 = reg.find_or_add_vertex(pb, 1e-4);
            let curve_3d = CurveGeom::Line { origin: pa, direction: pb - pa };
            let pcurve = Curve2d::Line {
                origin: (ua, va),
                direction: (ub - ua, vb - va),
            };
            let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve, true);
            wire_edges.push((ek, Orientation::Forward));
        }
        let outer = reg.wires.insert(BRepWire { edges: wire_edges });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer;
        }

        let n = fix_missing_seams(&mut reg, face_key);
        assert_eq!(
            n, 0,
            "seam at u=0 must be skipped when isoparam lies outside trim wire hull"
        );
    }

    #[test]
    fn cylinder_full_wrap_gets_periodic_seam() {
        let mut reg = BRepStore::new();
        let wire_key = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 5.0),
            outer_wire: wire_key,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let z0 = 0.0_f64;
        let z1 = 10.0_f64;
        let pts = [
            (0.0, z0),
            (std::f64::consts::TAU * 0.25, z0),
            (std::f64::consts::TAU * 0.5, z1),
            (0.0, z1),
        ];
        let mut wire_edges = Vec::new();
        for w in pts.windows(2) {
            let (ua, va) = w[0];
            let (ub, vb) = w[1];
            let pa = PVec3::new(5.0 * ua.cos(), 5.0 * ua.sin(), va);
            let pb = PVec3::new(5.0 * ub.cos(), 5.0 * ub.sin(), vb);
            let v0 = reg.find_or_add_vertex(pa, 1e-4);
            let v1 = reg.find_or_add_vertex(pb, 1e-4);
            let curve_3d = CurveGeom::Line { origin: pa, direction: pb - pa };
            let pcurve = Curve2d::Line {
                origin: (ua, va),
                direction: (ub - ua, vb - va),
            };
            let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve, true);
            wire_edges.push((ek, Orientation::Forward));
        }
        let outer = reg.wires.insert(BRepWire { edges: wire_edges });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer;
        }

        let n = fix_missing_seams(&mut reg, face_key);
        assert!(n >= 1, "wrap-around trim should get periodic seam, got {n}");
    }
}
