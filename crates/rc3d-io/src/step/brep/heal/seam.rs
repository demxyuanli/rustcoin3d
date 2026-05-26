//! Missing seam edges on closed parametric faces (OCCT ShapeFix_Face::FixMissingSeamMode).

use rc3d_core::math::Vec3;

use super::super::geom::{CurveGeom, SurfaceGeom};
use super::super::mesh::MESH_CLOSED_SURFACE_SEGS;
use super::super::registry::BRepRegistry;
use super::super::topo::{EdgeKey, FaceKey, Orientation};

/// Add parametric seam edges for closed faces and trimmed periodic faces.
pub fn fix_missing_seams(reg: &mut BRepRegistry, face_key: FaceKey) -> usize {
    let (surface, tolerance, wire_empty) = {
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

    // Idempotency: if seams were already fixed, skip.
    if !reg
        .faces
        .get(face_key)
        .map(|f| f.seam_edges.is_empty())
        .unwrap_or(true)
    {
        return 0;
    }

    if wire_empty {
        added += add_vertex_loop_seams(reg, face_key, &surface, tolerance);
    } else {
        added += fix_trimmed_periodic_seam(reg, face_key, &surface, tolerance);
    }

    added
}

fn add_vertex_loop_seams(
    reg: &mut BRepRegistry,
    face_key: FaceKey,
    surface: &SurfaceGeom,
    tolerance: f32,
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
        // Insert seam edges directly into the outer wire (OCC style).
        // For VERTEX_LOOP faces the wire is empty, so these become the wire.
        if let Some(face) = reg.faces.get(face_key) {
            if let Some(wire) = reg.wires.get_mut(face.outer_wire) {
                for &ek in &added {
                    wire.edges.push((ek, Orientation::Forward));
                }
            }
        }
        // Also populate seam_edges for backward compatibility (overlay, diagnostics).
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.seam_edges.extend(added.iter().copied());
        }
    }
    count
}

/// Insert u=period seam when a trimmed periodic face wire does not touch the seam.
fn fix_trimmed_periodic_seam(
    reg: &mut BRepRegistry,
    face_key: FaceKey,
    surface: &SurfaceGeom,
    tol: f32,
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

    let mut min_u = f32::MAX;
    let mut max_u = f32::MIN;
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
        for i in 0..=8 {
            let t = i as f32 / 8.0;
            let uv = pcurve.d0(t);
            min_u = min_u.min(uv.x);
            max_u = max_u.max(uv.x);
            has_uv = true;
        }
    }

    if !has_uv {
        return 0;
    }

    let seam_tol = tol.max(1e-3);
    let touches_low = min_u <= pr.u_min + seam_tol;
    let touches_high = max_u >= pr.u_max - seam_tol;
    if touches_low && touches_high {
        return 0;
    }

    let u_seam = if !touches_low { pr.u_min } else { pr.u_max - u_period * 0.001 };

    if let Some(ek) = build_u_isoparam_seam(reg, face_key, surface, u_seam, tol) {
        let wire_key = reg.faces.get(face_key).map(|f| f.outer_wire);
        if let Some(wk) = wire_key {
            if let Some(wire) = reg.wires.get_mut(wk) {
                wire.edges.push((ek, Orientation::Forward));
            }
        }
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.seam_edges.push(ek);
        }
        1
    } else {
        0
    }
}

fn seam_u(surface: &SurfaceGeom, normalized: f32) -> f32 {
    let pr = surface.param_range();
    pr.u_min + normalized * (pr.u_max - pr.u_min)
}

fn seam_v(surface: &SurfaceGeom, normalized: f32) -> f32 {
    let pr = surface.param_range();
    pr.v_min + normalized * (pr.v_max - pr.v_min)
}

/// Seam along fixed u, v spans native param range.
fn build_u_isoparam_seam(
    reg: &mut BRepRegistry,
    face_key: FaceKey,
    surface: &SurfaceGeom,
    u: f32,
    tol: f32,
) -> Option<EdgeKey> {
    let segs = MESH_CLOSED_SURFACE_SEGS;
    let pr = surface.param_range();
    let p_lo = surface.d0_native(u, pr.v_min);
    let p_hi = surface.d0_native(u, pr.v_max);
    let closed = (p_lo - p_hi).length() < tol.max(1e-4);

    let (mut pts_3d, mut pts_uv) = sample_isoparam(surface, u, true, segs, &pr);
    if pts_3d.len() < 2 {
        return None;
    }

    if closed {
        pts_3d.pop();
        pts_uv.pop();
    }

    let curve_3d = CurveGeom::Polyline { points: pts_3d.clone() };
    let pcurve = CurveGeom::Polyline { points: pts_uv };

    if closed {
        let vk = reg.find_or_add_vertex(p_lo, tol);
        Some(reg.add_seam_edge(vk, vk, curve_3d, tol, face_key, pcurve))
    } else {
        let v0 = reg.find_or_add_vertex(p_lo, tol);
        let v1 = reg.find_or_add_vertex(p_hi, tol);
        Some(reg.add_seam_edge(v0, v1, curve_3d, tol, face_key, pcurve))
    }
}

/// Seam along fixed v, u spans native param range.
fn build_v_isoparam_seam(
    reg: &mut BRepRegistry,
    face_key: FaceKey,
    surface: &SurfaceGeom,
    v: f32,
    tol: f32,
) -> Option<EdgeKey> {
    let segs = MESH_CLOSED_SURFACE_SEGS;
    let pr = surface.param_range();
    let p_lo = surface.d0_native(pr.u_min, v);
    let p_hi = surface.d0_native(pr.u_max, v);
    let closed = (p_lo - p_hi).length() < tol.max(1e-4);

    let (mut pts_3d, mut pts_uv) = sample_isoparam(surface, v, false, segs, &pr);
    if pts_3d.len() < 2 {
        return None;
    }

    if closed {
        pts_3d.pop();
        pts_uv.pop();
    }

    let curve_3d = CurveGeom::Polyline { points: pts_3d.clone() };
    let pcurve = CurveGeom::Polyline { points: pts_uv };

    if closed {
        let vk = reg.find_or_add_vertex(p_lo, tol);
        Some(reg.add_seam_edge(vk, vk, curve_3d, tol, face_key, pcurve))
    } else {
        let v0 = reg.find_or_add_vertex(p_lo, tol);
        let v1 = reg.find_or_add_vertex(p_hi, tol);
        Some(reg.add_seam_edge(v0, v1, curve_3d, tol, face_key, pcurve))
    }
}

fn sample_isoparam(
    surface: &SurfaceGeom,
    param: f32,
    hold_u: bool,
    segs: u32,
    pr: &crate::step::brep::geom::SurfaceParamRange,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut pts_3d = Vec::with_capacity(segs as usize + 1);
    let mut pts_uv = Vec::with_capacity(segs as usize + 1);
    for i in 0..=segs {
        let t = i as f32 / segs as f32;
        let (u, v) = if hold_u {
            (param, pr.v_min + t * (pr.v_max - pr.v_min))
        } else {
            (pr.u_min + t * (pr.u_max - pr.u_min), param)
        };
        pts_3d.push(surface.d0_native(u, v));
        pts_uv.push(Vec3::new(u, v, 0.0));
    }
    (pts_3d, pts_uv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::topo::{BRepFace, BRepWire, Orientation};

    #[test]
    fn sphere_vertex_loop_gets_seam_edge() {
        let mut reg = BRepRegistry::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Sphere {
                center: Vec3::ZERO,
                radius: 5.0,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
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
        let mut reg = BRepRegistry::new();
        let wire_key = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Cylinder {
                origin: Vec3::ZERO,
                axis: Vec3::Z,
                radius: 5.0,
            },
            outer_wire: wire_key,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
        });

        let z0 = 0.0f32;
        let z1 = 10.0f32;
        let u_mid = std::f32::consts::PI;
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
            let pa = Vec3::new(5.0 * ua.cos(), 5.0 * ua.sin(), va);
            let pb = Vec3::new(5.0 * ub.cos(), 5.0 * ub.sin(), vb);
            let v0 = reg.find_or_add_vertex(pa, 1e-4);
            let v1 = reg.find_or_add_vertex(pb, 1e-4);
            let curve_3d = CurveGeom::Line { origin: pa, direction: pb - pa };
            let pcurve = CurveGeom::Line {
                origin: Vec3::new(ua, va, 0.0),
                direction: Vec3::new(ub - ua, vb - va, 0.0),
            };
            let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve);
            wire_edges.push((ek, Orientation::Forward));
        }
        let outer = reg.wires.insert(BRepWire { edges: wire_edges });
        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = outer;
        }

        let n = fix_missing_seams(&mut reg, face_key);
        assert!(n >= 1, "trimmed cylinder should get periodic seam, got {n}");
    }
}
