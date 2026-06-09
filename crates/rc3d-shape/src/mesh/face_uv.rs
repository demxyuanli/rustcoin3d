//! BRepMesh_Face wire extraction — UV loops from shared edge discretization.
//! OCC: BRepMesh_Face + BRep_Tool::CurveOnSurface

use std::collections::HashMap;

use rc3d_core::math::Vec3;

use super::edge_disc::EdgePolygon;
use crate::topo::{EdgeKey, FaceKey, Orientation, WireKey};
use crate::geom::{build_ortho_axes, SurfaceGeom, plane_tangent_basis};
use crate::store::BRepStore;
use crate::topo::BRepFace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UvSource {
    Pcurve,
    Synthetic,
    SurfaceFill,
}

#[derive(Debug, Clone)]
pub struct UvVertex {
    pub global_idx: usize,
    pub uv: (f32, f32),
}

#[derive(Debug, Clone)]
pub struct UvLoop {
    pub boundary: Vec<UvVertex>,
}

#[derive(Debug, Clone)]
pub struct FaceUvLoops {
    pub outer: UvLoop,
    pub inners: Vec<UvLoop>,
    pub uv_source: UvSource,
}

impl FaceUvLoops {
    pub fn is_valid(&self) -> bool {
        let outer_uv: Vec<Option<(f32, f32)>> =
            self.outer.boundary.iter().map(|v| Some(v.uv)).collect();
        let inner_uv: Vec<Vec<(f32, f32)>> = self
            .inners
            .iter()
            .map(|l| l.boundary.iter().map(|v| v.uv).collect())
            .collect();
        loops_uv_valid(&outer_uv, &inner_uv)
    }

    /// True when boundary loops have enough vertices for earcut/fan fill (area may be zero).
    pub fn is_fillable(&self) -> bool {
        if self.outer.boundary.len() < 3 {
            return false;
        }
        self.inners
            .iter()
            .all(|l| l.boundary.len() >= 3 || l.boundary.is_empty())
    }
}

pub use crate::geom::signed_area_2d;

/// True when the outer UV loop has fewer than 3 vertices or zero signed area.
pub fn uv_loop_is_degenerate(loops: &FaceUvLoops) -> bool {
    if loops.outer.boundary.len() < 3 {
        return true;
    }
    let uv: Vec<(f32, f32)> = loops.outer.boundary.iter().map(|v| v.uv).collect();
    signed_area_2d(&uv).abs() <= 1e-6
}

pub fn loops_uv_valid(outer: &[Option<(f32, f32)>], inners: &[Vec<(f32, f32)>]) -> bool {
    if outer.len() < 3 || !outer.iter().all(|uv| uv.is_some()) {
        return false;
    }
    let flat: Vec<(f32, f32)> = outer.iter().map(|uv| uv.unwrap()).collect();
    if signed_area_2d(&flat).abs() <= 1e-10 {
        return false;
    }
    for inner in inners {
        if inner.len() < 3 || signed_area_2d(inner).abs() <= 1e-10 {
            return false;
        }
    }
    true
}

impl FaceUvLoops {
    pub fn revolution_native_uv_bounds_from_boundary(
        &self, face: &BRepFace, global_vertices: &[Vec3],
    ) -> Option<(f32, f32, f32, f32)> {
        if !matches!(face.surface, SurfaceGeom::Revolution { .. }) { return None; }
        let mut um = f32::MAX; let mut u_max = f32::MIN; let mut vm = f32::MAX; let mut v_max = f32::MIN; let mut n = 0usize;
        for v in self.outer.boundary.iter().chain(self.inners.iter().flat_map(|l| l.boundary.iter())) {
            let Some(p) = global_vertices.get(v.global_idx) else { continue; };
            let Some((u, vn)) = face.surface.revolution_native_uv_at(*p) else { continue; };
            um = um.min(u); u_max = u_max.max(u); vm = vm.min(vn); v_max = v_max.max(vn); n += 1;
        }
        if n < 2 { return None; }
        if (u_max - um) < 1e-8 && (v_max - vm) < 1e-8 { return None; }
        if (v_max - vm) < 1e-5 {
            if let Some((r0, r1)) = self.revolution_v_bounds_from_3d(face, global_vertices) {
                if r1 > r0 + 1e-5 { return Some((um, u_max, r0, r1)); }
            }
            return None;
        }
        Some((um, u_max, vm, v_max))
    }

    pub fn native_uv_bounds(&self) -> Option<(f32, f32, f32, f32)> {
        let mut u_min = f32::MAX; let mut u_max = f32::MIN;
        let mut v_min = f32::MAX; let mut v_max = f32::MIN; let mut c = 0usize;
        for v in self.outer.boundary.iter().chain(self.inners.iter().flat_map(|l| l.boundary.iter())) {
            u_min = u_min.min(v.uv.0); u_max = u_max.max(v.uv.0);
            v_min = v_min.min(v.uv.1); v_max = v_max.max(v.uv.1); c += 1;
        }
        if c < 3 { return None; }
        let du = u_max - u_min; let dv = v_max - v_min;
        if du < 1e-12 && dv < 1e-12 { return None; }
        Some((u_min, u_max, v_min, v_max))
    }
    pub fn uv_bounds_from_projection(&self, face: &BRepFace, vertices: &[Vec3]) -> Option<(f32, f32, f32, f32)> {
        let inv_tol = face.tolerance.max(1e-3);
        let mut um = f32::MAX; let mut u_max = f32::MIN; let mut vm = f32::MAX; let mut v_max = f32::MIN; let mut n = 0usize;
        for v in self.outer.boundary.iter().chain(self.inners.iter().flat_map(|l| l.boundary.iter())) {
            let Some(p) = vertices.get(v.global_idx) else { continue; };
            let Some(uv) = face.surface.project(*p).or_else(|| face.surface.inverse_native_uv(*p, inv_tol)) else { continue; };
            um = um.min(uv.0); u_max = u_max.max(uv.0); vm = vm.min(uv.1); v_max = v_max.max(uv.1); n += 1;
        }
        if n < 2 { None } else { Some((um, u_max, vm, v_max)) }
    }
    pub fn revolution_v_bounds_from_3d(&self, face: &BRepFace, vertices: &[Vec3]) -> Option<(f32, f32)> {
        let SurfaceGeom::Revolution { axis_origin, axis_dir, .. } = &face.surface else { return None; };
        let axis = axis_dir.normalize(); let (x_dir, y_dir) = build_ortho_axes(axis);
        let mut vm = f32::MAX; let mut v_max = f32::MIN; let mut any = false;
        for v in self.outer.boundary.iter().chain(self.inners.iter().flat_map(|l| l.boundary.iter())) {
            let Some(p) = vertices.get(v.global_idx) else { continue; };
            let rel = *p - *axis_origin; let radial = rel - axis * rel.dot(axis);
            if radial.length_squared() < face.tolerance * face.tolerance { continue; }
            let u = f32::atan2(radial.dot(y_dir), radial.dot(x_dir));
            let a = if u < 0.0 { u + std::f32::consts::TAU } else { u };
            vm = vm.min(a); v_max = v_max.max(a); any = true;
        }
        if any && v_max > vm + 1e-4 { Some((vm, v_max)) } else { None }
    }
}

/// True when stored revolution U span is too small for valid trim.
pub fn revolution_u_span_collapsed(du: f32) -> bool { du < std::f32::consts::TAU * 0.15 }

/// True when revolution boundary samples collapse in native V (axis angle).
pub fn revolution_boundary_v_collapsed(loops: &FaceUvLoops) -> bool {
    loops
        .native_uv_bounds()
        .map(|(_, _, v0, v1)| (v1 - v0).abs() < 1e-5)
        .unwrap_or(true)
}

/// Even-odd point-in-trim test (BRepClass_FaceClassifier equivalent, UV only).
pub fn point_in_trim(u: f32, v: f32, outer: &[(f32, f32)], holes: &[Vec<(f32, f32)>]) -> bool {
    if !pip_even_odd(u, v, outer) {
        return false;
    }
    for hole in holes {
        if pip_even_odd(u, v, hole) {
            return false;
        }
    }
    true
}

fn pip_even_odd(x: f32, y: f32, poly: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let n = poly.len();
    for i in 0..n {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % n];
        if ((y0 > y) != (y1 > y))
            && (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-20) + x0)
        {
            inside = !inside;
        }
    }
    inside
}

pub fn collect_face_loops(
    face_key: FaceKey,
    face: &BRepFace,
    reg: &BRepStore,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &super::edge_pool::FaceEdgeBoundaryIdx,
    global_vertices: &[Vec3],
) -> FaceUvLoops {
    let (mut outer, outer_pcurve) = collect_wire_loop(
        face_key,
        face.outer_wire,
        &face.surface,
        face.tolerance,
        reg,
        edge_polygons,
        edge_boundary_idx,
        global_vertices,
        false,
    );
    let mut inners = Vec::new();
    let mut any_pcurve = outer_pcurve;
    for &iw in &face.inner_wires {
        let (inner, inner_pcurve) = collect_wire_loop(
            face_key,
            iw,
            &face.surface,
            face.tolerance,
            reg,
            edge_polygons,
            edge_boundary_idx,
            global_vertices,
            true,
        );
        any_pcurve = any_pcurve && inner_pcurve;
        inners.push(inner);
    }

    let inv_tol = face.tolerance.max(1e-3);
    if !loops_uv_valid(
        &outer.boundary.iter().map(|v| Some(v.uv)).collect::<Vec<_>>(),
        &inners
            .iter()
            .map(|l| l.boundary.iter().map(|v| v.uv).collect())
            .collect::<Vec<_>>(),
    ) {
        rebuild_loop_uv_from_3d(&mut outer, &face.surface, global_vertices, inv_tol);
        for inner in &mut inners {
            rebuild_loop_uv_from_3d(inner, &face.surface, global_vertices, inv_tol);
        }
        ensure_loop_orientation(&mut outer.boundary, false);
        for inner in &mut inners {
            ensure_loop_orientation(&mut inner.boundary, true);
        }
        if !loops_uv_valid(
            &outer.boundary.iter().map(|v| Some(v.uv)).collect::<Vec<_>>(),
            &inners
                .iter()
                .map(|l| l.boundary.iter().map(|v| v.uv).collect())
                .collect::<Vec<_>>(),
        ) {
            rebuild_loop_uv_local_frame(&mut outer, global_vertices, Some(&face.surface));
            for inner in &mut inners {
                rebuild_loop_uv_local_frame(inner, global_vertices, Some(&face.surface));
            }
            ensure_loop_orientation(&mut outer.boundary, false);
            for inner in &mut inners {
                ensure_loop_orientation(&mut inner.boundary, true);
            }
        }
        any_pcurve = false;
    }

    let uv_source = if outer.boundary.len() >= 3 {
        let outer_uv: Vec<(f32, f32)> = outer.boundary.iter().map(|v| v.uv).collect();
        if signed_area_2d(&outer_uv).abs() > 1e-10 {
            if any_pcurve {
                UvSource::Pcurve
            } else {
                UvSource::Synthetic
            }
        } else if outer.boundary.len() >= 3 {
            UvSource::Synthetic
        } else {
            UvSource::SurfaceFill
        }
    } else {
        UvSource::SurfaceFill
    };

    inners.retain(|l| l.boundary.len() >= 3);

    FaceUvLoops {
        outer,
        inners,
        uv_source,
    }
}

pub(crate) fn collect_wire_loop(
    face_key: FaceKey,
    wire_key: WireKey,
    surface: &SurfaceGeom,
    inv_tol: f32,
    reg: &BRepStore,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &super::edge_pool::FaceEdgeBoundaryIdx,
    global_vertices: &[Vec3],
    is_hole: bool,
) -> (UvLoop, bool) {
    let mut boundary = Vec::new();
    let mut all_pcurve = true;

    let wire = match reg.wires.get(wire_key) {
        Some(w) => w,
        None => return (UvLoop { boundary }, false),
    };

    for &(ek, orient) in &wire.edges {
        let Some(poly) = edge_polygons.get(&ek) else {
            continue;
        };
        let pcurve_pts = poly.params_2d.get(&face_key);
        let mut indices: Vec<usize> = (0..poly.params_3d.len()).collect();
        if orient == Orientation::Reversed {
            indices.reverse();
        }
        for pi in indices {
            let Some(global_idx) = edge_boundary_idx.get(&(face_key, ek, pi)).copied() else {
                continue;
            };
            if boundary.last().map(|v: &UvVertex| v.global_idx) == Some(global_idx) {
                continue;
            }
            let uv = if let Some(pts) = pcurve_pts {
                pts.get(pi).map(|&(_, uv)| uv)
            } else {
                None
            };
            let pt_3d = global_vertices.get(global_idx).copied();
            let uv = if let Some(uv) = uv {
                let on_surf = surface.d0_native(uv.0, uv.1);
                let dev = pt_3d.map_or(0.0, |pt| (on_surf - pt).length());
                if dev <= inv_tol * 2.0 { uv } else {
                    all_pcurve = false;
                    match pt_3d {
                        Some(pt) => surface.project(pt).or_else(|| surface.inverse_native_uv(pt, inv_tol * 2.0)).unwrap_or(uv),
                        None => uv,
                    }
                }
            } else {
                all_pcurve = false;
                match pt_3d {
                    Some(pt) => surface.project(pt).or_else(|| surface.inverse_native_uv(pt, inv_tol * 2.0)).unwrap_or((0.0, 0.0)),
                    None => (0.0, 0.0),
                }
            };
            boundary.push(UvVertex { global_idx, uv });
        }
    }

    if boundary.len() >= 2 && boundary.first().map(|v| v.global_idx) == boundary.last().map(|v| v.global_idx) {
        boundary.pop();
    }

    ensure_loop_orientation(&mut boundary, is_hole);

    if boundary.len() >= 3 {
        let uv_flat: Vec<(f32, f32)> = boundary.iter().map(|v| v.uv).collect();
        if signed_area_2d(&uv_flat).abs() <= 1e-10 {
            all_pcurve = false;
            let mut loop_data = UvLoop { boundary };
            rebuild_loop_uv_from_3d(&mut loop_data, surface, global_vertices, inv_tol);
            if signed_area_2d(
                &loop_data
                    .boundary
                    .iter()
                    .map(|v| v.uv)
                    .collect::<Vec<_>>(),
            )
            .abs()
                <= 1e-10
            {
                rebuild_loop_uv_local_frame(&mut loop_data, global_vertices, Some(surface));
            }
            ensure_loop_orientation(&mut loop_data.boundary, is_hole);
            boundary = loop_data.boundary;
        }
    }

    (UvLoop { boundary }, all_pcurve)
}

pub(crate) fn assign_revolution_native_uv_along_wire(
    loop_data: &mut UvLoop,
    surface: &SurfaceGeom,
    global_vertices: &[Vec3],
) {
    let SurfaceGeom::Revolution {
        axis_origin,
        axis_dir,
        ..
    } = surface
    else {
        return;
    };
    let n = loop_data.boundary.len();
    if n < 3 {
        return;
    }
    let axis = axis_dir.normalize();
    let (x_dir, y_dir) = build_ortho_axes(axis);
    const TAU: f32 = std::f32::consts::TAU;

    let mut raw: Vec<(f32, f32)> = Vec::with_capacity(n);
    for v in &loop_data.boundary {
        let Some(pt) = global_vertices.get(v.global_idx) else {
            // SAFETY: This fallback should rarely trigger since boundary indices
            // should be valid. If this path is hit, the mesh topology may be corrupt.
            raw.push((0.0, 0.0));
            continue;
        };
        let u = surface
            .revolution_generatrix_u_at(*pt)
            .unwrap_or(0.0);
        let rel = *pt - *axis_origin;
        let radial = rel - axis * rel.dot(axis);
        let angle = if radial.length_squared() < 1e-10 {
            // Point lies on revolution axis — angle is undefined, mark as NaN
            // for later interpolation from neighbors.
            f32::NAN
        } else {
            let a = f32::atan2(radial.dot(y_dir), radial.dot(x_dir));
            if a < 0.0 { a + TAU } else { a }
        };
        raw.push((u, angle));
    }

    // Interpolate NaN angles from nearest valid neighbors (handles axis points).
    // NOTE: If ALL vertices have NaN angles (entire loop on axis), this falls back
    // to 0.0, which produces a degenerate UV mapping. This is acceptable for the
    // rare degenerate case but should be logged for diagnostics.
    let mut all_nan = true;
    for i in 0..n {
        if !raw[i].1.is_nan() {
            all_nan = false;
            continue;
        }
        for d in 1..n {
            let fwd = raw[(i + d) % n].1;
            if !fwd.is_nan() {
                raw[i].1 = fwd;
                break;
            }
            let bwd = raw[(i + n - d) % n].1;
            if !bwd.is_nan() {
                raw[i].1 = bwd;
                break;
            }
        }
        if raw[i].1.is_nan() {
            raw[i].1 = 0.0;
        }
    }
    if all_nan {
        log::warn!(
            "[Revolution UV] All vertices on axis — degenerate UV mapping for {} vertices",
            n
        );
    }

    loop_data.boundary[0].uv = (raw[0].0, raw[0].1);
    for i in 1..n {
        let v = get_mindiff(raw[i].1, loop_data.boundary[i - 1].uv.1, TAU);
        loop_data.boundary[i].uv = (raw[i].0, v);
    }
}

fn rebuild_loop_uv_from_3d(
    loop_data: &mut UvLoop,
    surface: &SurfaceGeom,
    global_vertices: &[Vec3],
    inv_tol: f32,
) {
    if matches!(surface, SurfaceGeom::Revolution { .. }) {
        assign_revolution_native_uv_along_wire(loop_data, surface, global_vertices);
        return;
    }

    // Pass 1: Independent projection — each vertex gets its own best UV
    for v in &mut loop_data.boundary {
        let Some(pt) = global_vertices.get(v.global_idx) else { continue; };
        // Skip expensive project() when PCurve UV already matches 3D point
        let surf_pt = surface.d0_native(v.uv.0, v.uv.1);
        if (*pt - surf_pt).length() <= inv_tol * 2.0 { continue; }
        if let Some(uv) = surface.project(*pt).or_else(|| surface.inverse_native_uv(*pt, inv_tol)) {
            v.uv = uv;
        }
    }

    // Pass 2: Periodic unwrapping — make UV continuous across period boundaries
    let up = surface.native_u_period();
    let vp = surface.native_v_period();
    if up.is_some() || vp.is_some() {
        for i in 1..loop_data.boundary.len() {
            let prev = loop_data.boundary[i - 1].uv;
            let mut cur = loop_data.boundary[i].uv;
            if let Some(pu) = up { cur.0 = get_mindiff(cur.0, prev.0, pu); }
            if let Some(pv) = vp { cur.1 = get_mindiff(cur.1, prev.1, pv); }
            loop_data.boundary[i].uv = cur;
        }
    }
}


/// Rebuild loop UV from 3D surface projection with periodic unwrap.
pub fn loops_native_surface_uv(
    loops: &FaceUvLoops,
    face: &BRepFace,
    global_vertices: &[Vec3],
) -> FaceUvLoops {
    let inv_tol = face.tolerance.max(1e-3);
    let mut out = loops.clone();
    rebuild_loop_uv_from_3d(&mut out.outer, &face.surface, global_vertices, inv_tol);
    ensure_loop_orientation(&mut out.outer.boundary, false);
    let mut inners = Vec::new();
    for inner in &loops.inners {
        let mut hole = inner.clone();
        rebuild_loop_uv_from_3d(&mut hole, &face.surface, global_vertices, inv_tol);
        ensure_loop_orientation(&mut hole.boundary, true);
        if hole.boundary.len() >= 3 {
            inners.push(hole);
        }
    }
    out.inners = inners;
    unwrap_periodic_uv_loops(&mut out, &face.surface);
    if out.uv_source != UvSource::Pcurve {
        out.uv_source = UvSource::Synthetic;
    }
    out
}

/// Fallback UV from a 3D orthonormal frame fitted to the loop (trim-only; not surface params).
pub(crate) fn rebuild_loop_uv_local_frame(
    loop_data: &mut UvLoop,
    global_vertices: &[Vec3],
    surface: Option<&SurfaceGeom>,
) {
    if let Some(SurfaceGeom::Plane {
        origin,
        normal: n,
        u_dir,
    }) = surface
    {
        let (u_axis, v_axis) = plane_tangent_basis(*n, *u_dir);
        for v in &mut loop_data.boundary {
            let Some(pt) = global_vertices.get(v.global_idx) else {
                continue;
            };
            let rel = *pt - *origin;
            v.uv = (rel.dot(u_axis), rel.dot(v_axis));
        }
        return;
    }

    let pts: Vec<Vec3> = loop_data
        .boundary
        .iter()
        .filter_map(|v| global_vertices.get(v.global_idx).copied())
        .collect();
    if pts.len() < 3 {
        return;
    }

    let mut normal = Vec3::ZERO;
    for i in 0..pts.len() {
        let p0 = pts[i];
        let p1 = pts[(i + 1) % pts.len()];
        normal.x += (p0.y - p1.y) * (p0.z + p1.z);
        normal.y += (p0.z - p1.z) * (p0.x + p1.x);
        normal.z += (p0.x - p1.x) * (p0.y + p1.y);
    }

    if normal.length_squared() < 1e-20 {
        return;
    }

    let normal = normal.normalize();
    let origin = pts[0];

    let mut tangent = Vec3::ZERO;
    let mut best_len = 0.0f32;
    for i in 0..pts.len() {
        let edge = pts[(i + 1) % pts.len()] - pts[i];
        let proj = edge - normal * edge.dot(normal);
        let len = proj.length_squared();
        if len > best_len {
            best_len = len;
            tangent = proj;
        }
    }
    if best_len < 1e-20 {
        let ref_dir = if normal.x.abs() < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        tangent = ref_dir - normal * ref_dir.dot(normal);
    }
    if tangent.length_squared() < 1e-20 {
        return;
    }
    let tangent = tangent.normalize();
    let bitangent = normal.cross(tangent);

    for v in &mut loop_data.boundary {
        let Some(pt) = global_vertices.get(v.global_idx) else {
            continue;
        };
        let rel = *pt - origin;
        v.uv = (rel.dot(tangent), rel.dot(bitangent));
    }
}

/// Outer CCW (positive area); inner CW (negative area).
pub(crate) fn ensure_loop_orientation(boundary: &mut Vec<UvVertex>, is_hole: bool) {
    if boundary.len() < 3 {
        return;
    }
    let uv: Vec<(f32, f32)> = boundary.iter().map(|v| v.uv).collect();
    let area = signed_area_2d(&uv);
    let want_positive = !is_hole;
    if (area > 0.0) != want_positive {
        boundary.reverse();
    }
}

/// Split outer boundary into open chains at large 3D edge jumps (cap/spiral junctions).
pub fn split_boundary_chains_at_3d_jumps(
    boundary: &[UvVertex],
    verts: &[Vec3],
    jump_ratio: f32,
) -> (Vec<Vec<UvVertex>>, usize) {
    let n = boundary.len();
    if n < 3 {
        return (Vec::new(), 0);
    }
    let mut lens = Vec::with_capacity(n);
    let mut dzs = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        if boundary[i].global_idx < verts.len() && boundary[j].global_idx < verts.len() {
            let pi = verts[boundary[i].global_idx];
            let pj = verts[boundary[j].global_idx];
            lens.push((pj - pi).length());
            dzs.push((pj.z - pi.z).abs());
        } else {
            lens.push(0.0);
            dzs.push(0.0);
        }
    }
    let mut sorted = lens.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[sorted.len() / 2];
    let len_thresh = (med * jump_ratio).max(med + 0.5).max(1.0);

    let mut dz_sorted = dzs.clone();
    dz_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dz_med = dz_sorted[dz_sorted.len() / 2];
    let dz_thresh = (dz_med * 15.0).max(8.0);

    let is_jump = |i: usize| lens[i] > len_thresh || dzs[i] > dz_thresh;
    let long_count = (0..n).filter(|&i| is_jump(i)).count();

    let mut chains: Vec<Vec<UvVertex>> = Vec::new();
    let mut current = vec![boundary[0].clone()];
    for i in 0..n {
        let j = (i + 1) % n;
        if is_jump(i) {
            if current.len() >= 3 {
                chains.push(current);
            }
            current = vec![boundary[j].clone()];
        } else if j != 0 {
            current.push(boundary[j].clone());
        }
    }
    if current.len() >= 3 {
        chains.push(current);
    }
    if chains.is_empty() {
        chains.push(boundary.to_vec());
    }
    (chains, long_count)
}

pub fn boundary_has_3d_jumps(boundary: &[UvVertex], verts: &[Vec3], jump_ratio: f32) -> bool {
    let (_, long_count) = split_boundary_chains_at_3d_jumps(boundary, verts, jump_ratio);
    long_count >= 2
}

/// Mixed cap+side topology: at least two large 3D jumps on the outer wire.
pub fn boundary_is_mixed(boundary: &[UvVertex], verts: &[Vec3]) -> bool {
    boundary_has_3d_jumps(boundary, verts, 8.0)
}

// ── Periodic UV Loop Unwrap ──────────────────────────────────

fn get_mindiff(u: f32, u0: f32, period: f32) -> f32 {
    (-4..=4)
        .map(|i| u + i as f32 * period)
        .min_by(|a, b| (a - u0).abs().partial_cmp(&(b - u0).abs()).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(u)
}

fn unwrap_loop_periodic(boundary: &mut [UvVertex], u_period: Option<f32>, v_period: Option<f32>) {
    if boundary.len() < 2 { return; }
    for i in 1..boundary.len() {
        let prev = boundary[i - 1].uv;
        let mut cur = boundary[i].uv;
        if let Some(pu) = u_period { cur.0 = get_mindiff(cur.0, prev.0, pu); }
        if let Some(pv) = v_period { cur.1 = get_mindiff(cur.1, prev.1, pv); }
        boundary[i].uv = cur;
    }
}

pub fn unwrap_periodic_uv_loops(loops: &mut FaceUvLoops, surface: &SurfaceGeom) {
    let up = surface.native_u_period();
    let vp = surface.native_v_period();
    unwrap_loop_periodic(&mut loops.outer.boundary, up, vp);
    for inner in &mut loops.inners {
        unwrap_loop_periodic(&mut inner.boundary, up, vp);
    }
}

/// True when consecutive UV samples jump more than half a native u period (seam not unwrapped).
pub fn uv_loop_has_u_seam_jump(boundary: &[UvVertex], u_period: f32) -> bool {
    if boundary.len() < 2 || u_period <= 1e-6 {
        return false;
    }
    let half = u_period * 0.5;
    for i in 1..boundary.len() {
        if (boundary[i].uv.0 - boundary[i - 1].uv.0).abs() > half {
            return true;
        }
    }
    let du = (boundary[0].uv.0 - boundary[boundary.len() - 1].uv.0).abs();
    du > half
}

/// Cylinder PCurve loops can be on-surface yet span the u seam; CDT then produces 3D spikes.
pub fn cylinder_loop_needs_uv_rebuild(loops: &FaceUvLoops, surface: &SurfaceGeom) -> bool {
    let SurfaceGeom::Cylinder { .. } = surface else {
        return false;
    };
    if uv_loop_is_degenerate(loops) {
        return true;
    }
    let Some(u_period) = surface.native_u_period() else {
        return false;
    };
    if uv_loop_has_u_seam_jump(&loops.outer.boundary, u_period) {
        return true;
    }
    let (u0, u1, _, _) = loops.native_uv_bounds().unwrap_or((0.0, 0.0, 0.0, 0.0));
    u1 - u0 > u_period * 0.75
}

fn reproject_analytic_loop_uv(
    loop_data: &mut UvLoop,
    surface: &SurfaceGeom,
    global_vertices: &[Vec3],
    inv_tol: f32,
) {
    for v in &mut loop_data.boundary {
        let Some(pt) = global_vertices.get(v.global_idx) else {
            continue;
        };
        if let Some(uv) = surface
            .project(*pt)
            .or_else(|| surface.inverse_native_uv(*pt, inv_tol))
        {
            v.uv = uv;
        }
    }
    let up = surface.native_u_period();
    let vp = surface.native_v_period();
    unwrap_loop_periodic(&mut loop_data.boundary, up, vp);
}

/// Rebuild cylinder trim loops from 3D wire order (always reproject u; do not trust PCURVE at seam).
pub fn repair_cylinder_uv_loops(
    loops: &FaceUvLoops,
    face: &BRepFace,
    global_vertices: &[Vec3],
) -> FaceUvLoops {
    let inv_tol = face.tolerance.max(1e-3);
    let mut out = loops.clone();
    reproject_analytic_loop_uv(&mut out.outer, &face.surface, global_vertices, inv_tol);
    ensure_loop_orientation(&mut out.outer.boundary, false);
    let mut inners = Vec::new();
    for inner in &loops.inners {
        let mut hole = inner.clone();
        reproject_analytic_loop_uv(&mut hole, &face.surface, global_vertices, inv_tol);
        ensure_loop_orientation(&mut hole.boundary, true);
        if hole.boundary.len() >= 3 && !uv_loop_is_degenerate(&FaceUvLoops {
            outer: hole.clone(),
            inners: vec![],
            uv_source: UvSource::Synthetic,
        }) {
            inners.push(hole);
        }
    }
    out.inners = inners;
    out.uv_source = UvSource::Synthetic;
    out
}

// ── Plane Cap UV Repair ──────────────────────────────────────

pub fn repair_plane_loop_uv(loop_data: &mut UvLoop, face: &BRepFace, global_vertices: &[Vec3]) {
    let SurfaceGeom::Plane { origin, normal, .. } = &face.surface else { return; };
    if loop_data.boundary.len() < 3 { return; }
    let n = normal.normalize();
    let p0 = global_vertices.get(loop_data.boundary[0].global_idx).copied().unwrap_or(*origin);
    let mut u_axis = Vec3::ZERO;
    for v in loop_data.boundary.iter().skip(1) {
        let Some(pt) = global_vertices.get(v.global_idx) else { continue; };
        let d = *pt - p0;
        let d_plane = d - n * d.dot(n);
        if d_plane.length_squared() > u_axis.length_squared() { u_axis = d_plane; }
    }
    if u_axis.length_squared() < 1e-12 {
        let (ua, va) = plane_tangent_basis(n, Vec3::X);
        u_axis = ua;
        let v_axis = va;
        for v in &mut loop_data.boundary {
            let Some(pt) = global_vertices.get(v.global_idx) else { continue; };
            let rel = *pt - p0;
            v.uv = (rel.dot(u_axis), rel.dot(v_axis));
        }
        return;
    }
    let u_axis = u_axis.normalize();
    let v_axis = n.cross(u_axis);
    for v in &mut loop_data.boundary {
        let Some(pt) = global_vertices.get(v.global_idx) else { continue; };
        let rel = *pt - p0;
        v.uv = (rel.dot(u_axis), rel.dot(v_axis));
    }
}

// ── UV Loops from Wire Edges ─────────────────────────────────

pub fn loops_from_wire_edges(
    face_key: FaceKey, surface: &SurfaceGeom, inv_tol: f32,
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &super::edge_pool::FaceEdgeBoundaryIdx,
    global_vertices: &[Vec3],
) -> Option<FaceUvLoops> {
    let mut boundary = Vec::new();
    let mut any_pcurve = true;
    for &(ek, ref pis) in wire_edges {
        let poly = edge_polygons.get(&ek)?;
        let pcurve_pts = poly.params_2d.get(&face_key);
        for &pi in pis {
            let global_idx = edge_boundary_idx.get(&(face_key, ek, pi)).copied()?;
            if boundary.last().map(|v: &UvVertex| v.global_idx) == Some(global_idx) { continue; }
            let uv = if let Some(pts) = pcurve_pts { pts.get(pi).map(|&(_, uv)| uv) } else { None };
            let uv = match uv {
                Some(uv) => uv,
                None => {
                    any_pcurve = false;
                    global_vertices.get(global_idx).and_then(|pt| {
                        surface.project(*pt).or_else(|| surface.inverse_native_uv(*pt, inv_tol))
                    })?
                }
            };
            boundary.push(UvVertex { global_idx, uv });
        }
    }
    if boundary.len() >= 2 && boundary.first().map(|v| v.global_idx) == boundary.last().map(|v| v.global_idx) {
        boundary.pop();
    }
    if boundary.len() < 3 { return None; }
    ensure_loop_orientation(&mut boundary, false);
    Some(FaceUvLoops { outer: UvLoop { boundary }, inners: Vec::new(), uv_source: if any_pcurve { UvSource::Pcurve } else { UvSource::Synthetic } })
}

pub fn revolution_loops_from_wire_edges(
    face_key: FaceKey, face: &BRepFace,
    wire_edges: &[(EdgeKey, Vec<usize>)],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &super::edge_pool::FaceEdgeBoundaryIdx,
    global_vertices: &[Vec3],
) -> Option<FaceUvLoops> {
    let mut boundary = Vec::new();
    let mut fail_reason: Option<&str> = None;
    for &(ek, ref pis) in wire_edges {
        let _poly = match edge_polygons.get(&ek) { Some(p) => p, None => { fail_reason = Some("no_poly"); break; } };
        for &pi in pis {
            let gi = match edge_boundary_idx.get(&(face_key, ek, pi)).copied() {
                Some(g) => g,
                None => {
                    fail_reason = Some("no_bidx");
                    break;
                }
            };
            if boundary.last().map(|v: &UvVertex| v.global_idx) == Some(gi) { continue; }
            let pt = match global_vertices.get(gi) { Some(p) => p, None => { fail_reason = Some("no_vtx"); break; } };
            let _ = pt;
            boundary.push(UvVertex { global_idx: gi, uv: (0.0, 0.0) });
        }
        if fail_reason.is_some() { break; }
    }
    if fail_reason.is_some() { return None; }
    if boundary.len() >= 2 && boundary.first().map(|v| v.global_idx) == boundary.last().map(|v| v.global_idx) {
        boundary.pop();
    }
    if boundary.len() < 3 { return None; }
    ensure_loop_orientation(&mut boundary, false);
    let mut loops = FaceUvLoops {
        outer: UvLoop { boundary },
        inners: Vec::new(),
        uv_source: UvSource::Synthetic,
    };
    assign_revolution_native_uv_along_wire(&mut loops.outer, &face.surface, global_vertices);
    Some(loops)
}

pub fn loops_from_boundary_indices(
    indices: &[usize], face: &BRepFace, global_vertices: &[Vec3],
) -> Option<FaceUvLoops> {
    let inv_tol = face.tolerance.max(1e-3);
    let mut boundary = Vec::new();
    // For Offset surfaces, project onto the basis surface first
    let proj_surface = if let SurfaceGeom::Offset { basis, .. } = &face.surface {
        basis.as_ref()
    } else {
        &face.surface
    };
    for &gi in indices {
        let pt = global_vertices.get(gi)?;
        let uv = if matches!(face.surface, SurfaceGeom::Revolution { .. }) {
            face.surface
                .revolution_native_uv_at(*pt)
                .or_else(|| face.surface.inverse_native_uv(*pt, inv_tol))
        } else {
            proj_surface
                .project(*pt)
                .or_else(|| proj_surface.inverse_native_uv(*pt, inv_tol))
        }?;
        boundary.push(UvVertex { global_idx: gi, uv });
    }
    if boundary.len() < 3 { return None; }
    ensure_loop_orientation(&mut boundary, false);
    let mut loops = FaceUvLoops { outer: UvLoop { boundary }, inners: Vec::new(), uv_source: UvSource::Synthetic };
    unwrap_periodic_uv_loops(&mut loops, &face.surface);
    Some(loops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::curve2d::Curve2d;
    use crate::mesh::edge_pool::FaceEdgeBoundaryIdx;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::store::BRepStore;
    use crate::topo::{BRepFace, BRepWire};

    #[test]
    fn signed_area_ccw_unit_square() {
        let uv = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(signed_area_2d(&uv) > 0.0);
    }

    #[test]
    fn boundary_uv_rejects_degenerate() {
        let uv = vec![Some((0.0, 0.0)), Some((0.0, 0.0)), Some((1.0, 0.0))];
        assert!(!loops_uv_valid(&uv, &[]));
    }

    #[test]
    fn point_in_trim_hole_excluded() {
        let outer = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let hole = vec![(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)];
        assert!(!point_in_trim(2.0, 2.0, &outer, &[hole.clone()]));
        assert!(point_in_trim(0.5, 0.5, &outer, &[hole]));
    }

    #[test]
    fn collect_face_loops_square_outer() {
        use crate::mesh::edge_disc::{discretize_edge, EdgeDiscConfig};

        let mut reg = BRepStore::new();
        let wire = reg.wires.insert(BRepWire { edges: vec![] });
        let face_key = reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane {
                origin: Vec3::ZERO,
                normal: Vec3::Z,
                u_dir: Vec3::X,
            },
            outer_wire: wire,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let mut edge_keys = Vec::new();
        let edges = [
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), (0.0, 0.0), (10.0, 0.0)),
            (Vec3::new(10.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
            (Vec3::new(10.0, 10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            (Vec3::new(0.0, 10.0, 0.0), Vec3::ZERO, (0.0, 10.0), (0.0, 0.0)),
        ];
        for (a, b, u0, u1) in edges {
            let v0 = reg.find_or_add_vertex(a, 1e-4);
            let v1 = reg.find_or_add_vertex(b, 1e-4);
            let curve_3d = CurveGeom::Line {
                origin: a,
                direction: b - a,
            };
            let pcurve = Curve2d::Line { origin: (u0.0, u0.1), direction: (u1.0 - u0.0, u1.1 - u0.1) };
            let ek = reg.add_edge_with_pcurve(v0, v1, curve_3d, 1e-4, face_key, pcurve);
            edge_keys.push(ek);
        }

        if let Some(face) = reg.faces.get_mut(face_key) {
            face.outer_wire = reg.wires.insert(BRepWire {
                edges: edge_keys
                    .iter()
                    .map(|&ek| (ek, Orientation::Forward))
                    .collect(),
            });
        }

        let config = EdgeDiscConfig::default();
        let mut edge_polygons = HashMap::new();
        let mut global_vertices = Vec::new();
        let mut edge_boundary_idx: FaceEdgeBoundaryIdx = HashMap::new();
        for &ek in &edge_keys {
            let poly = discretize_edge(ek, &reg, &config);
            for (pi, &(_, pt)) in poly.params_3d.iter().enumerate() {
                let gi = global_vertices.len();
                global_vertices.push(pt);
                edge_boundary_idx.insert((face_key, ek, pi), gi);
            }
            edge_polygons.insert(ek, poly);
        }

        let face = reg.faces.get(face_key).unwrap().clone();
        let loops = collect_face_loops(
            face_key,
            &face,
            &reg,
            &edge_polygons,
            &edge_boundary_idx,
            &global_vertices,
        );
        assert!(loops.is_valid());
        assert!(loops.outer.boundary.len() >= 4);
    }

    #[test]
    fn cylinder_seam_jump_detected() {
        let tau = std::f32::consts::TAU;
        let boundary = vec![
            UvVertex { global_idx: 0, uv: (0.1, 0.0) },
            UvVertex { global_idx: 1, uv: (tau - 0.1, 1.0) },
            UvVertex { global_idx: 2, uv: (0.2, 2.0) },
        ];
        assert!(uv_loop_has_u_seam_jump(&boundary, tau));
        let cyl = SurfaceGeom::cylinder(Vec3::ZERO, Vec3::Z, 1.0);
        let loops = FaceUvLoops {
            outer: UvLoop { boundary },
            inners: vec![],
            uv_source: UvSource::Pcurve,
        };
        assert!(cylinder_loop_needs_uv_rebuild(&loops, &cyl));
    }
}
