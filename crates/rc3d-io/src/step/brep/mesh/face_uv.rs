//! BRepMesh_Face wire extraction — UV loops from shared edge discretization.
//! OCC: BRepMesh_Face + BRep_Tool::CurveOnSurface

use std::collections::HashMap;

use rc3d_core::math::Vec3;

use super::edge_disc::EdgePolygon;
use crate::step::brep::topo::{EdgeKey, FaceKey, Orientation, WireKey};
use crate::step::brep::geom::{SurfaceGeom, plane_tangent_basis};
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::BRepFace;

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

pub fn signed_area_2d(uv: &[(f32, f32)]) -> f64 {
    let n = uv.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0f64;
    for i in 0..n {
        let (u0, v0) = uv[i];
        let (u1, v1) = uv[(i + 1) % n];
        a += u0 as f64 * v1 as f64 - u1 as f64 * v0 as f64;
    }
    a * 0.5
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
    reg: &BRepRegistry,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &HashMap<(EdgeKey, usize), usize>,
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

fn collect_wire_loop(
    face_key: FaceKey,
    wire_key: WireKey,
    surface: &SurfaceGeom,
    inv_tol: f32,
    reg: &BRepRegistry,
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    edge_boundary_idx: &HashMap<(EdgeKey, usize), usize>,
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
            let Some(global_idx) = edge_boundary_idx.get(&(ek, pi)).copied() else {
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
            let uv = match uv {
                Some(uv) => uv,
                None => {
                    all_pcurve = false;
                    match global_vertices.get(global_idx) {
                        Some(pt) => surface
                            .project(*pt)
                            .or_else(|| surface.inverse_native_uv(*pt, inv_tol))
                            .unwrap_or((0.0, 0.0)),
                        None => (0.0, 0.0),
                    }
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

fn rebuild_loop_uv_from_3d(
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
}

fn unwrap_loop_periodic(boundary: &mut [UvVertex], u_period: Option<f32>, v_period: Option<f32>) {
    if boundary.len() < 2 {
        return;
    }
    for i in 1..boundary.len() {
        let prev = boundary[i - 1].uv;
        let mut cur = boundary[i].uv;
        if let Some(pu) = u_period {
            while cur.0 - prev.0 > pu * 0.5 {
                cur.0 -= pu;
            }
            while prev.0 - cur.0 > pu * 0.5 {
                cur.0 += pu;
            }
        }
        if let Some(pv) = v_period {
            while cur.1 - prev.1 > pv * 0.5 {
                cur.1 -= pv;
            }
            while prev.1 - cur.1 > pv * 0.5 {
                cur.1 += pv;
            }
        }
        boundary[i].uv = cur;
    }
}

pub fn unwrap_periodic_uv_loops(loops: &mut FaceUvLoops, surface: &SurfaceGeom) {
    let u_period = surface.native_u_period();
    let v_period = surface.native_v_period();
    if u_period.is_none() && v_period.is_none() {
        return;
    }
    unwrap_loop_periodic(&mut loops.outer.boundary, u_period, v_period);
    for inner in &mut loops.inners {
        unwrap_loop_periodic(&mut inner.boundary, u_period, v_period);
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
fn rebuild_loop_uv_local_frame(
    loop_data: &mut UvLoop,
    global_vertices: &[Vec3],
    surface: Option<&SurfaceGeom>,
) {
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
        }
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
fn ensure_loop_orientation(boundary: &mut Vec<UvVertex>, is_hole: bool) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
    use crate::step::brep::registry::BRepRegistry;
    use crate::step::brep::topo::{BRepFace, BRepWire};

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
        use crate::step::brep::mesh::edge_disc::{discretize_edge, EdgeDiscConfig};

        let mut reg = BRepRegistry::new();
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
            let pcurve = CurveGeom::Line {
                origin: Vec3::new(u0.0, u0.1, 0.0),
                direction: Vec3::new(u1.0 - u0.0, u1.1 - u0.1, 0.0),
            };
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
        let mut edge_boundary_idx = HashMap::new();
        for &ek in &edge_keys {
            let poly = discretize_edge(ek, &reg, &config);
            for (pi, &(_, pt)) in poly.params_3d.iter().enumerate() {
                let gi = global_vertices.len();
                global_vertices.push(pt);
                edge_boundary_idx.insert((ek, pi), gi);
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
}
