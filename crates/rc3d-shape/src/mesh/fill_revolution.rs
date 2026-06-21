//! Revolution surface mesh generation (native UV grid + ruled grid).

use std::collections::{HashMap, HashSet};

use rc3d_core::math::{Real, PVec3};

use super::edge_disc::EdgePolygon;
use super::face_fill::{
    accumulate_normals, fix_tri_winding, polyline_surface_uv, surface_is_revolution_like,
    surface_uv_basis, FaceMeshRange,
};
use crate::geom::SurfaceGeom;
use crate::topo::{BRepFace, EdgeKey, FaceKey};

pub(crate) fn wire_native_v_span(samples: &[(Real, Real)]) -> (Real, Real) {
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for &(_, v) in samples {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}

pub(crate) fn nearest_periodic(value: Real, anchor: Real, period: Real) -> Real {
    if period <= 0.0 {
        return value;
    }
    let k = ((anchor - value) / period).round();
    value + k * period
}

pub(crate) fn normalized_v_bounds_for_period(uv0: &[(Real, Real)], uv1: &[(Real, Real)], period: Real) -> Option<(Real, Real)> {
    if period <= 0.0 {
        return None;
    }
    let mut values: Vec<Real> = uv0.iter().chain(uv1.iter()).map(|&(_, v)| v).collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let anchor = values[values.len() / 2];
    let mut lo = f64::INFINITY;
    let mut hi = f64::MIN;
    for &v in &values {
        let vn = nearest_periodic(v, anchor, period);
        lo = lo.min(vn);
        hi = hi.max(vn);
    }
    Some((lo, hi))
}

pub(crate) fn seam_face_v_bounds(
    face: &BRepFace,
    uv0: &[(Real, Real)],
    uv1: &[(Real, Real)],
    profile_sweep: bool,
) -> (Real, Real) {
    let pr = face.surface.param_range();
    if profile_sweep && surface_is_revolution_like(&face.surface) {
        return (pr.v_min, pr.v_max);
    }
    let (v0_lo, v0_hi) = wire_native_v_span(uv0);
    let (v1_lo, v1_hi) = wire_native_v_span(uv1);
    let mut v_lo = v0_lo.min(v1_lo);
    let mut v_hi = v0_hi.max(v1_hi);
    if surface_is_revolution_like(&face.surface) {
        if let Some(period) = surface_uv_basis(&face.surface).native_v_period() {
            if let Some((n_lo, n_hi)) = normalized_v_bounds_for_period(uv0, uv1, period) {
                v_lo = n_lo;
                v_hi = n_hi;
            }
            // Guardrail: avoid pathological multi-turn spans (e.g. 18*TAU) that explode grid cost.
            let max_span = period * 1.25;
            if v_hi - v_lo > max_span {
                v_lo = pr.v_min;
                v_hi = pr.v_max;
            }
        }
    }
    if v_hi - v_lo < 1e-5 {
        v_lo = pr.v_min;
        v_hi = pr.v_max;
    }
    (v_lo, v_hi)
}

/// Interpolate native U at a fixed revolution angle V along one wire's projected samples.
pub(crate) fn u_at_v_on_wire(samples: &[(Real, Real)], v: Real) -> Real {
    if samples.is_empty() {
        return 0.0;
    }
    if samples.len() == 1 {
        return samples[0].0;
    }
    let mut sorted: Vec<(Real, Real)> = samples.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    if v <= sorted[0].1 {
        return sorted[0].0;
    }
    if v >= sorted[sorted.len() - 1].1 {
        return sorted[sorted.len() - 1].0;
    }
    for w in sorted.windows(2) {
        let (u0, v0) = w[0];
        let (u1, v1) = w[1];
        if v >= v0 && v <= v1 {
            let t = if (v1 - v0).abs() > 1e-12 {
                (v - v0) / (v1 - v0)
            } else {
                0.0
            };
            return u0 * (1.0 - t) + u1 * t;
        }
    }
    sorted[0].0
}

/// Interpolate native V at a fixed generatrix U along one wire's projected samples.
pub(crate) fn v_at_u_on_wire(samples: &[(Real, Real)], u: Real) -> Real {
    if samples.is_empty() {
        return 0.0;
    }
    if samples.len() == 1 {
        return samples[0].1;
    }
    let mut sorted: Vec<(Real, Real)> = samples.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if u <= sorted[0].0 {
        return sorted[0].1;
    }
    if u >= sorted[sorted.len() - 1].0 {
        return sorted[sorted.len() - 1].1;
    }
    for w in sorted.windows(2) {
        let (u0, v0) = w[0];
        let (u1, v1) = w[1];
        if u >= u0 && u <= u1 {
            let t = if (u1 - u0).abs() > 1e-12 {
                (u - u0) / (u1 - u0)
            } else {
                0.0
            };
            return v0 * (1.0 - t) + v1 * t;
        }
    }
    sorted[0].1
}

pub(crate) fn wire_native_uv_spans(samples: &[(Real, Real)]) -> (Real, Real) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut u_min = f64::INFINITY;
    let mut u_max = f64::MIN;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::MIN;
    for &(u, v) in samples {
        u_min = u_min.min(u);
        u_max = u_max.max(u);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    (u_max - u_min, v_max - v_min)
}

/// True when wire samples run mainly along generatrix U with near-constant V (profile edge).
pub(crate) fn revolution_wire_is_generatrix_profile(samples: &[(Real, Real)]) -> bool {
    let (du, dv) = wire_native_uv_spans(samples);
    du > dv * 2.0 && du > 1e-4
}

pub(crate) fn uv_mindiff(u: Real, u0: Real, period: Real) -> Real {
    (-4..=4)
        .map(|i| u + i as Real * period)
        .min_by(|a, b| (a - u0).abs().partial_cmp(&(b - u0).abs()).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(u)
}

/// Revolution ruled strip: prefer PCurve UV on the face (OCC CurveOnSurface), then 3D fallback.
pub(crate) fn revolution_wire_uv_polyline(
    face_key: FaceKey,
    surface: &SurfaceGeom,
    ek: EdgeKey,
    pis: &[usize],
    edge_polygons: &HashMap<EdgeKey, EdgePolygon>,
    global_vertices: &[PVec3],
    curve_gi: &[usize],
) -> Vec<(Real, Real)> {
    if let Some(poly) = edge_polygons.get(&ek) {
        if let Some(pcurve) = poly.params_2d.get(&face_key) {
            let mut uv = Vec::with_capacity(pis.len());
            for &pi in pis {
                if let Some(&(_, uvi)) = pcurve.get(pi) {
                    uv.push(uvi);
                }
            }
            if uv.len() >= 2 {
                let v_period = surface_uv_basis(surface).native_v_period();
                for i in 1..uv.len() {
                    let prev = uv[i - 1];
                    let mut cur = uv[i];
                    if let Some(pv) = v_period {
                        cur.1 = uv_mindiff(cur.1, prev.1, pv);
                    }
                    uv[i] = cur;
                }
                return uv;
            }
        }
    }
    polyline_surface_uv(surface, curve_gi, global_vertices)
}

pub(crate) fn revolution_ruled_grid_from_uv(
    face: &BRepFace,
    uv0: &[(Real, Real)],
    uv1: &[(Real, Real)],
    nu: usize,
    nv: usize,
) -> Option<Vec<Vec<(PVec3, PVec3)>>> {
    if uv0.len() < 2 || uv1.len() < 2 {
        return None;
    }
    let u_min = uv0
        .iter()
        .chain(uv1.iter())
        .map(|p| p.0)
        .fold(f64::INFINITY, Real::min);
    let u_max = uv0
        .iter()
        .chain(uv1.iter())
        .map(|p| p.0)
        .fold(f64::MIN, Real::max);
    let v_min = uv0
        .iter()
        .chain(uv1.iter())
        .map(|p| p.1)
        .fold(f64::INFINITY, Real::min);
    let v_max = uv0
        .iter()
        .chain(uv1.iter())
        .map(|p| p.1)
        .fold(f64::MIN, Real::max);
    let du = u_max - u_min;
    let dv = v_max - v_min;
    let profile_sweep = revolution_wire_is_generatrix_profile(uv0)
        && revolution_wire_is_generatrix_profile(uv1);
    if std::env::var("SHAPE_FACE_DIAG").is_ok() {
        let (du0, dv0) = wire_native_uv_spans(uv0);
        let (du1, dv1) = wire_native_uv_spans(uv1);
        log::debug!(
            "[rev ruled] du={:.4} dv={:.4} profile={} uv0 du/dv={:.4}/{:.4} uv1 du/dv={:.4}/{:.4}",
            du, dv, profile_sweep, du0, dv0, du1, dv1
        );
    }
    let mut grid = vec![vec![(PVec3::ZERO, PVec3::Y); nv + 1]; nu + 1];
    // Same-edge seam (F+R): wires share UV so dv=0. Fill native (u,v) rectangle.
    // Sweep U uniformly from u_min to u_max (not from wire samples) so the
    // full face parameter range is covered even when the wire's UV samples
    // are non-uniform (e.g. BSpline pcurves on revolution surfaces).
    if dv < 1e-4 && du > 1e-6 {
        let (v_lo, v_hi) = seam_face_v_bounds(face, uv0, uv1, profile_sweep);
        if std::env::var("SHAPE_FACE_DIAG").is_ok() {
            eprintln!(
                "[rev ruled] seam structured u=[{:.4},{:.4}] v=[{:.4},{:.4}] profile={}",
                u_min, u_max, v_lo, v_hi, profile_sweep
            );
        }
        for i in 0..=nu {
            let u = u_min + (u_max - u_min) * i as Real / nu as Real;
            for j in 0..=nv {
                let v = v_lo + (v_hi - v_lo) * j as Real / nv as Real;
                let pt = face.surface.d0_native(u, v);
                let mut n = face.surface.normal_native(u, v);
                if n.length_squared() < 1e-12 {
                    n = PVec3::Y;
                }
                if !face.same_sense {
                    n = -n;
                }
                grid[i][j] = (pt, n);
            }
        }
        return Some(grid);
    }
    // Collapsed U: interpolate between the two wire polylines in UV.
    if du < 1e-6 {
        let r0 = resample_uv_polyline(uv0, nu);
        let r1 = resample_uv_polyline(uv1, nu);
        for i in 0..=nu {
            for j in 0..=nv {
                let s = j as Real / nv as Real;
                let (u0, v0) = r0[i];
                let (u1, v1) = r1[i];
                let u = u0 * (1.0 - s) + u1 * s;
                let v = v0 * (1.0 - s) + v1 * s;
                let pt = face.surface.d0_native(u, v);
                let mut n = face.surface.normal_native(u, v);
                if n.length_squared() < 1e-12 {
                    n = PVec3::Y;
                }
                if !face.same_sense {
                    n = -n;
                }
                grid[i][j] = (pt, n);
            }
        }
        return Some(grid);
    }
    if profile_sweep || dv <= du * 1.5 {
        // Two generatrix/profile wires: sweep V between wires at each native U.
        for i in 0..=nu {
            let u = u_min + du * i as Real / nu as Real;
            let v0 = v_at_u_on_wire(uv0, u);
            let v1 = v_at_u_on_wire(uv1, u);
            for j in 0..=nv {
                let s = j as Real / nv as Real;
                let v = v0 * (1.0 - s) + v1 * s;
                let pt = face.surface.d0_native(u, v);
                let mut n = face.surface.normal_native(u, v);
                if !face.same_sense {
                    n = -n;
                }
                grid[i][j] = (pt, n);
            }
        }
    } else {
        // Two circular wires: sweep U between wires at each native V.
        for j in 0..=nv {
            let v = v_min + dv * j as Real / nv as Real;
            let u0 = u_at_v_on_wire(uv0, v);
            let u1 = u_at_v_on_wire(uv1, v);
            for i in 0..=nu {
                let t = i as Real / nu as Real;
                let u = u0 * (1.0 - t) + u1 * t;
                let pt = face.surface.d0_native(u, v);
                let mut n = face.surface.normal_native(u, v);
                if !face.same_sense {
                    n = -n;
                }
                grid[i][j] = (pt, n);
            }
        }
    }
    Some(grid)
}

pub(crate) fn resample_uv_polyline(uvs: &[(Real, Real)], samples: usize) -> Vec<(Real, Real)> {
    if uvs.is_empty() || samples == 0 {
        return Vec::new();
    }
    if uvs.len() == 1 {
        return vec![uvs[0]; samples + 1];
    }
    let mut out = Vec::with_capacity(samples + 1);
    for i in 0..=samples {
        let t = i as Real / samples as Real;
        let f = t * (uvs.len() - 1) as Real;
        let k = f.floor() as usize;
        let j = (k + 1).min(uvs.len() - 1);
        let u = f - k as Real;
        let (a, b) = (uvs[k], uvs[j]);
        out.push((a.0 * (1.0 - u) + b.0 * u, a.1 * (1.0 - u) + b.1 * u));
    }
    out
}

/// Fill a revolution face in native (u,v) on the analytic surface (vertices stay on-surface).
/// Prefer `mesh_trimmed_uv_grid` + CDT; kept for tests and future closed patches.
#[allow(clippy::too_many_arguments)]
pub fn mesh_revolution_native_grid(
    face_key: FaceKey,
    face: &BRepFace,
    uv_bounds: (Real, Real, Real, Real),
    segs_u: u32,
    segs_v: u32,
    global_vertices: &mut Vec<PVec3>,
    global_normals: &mut Vec<PVec3>,
    all_indices: &mut Vec<i32>,
) -> FaceMeshRange {
    let first_tri = all_indices.len() / 4;
    let (u_min, u_max, v_min, v_max) = uv_bounds;
    if (u_max - u_min).abs() < 1e-6 || (v_max - v_min).abs() < 1e-5 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
            cdt_constraint_failures: 0,
        };
    }
    // Avoid ring-only mesh when native U is degenerate but V spans (use ruled path instead).
    if (u_max - u_min).abs() < 1e-4 && (v_max - v_min).abs() > 1e-3 {
        return FaceMeshRange {
            face_key,
            first_tri,
            tri_count: 0,
            boundary_global: HashSet::new(),
            max_chord_error: 0.0,
            cdt_constraint_failures: 0,
        };
    }
    let nu = segs_u.max(2) as usize;
    let nv = segs_v.max(2) as usize;
    let mut grid: Vec<Vec<usize>> = vec![vec![0; nv + 1]; nu + 1];

    for i in 0..=nu {
        let u = u_min + (u_max - u_min) * i as Real / nu as Real;
        for j in 0..=nv {
            let v = v_min + (v_max - v_min) * j as Real / nv as Real;
            let pt = face.surface.d0_native(u, v);
            let mut n = face.surface.normal_native(u, v);
            if !face.same_sense {
                n = -n;
            }
            let gi = global_vertices.len();
            global_vertices.push(pt);
            global_normals.push(n);
            grid[i][j] = gi;
        }
    }

    for i in 0..nu {
        for j in 0..nv {
            let i00 = grid[i][j] as i32;
            let i10 = grid[i + 1][j] as i32;
            let i11 = grid[i + 1][j + 1] as i32;
            let i01 = grid[i][j + 1] as i32;
            for (mut a, mut b, mut c) in [(i00, i10, i11), (i00, i11, i01)] {
                if a == b || b == c || c == a {
                    continue;
                }
                fix_tri_winding(
                    &mut a,
                    &mut b,
                    &mut c,
                    global_vertices,
                    &face.surface,
                    face.same_sense,
                );
                all_indices.extend_from_slice(&[a, b, c, -1]);
                accumulate_normals(a, b, c, global_vertices, global_normals);
            }
        }
    }

    FaceMeshRange {
        face_key,
        first_tri,
        tri_count: all_indices.len() / 4 - first_tri,
        boundary_global: HashSet::new(),
        max_chord_error: 0.0,
        cdt_constraint_failures: 0,
    }
}
