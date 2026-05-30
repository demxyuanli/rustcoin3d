//! Newton-Raphson projection onto curves and surfaces.
//!
//! Implements OCC `Extrema_ExtPC` / `Extrema_ExtPS` -style projection:
//! multi-start sampling → Newton-Raphson on stationarity equations →
//! dedup → return best candidates.
//!
//! Falls back to grid-search + coordinate-descent when Newton fails to converge.

use rc3d_core::math::Vec3;
use super::curve_eval::CurveGeom;
use super::surface_eval::{grid_project_2d, SurfaceGeom};

// ── Curve projection ──────────────────────────────────────────

/// Multi-start Newton-Raphson projection onto a curve.
///
/// Samples `n_seeds` uniformly distributed points across [0,1], runs
/// Newton-Raphson on f(t) = (C(t)-P)·C'(t) = 0 from each, deduplicates
/// converged stationary points, and returns up to `max_results` best
/// candidates as (t, distance²).
pub fn project_point_on_curve(
    curve: &CurveGeom,
    target: Vec3,
) -> Vec<(f32, f32)> {
    const SEEDS: usize = 4;
    const MAX_RESULTS: usize = 3;

    // Seed points: uniform + endpoints (6 total, down from 10)
    let mut seeds: Vec<f32> = Vec::with_capacity(SEEDS + 2);
    seeds.push(0.0);
    for i in 0..SEEDS {
        seeds.push((i + 1) as f32 / (SEEDS + 1) as f32);
    }
    seeds.push(1.0);

    // Newton-Raphson from each seed
    let mut converged: Vec<(f32, f32)> = Vec::new();
    for &seed in &seeds {
        if let Some((t, d2)) = newton_curve(curve, target, seed) {
            converged.push((t, d2));
        }
    }

    // Deduplicate: merge stationary points within 1e-4 of each other
    converged.sort_by(|a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut unique: Vec<(f32, f32)> = Vec::new();
    for (t, d2) in converged {
        if !unique.iter().any(|(u, _)| (u - t).abs() < 1e-4) {
            unique.push((t, d2));
        }
    }
    unique.truncate(MAX_RESULTS);

    // Fallback: if Newton produced nothing useful, sample densely
    if unique.is_empty() {
        return grid_fallback_curve(curve, target);
    }

    unique
}

/// Single-start Newton-Raphson on f(t) = (C(t) - P)·C'(t) = 0.
fn newton_curve(curve: &CurveGeom, target: Vec3, mut t: f32) -> Option<(f32, f32)> {
    const MAX_ITER: usize = 20;
    const TOL_F: f32 = 1e-10;
    const TOL_DT: f32 = 1e-12;

    for _ in 0..MAX_ITER {
        let (c, c1, c2) = curve.d012(t);
        let diff = c - target;

        // Stationarity condition: (C(t) - P)·C'(t) = 0
        let f_val = diff.dot(c1);
        // Derivative: C'(t)·C'(t) + (C(t)-P)·C''(t)
        let f_prime = c1.dot(c1) + diff.dot(c2);

        if f_prime.abs() < 1e-12 {
            break;
        }
        let dt = f_val / f_prime;
        t = (t - dt).clamp(0.0, 1.0);

        if f_val.abs() < TOL_F || dt.abs() < TOL_DT {
            return Some((t, (curve.d012(t).0 - target).length_squared()));
        }
    }

    // Check the final point
    let d2 = (curve.d012(t).0 - target).length_squared();
    let d2_start = (curve.d0(0.0) - target).length_squared();
    let d2_end = (curve.d0(1.0) - target).length_squared();
    // Result should be comparable to endpoints
    if d2 > d2_start.min(d2_end) * 3.0 {
        return None;
    }
    Some((t, d2))
}

/// Dense-grid fallback when Newton fails.
fn grid_fallback_curve(curve: &CurveGeom, target: Vec3) -> Vec<(f32, f32)> {
    const N: usize = 64;
    let mut best = (0.0f32, f32::MAX);

    for i in 0..=N {
        let t = i as f32 / N as f32;
        let d2 = (curve.d0(t) - target).length_squared();
        if d2 < best.1 { best = (t, d2); }
    }

    let mut t = best.0;
    let mut step = 1.0 / (N as f32 * 2.0);
    for _ in 0..5 {
        for &dt in &[-step, step] {
            let nt = (t + dt).clamp(0.0, 1.0);
            let d2 = (curve.d0(nt) - target).length_squared();
            if d2 < best.1 { best = (nt, d2); t = nt; }
        }
        step *= 0.5;
    }
    vec![best]
}

// ── Surface projection ────────────────────────────────────────

/// Multi-start Newton-Raphson projection onto a surface.
///
/// Samples a coarse grid (4×4 → extended to 8×8), runs Newton
/// from each, deduplicates, returns best candidates.
pub fn project_point_on_surface(
    surface: &SurfaceGeom,
    target: Vec3,
) -> Vec<(f32, f32, f32)> {
    const MAX_RESULTS: usize = 3;
    let range = surface.param_range();
    let u_lo = range.u_min;
    let u_hi = range.u_max;
    let v_lo = range.v_min;
    let v_hi = range.v_max;

    // Clamp unbounded ranges for seed sampling
    let v_lo_s = v_lo.max(-100.0);
    let v_hi_s = v_hi.min(100.0);

    // 3×3 grid = 9 seeds (down from 25). Newton converges quadratically
    // from nearby seeds; fewer seeds reduces redundant convergence.
    let coarse = 2;
    let n_seeds = (coarse + 1) * (coarse + 1);
    let mut candidates: Vec<(f32, f32, f32)> = Vec::with_capacity(n_seeds);

    for i in 0..=coarse {
        let u = u_lo + (u_hi - u_lo) * i as f32 / coarse as f32;
        for j in 0..=coarse {
            let v = v_lo_s + (v_hi_s - v_lo_s) * j as f32 / coarse as f32;
            if let Some(result) = newton_surface(surface, target, u, v) {
                candidates.push(result);
            }
        }
    }

    candidates.sort_by(|a, b| {
        a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut unique: Vec<(f32, f32, f32)> = Vec::new();
    for (u, v, d2) in candidates {
        if !unique.iter().any(|(pu, pv, _)| {
            (pu - u).abs() < 1e-5 && (pv - v).abs() < 1e-5
        }) {
            unique.push((u, v, d2));
        }
    }
    unique.truncate(MAX_RESULTS);

    // Fallback: if Newton produced nothing, use grid search
    if unique.is_empty() {
        let u_lo_e = u_lo.max(-100.0);
        let u_hi_e = u_hi.min(100.0);
        let v_lo_e = v_lo.max(-100.0);
        let v_hi_e = v_hi.min(100.0);
        let (u, v) = grid_project_2d(
            |uu, vv| surface.d0_native(uu, vv),
            u_lo_e, u_hi_e, v_lo_e, v_hi_e, target,
        );
        let d2 = (surface.d0_native(u, v) - target).length_squared();
        unique.push((u, v, d2));
    }

    unique
}

/// Single-start Newton-Raphson on the surface stationarity system:
///   f(u,v) = (S(u,v) - P) · ∂S/∂u = 0
///   g(u,v) = (S(u,v) - P) · ∂S/∂v = 0
fn newton_surface(
    surface: &SurfaceGeom,
    target: Vec3,
    mut u: f32,
    mut v: f32,
) -> Option<(f32, f32, f32)> {
    let range = surface.param_range();
    let u_lo = range.u_min.max(-1e3);
    let u_hi = range.u_max.min(1e3);
    let v_lo = range.v_min.max(-1e3);
    let v_hi = range.v_max.min(1e3);

    const MAX_ITER: usize = 20;
    const TOL_F: f32 = 1e-8;
    const TOL_DX: f32 = 1e-8;

    for _ in 0..MAX_ITER {
        let (s, su, sv, suu, suv, svv) = surface.d0_d1_d2_native(u, v);
        let diff = s - target;

        let f_val = diff.dot(su);
        let g_val = diff.dot(sv);

        // Jacobian: J = [su·su + diff·suu,  su·sv + diff·suv]
        //                [su·sv + diff·suv,  sv·sv + diff·svv]
        let j11 = su.dot(su) + diff.dot(suu);
        let j12 = su.dot(sv) + diff.dot(suv);
        let j22 = sv.dot(sv) + diff.dot(svv);

        let det = j11 * j22 - j12 * j12;
        if det.abs() < 1e-12 {
            break;
        }
        let inv_det = 1.0 / det;

        // Solve J · [du; dv] = -[f_val; g_val]
        let du = -(j22 * f_val - j12 * g_val) * inv_det;
        let dv = -(j11 * g_val - j12 * f_val) * inv_det;

        // Limit step size to stay near current estimate
        let max_du = (u_hi - u_lo).max(1.0) * 0.25;
        let max_dv = (v_hi - v_lo).max(1.0) * 0.25;
        u = (u + du.clamp(-max_du, max_du)).clamp(u_lo, u_hi);
        v = (v + dv.clamp(-max_dv, max_dv)).clamp(v_lo, v_hi);

        if f_val.abs() < TOL_F && g_val.abs() < TOL_F {
            break;
        }
        if du.abs() < TOL_DX && dv.abs() < TOL_DX {
            break;
        }
    }

    let s = surface.d0_native(u, v);
    let d2 = (s - target).length_squared();

    // Reject if result is significantly worse than domain corners,
    // which indicates Newton diverged or converged to a saddle point.
    let range = surface.param_range();
    let corners = [
        surface.d0_native(range.u_min, range.v_min),
        surface.d0_native(range.u_min, range.v_max),
        surface.d0_native(range.u_max, range.v_min),
        surface.d0_native(range.u_max, range.v_max),
    ];
    let min_corner_d2 = corners.iter()
        .map(|c| (c - target).length_squared())
        .fold(f32::MAX, |a, b| a.min(b));
    if d2 > min_corner_d2 * 3.0 && min_corner_d2 > 1e-6 {
        return None;
    }
    Some((u, v, d2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::curve_eval::build_ortho_axes;

    #[test]
    fn test_project_circle() {
        let (x_dir, y_dir) = build_ortho_axes(Vec3::Z);
        let circle = CurveGeom::Circle {
            center: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 2.0,
            x_dir,
            y_dir,
        };
        // Point at (0, 2.1, 0) — closest point should be near (0, 2, 0) at θ=π/2
        let results = project_point_on_curve(&circle, Vec3::new(0.0, 2.1, 0.0));
        assert!(!results.is_empty(), "should find at least one candidate");
        let (t, d2) = results[0];
        assert!((t - 0.25).abs() < 0.02, "quarter-turn expected, got t={:.4}", t);
        assert!(d2.sqrt() < 0.2, "distance should be near 0.1, got {:.4}", d2.sqrt());
    }

    #[test]
    fn test_project_line_endpoint() {
        let line = CurveGeom::Line {
            origin: Vec3::ZERO,
            direction: Vec3::new(10.0, 0.0, 0.0),
        };
        // Point at x=11 — closest is x=10 (t=1)
        let results = project_point_on_curve(&line, Vec3::new(11.0, 1.0, 0.0));
        assert!(!results.is_empty());
        let (t, d2) = results[0];
        assert!((t - 1.0).abs() < 0.01, "endpoint expected, got t={:.4}", t);
        assert!((d2.sqrt() - (1.0f32 + 1.0f32).sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_project_circle_exact() {
        let (x_dir, y_dir) = build_ortho_axes(Vec3::Z);
        let circle = CurveGeom::Circle {
            center: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 5.0,
            x_dir,
            y_dir,
        };
        // Point ON the circle at θ=0 → (5, 0, 0)
        let on_curve = Vec3::new(5.0, 0.0, 0.0);
        let results = project_point_on_curve(&circle, on_curve);
        assert!(!results.is_empty(), "should project onto exact point");
        let (t, d2) = results[0];
        assert!(d2 < 1e-3, "exact-on-curve point should have ~0 distance, got {}", d2);
        assert!(t < 0.01 || (t - 1.0).abs() < 0.01, "should be at t≈0, got t={}", t);
    }

    #[test]
    fn test_project_line_midpoint() {
        let line = CurveGeom::Line {
            origin: Vec3::new(1.0, 2.0, 3.0),
            direction: Vec3::new(6.0, 0.0, 0.0),
        };
        let results = project_point_on_curve(&line, Vec3::new(4.0, 2.0, 3.0));
        assert!(!results.is_empty());
        let (t, _) = results[0];
        assert!((t - 0.5).abs() < 0.05, "midpoint expected");
    }
}
