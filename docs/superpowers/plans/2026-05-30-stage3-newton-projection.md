# OCC Stage 3: Newton-Raphson Projection — Implementation Plan

> **Goal:** Replace brute-force grid search + coordinate-descent with Newton-Raphson for curve and surface projection, matching OCC Extrema framework behavior.

**Architecture:** New file `brep/geom/project.rs` contains the core Newton-Raphson algorithms. `curve_eval.rs` and `surface_eval.rs` route their existing calls through the new shared functions.

**Dependency:** Stage 2 analytical d2 (Task 5) must be complete — it is ✅.

---

### Task 1: Curve Newton-Raphson `project_point_on_curve` (Spec §3.1)

**File:** Create `crates/rc3d-io/src/step/brep/geom/project.rs`

- [ ] **Step 1: Create project.rs with function signature and seed sampling**

```rust
/// Multi-start Newton-Raphson projection onto a curve.
/// Returns up to `max_candidates` stationary points as (t, distance²).
pub fn project_point_on_curve(
    curve: &CurveGeom,
    target: Vec3,
    max_candidates: usize,
) -> Vec<(f32, f32)> {
    // Step 1: Sample seed points across [0, 1]
    const SEEDS: usize = 8;
    let mut seeds: Vec<f32> = Vec::with_capacity(SEEDS);
    for i in 0..SEEDS {
        seeds.push(i as f32 / (SEEDS - 1) as f32);
    }

    // Step 2: Newton-Raphson from each seed
    let mut converged: Vec<(f32, f32)> = Vec::new();
    for &seed in &seeds {
        if let Some((t, d2)) = newton_curve(curve, target, seed) {
            converged.push((t, d2));
        }
    }

    // Step 3: Deduplicate and sort by distance
    converged.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    converged.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);
    converged.truncate(max_candidates);
    converged
}
```

- [ ] **Step 2: Implement Newton iteration core**

```rust
/// Single-start Newton-Raphson on f(t) = (C(t) - P)·C'(t) = 0.
fn newton_curve(curve: &CurveGeom, target: Vec3, mut t: f32) -> Option<(f32, f32)> {
    const MAX_ITER: usize = 20;
    const TOL_F: f32 = 1e-10;
    const TOL_DT: f32 = 1e-12;

    for _ in 0..MAX_ITER {
        let c = curve.d0(t);
        let c1 = curve.d1(t);
        let c2 = curve.d2(t);
        let diff = c - target;

        // f(t) = (C(t) - P) · C'(t)
        let f_val = diff.dot(c1);
        // f'(t) = C'(t)·C'(t) + (C(t)-P)·C''(t)
        let f_prime = c1.dot(c1) + diff.dot(c2);

        if f_prime.abs() < 1e-12 {
            break; // degenerate
        }
        let dt = f_val / f_prime;
        t = (t - dt).clamp(0.0, 1.0);

        if f_val.abs() < TOL_F || dt.abs() < TOL_DT {
            return Some((t, (curve.d0(t) - target).length_squared()));
        }
    }
    // Check if the final t is reasonable
    let d2 = (curve.d0(t) - target).length_squared();
    let c0 = curve.d0(0.0);
    let c1 = curve.d0(1.0);
    // Guard: result should not be worse than endpoints
    if d2 > (c0 - target).length_squared().min((c1 - target).length_squared()) * 2.0 {
        return None;
    }
    Some((t, d2))
}
```

- [ ] **Step 3: Wire `find_param_on_curve` through `project_point_on_curve`**

Modify `find_param_on_curve` in `curve_eval.rs` to delegate to the new function:
```rust
pub fn find_param_on_curve(curve: &CurveGeom, target: Vec3, _n_samples: usize, _refine_iters: usize) -> f32 {
    let results = crate::step::brep::geom::project::project_point_on_curve(curve, target, 3);
    results.first().map(|r| r.0).unwrap_or(0.5)
}
```

- [ ] **Step 4: Add tests**

```rust
#[test]
fn test_project_circle_quarter() {
    let circle = CurveGeom::Circle { center: Vec3::ZERO, axis: Vec3::Z, radius: 1.0, x_dir: Vec3::X, y_dir: Vec3::Y };
    let results = project_point_on_curve(&circle, Vec3::new(0.0, 1.1, 0.0), 3);
    assert!(!results.is_empty());
    let (t, _) = results[0];
    assert!((t - 0.25).abs() < 0.01); // quarter turn
}

#[test]
fn test_project_bspline_high_curvature() {
    let bspline = CurveGeom::BSpline {
        degree: 3,
        control_points: vec![
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 3.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 3.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0), Vec3::new(5.0, 3.0, 0.0),
        ],
        knots: vec![0.0,0.0,0.0,0.0, 0.33, 0.66, 1.0,1.0,1.0,1.0],
        weights: None,
    };
    // Point near a valley of the S-curve
    let target = Vec3::new(2.5, -1.0, 0.0);
    let brute = find_param_on_curve(&bspline, target, 128, 8);
    let nr = project_point_on_curve(&bspline, target, 3);
    assert!(!nr.is_empty());
    // Newton should be within 1% of the brute-force result
    assert!((nr[0].0 - brute).abs() < 0.01);
}
```

- [ ] **Step 5: Test + commit**

```bash
cargo test -p rc3d-io --lib project && cargo test -p rc3d-io --test export_step_stl --release
git add crates/rc3d-io/src/step/brep/geom/
git commit -m "feat(step): Newton-Raphson curve projection (replaces brute-force grid)"
```

---

### Task 2: Surface Newton-Raphson `project_point_on_surface` (Spec §3.2)

**File:** Extend `crates/rc3d-io/src/step/brep/geom/project.rs`

- [ ] **Step 1: Implement 2×2 Newton step**

```rust
/// Single-start Newton-Raphson on the surface projection system:
/// f(u,v) = (S(u,v) - P)·∂S/∂u = 0
/// g(u,v) = (S(u,v) - P)·∂S/∂v = 0
fn newton_surface(
    surface: &SurfaceGeom, target: Vec3,
    mut u: f32, mut v: f32,
) -> Option<(f32, f32, f32)> {
    use crate::step::brep::geom::surface_eval::SurfaceParamRange;
    let range = surface.param_range();
    let u_lo = range.u_min.max(0.0); // handle cylinder unbounded
    let u_hi = range.u_max;
    let v_lo = range.v_min.max(-1e3);
    let v_hi = range.v_max.min(1e3);

    const MAX_ITER: usize = 20;
    const TOL_F: f32 = 1e-10;
    const TOL_DX: f32 = 1e-10;

    for _ in 0..MAX_ITER {
        let s = surface.d0_native(u, v);
        let (su, sv) = surface.d1_native(u, v);
        let (suu, suv, svv) = surface.d2(u, v);
        let diff = s - target;

        let f_val = diff.dot(su);
        let g_val = diff.dot(sv);

        // Jacobian: [ d²/ddu, d²/dudv; d²/dudv, d²/dvv ]
        // where each entry = (S-P)·d²S + dS·dS
        let j11 = su.dot(su) + diff.dot(suu);
        let j12 = su.dot(sv) + diff.dot(suv);
        let j22 = sv.dot(sv) + diff.dot(svv);

        let det = j11 * j22 - j12 * j12;
        if det.abs() < 1e-12 {
            break;
        }
        // Solve J·Δ = -F
        let inv_det = 1.0 / det;
        let du = -(j22 * f_val - j12 * g_val) * inv_det;
        let dv = -(j11 * g_val - j12 * f_val) * inv_det;

        // Limit step size to stay in domain
        let max_du = (u_hi - u_lo) * 0.3;
        let max_dv = (v_hi - v_lo) * 0.3;
        u = (u + du.clamp(-max_du, max_du)).clamp(u_lo, u_hi);
        v = (v + dv.clamp(-max_dv, max_dv)).clamp(v_lo, v_hi);

        if f_val.abs() < TOL_F && g_val.abs() < TOL_F { break; }
        if du.abs() < TOL_DX && dv.abs() < TOL_DX { break; }
    }
    let s = surface.d0_native(u, v);
    Some((u, v, (s - target).length_squared()))
}
```

- [ ] **Step 2: Implement multi-start projection**

```rust
pub fn project_point_on_surface(
    surface: &SurfaceGeom,
    target: Vec3,
    max_candidates: usize,
) -> Vec<(f32, f32, f32)> {
    let range = surface.param_range();
    // Multi-start from a 4×4 grid
    let coarse = 4;
    let mut candidates: Vec<(f32, f32, f32)> = Vec::with_capacity(coarse * coarse);
    for i in 0..=coarse {
        let u = range.u_min + range.u_span() * i as f32 / coarse as f32;
        for j in 0..=coarse {
            let v = range.v_min + range.v_span() * j as f32 / coarse as f32;
            if let Some(result) = newton_surface(surface, target, u, v) {
                candidates.push(result);
            }
        }
    }
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| {
        (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5
    });
    candidates.truncate(max_candidates);
    candidates
}
```

- [ ] **Step 3: Wire into SurfaceGeom::project for BSpline and Torus**

In `surface_eval.rs`, modify BSpline and Torus `project()` arms to use Newton:
```rust
SurfaceGeom::BSpline(nurbs) => {
    let candidates = project::project_point_on_surface(self, point, 3);
    candidates.first().map(|(u, v, _)| (*u, *v))
}
SurfaceGeom::Torus { .. } => {
    let candidates = project::project_point_on_surface(self, point, 3);
    candidates.first().map(|(u, v, _)| (u, v))
}
```
Keep analytical projection for Plane/Cylinder/Cone/Sphere. Only BSpline and Torus route through Newton.

- [ ] **Step 4: Tests + STL export + commit**

---

### Task 3: Cleanup and Fallback (Spec §3.3)

- [ ] **Step 1: Keep grid_search fallback** — If Newton returns fewer than expected candidates, fall back to grid_project_2d
- [ ] **Step 2: Remove old brute-force patterns** no longer needed  
- [ ] **Step 3: Final test suite** — all tests, STL export, verify grid_fallback_count doesn't regress
- [ ] **Step 4: Commit**
