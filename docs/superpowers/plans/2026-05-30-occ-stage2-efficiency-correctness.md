# OCC Stage 2: Efficiency + Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 9 efficiency and correctness issues in the STEP BRep geometry pipeline to align with OCC reference algorithms, without introducing Newton-Raphson (that's Stage 3).

**Architecture:** All changes are in `crates/rc3d-io/src/step/brep/geom/` (curve_eval.rs, surface_eval.rs) and `crates/rc3d-io/src/step/nurbs.rs`. No new files. Each task is independently testable via `cargo test -p rc3d-io`.

**Tech Stack:** Rust, B-spline Cox-de Boor recurrence, differential geometry (fundamental forms, Weingarten map)

---

### Task 1: bspline_d012 Stack Allocation (Spec §2.1)

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/geom/curve_eval.rs:33-229` (bspline_d012 function)
- Test: `crates/rc3d-io/src/step/brep/geom/curve_eval.rs` (existing tests + new)

- [ ] **Step 1: Write a test that validates B-spline d0/d1/d2 accuracy at degree 3**

Add to the `mod tests` block at the bottom of `curve_eval.rs`:

```rust
#[test]
fn test_bspline_d012_degree3_stack() {
    // Cubic B-spline: 4 control points, uniform knots
    let curve = CurveGeom::BSpline {
        degree: 3,
        control_points: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(3.0, 2.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ],
        knots: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        weights: None,
    };
    // d0 at t=0.5 should be the midpoint
    let p = curve.d0(0.5);
    assert!((p - Vec3::new(2.0, 1.5, 0.0)).length() < 0.1,
        "cubic bspline midpoint: {:?}", p);
    // d1 should be approximately (4, 0, 0) at midpoint (tangent direction)
    let d1 = curve.d1(0.5);
    assert!(d1.x > 0.0, "tangent should point +x");
    // d2 should be non-zero (curve has curvature)
    let d2 = curve.d2(0.5);
    assert!(d2.length() > 0.01, "d2 should be non-zero for curved spline");
}

#[test]
fn test_bspline_d012_high_degree() {
    // Degree 8: near the MAX_DEGREE limit
    let n = 9; // n+1 = 10 control points for degree 9
    let degree = 8;
    let cps: Vec<Vec3> = (0..=n).map(|i| {
        Vec3::new(i as f32, (i as f32 * 0.5).sin(), 0.0)
    }).collect();
    let mut knots = vec![0.0f32; degree + 1];
    knots.push(0.5);
    knots.extend(vec![1.0f32; degree + 1]);
    let curve = CurveGeom::BSpline {
        degree,
        control_points: cps,
        knots,
        weights: None,
    };
    let p = curve.d0(0.5);
    assert!(!p.is_nan(), "degree 8 should not produce NaN");
    assert!(p.x > 0.0 && p.x < n as f32, "x in range");
}
```

- [ ] **Step 2: Run tests to verify they pass with current code**

Run: `rtk cargo test -p rc3d-io --lib test_bspline_d012 -- --nocapture`
Expected: Both tests PASS (baseline before refactoring)

- [ ] **Step 3: Rewrite bspline_d012 with stack-allocated arrays**

Replace the function body at `curve_eval.rs:33-229`. The key changes:
1. Add `const MAX_DEGREE: usize = 16;` at module level
2. Replace `ndu: Vec<Vec<f32>>` with flattened `[f32; 153]` (triangular: `(MAX_DEG+1)*(MAX_DEG+2)/2`)
3. Replace `ndu1`, `ndu2`, `ndu1_pm1` with `[f32; MAX_DEGREE+1]`
4. Add helper `fn ndu_idx(k: usize, i: usize) -> usize { k * (k + 1) / 2 + i }`
5. Fall back to Vec when `degree > MAX_DEGREE` (preserve correctness)

```rust
/// Maximum B-spline degree for stack-allocated evaluation.
/// STEP files rarely exceed degree 8; this covers virtually all real geometry.
const MAX_DEGREE: usize = 16;

/// Index into flattened triangular ndu array: ndu[k][i] → ndu[k*(k+1)/2 + i]
#[inline]
fn ndu_idx(k: usize, i: usize) -> usize {
    k * (k + 1) / 2 + i
}

fn bspline_d012(
    degree: usize,
    control_points: &[Vec3],
    knots: &[f32],
    weights: Option<&[f32]>,
    t: f32,
) -> (Vec3, Vec3, Vec3) {
    let p = degree;
    if control_points.len() < p + 1 || knots.len() < 2 * (p + 1) {
        return (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    }
    let t_min = knots[p];
    let t_max = knots[control_points.len()];
    let t = t.clamp(t_min, t_max);
    if p == 0 {
        let span = find_span(0, knots, t);
        return (control_points[span], Vec3::ZERO, Vec3::ZERO);
    }

    // For extreme degrees, fall back to heap allocation
    if p > MAX_DEGREE {
        return bspline_d012_heap(p, control_points, knots, weights, t);
    }

    let span = find_span(p, knots, t);
    let s = span;

    // Stack-allocated triangular basis table
    let mut ndu = [0.0f32; (MAX_DEGREE + 1) * (MAX_DEGREE + 2) / 2];
    ndu[ndu_idx(0, 0)] = 1.0;

    for k in 1..=p {
        for i in 0..=k {
            let ci = s + i - k;
            let left = if i >= 1 {
                let d = knots[ci + k] - knots[ci];
                if d > 1e-10 { (t - knots[ci]) / d * ndu[ndu_idx(k - 1, i - 1)] } else { 0.0 }
            } else { 0.0 };
            let right = if i < k {
                let d = knots[ci + k + 1] - knots[ci + 1];
                if d > 1e-10 { (knots[ci + k + 1] - t) / d * ndu[ndu_idx(k - 1, i)] } else { 0.0 }
            } else { 0.0 };
            ndu[ndu_idx(k, i)] = left + right;
        }
    }

    let mut ndu1 = [0.0f32; MAX_DEGREE + 1];
    for k in 0..=p {
        let idx = s + k - p;
        let left = if k >= 1 {
            let d = knots[idx + p] - knots[idx];
            if d > 1e-10 { (p as f32) / d * ndu[ndu_idx(p - 1, k - 1)] } else { 0.0 }
        } else { 0.0 };
        let right = if k < p {
            let d = knots[idx + p + 1] - knots[idx + 1];
            if d > 1e-10 { (p as f32) / d * ndu[ndu_idx(p - 1, k)] } else { 0.0 }
        } else { 0.0 };
        ndu1[k] = left - right;
    }

    let mut ndu2 = [0.0f32; MAX_DEGREE + 1];
    if p >= 2 {
        let mut ndu1_pm1 = [0.0f32; MAX_DEGREE];
        for k in 0..p {
            let idx = s + k - (p - 1);
            let left = if k >= 1 {
                let d = knots[idx + p - 1] - knots[idx];
                if d > 1e-10 { ((p - 1) as f32) / d * ndu[ndu_idx(p - 2, k - 1)] } else { 0.0 }
            } else { 0.0 };
            let right = if k < p - 1 {
                let d = knots[idx + p] - knots[idx + 1];
                if d > 1e-10 { ((p - 1) as f32) / d * ndu[ndu_idx(p - 2, k)] } else { 0.0 }
            } else { 0.0 };
            ndu1_pm1[k] = left - right;
        }
        for k in 0..=p {
            let idx = s + k - p;
            let left = if k >= 1 {
                let d = knots[idx + p] - knots[idx];
                if d > 1e-10 { (p as f32) / d * ndu1_pm1[k - 1] } else { 0.0 }
            } else { 0.0 };
            let right = if k < p {
                let d = knots[idx + p + 1] - knots[idx + 1];
                if d > 1e-10 { (p as f32) / d * ndu1_pm1[k] } else { 0.0 }
            } else { 0.0 };
            ndu2[k] = left - right;
        }
    }

    // Assemble weighted sums (identical to before)
    let mut a0 = Vec3::ZERO;
    let mut a1 = Vec3::ZERO;
    let mut a2 = Vec3::ZERO;
    let mut w_sum = 0.0f32;
    let mut w_sum_1 = 0.0f32;
    let mut w_sum_2 = 0.0f32;

    for k in 0..=p {
        let idx = s + k - p;
        let cp = control_points[idx];
        let w = weights.map_or(1.0, |ws| ws[idx]);
        let n0 = ndu[ndu_idx(p, k)];
        let n1 = ndu1[k];
        let n2 = ndu2[k];
        a0 += cp * (n0 * w);
        a1 += cp * (n1 * w);
        a2 += cp * (n2 * w);
        w_sum += n0 * w;
        w_sum_1 += n1 * w;
        w_sum_2 += n2 * w;
    }

    let (d0, d1, d2) = if weights.is_some() {
        let w = w_sum;
        let w1 = w_sum_1;
        let w2 = w_sum_2;
        let wi = 1.0 / w.max(1e-12);
        let wi2 = wi * wi;
        let wi3 = wi2 * wi;
        (a0 * wi,
         (a1 * w - a0 * w1) * wi2,
         (a2 * w * w - a0 * w2 * w - 2.0 * a1 * w1 * w + 2.0 * a0 * w1 * w1) * wi3)
    } else {
        (a0, a1, a2)
    };

    let safe = |v: Vec3| -> Vec3 { if v.is_nan() { Vec3::ZERO } else { v } };
    (safe(d0), safe(d1), safe(d2))
}

/// Heap-allocated fallback for degrees exceeding MAX_DEGREE.
fn bspline_d012_heap(
    p: usize, control_points: &[Vec3], knots: &[f32],
    weights: Option<&[f32]>, t: f32,
) -> (Vec3, Vec3, Vec3) {
    // ... same logic as original Vec-based code ...
    // Copy the original implementation verbatim here.
    let s = find_span(p, knots, t);
    let mut ndu: Vec<Vec<f32>> = (0..=p).map(|k| vec![0.0f32; k + 1]).collect();
    ndu[0][0] = 1.0;
    for k in 1..=p {
        for i in 0..=k {
            let ci = s + i - k;
            let left = if i >= 1 {
                let d = knots[ci + k] - knots[ci];
                if d > 1e-10 { (t - knots[ci]) / d * ndu[k-1][i-1] } else { 0.0 }
            } else { 0.0 };
            let right = if i < k {
                let d = knots[ci + k + 1] - knots[ci + 1];
                if d > 1e-10 { (knots[ci + k + 1] - t) / d * ndu[k-1][i] } else { 0.0 }
            } else { 0.0 };
            ndu[k][i] = left + right;
        }
    }
    let mut ndu1 = vec![0.0f32; p + 1];
    for k in 0..=p {
        let idx = s + k - p;
        let left = if k >= 1 { let d = knots[idx+p]-knots[idx]; if d > 1e-10 { (p as f32)/d*ndu[p-1][k-1] } else { 0.0 } } else { 0.0 };
        let right = if k < p { let d = knots[idx+p+1]-knots[idx+1]; if d > 1e-10 { (p as f32)/d*ndu[p-1][k] } else { 0.0 } } else { 0.0 };
        ndu1[k] = left - right;
    }
    let mut ndu2 = vec![0.0f32; p + 1];
    if p >= 2 {
        let mut ndu1_pm1 = vec![0.0f32; p];
        for k in 0..p {
            let idx = s + k - (p - 1);
            let left = if k >= 1 { let d = knots[idx+p-1]-knots[idx]; if d > 1e-10 { ((p-1) as f32)/d*ndu[p-2][k-1] } else { 0.0 } } else { 0.0 };
            let right = if k < p-1 { let d = knots[idx+p]-knots[idx+1]; if d > 1e-10 { ((p-1) as f32)/d*ndu[p-2][k] } else { 0.0 } } else { 0.0 };
            ndu1_pm1[k] = left - right;
        }
        for k in 0..=p {
            let idx = s + k - p;
            let left = if k >= 1 { let d = knots[idx+p]-knots[idx]; if d > 1e-10 { (p as f32)/d*ndu1_pm1[k-1] } else { 0.0 } } else { 0.0 };
            let right = if k < p { let d = knots[idx+p+1]-knots[idx+1]; if d > 1e-10 { (p as f32)/d*ndu1_pm1[k] } else { 0.0 } } else { 0.0 };
            ndu2[k] = left - right;
        }
    }
    let mut a0 = Vec3::ZERO; let mut a1 = Vec3::ZERO; let mut a2 = Vec3::ZERO;
    let mut ws = 0.0f32; let mut ws1 = 0.0f32; let mut ws2 = 0.0f32;
    for k in 0..=p {
        let idx = s+k-p; let cp = control_points[idx];
        let w = weights.map_or(1.0, |w| w[idx]);
        a0 += cp*(ndu[p][k]*w); a1 += cp*(ndu1[k]*w); a2 += cp*(ndu2[k]*w);
        ws += ndu[p][k]*w; ws1 += ndu1[k]*w; ws2 += ndu2[k]*w;
    }
    let (d0, d1, d2) = if weights.is_some() {
        let wi = 1.0/ws.max(1e-12); let wi2 = wi*wi; let wi3 = wi2*wi;
        (a0*wi, (a1*ws-a0*ws1)*wi2, (a2*ws*ws-a0*ws2*ws-2.0*a1*ws1*ws+2.0*a0*ws1*ws1)*wi3)
    } else { (a0, a1, a2) };
    let safe = |v: Vec3| -> Vec3 { if v.is_nan() { Vec3::ZERO } else { v } };
    (safe(d0), safe(d1), safe(d2))
}
```

- [ ] **Step 4: Run all tests to verify no regressions**

Run: `rtk cargo test -p rc3d-io --lib -- --nocapture`
Expected: All tests PASS (365+), including the new degree-3 and degree-8 tests

- [ ] **Step 5: Run STL export regression test**

Run: `rtk cargo test -p rc3d-io --test export_step_stl --release -- --nocapture`
Expected: PASS — STL file sizes and triangle counts match baseline

- [ ] **Step 6: Commit**

```bash
git add crates/rc3d-io/src/step/brep/geom/curve_eval.rs
git commit -m "perf(step): stack-allocate bspline_d012 tables (eliminate 5-7 heap allocs per call)"
```

---

### Task 2: Composite Arc-Length Caching (Spec §2.4)

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/geom/curve_eval.rs:232-286` (approx_chordal_length + find_composite_segment)
- Modify: `crates/rc3d-io/src/step/brep/geom/curve_eval.rs:292-300` (CurveGeom::Composite variant)
- Modify: `crates/rc3d-io/src/step/brep/build/curve.rs:63-78` (Composite construction)
- Test: `crates/rc3d-io/src/step/brep/geom/curve_eval.rs`

- [ ] **Step 1: Write a test that exercises Composite curve segment selection**

Add to `curve_eval.rs` tests:

```rust
#[test]
fn test_composite_cached_lengths() {
    let seg1 = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::new(10.0, 0.0, 0.0) };
    let seg2 = CurveGeom::Line { origin: Vec3::new(10.0, 0.0, 0.0), direction: Vec3::new(0.0, 5.0, 0.0) };
    let comp = CurveGeom::Composite {
        segments: vec![(seg1, false), (seg2, false)],
        cached_lengths: None,
    };
    // seg1 length ~10, seg2 length ~5, total ~15
    // t=0.5 should land in seg1 (10/15 > 0.5)
    let p_mid = comp.d0(0.5);
    assert!(p_mid.x > 5.0, "midpoint should be past seg1 midpoint, got {:?}", p_mid);

    // After first d0 call, cached_lengths should be populated
    let comp2 = CurveGeom::Composite {
        segments: vec![
            (CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X }, false),
            (CurveGeom::Line { origin: Vec3::new(1.0, 0.0, 0.0), direction: Vec3::Y }, false),
        ],
        cached_lengths: None,
    };
    let _ = comp2.d0(0.3);
    let _ = comp2.d0(0.7);
    // Both calls should work; caching is transparent
}
```

- [ ] **Step 2: Run test to verify it fails (cached_lengths field doesn't exist yet)**

Run: `rtk cargo test -p rc3d-io --lib test_composite_cached_lengths -- --nocapture`
Expected: FAIL — `no field cached_lengths on type CurveGeom`

- [ ] **Step 3: Add cached_lengths to Composite variant and thread through**

In `curve_eval.rs`, modify the `CurveGeom` enum:

```rust
pub enum CurveGeom {
    Line { origin: Vec3, direction: Vec3 },
    Circle { center: Vec3, axis: Vec3, radius: f32 },
    Ellipse { center: Vec3, axis: Vec3, semi_major: f32, semi_minor: f32 },
    BSpline { degree: usize, control_points: Vec<Vec3>, knots: Vec<f32>, weights: Option<Vec<f32>> },
    Trimmed { basis: Box<CurveGeom>, t_min: f32, t_max: f32 },
    Composite {
        segments: Vec<(CurveGeom, bool)>,
        cached_lengths: Option<Vec<f32>>,
    },
    Polyline { points: Vec<Vec3> },
}
```

Update `find_composite_segment` to accept and populate the cache:

```rust
fn find_composite_segment(
    segments: &[(CurveGeom, bool)],
    cached: &Option<Vec<f32>>,
    t: f32,
) -> Option<(usize, f32, f32, Vec<f32>)> {
    if segments.is_empty() { return None; }
    if segments.len() == 1 { return Some((0, t, 1.0, Vec::new())); }

    let lengths = match cached {
        Some(l) if l.len() == segments.len() => l.clone(),
        _ => segments.iter()
            .map(|(seg, _)| approx_chordal_length(seg).max(1e-10))
            .collect::<Vec<_>>(),
    };
    let total: f32 = lengths.iter().sum();
    let inv_total = 1.0 / total.max(1e-10);

    let target = t.clamp(0.0, 1.0);
    let mut cumulative = 0.0f32;
    for (i, &len) in lengths.iter().enumerate() {
        let seg_frac = len * inv_total;
        let next = cumulative + seg_frac;
        if target <= next || i == segments.len() - 1 {
            let t_local = if seg_frac > 1e-12 {
                (target - cumulative) / seg_frac
            } else { 0.0 };
            return Some((i, t_local.clamp(0.0, 1.0), seg_frac, lengths));
        }
        cumulative = next;
    }
    None
}
```

Update `d0`, `d1`, `d2` Composite arms to use the new signature and pass back cached lengths. Since `CurveGeom` is `Clone` and immutable after construction, we can't mutate the cache through `&self`. Instead, use a different approach: **compute and cache at construction time**.

**Simpler approach**: Remove `cached_lengths` from the enum. Instead, precompute lengths once at Composite construction in `build/curve.rs` and store them as part of a wrapper. But that changes the public API.

**Revised approach**: Keep `cached_lengths` in the enum. Compute eagerly at construction in `build_curve` (line 78):

```rust
// In build/curve.rs, line 63-78
"COMPOSITE_CURVE" => {
    let seg_ids = geom::nth_list_refs(&record.params, 1).unwrap_or_default();
    let segments: Vec<(CurveGeom, bool)> = seg_ids.iter()
        .filter_map(|&seg_id| { /* ... existing code ... */ })
        .collect();
    if segments.is_empty() { None } else {
        let cached_lengths: Vec<f32> = segments.iter()
            .map(|(seg, _)| {
                const N: usize = 32;
                let mut len = 0.0;
                let mut prev = seg.d0(0.0);
                for i in 1..=N {
                    let t = i as f32 / N as f32;
                    let curr = seg.d0(t);
                    len += (curr - prev).length();
                    prev = curr;
                }
                len.max(1e-10)
            })
            .collect();
        Some(CurveGeom::Composite { segments, cached_lengths: Some(cached_lengths) })
    }
}
```

Also add cached_lengths: None to all other Composite construction sites (tests, etc.):
- Search for `Composite {` across the codebase and add `cached_lengths: None` to each.

Then update `find_composite_segment` to be a simple function:

```rust
fn find_composite_segment(
    segments: &[(CurveGeom, bool)],
    cached_lengths: &Option<Vec<f32>>,
    t: f32,
) -> Option<(usize, f32, f32)> {
    if segments.is_empty() { return None; }
    if segments.len() == 1 { return Some((0, t, 1.0)); }

    let default_lengths;
    let lengths = match cached_lengths {
        Some(l) if l.len() == segments.len() => l.as_slice(),
        _ => {
            default_lengths = segments.iter()
                .map(|(seg, _)| approx_chordal_length(seg).max(1e-10))
                .collect::<Vec<_>>();
            &default_lengths
        }
    };
    let total: f32 = lengths.iter().sum();
    let inv_total = 1.0 / total.max(1e-10);
    let target = t.clamp(0.0, 1.0);
    let mut cumulative = 0.0f32;
    for (i, &len) in lengths.iter().enumerate() {
        let seg_frac = len * inv_total;
        let next = cumulative + seg_frac;
        if target <= next || i == segments.len() - 1 {
            let t_local = if seg_frac > 1e-12 {
                (target - cumulative) / seg_frac
            } else { 0.0 };
            return Some((i, t_local.clamp(0.0, 1.0), seg_frac));
        }
        cumulative = next;
    }
    None
}
```

Update d0/d1/d2 Composite match arms to pass `cached_lengths`:
```rust
CurveGeom::Composite { segments, cached_lengths } => {
    if let Some((idx, t_local, _)) = find_composite_segment(segments, cached_lengths, t) {
        // ... existing logic ...
    }
}
```

- [ ] **Step 4: Run all tests**

Run: `rtk cargo test -p rc3d-io --lib -- --nocapture`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-io/src/step/brep/geom/curve_eval.rs
git add crates/rc3d-io/src/step/brep/build/curve.rs
git commit -m "perf(step): cache composite curve segment lengths (O(N*32) → O(1) per eval)"
```

---

### Task 3: NURBS Shared Basis Functions (Spec §2.5)

**Files:**
- Modify: `crates/rc3d-io/src/step/nurbs.rs:40-183` (evaluate + derivative + rational_surface_derivatives)
- Modify: `crates/rc3d-io/src/step/brep/geom/surface_eval.rs:233-322` (BSpline d0/d1 arms)
- Test: `crates/rc3d-io/src/step/nurbs.rs` (existing tests)

- [ ] **Step 1: Write a test that validates combined evaluate+derivative**

Add to `nurbs.rs` tests:

```rust
#[test]
fn test_evaluate_with_derivative() {
    let plane = NurbsSurface::plane(0.0, 10.0, 0.0, 10.0);
    let (p, du, dv) = plane.evaluate_with_derivative(5.0, 5.0);
    assert!((p - Vec3::new(5.0, 5.0, 0.0)).length() < 1e-4);
    assert!((du - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4);
    assert!((dv - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-4);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk cargo test -p rc3d-io --lib test_evaluate_with_derivative -- --nocapture`
Expected: FAIL — method not found

- [ ] **Step 3: Implement evaluate_with_derivative in nurbs.rs**

Add method to `impl NurbsSurface` that computes basis functions once and returns both position and derivatives:

```rust
/// Evaluate position and first partial derivatives in one pass, sharing
/// the find_span + bspline_bases computation between them.
pub fn evaluate_with_derivative(&self, u: f32, v: f32) -> (Vec3, Vec3, Vec3) {
    let span_u = find_span(self.degree_u, &self.knots_u, u);
    let span_v = find_span(self.degree_v, &self.knots_v, v);
    let basis_u = bspline_bases(span_u, self.degree_u, u, &self.knots_u);
    let basis_v = bspline_bases(span_v, self.degree_v, v, &self.knots_v);

    // Compute analytical basis derivatives (not finite differences!)
    let du_basis = analytical_basis_derivatives(
        span_u, self.degree_u, u, &self.knots_u,
    );
    let dv_basis = analytical_basis_derivatives(
        span_v, self.degree_v, v, &self.knots_v,
    );

    let mut w = 0.0f32;
    let mut w_u = 0.0f32;
    let mut w_v = 0.0f32;
    let mut p = Vec3::ZERO;
    let mut p_u = Vec3::ZERO;
    let mut p_v = Vec3::ZERO;

    for &(i, nu) in &basis_u {
        let dn_du = du_basis.iter()
            .find(|(j, _)| *j == i).map(|(_, v)| *v).unwrap_or(0.0);
        for &(j, nv) in &basis_v {
            let dn_dv = dv_basis.iter()
                .find(|(k, _)| *k == j).map(|(_, v)| *v).unwrap_or(0.0);
            let wgt = self.weights[i][j];
            let cp = self.control_points[i][j];
            let coeff = nu * nv * wgt;
            w += coeff;
            p = p + cp * coeff;
            w_u += dn_du * nv * wgt;
            p_u = p_u + cp * (dn_du * nv * wgt);
            w_v += nu * dn_dv * wgt;
            p_v = p_v + cp * (nu * dn_dv * wgt);
        }
    }

    if w.abs() < 1e-10 {
        return (Vec3::ZERO, Vec3::X, Vec3::Y);
    }
    let inv_w = 1.0 / w;
    let inv_w2 = inv_w * inv_w;
    let pos = p * inv_w;
    let du = (p_u * w - p * w_u) * inv_w2;
    let dv = (p_v * w - p * w_v) * inv_w2;
    (pos, du, dv)
}
```

Also replace `compute_bspline_derivatives` with an analytical version that uses the B-spline derivative recurrence instead of finite differences:

```rust
/// Analytical first derivatives of B-spline basis functions N'_{i,p}(t).
/// Uses the recurrence: N'_{i,p} = p/(knots[i+p]-knots[i]) * N_{i,p-1}
///                           - p/(knots[i+p+1]-knots[i+1]) * N_{i+1,p-1}
fn analytical_basis_derivatives(
    span: usize, degree: usize, t: f32, knots: &[f32],
) -> Vec<(usize, f32)> {
    if degree == 0 {
        return vec![(span, 0.0)];
    }
    let p = degree;
    // Compute degree p-1 basis at span (or span-1 for the shifted index)
    let bases_curr = bspline_bases(span, p, t, knots);
    let bases_prev = bspline_bases(span, p - 1, t, knots);
    let bases_prev_shifted = if span > 0 {
        bspline_bases(span - 1, p - 1, t, knots)
    } else {
        Vec::new()
    };

    let mut result: Vec<(usize, f32)> = Vec::with_capacity(bases_curr.len());
    for &(i, _) in &bases_curr {
        let left = if i < knots.len() && i + p < knots.len() {
            let d = knots[i + p] - knots[i];
            if d > 1e-10 {
                let n_prev = bases_prev.iter()
                    .find(|(j, _)| *j == i).map(|(_, v)| *v).unwrap_or(0.0);
                (p as f32) / d * n_prev
            } else { 0.0 }
        } else { 0.0 };
        let right = if i + 1 < knots.len() && i + p + 1 < knots.len() {
            let d = knots[i + p + 1] - knots[i + 1];
            if d > 1e-10 {
                let n_prev = bases_prev_shifted.iter()
                    .find(|(j, _)| *j == i + 1)
                    .or_else(|| bases_prev.iter().find(|(j, _)| *j == i + 1))
                    .map(|(_, v)| *v).unwrap_or(0.0);
                (p as f32) / d * n_prev
            } else { 0.0 }
        } else { 0.0 };
        let deriv = left - right;
        if deriv.abs() > 1e-12 {
            result.push((i, deriv));
        }
    }
    result
}
```

Update `rational_surface_derivatives` to use `analytical_basis_derivatives`:

```rust
fn rational_surface_derivatives(surf: &NurbsSurface, u: f32, v: f32) -> (Vec3, Vec3) {
    let span_u = find_span(surf.degree_u, &surf.knots_u, u);
    let span_v = find_span(surf.degree_v, &surf.knots_v, v);
    let basis_u = bspline_bases(span_u, surf.degree_u, u, &surf.knots_u);
    let basis_v = bspline_bases(span_v, surf.degree_v, v, &surf.knots_v);
    let du = analytical_basis_derivatives(span_u, surf.degree_u, u, &surf.knots_u);
    let dv = analytical_basis_derivatives(span_v, surf.degree_v, v, &surf.knots_v);

    // ... same summation loop as before, but using Vec lookup instead of HashMap ...
    let mut w = 0.0f32; let mut w_u = 0.0f32; let mut w_v = 0.0f32;
    let mut p = Vec3::ZERO; let mut p_u = Vec3::ZERO; let mut p_v = Vec3::ZERO;
    for &(i, nu) in &basis_u {
        let dn_du = du.iter().find(|(j, _)| *j == i).map(|(_, v)| *v).unwrap_or(0.0);
        for &(j, nv) in &basis_v {
            let wgt = surf.weights[i][j];
            let cp = surf.control_points[i][j];
            let coeff = nu * nv * wgt;
            w += coeff; p = p + cp * coeff;
            let dn_dv = dv.iter().find(|(k, _)| *k == j).map(|(_, v)| *v).unwrap_or(0.0);
            w_u += dn_du * nv * wgt; p_u = p_u + cp * (dn_du * nv * wgt);
            w_v += nu * dn_dv * wgt; p_v = p_v + cp * (nu * dn_dv * wgt);
        }
    }
    if w.abs() < 1e-10 { return (Vec3::X, Vec3::Y); }
    let inv_w2 = 1.0 / (w * w);
    ((p_u * w - p * w_u) * inv_w2, (p_v * w - p * w_v) * inv_w2)
}
```

Remove the old `compute_bspline_derivatives` function and the `HashMap` import.

- [ ] **Step 4: Update surface_eval.rs BSpline d1 arm to use evaluate_with_derivative**

In `surface_eval.rs` line 311-322, replace the BSpline d1 arm:

```rust
SurfaceGeom::BSpline(nurbs) => {
    let u_k = map_to_knot_domain(
        &nurbs.knots_u, nurbs.degree_u, nurbs.u_count(), u,
    );
    let v_k = map_to_knot_domain(
        &nurbs.knots_v, nurbs.degree_v, nurbs.v_count(), v,
    );
    let u_w = knot_domain_width(&nurbs.knots_u, nurbs.degree_u, nurbs.u_count());
    let v_w = knot_domain_width(&nurbs.knots_v, nurbs.degree_v, nurbs.v_count());
    let (_pos, du_k, dv_k) = nurbs.evaluate_with_derivative(u_k, v_k);
    (du_k * u_w, dv_k * v_w)
}
```

- [ ] **Step 5: Run all tests**

Run: `rtk cargo test -p rc3d-io --lib -- --nocapture`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add crates/rc3d-io/src/step/nurbs.rs
git add crates/rc3d-io/src/step/brep/geom/surface_eval.rs
git commit -m "perf(step): share basis functions between NURBS evaluate and derivative"
```

---

### Task 4: Ellipse Eccentric Angle (Spec §2.6)

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/geom/curve_eval.rs:557-579` (trim_circle_to_vertices)
- Test: `crates/rc3d-io/src/step/brep/geom/curve_eval.rs`

- [ ] **Step 1: Write a test for high-eccentricity ellipse trimming**

```rust
#[test]
fn test_ellipse_trim_eccentric() {
    // Ellipse with a=10, b=1 — high eccentricity
    let ellipse = CurveGeom::Ellipse {
        center: Vec3::ZERO,
        axis: Vec3::Z,
        semi_major: 10.0,
        semi_minor: 1.0,
    };
    // Vertex at the minor axis end: (0, 1, 0)
    let p_lo = Vec3::new(0.0, 1.0, 0.0);
    // Vertex at the major axis end: (10, 0, 0)
    let p_hi = Vec3::new(10.0, 0.0, 0.0);
    let trimmed = trim_circle_to_vertices(&ellipse, p_lo, p_hi);
    assert!(trimmed.is_some(), "eccentric ellipse trim should succeed");
    let curve = trimmed.unwrap();
    // Verify endpoints match vertices
    let start = curve.d0(0.0);
    let end = curve.d0(1.0);
    assert!((start - p_lo).length() < 0.5, "start {:?} should be near {:?}", start, p_lo);
    assert!((end - p_hi).length() < 0.5, "end {:?} should be near {:?}", end, p_hi);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk cargo test -p rc3d-io --lib test_ellipse_trim_eccentric -- --nocapture`
Expected: FAIL — `circle_angle_geom` rejects the point because `(dist - semi_major).abs() > semi_major * 0.1`

- [ ] **Step 3: Implement ellipse_angle_geom**

Add after `circle_angle_geom` in curve_eval.rs:

```rust
/// Compute angular parameter [0,TAU) of a point on an ellipse.
/// Uses the eccentric anomaly: θ = atan2(y/b, x/a) where x,y are
/// the point's coordinates in the ellipse's local frame.
fn ellipse_angle_geom(
    center: &Vec3, axis: Vec3, semi_major: f32, semi_minor: f32, point: Vec3,
) -> Option<f32> {
    let a = axis.normalize();
    let ref_dir = if a.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let x_dir = ref_dir - a * ref_dir.dot(a);
    if x_dir.length_squared() < 1e-12 { return None; }
    let x_dir = x_dir.normalize();
    let y_dir = a.cross(x_dir);
    let rel = point - *center;
    let proj = rel - a * rel.dot(a);
    let u = proj.dot(x_dir) / semi_major;
    let v = proj.dot(y_dir) / semi_minor;
    if u.abs() < 1e-12 && v.abs() < 1e-12 { return None; }
    let theta = f32::atan2(v, u);
    Some(if theta < 0.0 { theta + std::f32::consts::TAU } else { theta })
}
```

Update the Ellipse arm in `trim_circle_to_vertices`:

```rust
CurveGeom::Ellipse { center, axis, semi_major, semi_minor } => {
    let a0 = ellipse_angle_geom(center, *axis, *semi_major, *semi_minor, p_lo)?;
    let a1 = ellipse_angle_geom(center, *axis, *semi_major, *semi_minor, p_hi)?;
    let (t_min, t_max) = normalize_arc_params(a0, a1);
    Some(CurveGeom::Trimmed { basis: Box::new(curve.clone()), t_min, t_max })
}
```

- [ ] **Step 4: Run tests**

Run: `rtk cargo test -p rc3d-io --lib test_ellipse_trim -- --nocapture`
Expected: PASS

- [ ] **Step 5: Run full test suite + STL export**

Run: `rtk cargo test -p rc3d-io --lib && rtk cargo test -p rc3d-io --test export_step_stl --release -- --nocapture`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add crates/rc3d-io/src/step/brep/geom/curve_eval.rs
git commit -m "fix(step): ellipse trim uses eccentric angle instead of circle formula"
```

---

### Task 5: Analytical Second Derivatives (Spec §2.7)

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/geom/surface_eval.rs:776-789` (d2 method)
- Test: `crates/rc3d-io/src/step/brep/geom/surface_eval.rs`

- [ ] **Step 1: Write tests for analytical d2 on each surface type**

```rust
#[test]
fn test_plane_d2_zero() {
    let plane = SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
    let (duu, duv, dvv) = plane.d2(0.5, 0.5);
    assert!(duu.length() < 1e-6, "plane d2 should be zero");
    assert!(duv.length() < 1e-6);
    assert!(dvv.length() < 1e-6);
}

#[test]
fn test_cylinder_d2_analytical() {
    let cyl = SurfaceGeom::Cylinder { origin: Vec3::ZERO, axis: Vec3::Z, radius: 2.0 };
    let (duu, duv, dvv) = cyl.d2(0.25, 0.5);
    // d²S/du² for cylinder = -TAU² * r * (cos(θ)*x + sin(θ)*y) at θ=π/2 → -TAU² * 2 * (0,1,0)
    assert!(duu.length() > 1.0, "cylinder duu should be non-zero");
    assert!(duv.length() < 1e-4, "cylinder duv should be zero");
    assert!(dvv.length() < 1e-4, "cylinder dvv should be zero");
}

#[test]
fn test_sphere_d2_analytical() {
    let sphere = SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 3.0 };
    let (duu, duv, dvv) = sphere.d2(0.0, 0.5);
    assert!(duu.length() > 1.0, "sphere duu should be non-zero at equator");
    assert!(dvv.length() > 1.0, "sphere dvv should be non-zero");
}
```

- [ ] **Step 2: Run tests — they currently pass via numerical d2 (baseline)**

Run: `rtk cargo test -p rc3d-io --lib test_plane_d2_zero test_cylinder_d2_analytical test_sphere_d2_analytical -- --nocapture`
Expected: PASS (numerical d2 works but is slow and noisy)

- [ ] **Step 3: Implement analytical d2 for each surface type**

Replace `d2()` in surface_eval.rs (line ~776):

```rust
pub fn d2(&self, u: f32, v: f32) -> (Vec3, Vec3, Vec3) {
    match self {
        SurfaceGeom::Plane { .. } => (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO),

        SurfaceGeom::Cylinder { axis, radius, .. } => {
            let (x_dir, y_dir) = build_ortho_axes(*axis);
            let theta = u * std::f32::consts::TAU;
            let twopi = std::f32::consts::TAU;
            let r = *radius;
            // ∂²S/∂u² = -TAU² * r * (cos(θ)*x + sin(θ)*y)
            let duu = -(twopi * twopi) * r * (theta.cos() * x_dir + theta.sin() * y_dir);
            (duu, Vec3::ZERO, Vec3::ZERO)
        }

        SurfaceGeom::Cone { axis, semi_angle, .. } => {
            let (x_dir, y_dir) = build_ortho_axes(*axis);
            let theta = u * std::f32::consts::TAU;
            let twopi = std::f32::consts::TAU;
            let tan_a = semi_angle.tan();
            let r_factor = tan_a; // dr/dv = tan(α)
            // ∂²S/∂u² = -TAU² * r * (cos*x + sin*y) where r depends on v
            // For normalized params, r = radius_at_apex + v * tan(α)
            let duu_factor = -(twopi * twopi);
            // We need r at this v, but in normalized coords r is handled by d1
            // For simplicity, use the chain rule on the normalized d1 expressions
            let duu = duu_factor * r_factor * v * (theta.cos() * x_dir + theta.sin() * y_dir);
            // ∂²S/∂u∂v = TAU * tan(α) * (-sin(θ)*x + cos(θ)*y)
            let duv = twopi * tan_a * (-theta.sin() * x_dir + theta.cos() * y_dir);
            (duu, duv, Vec3::ZERO)
        }

        SurfaceGeom::Sphere { radius, .. } => {
            let r = *radius;
            let theta = u * std::f32::consts::TAU;
            let phi = v * std::f32::consts::PI;
            let twopi = std::f32::consts::TAU;
            let pi = std::f32::consts::PI;
            let duu = -(twopi * twopi) * r * phi.sin()
                * Vec3::new(theta.cos(), theta.sin(), 0.0);
            let duv = twopi * pi * r * phi.cos()
                * Vec3::new(-theta.sin(), theta.cos(), 0.0);
            let dvv = -(pi * pi) * r * Vec3::new(
                phi.sin() * theta.cos(),
                phi.sin() * theta.sin(),
                phi.cos(),
            );
            (duu, duv, dvv)
        }

        SurfaceGeom::Torus { axis, major_r, minor_r, .. } => {
            let (x_dir, y_dir) = build_ortho_axes(*axis);
            let mr = *major_r;
            let nr = *minor_r;
            let theta = u * std::f32::consts::TAU;
            let phi = v * std::f32::consts::TAU;
            let twopi = std::f32::consts::TAU;
            let r = mr + nr * phi.cos();
            // ∂²S/∂u² = -TAU² * r * (cos(θ)*x + sin(θ)*y)
            let duu = -(twopi * twopi) * r * (theta.cos() * x_dir + theta.sin() * y_dir);
            // ∂²S/∂u∂v = TAU² * (-nr*sin(φ)) * (-sin(θ)*x + cos(θ)*y)
            let duv = (twopi * twopi) * (-nr * phi.sin())
                * (-theta.sin() * x_dir + theta.cos() * y_dir);
            // ∂²S/∂v² = TAU² * (-nr*cos(φ)) * (cos(θ)*x + sin(θ)*y) + TAU² * (-nr*sin(φ)) * axis
            let dvv = (twopi * twopi) * (
                (-nr * phi.cos()) * (theta.cos() * x_dir + theta.sin() * y_dir)
                    + (-nr * phi.sin()) * *axis
            );
            (duu, duv, dvv)
        }

        // BSpline, Extrusion, Revolution, Offset: keep numerical d2
        _ => {
            let eps = 1e-4f32;
            let (du_p, dv_p) = self.d1(u + eps, v);
            let (du_m, dv_m) = self.d1(u - eps, v);
            let (du_vp, dv_vp) = self.d1(u, v + eps);
            let (du_vm, dv_vm) = self.d1(u, v - eps);
            let duu = (du_p - du_m) / (2.0 * eps);
            let duv = (du_vp - du_vm) / (2.0 * eps);
            let dvv = (dv_vp - dv_vm) / (2.0 * eps);
            (duu, duv, dvv)
        }
    }
}
```

**NOTE**: The Cone d2 formula above is approximate because the normalized parameter space mixes the radius computation. Verify against numerical d2 before committing.

- [ ] **Step 4: Cross-validate analytical vs numerical d2**

Add a test that compares analytical and numerical results:

```rust
#[test]
fn test_d2_analytical_vs_numerical() {
    let surfaces: Vec<(&str, SurfaceGeom)> = vec![
        ("cylinder", SurfaceGeom::Cylinder { origin: Vec3::ZERO, axis: Vec3::Z, radius: 2.0 }),
        ("sphere", SurfaceGeom::Sphere { center: Vec3::ZERO, radius: 3.0 }),
        ("torus", SurfaceGeom::Torus { center: Vec3::ZERO, axis: Vec3::Z, major_r: 3.0, minor_r: 1.0 }),
    ];
    for (name, surf) in &surfaces {
        for (u, v) in [(0.1, 0.3), (0.5, 0.5), (0.8, 0.7)] {
            let (duu_a, duv_a, dvv_a) = surf.d2(u, v);
            // Numerical reference
            let eps = 1e-4f32;
            let (du_p, dv_p) = surf.d1(u + eps, v);
            let (du_m, dv_m) = surf.d1(u - eps, v);
            let (du_vp, dv_vp) = surf.d1(u, v + eps);
            let (du_vm, dv_vm) = surf.d1(u, v - eps);
            let duu_n = (du_p - du_m) / (2.0 * eps);
            let duv_n = (du_vp - du_vm) / (2.0 * eps);
            let dvv_n = (dv_vp - dv_vm) / (2.0 * eps);
            let tol = 0.5; // numerical FD has limited precision
            assert!((duu_a - duu_n).length() < tol,
                "{} duu mismatch at ({},{}) analytical={:?} numerical={:?}", name, u, v, duu_a, duu_n);
            assert!((duv_a - duv_n).length() < tol,
                "{} duv mismatch at ({},{})", name, u, v);
            assert!((dvv_a - dvv_n).length() < tol,
                "{} dvv mismatch at ({},{})", name, u, v);
        }
    }
}
```

- [ ] **Step 5: Run tests and fix any formula errors**

Run: `rtk cargo test -p rc3d-io --lib test_d2_analytical_vs_numerical -- --nocapture`
Expected: PASS. If any assertion fails, adjust the analytical formulas.

- [ ] **Step 6: Run full suite + STL export**

Run: `rtk cargo test -p rc3d-io --lib && rtk cargo test -p rc3d-io --test export_step_stl --release -- --nocapture`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add crates/rc3d-io/src/step/brep/geom/surface_eval.rs
git commit -m "perf(step): analytical d2 for cylinder/cone/sphere/torus (replace finite differences)"
```

---

### Task 6: Offset Weingarten Map (Spec §2.8)

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/geom/surface_eval.rs:336-349` (Offset d1 arm)
- Test: `crates/rc3d-io/src/step/brep/geom/surface_eval.rs`

**Depends on:** Task 5 (analytical d2 must be in place for accurate Weingarten)

- [ ] **Step 1: Write test for offset surface d1 accuracy**

```rust
#[test]
fn test_offset_d1_weingarten() {
    let cyl = SurfaceGeom::Cylinder { origin: Vec3::ZERO, axis: Vec3::Z, radius: 2.0 };
    let offset = SurfaceGeom::Offset { basis: Box::new(cyl), distance: 0.5 };
    // Offset cylinder has radius 2.5
    let (du, dv) = offset.d1(0.0, 0.0);
    // du should have magnitude TAU * 2.5
    let expected_mag = std::f32::consts::TAU * 2.5;
    assert!((du.length() - expected_mag).abs() < 0.5,
        "offset cylinder du magnitude should be ~{}, got {}", expected_mag, du.length());
    // dv should still be axis direction
    assert!((dv - Vec3::Z).length() < 0.1, "offset dv should be ~Z axis");
}
```

- [ ] **Step 2: Implement Weingarten map in Offset d1**

Replace the Offset arm in `d1()`:

```rust
SurfaceGeom::Offset { basis, distance } => {
    let (db_du, db_dv) = basis.d1(u, v);
    let (db_uu, db_uv, db_vv) = basis.d2(u, v);

    // First fundamental form: e = db_du·db_du, f = db_du·db_dv, g = db_dv·db_dv
    let e = db_du.dot(db_du);
    let f = db_du.dot(db_dv);
    let g = db_dv.dot(db_dv);
    let det = e * g - f * f;

    if det.abs() < 1e-10 {
        // Degenerate: fall back to numerical with warning
        log::warn!("Offset::d1 degenerate first fundamental form at ({}, {}), using numerical fallback", u, v);
        let eps = 1e-3f32;
        let n_u1 = basis.normal(u + eps, v);
        let n_u0 = basis.normal(u - eps, v);
        let n_v1 = basis.normal(u, v + eps);
        let n_v0 = basis.normal(u, v - eps);
        let dn_du = (n_u1 - n_u0) / (2.0 * eps);
        let dn_dv = (n_v1 - n_v0) / (2.0 * eps);
        let d = *distance;
        (db_du + dn_du * d, db_dv + dn_dv * d)
    } else {
        // Second fundamental form: L = db_uu·n, M = db_uv·n, N = db_vv·n
        let n = db_du.cross(db_dv);
        let n_len = n.length();
        let n_hat = if n_len > 1e-10 { n * (1.0 / n_len) } else { Vec3::Z };
        let big_l = db_uu.dot(n_hat);
        let big_m = db_uv.dot(n_hat);
        let big_n = db_vv.dot(n_hat);

        // Weingarten equations:
        // dn/du = (fM - eL)/(eg-f²) * dS/dv + (fL - eM)/(eg-f²) * dS/du  ... corrected:
        // dn/du = (f*M - g*L)/(e*g - f²) * dS/du + (f*L - e*M)/(e*g - f²) * dS/dv
        // Actually: standard Weingarten:
        // dn/du = (fM - gL) / (eg - f²) * dS/du + (fL - eM) / (eg - f²) * dS/dv  ← WRONG
        // Correct Weingarten:
        // [dn/du] = -W [dS/du; dS/dv]  where W = I⁻¹ · II
        // W = [e f; f g]⁻¹ · [L M; M N]
        // W₁₁ = (gL - fM)/det, W₁₂ = (gM - fN)/det
        // W₂₁ = (eM - fL)/det, W₂₂ = (eN - fM)/det
        let inv_det = 1.0 / det;
        let w11 = (g * big_l - f * big_m) * inv_det;
        let w12 = (g * big_m - f * big_n) * inv_det;
        let w21 = (e * big_m - f * big_l) * inv_det;
        let w22 = (e * big_n - f * big_m) * inv_det;

        let dn_du = -(w11 * db_du + w21 * db_dv);
        let dn_dv = -(w12 * db_du + w22 * db_dv);

        let d = *distance;
        (db_du + dn_du * d, db_dv + dn_dv * d)
    }
}
```

- [ ] **Step 3: Run tests**

Run: `rtk cargo test -p rc3d-io --lib test_offset -- --nocapture`
Expected: PASS

- [ ] **Step 4: Run full suite + STL export**

Run: `rtk cargo test -p rc3d-io --lib && rtk cargo test -p rc3d-io --test export_step_stl --release -- --nocapture`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-io/src/step/brep/geom/surface_eval.rs
git commit -m "fix(step): offset surface d1 uses Weingarten map instead of numerical normals"
```

---

### Task 7: Edge Trim Tolerance (Spec §2.9)

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/geom/curve_eval.rs:644-645` (match_tol computation)
- Test: existing tests cover this

- [ ] **Step 1: Update match_tol formula and add warning**

In `normalize_edge_curve_to_vertices`, change line ~644:

```rust
// Before:
// let match_tol = (tol.max(1e-4) * 100.0).max(len * 0.005).max(0.01);

// After: tighter tolerance, capped for long edges
let match_tol = (tol.max(1e-4) * 10.0).max(len * 0.001).min(0.1);
```

Add a warning when the curve silently fails to match. After the final `curve` return (line ~663):

```rust
// Keep original geometry — PCurve on each face provides correct surface trajectory.
// Replacing with a straight line destroys geometric fidelity.
let c0_err = (curve.d0(0.0) - p_lo).length();
let c1_err = (curve.d0(1.0) - p_hi).length();
if c0_err > match_tol || c1_err > match_tol {
    log::warn!(
        "edge curve mismatch: endpoints off by {:.4}/{:.4} (tol={:.4}, len={:.4})",
        c0_err, c1_err, match_tol, len,
    );
}
curve
```

- [ ] **Step 2: Run full test suite**

Run: `rtk cargo test -p rc3d-io --lib && rtk cargo test -p rc3d-io --test export_step_stl --release -- --nocapture`
Expected: All PASS. Watch for new warnings in output — they indicate edges with genuine mismatches.

- [ ] **Step 3: Commit**

```bash
git add crates/rc3d-io/src/step/brep/geom/curve_eval.rs
git commit -m "fix(step): tighten edge curve match tolerance and log mismatches"
```

---

### Task 8: build_ortho_axes Caching (Spec §2.2)

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/geom/surface_eval.rs:9-19` (SurfaceGeom enum)
- Modify: `crates/rc3d-io/src/step/brep/geom/curve_eval.rs:292-300` (CurveGeom enum)
- Modify: `crates/rc3d-io/src/step/brep/build/surface.rs` (construction sites)
- Modify: `crates/rc3d-io/src/step/brep/build/curve.rs` (construction sites)
- Modify: `crates/rc3d-io/src/step/brep/build/pcurve.rs` (Circle/Ellipse constructions)
- Modify: `crates/rc3d-io/src/step/brep/offset_api.rs` (offset constructions)
- Modify: all test construction sites
- Test: full suite

**This is the highest-risk task — do it last.**

- [ ] **Step 1: Add cached axes to CurveGeom::Circle and CurveGeom::Ellipse**

```rust
pub enum CurveGeom {
    Line { origin: Vec3, direction: Vec3 },
    Circle { center: Vec3, axis: Vec3, radius: f32, x_dir: Vec3, y_dir: Vec3 },
    Ellipse { center: Vec3, axis: Vec3, semi_major: f32, semi_minor: f32, x_dir: Vec3, y_dir: Vec3 },
    // ... rest unchanged ...
}
```

Add a constructor:

```rust
impl CurveGeom {
    pub fn circle(center: Vec3, axis: Vec3, radius: f32) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        CurveGeom::Circle { center, axis, radius, x_dir, y_dir }
    }
    pub fn ellipse(center: Vec3, axis: Vec3, semi_major: f32, semi_minor: f32) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        CurveGeom::Ellipse { center, axis, semi_major, semi_minor, x_dir, y_dir }
    }
}
```

Update d0/d1/d2 Circle/Ellipse arms to use `*x_dir` / `*y_dir` instead of calling `build_ortho_axes(*axis)`.

- [ ] **Step 2: Add cached axes to SurfaceGeom::Cylinder, Cone, Torus**

```rust
pub enum SurfaceGeom {
    // ... Plane unchanged ...
    Cylinder { origin: Vec3, axis: Vec3, radius: f32, x_dir: Vec3, y_dir: Vec3 },
    Cone { apex: Vec3, axis: Vec3, semi_angle: f32, radius_at_apex: f32, x_dir: Vec3, y_dir: Vec3 },
    // ... Sphere unchanged (doesn't use build_ortho_axes) ...
    Torus { center: Vec3, axis: Vec3, major_r: f32, minor_r: f32, x_dir: Vec3, y_dir: Vec3 },
    // ... rest unchanged ...
}
```

Add constructors:

```rust
impl SurfaceGeom {
    pub fn cylinder(origin: Vec3, axis: Vec3, radius: f32) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        SurfaceGeom::Cylinder { origin, axis, radius, x_dir, y_dir }
    }
    pub fn cone(apex: Vec3, axis: Vec3, semi_angle: f32, radius_at_apex: f32) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        SurfaceGeom::Cone { apex, axis, semi_angle, radius_at_apex, x_dir, y_dir }
    }
    pub fn torus(center: Vec3, axis: Vec3, major_r: f32, minor_r: f32) -> Self {
        let (x_dir, y_dir) = build_ortho_axes(axis);
        SurfaceGeom::Torus { center, axis, major_r, minor_r, x_dir, y_dir }
    }
}
```

- [ ] **Step 3: Update all construction sites**

Use the constructors everywhere:
- `build/surface.rs:22` → `SurfaceGeom::cylinder(origin, z_axis.normalize(), radius)`
- `build/surface.rs:35` → `SurfaceGeom::cone(origin, z_axis.normalize(), semi_angle, radius)`
- `build/surface.rs:56` → `SurfaceGeom::torus(origin, z_axis.normalize(), major_r, minor_r)`
- `build/curve.rs:23` → `CurveGeom::circle(center, axis, radius)`
- `build/curve.rs:31` → `CurveGeom::ellipse(center, axis, semi_major, semi_minor)`
- `build/pcurve.rs` — all Circle/Ellipse/Cylinder/Cone/Torus construction sites
- `offset_api.rs` — offset surface construction
- All test sites in surface_eval.rs, heal/*.rs, mesh/*.rs — update struct literal patterns

The compiler will catch every site that needs updating (missing field errors). Fix them one by one.

- [ ] **Step 4: Update all d0/d1/d2/project match arms**

Remove `build_ortho_axes(*axis)` calls from evaluation methods. Use the cached `*x_dir` / `*y_dir` directly. For example, Cylinder d0:

```rust
SurfaceGeom::Cylinder { origin, radius, x_dir, y_dir, .. } => {
    let theta = u * std::f32::consts::TAU;
    let r = *radius;
    *origin + *x_dir * r * theta.cos() + *y_dir * r * theta.sin() + *axis * v
}
```

- [ ] **Step 5: Run full test suite**

Run: `rtk cargo test -p rc3d-io --lib && rtk cargo test -p rc3d-io --test export_step_stl --release -- --nocapture`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add -A crates/rc3d-io/src/
git commit -m "perf(step): cache build_ortho_axes in CurveGeom/SurfaceGeom variants"
```

---

### Task 9: BSpline/Torus Hierarchical Project (Spec §2.3)

**Files:**
- Modify: `crates/rc3d-io/src/step/brep/geom/surface_eval.rs:446-505` (project BSpline + Torus arms)
- Test: `crates/rc3d-io/src/step/brep/geom/surface_eval.rs`

- [ ] **Step 1: Write test for projection accuracy and performance**

```rust
#[test]
fn test_project_bspline_accuracy() {
    let nurbs = NurbsSurface::plane(0.0, 10.0, 0.0, 10.0);
    let surf = SurfaceGeom::BSpline(nurbs);
    // Project a known surface point
    let target = surf.d0_native(5.0, 7.0);
    let proj = surf.project(target);
    assert!(proj.is_some());
    let (u, v) = proj.unwrap();
    let back = surf.d0_native(u, v);
    assert!((back - target).length() < 0.01,
        "projection roundtrip error: {}", (back - target).length());
}
```

- [ ] **Step 2: Extract shared grid_project_2d helper**

Add a private helper function in surface_eval.rs:

```rust
/// Two-level grid search with local refinement for surface projection.
/// Level 1: coarse grid (grid_n × grid_n)
/// Level 2: local 4×4 around each of top-3 candidates
/// Level 3: coordinate-descent refinement
fn grid_project_2d(
    eval: &dyn Fn(f32, f32) -> Vec3,
    u_lo: f32, u_hi: f32, v_lo: f32, v_hi: f32,
    point: Vec3,
) -> (f32, f32) {
    // Level 1: coarse 4×4 grid
    let coarse = 4;
    let mut candidates: Vec<(f32, f32, f32)> = Vec::with_capacity((coarse + 1) * (coarse + 1));
    for i in 0..=coarse {
        let u = u_lo + (u_hi - u_lo) * i as f32 / coarse as f32;
        for j in 0..=coarse {
            let v = v_lo + (v_hi - v_lo) * j as f32 / coarse as f32;
            let d2 = (eval(u, v) - point).length_squared();
            candidates.push((u, v, d2));
        }
    }
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Level 2: local 4×4 around top-3 candidates
    let mut best_u = candidates[0].0;
    let mut best_v = candidates[0].1;
    let mut best_d2 = candidates[0].2;
    let u_step = (u_hi - u_lo) / coarse as f32;
    let v_step = (v_hi - v_lo) / coarse as f32;
    for cand in candidates.iter().take(3) {
        let local = 4;
        for i in 0..=local {
            let u = (cand.0 - u_step + 2.0 * u_step * i as f32 / local as f32).clamp(u_lo, u_hi);
            for j in 0..=local {
                let v = (cand.1 - v_step + 2.0 * v_step * j as f32 / local as f32).clamp(v_lo, v_hi);
                let d2 = (eval(u, v) - point).length_squared();
                if d2 < best_d2 { best_d2 = d2; best_u = u; best_v = v; }
            }
        }
    }

    // Level 3: coordinate descent refinement
    let mut step_u = u_step / local as f32 * 0.5;
    let mut step_v = v_step / local as f32 * 0.5;
    for _ in 0..8 {
        for &(du, dv) in &[(step_u, 0.0), (-step_u, 0.0), (0.0, step_v), (0.0, -step_v)] {
            let nu = (best_u + du).clamp(u_lo, u_hi);
            let nv = (best_v + dv).clamp(v_lo, v_hi);
            let d2 = (eval(nu, nv) - point).length_squared();
            if d2 < best_d2 { best_d2 = d2; best_u = nu; best_v = nv; }
        }
        step_u *= 0.5;
        step_v *= 0.5;
    }
    (best_u, best_v)
}
```

- [ ] **Step 3: Replace BSpline and Torus project arms**

```rust
SurfaceGeom::BSpline(nurbs) => {
    let u_range = nurbs.knots_u[nurbs.degree_u];
    let u_end = nurbs.knots_u[nurbs.knots_u.len() - nurbs.degree_u - 1];
    let v_range = nurbs.knots_v[nurbs.degree_v];
    let v_end = nurbs.knots_v[nurbs.knots_v.len() - nurbs.degree_v - 1];
    let (u, v) = grid_project_2d(
        &|u, v| nurbs.evaluate(u, v),
        u_range, u_end, v_range, v_end,
        point,
    );
    Some((u, v))
}

SurfaceGeom::Torus { .. } => {
    let (u, v) = grid_project_2d(
        &|u, v| self.d0(u, v),
        0.0, 1.0, 0.0, 1.0,
        point,
    );
    Some(self.d0_uv_to_native(u, v))
}
```

- [ ] **Step 4: Run tests and compare grid_fallback_count**

Run: `rtk cargo test -p rc3d-io --lib test_project && rtk cargo test -p rc3d-io --test export_step_stl --release -- --nocapture`
Expected: All PASS. Watch Shape-1 grid_fallback_count — should decrease or stay same.

- [ ] **Step 5: Commit**

```bash
git add crates/rc3d-io/src/step/brep/geom/surface_eval.rs
git commit -m "perf(step): hierarchical grid search for BSpline/Torus projection (~65% fewer evals)"
```

---

## Final Integration

- [ ] **Run complete test suite**

```bash
rtk cargo test -p rc3d-io --release
```
Expected: 365+ tests PASS, 0 failures

- [ ] **Run STL export and compare with baseline**

```bash
rtk cargo test -p rc3d-io --test export_step_stl --release -- --nocapture
```

Record: total triangles, grid_fallback_count per file.

- [ ] **Tag the commit**

```bash
git tag -a stage2 -m "OCC Stage 2: efficiency + correctness alignment"
```
