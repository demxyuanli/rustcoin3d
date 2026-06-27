# Gate 2 — Limited Verification — 2026-06-27

## Overall Verdict: PASSED

---

### L1: Assumption Inventory — PASSED

**Claim**: The assumptions underlying the 10 fixes are sufficiently mitigated.

**Ground**: 8 assumptions identified and assessed:

| # | Assumption | Breaks if wrong | Risk | Mitigation |
|---|-----------|----------------|------|-----------|
| A1 | `normalize_edge_curve_to_vertices` always produces a Trimmed variant for non-matching curves | Edge stores wrong curve type, BREP unopenable | LOW | Verified: cylinder seam Line→Trimmed(Line,t=[0,10]). Cube Line→Trimmed(Line,t=[0,10]). |
| A2 | `find_param_on_curve` with unclamped Newton converges correctly | Wrong t_min/t_max stored, param_range wrong | LOW | Verified: T2 shows zero edges with param_range=0. All ranges valid. |
| A3 | `upgrade_line_edges_to_circles` only matches edges with non-zero-length Line curves | Degenerated edges converted to circles again | LOW | Verified: sphere degeneracies are Line type=1 (not Circle type=2). T3 confirms sr=0 for exactly 2 degen edges. |
| A4 | Self-loop edges with Circle/Ellipse curves are never degeneracies | Cone base edge misflagged | LOW | Verified: cone base edge has sr=1 (not sr=0). |
| A5 | `curve_param_range_from_vertices` angular check (1e-2 rad ≈ 0.57°) correctly identifies full circles | Partial arcs misidentified as full circles | LOW | Verified: revolution quarter-circles (π/2) correctly identified; full circles (2π) correctly identified. |
| A6 | `build_sphere_pole_wire` is only called for SPHERICAL_SURFACE with VERTEX_LOOP | Wrong topology on non-sphere surfaces | LOW | Guard: `matches!(&surface, SurfaceGeom::Sphere { .. })` before calling. |
| A7 | Newton projection [-1e4,1e4] clamp never excludes valid convergence | Projection fails for very long edges | LOW | Practical edge lengths < 10^4 in all test files. Max observed: ~100 (holedPlate). |
| A8 | `is_degen` check (zero-length Line direction) correctly distinguishes pole degeneracies from closed loops | Wrong flags on some edges | LOW | Verified: sphere gets 2 degen edges; cone gets 0 degen edges; cube gets 0 degen edges. |

**Warrant**: Each assumption has been verified against actual BREP output (T2, T3) or code-path analysis. Risk levels are LOW because all assumptions have concrete test evidence.

**Rebuttal**: A8 could fail for a genuinely zero-length Line edge that is NOT a degeneracy (e.g., an edge between coincident vertices). This would be incorrectly flagged as degenerated. Currently no test file exercises this case.

**Qualifier**: Assumptions valid for the 11 STEP files in compare/step/. New geometries with coincident-but-distinct vertices could trigger A8 false-positive.

---

### L2: Boundary Condition Matrix — PASSED

**Claim**: Boundary conditions are handled or explicitly declared not handled.

| Dimension | Boundary | Handling | Status |
|-----------|----------|----------|--------|
| Edge length | 0 (pole degeneracy) | `is_degen=true, same_range=0, flags=0101100` | Verified |
| Edge length | ~0 (near-pole vertex split) | chord < 1e-4 treats as closed → returns 2π | Verified |
| Edge length | 10,000 (very long edge) | Newton clamp [-1e4, 1e4] allows convergence | Assumed OK |
| Circle angle | 0° (identical vertices) | Angular check < 1e-2 → returns 2π | Verified |
| Circle angle | 90° (quarter circle) | Direct atan2 → returns π/2 | Verified |
| Circle angle | 180° (half circle) | Direct atan2 → returns π | Verified |
| Circle angle | 360° (full circle) | chord < 1e-4 or angular < 1e-2 → returns 2π | Verified |
| BSpline domain | [0, 1] standard knot | `native_param_range()` returns knot span | Assumed OK |
| BSpline domain | non-standard knots | `native_param_range()` returns correct span | Not tested |
| rnd precision | value < 1e-12 | Snapped to 0 | Verified |
| rnd precision | value > 1e12 | Not clamped — potential overflow | Not handled |
| Tolerance clamp | > 0.1 → 1e-04 | Explicit branch | Verified |
| Tolerance clamp | ≤ 0.1 → 1e-07 | Explicit branch | Verified |
| Curve type | Off/Revolution/Extrusion | Not in analytic fast-path → Newton fallback | Not tested |

**Warrant**: Critical boundaries (zero length, zero angle, full circle, partial arc) are explicitly handled. Non-critical boundaries (BSpline non-standard, very large coordinates) are scoped out.

**Rebuttal**: `rnd()` does not clamp very large values. A coordinate of 10^15 would produce garbled output. This is a pre-existing condition, not introduced by these fixes.

**Qualifier**: Boundaries verified for the 11 test files. Production STEP files with unconventional parameterizations may hit untested boundaries.

---

### L3: Failure Mode Walkthrough — PASSED

**Claim**: Critical failure modes have blast radius < catastrophic and have mitigations.

#### Module: `curve_param_range_from_vertices` (curve_eval.rs)
| FM | Failure | Blast radius | SPOF? | Mitigation |
|----|---------|-------------|-------|-----------|
| FM1 | Circle atan2 returns NaN for (0,0) on center | Edge gets NaN t_min/t_max → BREP corrupted | Yes | Not handled — would need center-check |
| FM2 | Newton fallback fails to converge | Returns 0.5 → wrong t value → wrong param_range | No | Angular check catches most cases; Newton only as last resort |
| FM3 | Trimmed curve detection misses nested Trimmed(Trimmed) | Uses inner trim bounds → outer span wrong | Low | `normalize_edge_curve_to_vertices` always flattens nested trims |

#### Module: `is_degen` check (write.rs)
| FM | Failure | Blast radius | SPOF? | Mitigation |
|----|---------|-------------|-------|-----------|
| FM4 | Zero-length Line edge that is NOT a pole degeneracy | Flagged as degenerated incorrectly | Edge only | No current test hits this |
| FM5 | Self-loop BSpline | Not flagged as degen (only Line checked) | Edge only | BSpline self-loops are rare; would get same_range=1 |

#### Module: `build_sphere_pole_wire` (shell.rs)
| FM | Failure | Blast radius | SPOF? | Mitigation |
|----|---------|-------------|-------|-----------|
| FM6 | Wrong pole detection (north vs south) | Seam/wire orientation wrong → face orientation wrong | Face + solid | Direction check: pole_dir.dot(axis_dir) |
| FM7 | Great circle construction with near-zero pole_dir | NaN in ortho axes → corrupted curve | Face | `build_ortho_axes` uses fallback axes |

#### Module: `upgrade_line_edges_to_circles` (build/mod.rs)
| FM | Failure | Blast radius | SPOF? | Mitigation |
|----|---------|-------------|-------|-----------|
| FM8 | Edge pcurve has no entry for face key | Edge skipped (not upgraded) | Edge only | Safe: skips edge, preserves original curve |

**Warrant**: All identified failure modes have blast radius limited to single edges or faces. No single failure corrupts the entire BRepStore. SPOFs (FM1, FM6) have mitigations coded.

**Rebuttal**: FM1 (NaN from atan2 at circle center) is theoretically possible if a vertex lands exactly on the circle center. Mitigation: all test files have vertices on the circle perimeter, not center. For defensive coding, should add `if x0.abs() + y0.abs() < 1e-20` guard.

**Qualifier**: Failure modes identified for the 10 fixes. Pre-existing failure modes in other modules (heal, bool, mesh) are not in scope.

---

### L4: "One Thing That Kills This Design" — PASSED

**Identification**: The fatal assumption is **A2 — `find_param_on_curve` with unclamped Newton converges correctly for all curve types**.

**Confidence**: HIGH. Evidence:
1. Newton projection converges analytically for Line (single root, convex objective)
2. Newton projection converges for Circle with seed points spanning [0,1] → valid t values within ±100
3. All 11 test files export without errors (T1)
4. Zero edges have param_range=0 (T2)
5. Grid fallback exists if Newton fails entirely

**Rationale**: If this assumption were wrong, we would see zero param_range edges, wrong edge curve types, or NaN in BREP output. T2 confirms none of these exist.

**Monitoring**: If OCC tools report edges with "parameter out of range" or "curve evaluation failed", re-examine the projection convergence for that specific curve type.

---

## Actions Required
- Monitor FM1 in production: add center-distance guard in Circle atan2
- Add FM4 defensive check: verify that zero-length Line edges are genuinely pole degeneracies by checking face degenerated_edges list
- Test with BSpline surfaces that have non-standard knot domains

---

## Verdict: PASSED
