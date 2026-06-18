# Deep Verification Report — OCCT 7.8.0 Alignment

Date: 2026-06-18. 5-system deep code review against OCCT 7.8.0 source.

## Summary

29 modules reviewed. 3 ISSUE (bugs affecting correctness), 10 MINOR (edge cases/cosmetic), 16 PASS.

## ISSUE findings (recommend fixing)

### 1. Boolean processing order (`pave_filler.rs`)
- FF runs 2nd in the pipeline; OCCT runs it LAST after VE/EE/EF/VF
- Missing `UpdatePaveBlocksWithSDVertices()` between phases
- Impact: SD vertices found in VE/EE/VF don't propagate to FF pave blocks
- Fix: move FF to after VF; add SD update after each phase

### 2. FixNotchedEdges semantics (`wire_ops.rs`)
- OCCT: detects a notch WITHIN a single edge, splits it
- rustcoin3d: detects angle between TWO consecutive edges, MERGES them into a straight line
- Impact: merges replace original curved geometry with straight Line, losing curvature
- Fix: redesign as edge-splitting operation rather than edge-merging

### 3. BSpline curve attribute ordering (`parametric.rs`)
- STEP B-spline curve attributes swapped: `control_points` and `curve_form` positions reversed
- Extra spurious field `.UNSPECIFIED.` inserted
- Impact: generated STEP files fail validation; BSpline curves cannot roundtrip
- Fix: reorder `emit_bspline_curve` attributes to match ISO 10303-42

## MINOR findings (edge cases, cosmetic)

| Module | Finding | Severity |
|--------|---------|----------|
| `canonical.rs` | Sphere detection uses centroid-distance; fails for partial patches (OCCT uses LS+PSO) | Low |
| `coplanar.rs` | `polygon_union` returns unmerged polygons; geometry duplication | Medium |
| `model_preprocessor.rs` | Missing angle/parallel/loop-area robustness filters from OCCT | Low |
| `continuity.rs` | Misnamed (detects C1 not C0); uniform sampling less precise than knot analysis | Low |
| `properties.rs` | Naming misleading (midpoint not actual GK); different convergence criterion | Low |
| `curve2d.rs` | Weights ignored for NURBS Bezier extraction; degree>3 CPs discarded | Medium |
| `edge_disc.rs` | No ratio tolerance for reuse (OCCT uses 10%); exact comparison causes extra work | Low |
| `iges_writer.rs` | Form numbers hardcoded to 0; Cylinder generatrix DE=0 | Low |
| `bopds.rs` | FaceInfo simplified (no In/On/Sc separation); PaveBlocks point-based | Low |
| `unify_same_domain.rs` | Transitive-only neighbors may skip merge in cluster | Low |

## PASS findings (verified correct)

| Category | Modules | Count |
|----------|---------|-------|
| Boolean | builder_face, builder_solid, intersect (ellipse formula, SSI) | 3 |
| Heal | unify_same_domain, solid_fix, shell_fix, face_fix | 4 |
| Mesh | post_process | 1 |
| I/O | iges, vrml, brep_binary | 3 |
| Geom | curve_eval (NaN handling) | 1 |
| Other | bopds (partial), canonical (plane/cylinder) | 2 |

## Fix priority

| # | Module | Effort | Impact |
|---|--------|--------|--------|
| 1 | BSpline curve attribute order | 30min | STEP roundtrip correctness |
| 2 | PaveFiller processing order | 2h | Boolean edge-case correctness |
| 3 | FixNotchedEdges redesign | 3h | Wire healing correctness |
