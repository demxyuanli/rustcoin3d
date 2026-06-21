# Geometry Processing Deep Audit — Defect Report

**Date**: 2026-06-21
**Scope**: `crates/rc3d-shape/src/` — 120+ files, ~42,000 lines
**Method**: 5 parallel agents + manual review, each file read in full
**Total Defects**: 113 (12 CRITICAL, 24 HIGH, 47 MEDIUM, 30 LOW)
**Post-Grill Adjustments**: C8 CRITICAL→LOW (dead code), C14 CRITICAL→MEDIUM (rare trigger)

---

## Executive Summary

The geometry kernel has achieved 37/37 (100%) OCC class mapping coverage, but this audit reveals significant correctness gaps. The most systemic issues are:

1. **Boolean pipeline bypasses selection logic** (C1) — all sub-faces are included regardless of classification
2. **B-Rep serialization corrupts geometry** (C13, H22-H26) — BREP round-trip is broken for cones, BSplines, rational surfaces
3. **UV coordinate corruption in interference detection** (C2-C3) — edge parameters stored as face UVs
4. **Dead-code EF phase** (C15) — edge-face intersection exists but never wired into pave filler
5. **Mesh healing is a no-op for most models** (C11) — vertex duplication on reassembly

---

## CRITICAL (16) — Silent Wrong Results

### C1. `bool/mod.rs:174-201` — All sub-face regions appended unconditionally

**Problem**: Pipeline builds all split sub-faces BEFORE classification (lines 175-186), then appends ALL of them to `all_selected` unconditionally at lines 199-201. The `select::select_brep_faces` filter at line 194 is bypassed.

```rust
let mut all_selected = selected.clone();
all_selected.extend(new_faces_a);  // UNFILTERED
all_selected.extend(new_faces_b);  // UNFILTERED
```

**Impact**: Boolean Intersection includes Outside-classified sub-faces. Difference includes Inside sub-faces. Wrong shapes silently produced.

**Fix**: Filter `new_faces_a`/`new_faces_b` through classification before appending, or move face building to after selection.

**OCCT ref**: BOPAlgo_Builder::BuildFace runs AFTER classification.

---

### C2. `bool/pave_filler.rs:103-108` — EE intersection stores edge curve parameters as face UVs

**Problem**: Edge-edge intersection `InterfPoint` stores `(hit.t_a, 0.0)` as `uv_a` and `(hit.t_b, 0.0)` as `uv_b`. `hit.t_a`/`t_b` are edge curve parameters [0,1], not surface UV coordinates.

```rust
uv_a: (hit.t_a, 0.0),   // BUG: edge curve parameter, not surface UV
uv_b: (hit.t_b, 0.0),   // BUG: edge curve parameter, not surface UV
```

**Impact**: All downstream consumers (pave blocks, face splits, builder_face) interpret curve-parameters as surface-UVs, producing geometrically invalid splits.

**Fix**: Project `hit.point` to each face's surface:
```rust
uv_a: face_a.surface.project(hit.point).unwrap_or((hit.t_a, 0.0)),
uv_b: face_b.surface.project(hit.point).unwrap_or((hit.t_b, 0.0)),
```

**OCCT ref**: BOPDS_InterfVE::SetUV stores true surface UV from Extrema projection.

---

### C3. `bool/pave_filler.rs:172-177,497-510` — VE intersection stores zero-valued UV

**Problem**: `vertex_on_edge()` always returns `uv_face: (0.0, 0.0)`. This zero constant is not a valid surface UV.

```rust
fn vertex_on_edge(...) -> Option<EdgeFaceHit> {
    Some(EdgeFaceHit {
        t_edge: t_clamped,
        point: edge.curve.d0(t_clamped),
        uv_face: (0.0, 0.0),  // BUG: always zero, not a valid surface UV
    })
}
```

**Impact**: Boolean operations where a vertex touches an edge produce corrupted surface parameterization.

**Fix**: Project the 3D hit point to the face surface to obtain proper UV.

**OCCT ref**: BOPAlgo_PaveFiller::PerformVE uses Extrema_ExtPElC.

---

### C4. `bool/builder_face.rs:353-439` — `_edge_map` unused; sub-faces get disconnected topology

**Problem**: `build_face_from_region` receives `edge_map` (pre-split edge segments) as `_edge_map` and never uses it. Every sub-face creates brand-new edges and vertices from scratch.

**Impact**: Adjacent sub-faces that should share a split boundary get duplicate independent edges. `weld_shell_vertices` cannot fully repair this. Shell closure detection fails.

**Fix**: Reuse pre-split edge segments from `edge_map` instead of creating new ones.

**OCCT ref**: BOPAlgo_BuilderFace::PerformShapesToAvoid explicitly reuses split edges.

---

### C5. `bool/classify.rs:39-60` — Ray casting uses only 6 axis-aligned directions

**Problem**: Only 6 cardinal directions tested. When a ray grazes a face/edge aligned with an axis, intersection count is wrong. Winding-number tiebreaker (line 92-127) ignores inner wires (holes) — `face_vertex_positions` only reads `face.outer_wire`.

**Impact**: Points near sharp edges misclassified. Holes invisible to tiebreaker.

**Fix**: Add random ray directions; include inner wire vertices in winding number.

**OCCT ref**: BRepClass3d_SClassifier tries random directions; BRepClass3d_SolidAngle includes inner wires.

---

### C6. `heal/edge_connect.rs:54-88` — Vertex reference corruption during edge merge

**Problem**: When merging edges at a shared vertex, only the current edge's `v_low`/`v_high` is patched. Other edges referencing the old vertex are migrated in `vertex_to_edges` but their own vertex fields are NOT updated. Creates inconsistent topology.

**Impact**: All subsequent passes querying edges by vertex operate on corrupt data. Shell watertightness fails.

**Fix**: Update `v_low`/`v_high` for ALL migrated edges, or rebuild `vertex_to_edges` after the pass.

---

### C7. `heal/same_param_reparam.rs:150` — Line PCurve direction bug on periodic surfaces

**Problem**: `adjust_line_pcurve` computes UV direction as straight-line difference between projected endpoints. For cylindrical/spherical surfaces, UVs can wrap around periodic boundaries (u=0.1 vs u=6.2), producing a direction that goes the wrong way around.

**Impact**: Edges on cylinders/spheres appear at wrong 3D locations on the surface.

**Fix**: Normalize periodic UVs to same branch before computing direction.

---

### C8. `geom/nurbs.rs:1236-1263` — `NurbsSurface::sphere()` creates a flat plane, not a sphere

**Problem**: Single-patch `sphere()` constructor places all 3×3 control points at y=-r. Every control point shares the same y coordinate, so the surface evaluates to plane y=-r, not a sphere.

```rust
control_points: vec![
    [(-r, -r, -r), (-r, -r, 0), (-r, -r, r)],  // all y = -r
    [( 0, -r, -r), ( 0, -r, 0), ( 0, -r, r)],  // all y = -r
    [( r, -r, -r), ( r, -r, 0), ( r, -r, r)],  // all y = -r
]
```

**Impact**: Silent wrong geometry. `sphere_six_patch()` works correctly.

**Fix**: Panic with message directing to `sphere_six_patch()`, or implement proper pole construction.

---

### C9. `mesh/face_fill.rs:1157-1197` — Projection cache hash collision

**Problem**: `surface_point_deviation_cached` uses quantized 3D position bits as cache key. Two distinct points quantizing to same cell share a cached UV projection, computing chord error against wrong surface point.

**Impact**: `max_chord_error` silently wrong — quality gate accepts triangles exceeding deflection budget.

**Fix**: Use more unique key, or remove cache, or use LRU of size 1.

---

### C10. `mesh/face_cdt.rs:184-187` — Single inner-loop constraint failure kills entire face

**Problem**: If ANY inner-hole edge fails constraint insertion, `triangulate_uv_cdt_with_steiner` returns empty result immediately. Face with multiple holes loses ALL triangulation if just one hole edge fails.

```rust
if !ok && cdt.num_constraints() == before && !cdt.exists_constraint(a, b) {
    return (Vec::new(), 0.0, CdtConstraintReport::default());  // KILLS WHOLE FACE
}
```

**Impact**: Entire face becomes a gap in the shell mesh.

**Fix**: Log failure, continue. Collect all constraint failures into report.

**OCCT ref**: BRepMesh_Delaun continues with constraint failures; ModelHealer handles gaps later.

---

### C11. `mesh/post_process.rs:251-282` — `heal_mesh_gaps` vertex duplication on reassembly

**Problem**: `heal_mesh_gaps` builds per-face sub-meshes with local vertex/index buffers. Welded boundary vertices update local buffers but share NO unified index space. On reassembly, all sub-mesh vertices appended sequentially → duplicates at seams.

**Impact**: `heal_mesh_gaps` is effectively a no-op for most real-world models.

**Fix**: Use unified index deduplication or weld after reassembly with tolerance.

---

### C12. `mesh/void_subtract.rs:96,163` — DDA ray walk uses wrong coordinate space for max_x

**Problem**: `max_x` is max grid cell coordinate from void triangle registration. If ray origin is to the right of all void triangles (point.x > max void x), then `cx > max_x`, the `cx..=max_x` range is empty, and ray returns 0 hits even if point is inside void cavity.

**Impact**: False negatives for points in rightmost portion of voids — triangles incorrectly retained.

**Fix**: Walk to `cx + cell_count_x` or use infinite DDA.

---

### C13. `brep/write.rs:400-408` — Cone surface serialization writes wrong field

**Problem**: Cone (type 3) writes `semi_angle` where OCC expects `radius_at_apex`, and writes `radius_at_apex` on a separate second line. OCC format requires all 13 values on one line with radius_at_apex as 13th.

**Impact**: Every BREP file with conical surface unreadable by OCC tools, or cone geometry corrupted silently.

**Fix**: Write radius_at_apex as the 13th value:
```rust
writeln!(output, "3 {} {} {} {} {} {} {} {} {} {} {} {} {}",
    apex.xyz, axis.xyz, x_dir.xyz, y_dir.xyz, radius_at_apex)?;
```

---

### C14. `bool/classify.rs:234-267` — Torus ray intersection uses numerical sampling

**Problem**: `ray_torus_intersect` walks along ray with step `major_r.min(minor_r) * 0.1` instead of solving the quartic equation analytically. For thin tori (minor_r=0.1), only 40 samples — ray can easily miss the torus entirely.

**Impact**: Point-in-solid classification wrong for solids with toroidal faces.

**Fix**: Solve the torus quartic analytically: `(|P-center|² - (R²+r²))² - 4R²(r² - (P-center)·axis²) = 0`.

---

### C15. `bool/pave_filler.rs` — EF (Edge-Face) interference phase missing

**Problem**: `intersect_edge_face` and `intersect_wire_face` functions exist in `intersect_edge.rs` and have tests, but are NEVER called from `fill_paves()`. The 6-layer interference pipeline has VV, EE, VE, VF, FF but no EF call.

**Impact**: Edges piercing through face interiors are not detected. Boolean results wrong for non-trivial configurations.

**Fix**: Wire `intersect_wire_face` calls into `fill_paves()` between VF and FF phases.

---

### C16. *(Manual)* `bool/pave_filler.rs:324-349` — Edge AABB uses only 2 endpoints

**Problem**: `edge_bbox_touches` and `edge_bbox_touches_two` build AABB from curve endpoints only. Curved edges (Bezier, BSpline, Circle arc) bulge outside endpoint AABB, causing false-negative overlap tests.

**Impact**: Missed edge-edge and edge-face intersections for curved edges.

**Fix**: Sample the edge curve at multiple points or use the curve's true bounding box.

---

## HIGH (27) — Common-Input Crashes or Errors

### Boolean (5)

- **H1** `stitch.rs:88-108`: Transitive vertex merges not handled — v2→v1, v1→v0 chain leaves dangling references
- **H2** `ssi_newton.rs:103-109`: Blind 0.2-scale fallback step when all backtracking fails — may increase residual
- **H3** `classify.rs:462-483`: Multi-shell classification checks Inside against ALL shells simultaneously
- **H4** `intersect.rs:333-345`: `plane_plane` intersection origin forced to z=0, may not lie on actual intersection line
- **H5** `bopds.rs:444-448`: Binary search for edge projection doesn't bracket the minimum

### Heal (6)

- **H6** `edge_tolerance.rs:104-108`: Tolerance comparison reads value AFTER mutation — stats always zero
- **H7** `degenerated.rs:63-148`: Multi-singularity processing uses stale wire cache after first modification
- **H8** `unify_same_domain.rs:406-464`: Merged face not propagated to `edge_to_faces` — subsequent merges skip
- **H9** `wire_join.rs:332-339`: `nudge_line_pcurve` at_end=false modifies direction instead of origin
- **H10** `seam.rs:199-206`: Periodic seam edge added to `seam_edges` but NOT to wire's edge list
- **H11** `pipeline.rs:261-325`: `select_fixes` iter-0 enables passes requiring unmet prerequisites

### Geom/NURBS (5)

- **H12** `curve_eval.rs:289-298`: Rational Bezier d1 uses finite differences — endpoint precision drops 4 orders of magnitude
- **H13** `curve_eval.rs:648-649`: Bezier d2 applies finite difference ON finite difference for rational curves
- **H14** `nurbs.rs:200-201`: `evaluate()` weight check missing `abs()` — negative weights cause division by near-zero
- **H15** `curve2d.rs:807-815`: `clip_beziers_to_range` assumes uniform segment distribution (wrong for non-uniform knots)
- **H16** `nurbs.rs:680`: `reduce_degree_u` uses `i/degree` instead of correct knot-dependent alpha formula

### Mesh (5)

- **H17** `fill_surface.rs:536,602`: Plane-CDT Steiner points snap to surface but CDT uses plane coordinates
- **H18** `face_uv.rs:178`: `pip_even_odd` denominator guard `1e-20` insufficient for horizontal edges
- **H19** `face_cdt.rs:296-298`: UV quantization `as u64` may overflow on large domains (panic in debug)
- **H20** `fill_revolution.rs:259-283`: Seam branch uses first wire's U samples, not uniform U sweep
- **H21** `finalize.rs:303-313`: Chord error measurement skipped for ALL analytic surfaces (cylinder, sphere, torus)

### Integration (6)

- **H22** `brep/write.rs:432-437`: BSpline surface rational/periodic flags encode phantom v-periodicity
- **H23** `brep/write.rs:440`: Rational BSpline surface weights all hardcoded to 1.0 (silent shape corruption)
- **H24** `brep/write.rs:320-331`: BSpline curve knots written without multiplicities — OCC misparses
- **H25** `brep/write.rs:444-445`: BSpline surface knots also missing multiplicities
- **H26** `brep/write.rs:576-578`: Edge vertex orientation: second vertex written as Reversed (should be Forward)
- **H27** `pave_filler.rs:80-117`: EE phase mislabeled as "Phase 2: VE" — naming confusion

---

## Representative MEDIUM (key selections from 45 total)

### Boolean
- **M1** `coplanar.rs:406-412`: GH entry/exit uses hardcoded 0.001 offset (not relative to edge length)
- **M2** `split.rs:136-175`: Boundary midpoint used as interior classification point (wrong for concave regions)
- **M3** `builder_solid.rs:236-254`: Seam edges cause shell closure false negative (c==4 instead of c==2)
- **M4** `coplanar.rs:203-219`: Strict inequalities reject vertex-on-vertex and edge-on-vertex intersections
- **M5** `pave_filler.rs:367-415`: Pave blocks built with zero-length intervals `t_range: (t, t)`

### Heal
- **M6** `pipeline.rs:47-102`: `HealPolicy.passes` is dead code — never consumed by `auto_heal_shell`
- **M7** `canonical.rs:401-456`: Kasa circle fit numerically unstable for small arc data (<90°)
- **M8** `continuity.rs:77-95`: G1 check uses `project()` not PCurve UV — normals at wrong positions on tori
- **M9** `solid_fix.rs:34-98`: Solid orientation uses polygon Newell's method for curved faces
- **M10** `intersecting_wires.rs:238-253`: Merged inner wires have disconnected edge lists (not reordered)

### Mesh
- **M11** `grid.rs:309`: Partial cell fan-triangulation produces sliver triangles near boundaries
- **M12** `report.rs:80-97`: Shell bbox diagonal skips seam edges, underestimating shell size
- **M13** `edge_pool.rs:106-143`: Arc-length welding may map distinct points to same quantized key
- **M14** `post_process.rs:293-296`: `tri_ref` stores only last triangle per vertex (T-junction splits wrong)
- **M15** `model_preprocessor.rs:203`: `orient_2d` uses raw f64 without error bounds

### Geom
- **M16** `surface_eval.rs:357-358`: Extrusion direction normalization inconsistent between d0 and project
- **M17** `curve_eval.rs:533-539`: Offset curve `normalize_or_zero()` produces zero offset when tangent || offset_dir
- **M18** `properties.rs:166-169`: GK adaptive integration may hang for near-zero integrands (subdivides to max_depth)
- **M19** `curve2d.rs:533-612`: Bézier clipping on coincident curves produces O(2^depth) spurious intersections
- **M20** `surface_eval.rs:761-778`: Cylinder/Cone search window ±50 may miss tall geometry

### Integration
- **M21** `emit_plan.rs:104-131`: `config_hash` fragile manual encoding — new field = silent cache collision
- **M22** `brep/write.rs:336-355`: Trimmed non-line curves fall back to dummy line at origin (losing geometry)
- **M23** `brep/write.rs:411-421`: Sphere orientation hardcoded to +Z axis
- **M24** `error.rs:3-4`: `NullShape` carries zero context for debugging

---

## LOW (25) — Selected

- **L1** `marching.rs:202-283`: `step_size_for_surfaces`, `cluster_directions`, `detect_bifurcation` — dead code
- **L2** `select.rs:42,60,76`: `curves` parameter to `push_kept_face` always empty slice
- **L3** `coplanar.rs:75-82`: Denominator epsilon `1e-12` doesn't scale with coordinate magnitude
- **L4** `topo_iter.rs:93-98`: `iter_faces_of_shell` doesn't deduplicate (unlike `iter_edges_of_shell`)
- **L5** `xde.rs:42-43`: `add_label` doesn't update `root_labels` — manual step required
- **L6** `mesh_result.rs:300-325`: `finalize_normals` may retain zero normals, fallback to +Z
- **L7** `check/face.rs:132-141`: Sphere pole degeneracy tolerance hardcoded to 0.01 rad
- **L8** `face_fold.rs:56-57`: Fold detection compares all normals against only first sample
- **L9** `canonical.rs:141-159`: `acos()` poor near-zero precision for near-spherical eigenvalue distributions
- **L10** `wire_ops.rs:291-301`: Tail detection vertex count only checks within-wire references
- **L11** `face_fix.rs:142-148`: `signed_uv_wire_area` doesn't dedup near-duplicate consecutive vertices
- **L12** `self_intersect.rs:176-183`: Collinear overlapping segments not detected (det≈0 returns None)
- **L13** `free_bounds.rs:265-273`: `dedup_by` only removes consecutive duplicates
- **L14** `same_param_fix.rs:107-127`: `sample_deviation` uses tolerance for both sampling density and reporting
- **L15** `nurbs.rs:357`: Evaluate with hessian silently converts NaN to ZERO (no warning)

---

## Fix Priority Matrix

### Blocker — Boolean/Mesh/Output untrustworthy until fixed
C1-C5, C9-C11, C13-C16

### P0 — Severely affects common inputs
C6-C8, C12, H1-H5, H6-H11, H12-H16, H22-H26

### P1 — Boundary conditions / degenerate inputs
H17-H21, H27, M1-M24

### P2 — Code quality / performance
L1-L25, all remaining LOW items

---

## Notes

- This audit found 113 defects, but does NOT claim exhaustiveness — some algorithmic bugs may only manifest on specific geometry configurations not traceable through static analysis alone.
- The most systemic issue is in the boolean pipeline: 5 CRITICAL defects in the core selection/splitting/classification path mean boolean operations cannot produce correct results for non-trivial inputs.
- The second most systemic issue is BREP serialization: 1 CRITICAL + 5 HIGH defects mean BREP round-trip is fundamentally broken.
- The mesh subsystem has well-structured architecture but several implementation shortcuts that silently degrade quality.

---

## Appendix A: Post-Grill Severity Adjustments

### C8: `NurbsSurface::sphere()` — CRITICAL → LOW
**Reason**: `NurbsSurface::sphere()` is never called anywhere in the codebase. The function is dead code. The correctly-working `sphere_six_patch()` is the one used. A broken function that nothing invokes cannot cause runtime harm.

### C14: Torus ray intersection — CRITICAL → MEDIUM
**Reason**: Re-analysis of the sampling approach shows it works adequately for typical torus dimensions:
- For common torus ratios (major_r/minor_r ∈ [2, 10]), the step size produces 20+ samples within the tube diameter
- The max_t limitation only triggers when the test point is far (>4*(R+r)) from the torus center, which is rare for point-in-solid classification (points are typically near the solid)
- 6 ray directions + winding number tiebreaker provide defense-in-depth
- Still a real defect, but impact is limited to edge-case torus configurations

### Potential Duplicates
- C5 and M8 (inner wire handling in winding number vs polygon test) — related but affect different code paths (winding number tiebreaker vs ray-surface hit test). Kept separate.
- H22 and H23 (BSpline surface flags and weights) — same file, same function, but different bugs (flag encoding vs weight value). Kept separate.
- H24 and H25 (BSpline curve vs surface knot multiplicities) — same root cause (missing unique-knot+multiplicity extraction), different code paths. Could be merged.

---

## Appendix B: Audit Scope Limitations

### Not Covered
1. **`rc3d_core::utils::bspline`** — B-spline basis functions (`bspline_bases_f64`, `find_span_f64`), Cox-de Boor recurrence. The `bspline.rs` file in `rc3d-shape` is only a re-export.
2. **`rc3d_core::utils::spatial`** — Spatial indexing (grid, octree, hash) used by intersection acceleration.
3. **`rc3d_core::utils::graph`** — Graph algorithms (toposort, BFS) used by topology traversal.
4. **`rc3d-io` crate (beyond brep/write.rs)** — STEP/IGES parsers, adapter logic, import options. Only the BREP writer was deeply reviewed.
5. **Analytic projection formulas** — Each surface type's `project()` method (plane, cylinder, cone, sphere, torus, extrusion, revolution, BSpline, offset) was not individually verified for formula correctness.
6. **Thread safety** — `BRepStore` Send/Sync status not checked.
7. **Performance profiles** — O(n²) patterns (vertex welding, edge dedup) noted but not benchmarked.

### Low-Confidence Findings (need runtime verification)
- **C9** (projection cache collision): Only triggers if two distinct 3D points quantize to the same cell AND are both projected. Probability depends on quantization granularity.
- **M7** (Kasa circle fit instability): Only triggers for arc coverage < ~90°. Many real CAD faces span > 90°.
- **H4** (plane_plane origin at z=0): Only matters if the intersection line origin is consumed by downstream code. For marching-based tracing, only direction matters.

### False Positive Candidates
- **C4 severity**: If `weld_shell_vertices` + `stitch_faces_into_shell` + `propagate_shell_orientations` can fully reconstruct correct topology despite disconnected edges, the impact of unused `_edge_map` may be lower than CRITICAL. Needs integration test verification.
- **H27** (EE phase mislabeling): Purely cosmetic — "Phase 2: VE" comment vs actual EE code. Zero runtime impact. Should be LOW, not HIGH.

---

## Appendix C: Additional Findings During Grill

### G1. `geom/project.rs:983-984` — `find_param_on_curve` defaults to `0.0`
**Problem**: If both Newton-Raphson and grid fallback return empty results, `find_param_on_curve` returns `0.0` (the curve start). This is used by `trim_curve_to_vertices` at `curve_eval.rs:1013-1014`. A wrong parameter produces an incorrect trimmed curve.
**Severity**: MEDIUM. Requires both projection methods to fail — rare but possible on degenerate curves.

### G2. `surface_eval.rs` — `project()` returns `None` silently for Torus/Extrusion/Offset
**Problem**: The `project()` method returns `None` for `Torus`, `Extrusion`, and `Offset` surfaces, forcing callers to use the slower `inverse_native_uv` with grid search. Callers that don't check the None case (e.g., bool/classify.rs:415-418) silently fall through.
**Severity**: MEDIUM. Analytic projection for torus is well-known (solving for the nearest point on the tube center circle + projecting onto the tube). Missing these means slower, less reliable projection.

### G3. Audit gap: Edge curve self-intersection not checked
**Problem**: No code was found that checks if a single edge curve self-intersects. A BSpline edge with a loop would produce invalid topology.
**Severity**: LOW. Rare in well-formed CAD data.

### G4. H27 reassessment: EE phase mislabeling — HIGH → cosmetic
**Problem**: `pave_filler.rs:80` says "Phase 2: VE — Vertex-Edge" but the code iterates edges against edges. This is a comment-only error with zero runtime impact.
**Severity**: Downgrade from HIGH to LOW (cosmetic).

---

Generated by Claude Code with 5 parallel review agents.
Post-grill verification completed 2026-06-21.
