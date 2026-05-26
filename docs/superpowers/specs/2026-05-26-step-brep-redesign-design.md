# STEP B-Rep Pipeline — Redesigned Architecture

**Date:** 2026-05-26
**Status:** Draft — redesign from two rounds of audit
**Supersedes:** `2026-05-25-step-brep-face-fill-design.md` (original spec; most tasks implemented, gaps found)

**Scope:** Close remaining gaps between rc3d-io BRep pipeline and OCC's `STEPControl_Reader → ShapeFix → BRepCheck → BRepMesh_IncrementalMesh`.

---

## 1. Current State (post-audit)

The P0/P1 tasks from the original spec are implemented: all 9 surface types have `project()`, `param_range()`, `d0_native()`, `normal_native()`; edge discretization with PCURVE priority works; `face_uv` + `face_fill` + `face_cdt` wire extraction and triangulation exist; `refine_mesh_interior` preserves watertight boundaries; `optimize_mesh` flips non-Delaunay edges; `fix_missing_seams` handles VERTEX_LOOP and trimmed periodic faces.

Two audit rounds found these gaps:

| # | Gap | Root cause |
|---|-----|-----------|
| G1 | Seam edges never merged into UV loops | `fix_missing_seams` stores in `face.seam_edges`; `collect_face_loops` reads `wire.edges` only |
| G2 | `grid_fallback_3d` ignores surface geometry | Takes `_face: &BRepFace` (unused), computes best-fit plane, never evaluates surface |
| G3 | `check_shell` is dead code | `HealConfig::run_check` defaults to `false`; `heal_shell` never calls it |
| G4 | Refinement disabled by default | `RefineConfig::enable_post_refine=false`, `max_iterations=0` |
| G5 | Deflection is post-triangulation, not insert-time | `adapt_tris_to_deflection` splits existing tris; OCC inserts Steiner points BEFORE Delaunay |
| G6 | SameParameter runs inside mesh, not heal | OCC `ShapeFix_SameParameter` is pre-mesh; ours runs inside `mesh_brep_shell` |
| G7 | No angular deflection control | Only linear chord error; OCC checks both linear and angular |
| G8 | RemoveDegenerated not wired | Function exists in plan pseudocode; never called in pipeline |
| G9 | `relative_deflection=0.0` disables deflection scaling | OCC `IMeshTools_Parameters::Relative` is on by default |
| G10 | No T4 OCC reference comparison | Planned (Task 16), not implemented |

---

## 2. Redesigned Architecture

### 2.1 Pipeline stages (OCC order)

```
StepToTopoDS (build_brep)
  → build surfaces + curves + PCURVEs
  → no changes needed — already handles 9 surface + 7 curve types

ShapeFix (heal_shell)
  → reorder_wire_edges          ← existing
  → close_wire_gaps             ← existing
  → fix_shell_orientation       ← existing
  → fix_vertex_tolerance        ← NEW
  → fix_small_area              ← NEW, returns list of skipped face keys
  → fix_missing_seams           ← MODIFIED: inserts seam edges into wire.edges
  → check_shell                 ← MODIFIED: always runs; adds self-intersection,
                                   non-manifold, face-area checks

BRepCheck (check_shell — called from within heal_shell)
  → wire closure                ← existing
  → missing edges               ← existing
  → seam edge validity          ← existing
  → face area > 0               ← NEW
  → UV self-intersection        ← NEW
  → non-manifold edge count     ← NEW

BRepMesh (mesh_brep_shell)
  → edge discretization         ← existing (PCURVE priority, adaptive)
  → shared boundary vertex pool ← existing (quantized hash dedup)
  → per-face triangulation      ← REDESIGNED (see §2.2)
  → RemoveDegenerated           ← NEW (wired after triangulation + after refine)
  → shell optimize              ← existing (edge-flip)

Assembly + Materials            ← NEW (§2.3)
  → assembly hierarchy          ← build parent/child tree from NAUO entities
  → color/material transfer     ← extract from STYLED_ITEM chain
```

### 2.2 Per-face triangulation (redesigned)

Two paths, same core algorithm:

```
collect_face_loops(face, wire)
  │  (seam edges already IN wire — no separate seam_edges iteration)
  │
  ├─ wire is empty (VERTEX_LOOP)
  │   → mesh_closed_surface (analytic UV grid, 64×64)
  │
  ├─ UV loops valid (has PCURVE data, non-degenerate area)
  │   → face_fill_cdt:
  │       1. Build initial CDT from UV boundary vertices + constraints
  │       2. Insert Steiner points iteratively (after initial CDT,
  │          before final triangle extraction):
  │          a. For each triangle: check linear deflection at edge midpoints
  │             AND angular deflection between edge endpoint normals
  │          b. If either > threshold AND edge > MinSize:
  │             - compute Steiner point as surface.d0_native(uv_centroid)
  │             - insert into CDT with area constraint
  │          c. Repeat until no splits or max_iters reached
  │       3. Extract triangles; cull those outside UV trim domain
  │       4. fix_winding against surface normal at centroid
  │       5. Emit indices + accumulate normals
  │   Fallback if CDT constraints fail: earcut with same Steiner loop
  │
  └─ UV loops invalid (projection failed, zero-area loop)
      → surface_fill_3d:
          1. Project boundary 3D points to surface → UV
          2. If >70% of projections succeed: use projected UVs for CDT + Steiner
          3. If ≤70% succeed: build planar parameterization from
             boundary vertices, then CDT + surface.project() for Steiner points
          4. All interior points evaluated via surface.d0_native()
```

Key changes from current code:
- CDT is PRIMARY triangulator (was fallback). Earcut is ultimate fallback if CDT constraints fail.
- Steiner points inserted BEFORE triangulation is complete (was: triangulate then centroid-subdivide).
- `adapt_tris_to_deflection` is removed. Steiner loop replaces it.
- Angular deflection checked alongside linear deflection.
- `grid_fallback_3d` replaced by `surface_fill_3d` — always evaluates surface for interior points.
- `UvSource::GridFallback` renamed to `UvSource::SurfaceFill`.
- Post-triangulation refine (`refine_mesh_interior`) remains as a safety net for faces
  where the Steiner loop did not fully converge (e.g., hit max_adapt_iterations).

### 2.3 Assembly hierarchy + material transfer

**Assembly:** Parse `NEXT_ASSEMBLY_USAGE_OCCURRENCE` entities to build parent/child tree. Each node holds an optional transform (from `ITEM_DEFINED_TRANSFORMATION` or `AXIS2_PLACEMENT_3D`) and a shell reference. In `step/mod.rs`, emit `Separator → Transform → [Material → Coordinate3 → IndexedFaceSet]` for each node, recursively.

**Materials:** Walk `STYLED_ITEM → PRESENTATION_STYLE_ASSIGNMENT → SURFACE_STYLE_FILL_AREA → FILL_AREA_STYLE_COLOUR → COLOUR_RGB` chain for each face. Add `color: Option<[f32;3]>` to BRepFace. In `step/mod.rs`, use face color for `MaterialNode::diffuse_color`. Default gray for uncolored faces.

### 2.4 T4 OCC reference gate

New file `mesh/t4_quality.rs`:
- `load_reference_mesh(path) → MeshResult` — parse OCC-exported golden mesh JSON
- `hausdorff_p95(engine: &MeshResult, reference: &MeshResult) → f32` — 95th-percentile Hausdorff distance
- `#[test] #[ignore]` tests for T1 files (Cube.step, cs.step) with deflection-band assertions

---

## 3. OCC Pipeline Mapping (complete)

| OCC stage | Engine module | Status |
|-----------|--------------|--------|
| STEPControl_Reader::ReadFile | `parser::parse_exchange` | Done |
| StepToTopoDS::Transfer — surface | `build.rs:build_surface` (9 types) | Done |
| StepToTopoDS::Transfer — curve | `build.rs:build_curve` (7 types) | Done |
| StepToTopoDS::Transfer — PCurve | `build.rs:resolve_edge_pcurve` | Done |
| ShapeFix_Wire::FixReorder | `heal/reorder.rs` | Done |
| ShapeFix_Wire::FixGaps | `heal/gap.rs` | Done |
| ShapeFix_Shape::FixShellOrientation | `heal/orient.rs` | Done |
| ShapeFix_Edge::FixVertexTolerance | `heal/mod.rs` (NEW) | Gap |
| ShapeFix_Face::FixSmallArea | `heal/mod.rs` (NEW) | Gap |
| ShapeFix_Face::FixMissingSeam | `heal/seam.rs` (MODIFY: insert into wire) | Gap |
| ShapeFix_SameParameter | `mesh/same_param.rs` (keep in mesh) | Deviated* |
| BRepCheck_Wire (closure) | `heal/check.rs` | Done |
| BRepCheck_Face (area > 0) | `heal/check.rs` (NEW) | Gap |
| BRepCheck_SelfIntersection | `heal/check.rs` (NEW) | Gap |
| BRepCheck_NonManifold | `heal/check.rs` (NEW) | Gap |
| BRepMesh_EdgeDiscret | `mesh/edge_disc.rs` | Done |
| BRepMesh_Face (UV loops) | `mesh/face_uv.rs` | Done |
| BRepMesh_Delaun + Steiner | `mesh/face_cdt.rs` (REDESIGN) | Gap |
| BRepMesh::RemoveDegenerated | `mesh/mod.rs` (NEW) | Gap |
| BRepMesh_Optimize | `mesh/optimize.rs` | Done |
| IMeshTools_Parameters (relative) | `mesh/report.rs:apply_relative_deflection` | Fix† |
| IMeshTools_Parameters (angular) | `mesh/refiner.rs` (NEW field) | Gap |
| STEPCAFControl_Reader (assembly) | `assembly.rs` (EXTEND) | Gap |
| XCAF (colors/materials) | `topology.rs + build.rs` (NEW) | Gap |

\* SameParameter kept in mesh because it operates on discretized points. Curve-level snapping (OCC approach) is a separate feature. This is a documented deviation.
† `relative_deflection` fixed to use `min()` (tightens for small parts) instead of overwriting explicit config.

---

## 4. Config Defaults (fixed)

```rust
EdgeDiscConfig {
    deflection: 0.01,          // 1cm absolute (was 0.1)
    angle_deflection: 0.1,     // unchanged
    min_points: 2,             // unchanged
    max_points: 256,           // unchanged
}

FaceFillConfig {
    enable_interior: true,
    deflection_interior: 0.01, // 1cm absolute (was 0.05)
    min_size: 1e-3,
    min_size_relative: 0.01,
    max_adapt_iterations: 8,   // for Steiner loop (was 3 — too few)
}

RefineConfig {
    enable_post_refine: true,  // was false
    max_deflection: 0.01,      // 1cm (was 0.05)
    max_iterations: 4,         // was 0 (disabled!)
    skip_refine_above: 512,
    max_tris: 8192,
    angular_deflection: 0.2,   // NEW: radians (~11.5°)
}

BRepMeshConfig {
    edge: EdgeDiscConfig::default(),
    face: FaceFillConfig::default(),
    refine: RefineConfig::default(),
    optimize: OptimizeConfig::default(),
    relative_deflection: 0.005, // 0.5% of bbox diagonal (was 0.0 — off)
    same_parameter_tol: 1e-4,
}

HealConfig {
    gap_tolerance: 1e-4,
    fix_orientation: true,
    fix_reorder: true,
    fix_missing_seams: true,
    run_check: true,           // was false
    fix_vertex_tolerance: true, // NEW
    fix_small_area: true,       // NEW
}
```

---

## 5. Known Deviations from OCC

| Deviation | Reason |
|-----------|--------|
| SameParameter in mesh, not heal | Operates on discretized points; curve-level snapping is future work |
| Seam edges appended to wire, not inserted at parametric position | CDT handles non-simple polygons; insertion-at-position is future work |
| No `GeomConvert` unified NURBS cache | BSpline surfaces evaluated directly; conversion not needed for mesh |
| No `BRepMesh::UpdateSurface` cache | Single-use mesh; incremental update not needed |
| No unit detection for AutoRelativeMode | Relative deflection coefficient fixed; user can override |

---

## 6. Scope and Phasing

### Phase A — Core correctness (must do first)
1. `fix_missing_seams`: insert seam edges into `wire.edges` instead of `face.seam_edges`
2. `collect_face_loops`: remove seam_edges iteration (no longer needed)
3. `check_shell`: always run; add face-area, self-intersection, non-manifold checks
4. CDT-first + Steiner loop: merge `triangulate_uv_cdt` + `fill_trimmed` + `adapt_tris_to_deflection` into one iterative function
5. `surface_fill_3d`: replace `grid_fallback_3d` with surface-aware fill
6. `RemoveDegenerated`: wire after triangulation AND after refine
7. Fix config defaults (all items in §4)
8. Fix `relative_deflection` to use `min()` instead of override

### Phase B — Quality (do second)
9. Angular deflection in Steiner loop
10. `fix_vertex_tolerance` in heal_shell
11. `fix_small_area` in heal_shell
12. `UvSource::GridFallback` → `UvSource::SurfaceFill`

### Phase C — Completeness (do last)
13. Assembly hierarchy (parent/child from NAUO)
14. Color/material transfer (STYLED_ITEM chain)
15. T4 OCC reference gate (t4_quality.rs)
16. AP242 entity name aliases (if any found during testing)

---

## 7. Test Plan

| Tier | Files | Assertions |
|------|-------|-----------|
| T1 | Cube.step, cs.step | >0 tris, watertight, no grid fallback |
| T2 | Shape.step, Shape-1.step, Shape-2.step | >100 tris per face, grid_fallback_rate < 10%, meshed_faces = face_count |
| T3 | AssemblyExample-Assembly.step | >0 tris per component, transform applied |
| T4 | Reference meshes from OCC | Hausdorff p95 ≤ 2× linear_deflection |

---

## 8. Implementation Notes

- `heal_shell` returns `HealReport` with new field `skip_face_keys: Vec<FaceKey>` from `fix_small_area`. `mesh_brep_shell` skips these faces.
- `HealConfig::run_check` field removed — `check_shell` always runs. Caller-side `if heal_config.run_check` block in `step/mod.rs` removed.
- `HealConfig::fix_vertex_tolerance` and `HealConfig::fix_small_area` fields added.
- `BRepFace.seam_edges` field retained for backward compatibility but no longer populated by `fix_missing_seams`. Can be removed in cleanup pass.
- CDT-first means `triangulate_uv_cdt` is now the PRIMARY triangulator. It must handle all face types. Earcut is emergency fallback only when CDT constraint insertion fails (should be rare with well-formed UV loops).
