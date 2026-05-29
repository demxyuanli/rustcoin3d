# Phase 2: Mesh Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Prevent CDT failures from invalid trim domains — self-intersecting wires, missing 2D edges, degeneracies.

**Reference mesh pipeline (OCC BRepMesh_IncrementalMesh, OCCT 7.x):**
1. Discretize each `TopoDS_Edge` on face PCurve with linear/angular deflection (`edge_disc.rs`).
2. Build 2D trim loops in parametric (u,v) (`face_uv.rs`, `loops_native_surface_uv` for Revolution/BSpline).
3. Constrained Delaunay triangulation in UV + deflection Steiner refinement (`spade` CDT in `face_cdt.rs`; `earcutr` fallback).
4. Optional trimmed parametric grid clipped with `point_in_trim` (`mesh_trimmed_uv_grid`); **not** full-rectangle native grids or 3D ruled blends.

**Rust ecosystem (reference only):** `spade` (CDT, already used), `earcutr` (polygon fallback), pure-Rust `brepkit` tessellation, or `cadrum`/OCCT bindings for ground-truth comparison — in-tree path stays OCC-aligned without OCCT link.

**Architecture:** Four new heal modules (edge_curve, lacking, self_intersect, degenerated) + `BRepFace.degenerated_edges` field. All depend on Phase 1's PCURVE pathway.

**Dependencies:** Phase 1 (FixConnected, PCURVE pathway, expanded BRepCheck).

---

### Task 1: FixEdgeCurves — Edge Curve to Vertex Alignment

**Files:** Create `heal/edge_curve.rs` (~150 lines), modify `heal/mod.rs`

- Edge curve endpoint adjustment: LINE (recompute), CIRCLE/ELLIPSE (project vertices, update arc angles), B-SPLINE (translate control polygon)
- Signature: `fix_edge_curves(shell_key, reg, tolerance) -> usize`
- 3 tests: line endpoints, circle arc, bspline translate
- Integration: after Phase 1 fixes, before other Phase 2 fixes

### Task 2: FixLacking — Missing 2D Edge Detection

**Files:** Create `heal/lacking.rs` (~120 lines), modify `heal/mod.rs`

- Detect 3D-connected but 2D-disconnected edge pairs
- Small UV gaps: increase edge tolerance; large gaps: insert new PCurve edge
- Signature: `fix_lacking_edges(wire_key, face_key, reg, tol_3d, tol_uv) -> LackingReport`
- 3 tests: small gap tolerance fix, large gap new edge, no 3D connection skip

### Task 3: FixSelfIntersection — UV Boundary Self-Intersection Repair

**Files:** Create `heal/self_intersect.rs` (~250 lines), modify `heal/mod.rs`

- Compute non-adjacent PCurve segment intersections in UV space
- Split edges at intersection points, rebuild wire
- Fallback: >50 intersections → mark face failed
- Signature: `fix_self_intersecting_wire(wire_key, face_key, reg) -> SelfIntersectReport`
- 4 tests: simple cross, multiple intersections, too many, no false positive on adjacent

### Task 4: FixDegenerated — Singularity Detection + Degenerated Edge Creation

**Files:** Create `heal/degenerated.rs` (~180 lines), modify `heal/mod.rs`, `brep/topo.rs` (+5 lines for BRepFace field), `brep/registry.rs` (+15 lines)

- Detect surface singularities per type: sphere poles, cone apex, revolution axis, BSpline numerical
- Create degenerated edges (v_low == v_high, real UV extent)
- Add to `BRepFace.degenerated_edges: Vec<EdgeKey>`
- 4 tests: sphere pole, cone apex, create degenerated edge, plane no false positive

### Task 5: Integration test + full regression

**Files:** Extend `tests/step_files.rs`

- Run full Phase 2 pipeline on cs.step, verify mesh output
- Run all existing tests for regression

---

**Total:** 4 new files, 4 modified files, ~750 net lines
