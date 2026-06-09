# STEP B-Rep OCC Algorithm Gap Closure — Complete Technical Specification

**Date**: 2026-05-26
**Status**: Superseded by `docs/superpowers/specs/2026-06-08-step-universal-pipeline-design.md`
**Scope**: 4-phase implementation plan to close gaps between rc3d STEP pipeline and OCC reference

---

## Architecture Context

The existing pipeline flows: `parser.rs` → `build.rs` (4-pass B-Rep) → `heal/` (6 passes) → `mesh/` (edge disc → face fill → CDT → refine → optimize). This spec adds new heal passes and modifies mesh-stage behavior without restructuring the pipeline topology.

### Key Invariants Preserved
- SlotMap-based BRepRegistry (no TopoDS shared-topology migration)
- `[i0, i1, i2, -1]` triangle index format
- spade CDT as the core triangulator
- Configurable heal passes (existing `HealConfig` pattern)

---

# Phase 1: Heal Infrastructure + Core Correctness

**Goal**: Establish the foundation — vertex connectivity, gap closure in both 3D and 2D, small edge removal, shifted PCurve detection, expanded topology checks, and a PCURVE modification pathway independent of discretization.

**Dependencies**: None (builds on existing heal framework)

## 1.1 FixConnected — Topological Vertex Sharing at Junctions

### File: `crates/rc3d-io/src/step/brep/heal/connected.rs` (new)

### Purpose
OCC ShapeFix_Wire::FixConnected forces adjacent edges in a wire to share the same topological vertex. Currently `reorder.rs` only sorts edges into a connected chain but doesn't merge vertex keys. After gap closing or orientation fixes, edges that meet at the same 3D point may still reference different vertex keys, causing the gap detector to report false positives.

### Types

```rust
/// Result of a FixConnected pass.
#[derive(Debug, Default)]
pub struct ConnectedReport {
    /// Number of vertex pairs merged.
    pub merged_vertices: usize,
    /// Edge junctions that were already connected.
    pub already_connected: usize,
}
```

### Core Function

```rust
/// Merge vertices at adjacent edge junctions within a wire.
/// Walks the wire edge list; for each consecutive pair (including last→first
/// if closed), checks whether the end vertex of edge_i and start vertex of
/// edge_{i+1} are at the same 3D position within tolerance. If so, replaces
/// all references to the second vertex with the first.
///
/// This is a registry-mutating operation: it updates edge.v_low/v_high and
/// re-keys vertices.
pub fn fix_connected_wire(
    wire_key: WireKey,
    reg: &mut BRepRegistry,
    tolerance: f32,
) -> ConnectedReport
```

### Algorithm

1. Clone the wire's edge list.
2. For each consecutive edge pair `(ek_i, ek_{i+1})`:
   a. Get `v_end` of edge_i (accounting for orientation) and `v_start` of edge_{i+1}.
   b. Compute 3D distance between the two vertex positions.
   c. If `distance > 0 && distance < tolerance`: record a merge pair `(v_end, v_start)`.
   d. If `distance == 0` but vertex keys differ: merge immediately (exact match).
3. For closed wires: also check last-edge-end vs first-edge-start.
4. Apply merges: for each `(keep, replace)` pair, update all edges referencing `replace` to reference `keep` instead. Remove the `replace` vertex from the registry.
5. Also fix PCurve endpoints: if PCurve of edge_i at t=1 maps to a UV point that's "near" the PCurve of edge_{i+1} at t=0, nudge them to match exactly.

### Integration
- Called from `heal_shell()` in `heal/mod.rs` BEFORE `reorder_wire_edges`
- New config flag: `HealConfig::fix_connected: bool` (default: true)

### Error Handling
- If a vertex to be removed is referenced by edges outside this wire, skip that merge (non-manifold junction). Log a warning.
- If both vertices are referenced by >2 faces, skip (would create non-manifold topology).

### Test Cases
1. `test_fix_connected_adjacent_edges` — wire with two edges sharing a 3D point but different vertex keys; after fix, both reference the same key.
2. `test_fix_connected_closed_wire` — closed wire with last→first gap = 0 but different vertex keys.
3. `test_fix_connected_non_manifold_skip` — vertex shared by 3 faces; merge is skipped.
4. `test_fix_connected_outside_tolerance` — gap > tolerance; no merge.

---

## 1.2 FixSmall — Small Edge Removal

### File: `crates/rc3d-io/src/step/brep/heal/small.rs` (new)

### Purpose
Remove null-length or near-null-length edges from wires. Such edges cause CDT constraint insertion failures (zero-length constraints are invalid) and produce degenerate triangles in the output mesh.

### Types

```rust
#[derive(Debug, Default)]
pub struct SmallEdgeReport {
    pub removed_edges: usize,
    pub merged_wires: usize,  // wires that became empty and were merged into parent
}
```

### Core Function

```rust
/// Remove edges shorter than `min_length` from a wire.
/// When an edge is removed, its predecessor and successor are connected
/// directly. If the wire ends up with < 2 edges, it is marked for removal.
///
/// Returns the updated edge list and a report.
pub fn remove_small_edges(
    wire_key: WireKey,
    reg: &mut BRepRegistry,
    min_length: f32,
) -> Option<Vec<(EdgeKey, Orientation)>>  // None = wire should be removed entirely
```

### Algorithm

1. For each edge in the wire:
   a. Compute edge curve length at t=0 and t=1 (or use vertex positions).
   b. If length < min_length:
      - Record the edge for removal.
      - Connect predecessor to successor: update predecessor's end vertex to match successor's end vertex (or vice versa, choosing the more stable vertex).
   c. If the edge is a seam edge, do NOT remove (seam edges may be zero-length in 3D but have valid UV extent).
2. For inner wires: if all edges are removed, the wire is deleted.
3. For outer wires: if fewer than 3 edges remain, mark the face as `skip_face_keys`.

### Integration
- Called from `heal_shell()` AFTER `fix_connected_wire`, BEFORE `close_wire_gaps`
- Reuses `fix_small_area` logic for face-level removal decisions

### Edge Cases
- **Seam edges on closed surfaces**: These are short in 3D but critical for mesh topology — identified by `face.seam_edges.contains(ek)` and skipped.
- **Single-edge wire on a closed surface (sphere vertex_loop)**: The wire has 0 edges, face is a closed surface — handled by existing `mesh_closed_surface` path. Do not remove the face.

### Test Cases
1. `test_remove_zero_length_edge` — wire with an edge where v_low == v_high; edge removed, neighbor edges reconnected.
2. `test_skip_seam_edge` — seam edge with zero 3D length; not removed.
3. `test_small_inner_wire_removed` — inner wire with all edges below threshold; wire deleted.

---

## 1.3 FixGaps2d — UV-Space Gap Closure

### File: Extends `crates/rc3d-io/src/step/brep/heal/gap.rs`

### Purpose
The existing `close_wire_gaps` only checks 3D vertex proximity. But a wire can be connected in 3D while having a gap in UV parameter space — this is typical when adjacent edges on a periodic surface have PCurves that nearly meet but don't quite touch due to floating-point error. OCC's FixGaps2d addresses this specifically.

### New Function

```rust
/// Close gaps between PCurve endpoints of adjacent edges in UV space.
/// For each adjacent pair (including last→first for closed wires), if the
/// 3D endpoints are connected (gap < tol_3d) but the UV endpoints differ
/// by more than tol_uv, adjust the PCurve of one edge to meet the other.
///
/// Strategy: trim/extend the PCurve at the gap end via linear extrapolation
/// of the last segment, clamped to tol_uv * 2.0 max adjustment.
pub fn close_wire_gaps_2d(
    wire_key: WireKey,
    reg: &mut BRepRegistry,
    tol_3d: f32,
    tol_uv: f32,
) -> usize  // number of gaps closed
```

### Algorithm

1. For each adjacent edge pair (i, i+1) including wrap-around for closed wires:
   a. Get edge_i end vertex and edge_{i+1} start vertex — if 3D distance > tol_3d, skip.
   b. Get PCurve for face at edge_i t=1 and edge_{i+1} t=0.
   c. Compute UV distance between these PCurve endpoints.
   d. If UV distance > tol_uv:
      - Determine which edge has the "looser" PCurve (farther from the ideal meeting point).
      - Extend/shorten that edge's PCurve parameter range by up to 2×tol_uv to close the gap.
      - Update the edge's PCurve in the registry.

2. For inner wires: same logic, but the wire is not closed (no wrap-around check).

### Integration
- Called from `heal_shell()` AFTER `remove_small_edges`, BEFORE `fix_missing_seams`
- New config: `HealConfig::uv_gap_tolerance: f32` (default: 1e-5)

### Test Cases
1. `test_close_uv_gap_adjacent` — two edges meeting in 3D, UV endpoints 1e-4 apart; gap closed.
2. `test_no_close_when_3d_disconnected` — 3D gap > tol_3d; UV check skipped.

---

## 1.4 FixShifted — PCurve Period-Range Shift Detection

### File: `crates/rc3d-io/src/step/brep/heal/shifted.rs` (new)

### Purpose
On periodic surfaces (cylinder U, torus U/V, sphere U), a PCurve may be shifted by a full period relative to the actual trim boundary. This happens when the STEP file's PCurve representation uses a different parameter origin than the surface. The result: the UV trim domain has one edge far from the others, causing CDT to fill the wrong region.

### Types

```rust
#[derive(Debug, Default)]
pub struct ShiftedReport {
    pub shifts_applied: usize,
}

/// Describes a detected period shift for a specific PCurve.
struct PeriodShift {
    face_key: FaceKey,
    edge_key: EdgeKey,
    shift_u: f64,  // ±period in U
    shift_v: f64,  // ±period in V
}
```

### Core Function

```rust
/// Detect PCurves shifted by a surface period relative to the wire's
/// expected UV domain. Applies corrections in-place on edge PCurves.
pub fn fix_shifted_pcurves(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepRegistry,
) -> ShiftedReport
```

### Algorithm

1. Determine the surface's period(s): U-period for cylinders/tori/spheres, V-period for tori.
2. For each edge in the wire, get its PCurve for `face_key`.
3. Check whether the PCurve's midpoint is "far" from the wire's expected UV centroid:
   a. Compute UV bounding box of all PCurve endpoints.
   b. If any PCurve's midpoint is >0.8×period from the centroid, test whether shifting by ±period brings it closer.
4. If a shift correction is found: adjust the PCurve's UV coordinates by adding/subtracting the period.
5. Optionally: also check inner wires (each inner wire forms its own UV cluster).

### Integration
- Called from `heal_shell()` AFTER `close_wire_gaps_2d`
- New config: `HealConfig::fix_shifted: bool` (default: true)

### Edge Cases
- **Torus V-period**: a PCurve may need both U and V correction simultaneously.
- **Sphere pole**: PCurves near the pole may appear shifted in both U and V — flag for FixDegenerated (Phase 2) instead.

### Test Cases
1. `test_shifted_cylinder_pcurve` — cylinder face where one edge's PCurve is offset by 2π in U.
2. `test_shifted_torus_pcurve` — torus with combined U+V shift.
3. `test_no_false_positive` — valid PCurves not incorrectly shifted.

---

## 1.5 BRepCheck Expansion — Topology Validation Upgrades

### File: Extends `crates/rc3d-io/src/step/brep/heal/check.rs`

### Purpose
The existing `check_face` validates 5 properties. This expansion adds 4 more checks aligned with OCC BRepCheck_Analyzer, providing the diagnostics needed for Phase 2+ fixes.

### New Checks

#### 5. Edge Tolerance Validity
```rust
fn check_edge_tolerance(ek: EdgeKey, reg: &BRepRegistry) -> Option<String>
```
- Computes edge curve length via `edge.curve.length()` (or endpoint distance for lines).
- If `edge.tolerance > edge_length * 10.0`: warns of oversized tolerance relative to edge.
- If `edge.tolerance < 1e-12`: warns of zero/negative tolerance.

#### 6. Surface Singularity Detection
```rust
fn check_surface_singularities(
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<String>
```
- Evaluates the surface's partial derivatives at parameter extremes.
- If `|dS/du| ≈ 0` or `|dS/dv| ≈ 0` at a point on the trim boundary: reports a potential degeneracy.
- Flags the face for FixDegenerated (Phase 2).

#### 7. Parameter Range Validity
```rust
fn check_parameter_range(
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<String>
```
- Checks that all PCurve parameters for the face lie within the surface's natural domain or within one period of it.
- Reports edges whose PCurve t-range maps to UV coordinates outside expected bounds.

#### 8. Wire Orientation Consistency
```rust
fn check_wire_orientation(
    face_key: FaceKey,
    reg: &BRepRegistry,
) -> Vec<String>
```
- Computes signed area of the outer wire in UV space (via shoelace formula).
- Positive area → CCW (correct for outer when same_sense=true).
- Inner wires should have opposite sign.
- Reports mismatches to `report.warnings`.

### Integration
- All checks called from `check_face()` (no signature change)
- Results flow into existing `CheckReport` structure

### Test Cases
1. `test_edge_tolerance_oversized` — edge with tolerance 100× length; warning generated.
2. `test_surface_singularity_sphere_pole` — sphere at u=π/2; degeneracy reported.
3. `test_parameter_range_oob` — PCurve UV outside surface domain; warning generated.
4. `test_wire_orientation_inconsistent` — outer wire CW, inner wire CCW; warnings for both.

---

## 1.6 PCURVE Modification Pathway

### File: Extends `crates/rc3d-io/src/step/brep/heal/mod.rs`

### Purpose
Currently, heal passes cannot modify PCurves — they can only reorder edges, close 3D gaps, fix orientation, and add seams. FixGaps2d, FixShifted, and Phase 2's FixEdgeCurves all need to modify raw PCurve geometry. This item adds the infrastructure to do so.

### New Type

```rust
/// A lightweight wrapper enabling PCurve modification during healing.
/// Stored temporarily; committed back to the edge when the heal pass completes.
struct PCurveEdit {
    edge_key: EdgeKey,
    face_key: FaceKey,
    /// Replacement PCurve. None = remove existing.
    new_pcurve: Option<P curveGeom>,
    /// If set, also update the edge's 3D curve range.
    new_t_range: Option<(f32, f32)>,
}
```

### Registry Extension

```rust
impl BRepRegistry {
    /// Replace a PCurve for (edge, face). Returns the old PCurve, if any.
    pub fn set_pcurve(
        &mut self,
        ek: EdgeKey,
        face_key: FaceKey,
        pcurve: PCurveGeom,
    ) -> Option<PCurveGeom> { .. }

    /// Get mutable access to a PCurve for in-place adjustment.
    pub fn pcurve_mut(
        &mut self,
        ek: EdgeKey,
        face_key: FaceKey,
    ) -> Option<&mut PCurveGeom> { .. }
}
```

### Design Decision
PCurves are stored in `BRepEdge.pcurves: HashMap<FaceKey, PCurveGeom>`. The new methods expose direct mutation of this map, gated through `&mut self` on the registry. No interior mutability — heal passes already take `&mut BRepRegistry`.

### Test Cases
1. `test_set_pcurve_replace` — replace a PCurve, verify through registry read-back.
2. `test_set_pcurve_new_face` — add PCurve for a face that didn't have one.

---

## Phase 1 File Manifest

| File | Action | Lines |
|------|--------|-------|
| `heal/connected.rs` | New | ~100 |
| `heal/small.rs` | New | ~100 |
| `heal/gap.rs` | Extend (+~80) | — |
| `heal/shifted.rs` | New | ~80 |
| `heal/check.rs` | Extend (+~120) | — |
| `heal/mod.rs` | Extend (+~80) | — |
| `heal/mod.rs` (HealConfig + heal_shell) | Modify (+~40) | — |
| `brep/registry.rs` | Modify (+~20) | — |
| **Total** | | **~620** |

## Phase 1 Test Plan

| Test | File | What it verifies |
|------|------|------------------|
| `test_fix_connected_*` (×3) | `tests/step_heal.rs` | Vertex merging at junctions |
| `test_remove_small_*` (×3) | `tests/step_heal.rs` | Small edge removal |
| `test_close_uv_gap_*` (×2) | `tests/step_heal.rs` | UV gap closure |
| `test_shifted_*` (×3) | `tests/step_heal.rs` | PCurve period correction |
| `test_check_edge_tolerance` | `tests/step_heal.rs` | Tolerance validation |
| `test_check_singularity` | `tests/step_heal.rs` | Singularity detection |
| `test_step_corpus_regression` | `tests/step_files.rs` | All existing STEP files still import |

---

# Phase 2: Mesh Correctness — Self-Intersection, Degeneracy, Missing Edges

**Goal**: Prevent CDT failures caused by invalid trim domains. Self-intersecting wires, missing 2D edges, and unhandled degeneracies all cause `try_add_constraint` to fail silently, producing empty meshes.

**Dependencies**: Phase 1 (uses PCURVE pathway, FixConnected, expanded BRepCheck).

## 2.1 FixEdgeCurves — Edge Curve to Vertex Alignment

### File: `crates/rc3d-io/src/step/brep/heal/edge_curve.rs` (new)

### Purpose
After Phase 1's gap closing and vertex merging, edge 3D curve endpoints may no longer exactly match their vertex positions. OCC's FixEdgeCurves adjusts the curve parameterization to restore alignment.

### Core Function

```rust
/// Adjust edge 3D curves so their t=0 and t=1 positions match their
/// vertex positions within tolerance. Strategy depends on curve type:
///
/// - LINE: re-compute from vertex positions (trivial).
/// - CIRCLE/ELLIPSE: recompute arc from center + radius + vertex projections.
/// - B-SPLINE: translate control polygon so endpoints match vertices,
///   then optionally re-fit if deviation exceeds tolerance.
/// - INTERSECTION/PROJECTION: adjust parameter range (no geometry change needed).
///
/// Returns count of edges adjusted.
pub fn fix_edge_curves(
    shell_key: ShellKey,
    reg: &mut BRepRegistry,
    tolerance: f32,
) -> usize
```

### Algorithm by Curve Type

1. **Line**: If `|curve.d0(0) - v_low| > tol` or `|curve.d0(1) - v_high| > tol`: recompute the line from vertex positions (no other geometry data needed).

2. **Circle/Ellipse**: Project vertices onto the circle/ellipse. Compute new start/end angles. Update the curve's parameter range.

3. **B-Spline**: Translate the control polygon so the first control point matches v_low and the last matches v_high. If this changes the curve shape by more than tol at any knot, log a warning (the source data has conflicting geometry). Check chord error at mid-span; if > 2×tolerance, re-fit.

4. **Intersection/Projection**: These are defined by implicit geometry. Adjust the parameter range only.

### Integration
- Called from `heal_shell()` AFTER Phase 1 fixes (connected → gaps2d → shifted), BEFORE Phase 2 fixes

### Test Cases
1. `test_fix_line_endpoints` — line edge with endpoints 0.01 from vertices; corrected.
2. `test_fix_circle_arc` — circular edge; vertex projections update arc angles.
3. `test_fix_bspline_translate` — B-spline edge; control polygon translated.

---

## 2.2 FixLacking — Missing 2D Edge Detection

### File: `crates/rc3d-io/src/step/brep/heal/lacking.rs` (new)

### Purpose
OCC FixLacking detects cases where edges are connected in 3D (their 3D curves meet) but disconnected in UV space (PCurves end far apart). This indicates a missing PCurve segment — the STEP file may have omitted a small connecting PCurve. The fix either adds the missing PCurve or increases the edge tolerance to bridge the gap.

### Core Function

```rust
/// Detect and fix 2D disconnections between adjacent edges where 3D is connected.
/// Strategy: for each adjacent pair where 3D gap < tol_3d but 2D gap > tol_uv:
///   1. If the 2D gap is < 10×tol_uv: increase edge tolerance to bridge it.
///   2. If the 2D gap is larger: insert a new line-segment PCurve connecting
///      the two UV endpoints (this adds a new edge to the wire).
///
/// Returns report with counts of tolerance-fixed and edge-added fixes.
pub fn fix_lacking_edges(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepRegistry,
    tol_3d: f32,
    tol_uv: f32,
) -> LackingReport
```

### Algorithm

1. For each adjacent edge pair (including wrap-around):
   a. Check 3D connectivity (already guaranteed by FixConnected, but verify).
   b. Get PCurve UV endpoints.
   c. If UV gap > tol_uv: apply fix strategy based on gap size.
2. When inserting a new edge:
   a. Build a line-segment PCurve in UV space.
   b. Build corresponding 3D curve via `surface.d0()` sampling.
   c. Insert into registry, update wire edge list.

### Integration
- Called from `heal_shell()` AFTER `fix_edge_curves`

### Test Cases
1. `test_lacking_small_gap_tolerance` — UV gap < 10×tol; tolerance increased.
2. `test_lacking_large_gap_new_edge` — UV gap > 10×tol; new edge inserted.
3. `test_lacking_no_3d_connection` — 3D disconnected; skipped (not a lacking case).

---

## 2.3 FixSelfIntersection — UV Boundary Self-Intersection Repair

### File: `crates/rc3d-io/src/step/brep/heal/self_intersect.rs` (new)

### Purpose
The existing `check_uv_self_intersection` detects but does not fix self-intersecting UV boundaries. A self-intersecting trim domain is invalid for CDT — it produces triangles outside the intended face region (or fails constraint insertion entirely).

### Types

```rust
#[derive(Debug)]
struct IntersectionPoint {
    /// Parameter along segment_a (0..1)
    ta: f32,
    /// Parameter along segment_b (0..1)
    tb: f32,
    /// UV coordinates of intersection
    uv: (f64, f64),
}

#[derive(Debug, Default)]
pub struct SelfIntersectReport {
    pub intersections_found: usize,
    pub edges_split: usize,
    pub wires_rebuilt: bool,
}
```

### Core Function

```rust
/// Detect and fix self-intersections in a face's outer wire UV boundary.
///
/// Strategy:
///   1. Find all intersection points between non-adjacent PCurve segments.
///   2. At each intersection, split both involved edges at the intersection
///      parameter, creating 4 edges from 2.
///   3. Rebuild the wire with the split edges in correct order.
///   4. Re-run check_uv_self_intersection; if still intersecting, mark face
///      as failed (skip_face_keys).
///
/// Returns the updated wire edge list (may differ in length from input).
pub fn fix_self_intersecting_wire(
    wire_key: WireKey,
    face_key: FaceKey,
    reg: &mut BRepRegistry,
) -> SelfIntersectReport
```

### Algorithm Detail

1. **Detection**: Compute precise intersection of all non-adjacent PCurve segment pairs in UV space. Use rational arithmetic for colinearity checks; avoid floating-point false positives.
2. **Intersection parameter computation**: For each intersecting pair `(seg_a, seg_b)`:
   - Compute intersection point in UV.
   - Back-project to find parameter `t` along each segment's PCurve.
   - Discard intersections at endpoints (t ≈ 0 or t ≈ 1) — these are just shared vertices.
3. **Edge splitting**: For each edge with intersection parameters `{t_1, t_2, ...}`:
   - Sort the parameters.
   - Create sub-edges for each interval `[0, t_1], [t_1, t_2], ..., [t_n, 1]`.
   - Each sub-edge gets a new EdgeKey with its own PCurve trimmed to the parameter range.
   - The 3D curve is similarly split.
4. **Wire rebuilding**: Replace the original edges with the split edges, maintaining orientation.
5. **Fallback**: If the wire has > 50 intersections, don't attempt repair — mark face as failed.

### Error Handling
- If splitting creates an edge shorter than `1e-6` in UV space, skip that split.
- If rebuilding produces a wire with < 3 edges, mark face as failed.

### Test Cases
1. `test_fix_self_intersect_simple_cross` — wire with one self-crossing; split into 4 edges.
2. `test_fix_self_intersect_multiple` — wire with 3 intersections; all resolved.
3. `test_fix_self_intersect_too_many` — >50 intersections; face marked failed.
4. `test_no_false_positive_adjacent` — adjacent edges not flagged as intersecting.

---

## 2.4 FixDegenerated — Singularity Detection + Degenerated Edge Creation

### File: `crates/rc3d-io/src/step/brep/heal/degenerated.rs` (new)

### Purpose
Parametric surface singularities (sphere pole, cone apex, torus inner equator) are points where the surface normal is undefined and the partial derivative vanishes. At these points, edges should be "degenerated" — they have zero 3D length but a defined UV extent. Without degenerated edge handling, the CDT receives a degenerate constraint that causes triangulation to fail.

This Phase 2 component handles **detection and entity creation** only. Phase 3 handles CDT integration.

### Types

```rust
/// A degenerated edge links a regular vertex to a singularity point.
/// In 3D, v_low == v_high (zero length). In UV, it has real extent.
#[derive(Debug, Clone)]
pub struct DegeneratedEdgeInfo {
    /// The degenerated edge key (v_low == v_high).
    pub edge_key: EdgeKey,
    /// The singularity point in 3D.
    pub singularity_3d: Vec3,
    /// The singularity in UV parameter space.
    pub singularity_uv: (f32, f32),
    /// The other (regular) vertex of this degenerated edge.
    pub regular_vertex: VertexKey,
}

#[derive(Debug, Default)]
pub struct DegeneratedReport {
    pub degeneracies_found: usize,
    pub degenerate_edges_created: usize,
}
```

### Core Function

```rust
/// Detect singularities on the face's surface and create degenerated edges
/// for wire segments that pass through singular points.
///
/// Algorithm:
///   1. Evaluate surface derivatives across the face's UV domain.
///   2. Find points where |dS/du × dS/dv| ≈ 0 (singular points).
///   3. For each wire edge whose PCurve passes through or near a singular
///      point (UV distance < 1e-5): split the edge at the singularity,
///      creating a degenerated segment.
///   4. Add degenerated edges to face.degenerated_edges list.
pub fn fix_degenerated_edges(
    face_key: FaceKey,
    reg: &mut BRepRegistry,
) -> DegeneratedReport
```

### Singularity Detection Per Surface Type

| Surface | Singularities | Detection |
|---------|---------------|-----------|
| Sphere | u=0, u=π (poles) | dS/dv = 0 at poles |
| Cone | u=0 (apex) | dS/du = 0 at apex |
| Torus | None | Torus is regular everywhere |
| Cylinder | None | Cylinder is regular (seam, not degenerate) |
| B-Spline | Computed numerically | Sample derivative grid, find near-zero regions |
| Revolution | Computed numerically | Same as B-Spline for the profile curve |

### Integration
- Called from `heal_shell()` AFTER `fix_self_intersecting_wire`
- Adds edges to `BRepFace.degenerated_edges: Vec<EdgeKey>` (new field)

### BRepFace Schema Change
```rust
pub struct BRepFace {
    // ... existing fields ...
    /// Degenerated edges at surface singularities (Phase 2+).
    pub degenerated_edges: Vec<EdgeKey>,
}
```

### Test Cases
1. `test_detect_sphere_pole_degeneracy` — sphere face; poles identified.
2. `test_detect_cone_apex_degeneracy` — cone face; apex identified.
3. `test_create_degenerated_edge` — degenerated edge entity created with v_low=v_high.
4. `test_no_false_degeneracy_plane` — plane face; no degeneracies detected.

---

## Phase 2 File Manifest

| File | Action | Lines |
|------|--------|-------|
| `heal/edge_curve.rs` | New | ~150 |
| `heal/lacking.rs` | New | ~120 |
| `heal/self_intersect.rs` | New | ~250 |
| `heal/degenerated.rs` | New | ~180 |
| `heal/mod.rs` (heal_shell) | Modify (+~30) | — |
| `brep/topo.rs` (BRepFace field) | Modify (+~5) | — |
| `brep/registry.rs` (degenerated edges) | Modify (+~15) | — |
| **Total** | | **~750** |

## Phase 2 Test Plan

| Test | What it verifies |
|------|------------------|
| `test_fix_line_endpoints` | Edge curves aligned to vertices |
| `test_fix_lacking_*` (×3) | Missing 2D edge detection |
| `test_fix_self_intersect_*` (×4) | Self-intersection repair |
| `test_fix_degenerated_*` (×4) | Degeneracy detection + entity creation |
| Shape corpus regression | 6 existing STEP test files, all mesh successfully |

---

# Phase 3: Mesh Quality — CDT Integration + Better Steiner + Intersecting Wires

**Goal**: Triangle quality improvements. CDT handles degenerated edges correctly. Steiner refinement uses edge midpoints. Intersecting wire repair enables correct meshing of faces with complex trim.

**Dependencies**: Phase 2 (degenerated edge entities exist, self-intersection fixed).

## 3.1 Degenerated Edge CDT Integration

### File: Modifies `crates/rc3d-io/src/step/brep/mesh/face_cdt.rs`

### Purpose
When a face has degenerated edges (created by Phase 2), the CDT must handle them correctly: they are UV-space constraint edges but their 3D mapping collapses to a single point.

### Changes to `triangulate_uv_cdt_with_steiner`

1. **Constraint insertion**: Degenerated edges are added as CDT constraints like regular edges. Their UV extent is real, so they define a valid constraint in UV space.

2. **Triangle extraction**: When extracting triangles from the CDT, triangles that have all 3 vertices on a degenerated edge (i.e., all map to the same 3D point) are skipped — they are zero-area in 3D.

3. **Steiner refinement**: Steiner points are NOT inserted on degenerated edges (they wouldn't improve the 3D mesh since all points on the edge map to the same 3D location).

```rust
// In triangulate_uv_cdt_with_steiner, after inserting outer boundary:

// Insert degenerated edges as constraints
for &dek in &face.degenerated_edges {
    if let Some(edge) = reg.edges.get(dek) {
        if let Some(pcurve) = edge.pcurves.get(&face_key) {
            let uv_start = pcurve.d0(0.0);
            let uv_end = pcurve.d0(1.0);
            // Insert start and end vertices into CDT
            let h_start = insert_uv(&mut cdt, (uv_start.x, uv_start.y), ...);
            let h_end = insert_uv(&mut cdt, (uv_end.x, uv_end.y), ...);
            // Add constraint between them
            let _ = cdt.try_add_constraint(h_start, h_end);
        }
    }
}
```

### Test Cases
1. `test_cdt_with_degenerated_sphere_pole` — sphere face with degenerated edges at poles; CDT produces valid triangles, no zero-area triangles at pole.
2. `test_cdt_with_degenerated_cone_apex` — cone face; apex singularity correctly triangulated.

---

## 3.2 Steiner Edge-Midpoint Insertion

### File: Modifies `crates/rc3d-io/src/step/brep/mesh/face_cdt.rs`

### Purpose
The current Steiner refinement inserts only triangle centroids (face_cdt.rs:241-244). OCC's strategy is more sophisticated: if a single edge of a triangle fails the deflection check, insert at the midpoint of that edge rather than the centroid. Centroids are only used when all three edges fail.

### Algorithm Change

Replace the centroid-only logic in the Steiner loop:

```rust
// Current (simplified):
if tri_split {
    let cu = (uv0.0 + uv1.0 + uv2.0) / 3.0;
    let cv = (uv0.1 + uv1.1 + uv2.1) / 3.0;
    splits.push((cu as f64, cv as f64));
}

// New: edge-midpoint preference
if tri_split {
    // Collect which edges failed
    let mut failed_edges: Vec<usize> = Vec::new(); // 0=01, 1=12, 2=20
    for (ei, (a, b, uva, uvb)) in edge_checks.iter().enumerate() {
        let dev = ((*a + *b) * 0.5 - mid_on_surf).length();
        if dev > config.deflection_interior && edge_len > min_sz {
            failed_edges.push(ei);
        }
    }
    
    if failed_edges.len() == 1 {
        // Insert at midpoint of the worst edge
        let uv_edge_mid = match failed_edges[0] {
            0 => ((uv0.0 + uv1.0) * 0.5, (uv0.1 + uv1.1) * 0.5),
            1 => ((uv1.0 + uv2.0) * 0.5, (uv1.1 + uv2.1) * 0.5),
            _ => ((uv2.0 + uv0.0) * 0.5, (uv2.1 + uv0.1) * 0.5),
        };
        splits.push((uv_edge_mid.0 as f64, uv_edge_mid.1 as f64));
    } else {
        // Multiple edges or all edges fail: use centroid
        let cu = (uv0.0 + uv1.0 + uv2.0) / 3.0;
        let cv = (uv0.1 + uv1.1 + uv2.1) / 3.0;
        splits.push((cu as f64, cv as f64));
    }
}
```

### Additionally: Sort Splits by Deflection

Before truncation, sort `splits` by descending chord error. This ensures the worst triangles are refined first:

```rust
// After collecting splits, before truncate:
splits.sort_unstable_by(|a, b| {
    let da = chord_error_at_uv(a, &face.surface, ...);
    let db = chord_error_at_uv(b, &face.surface, ...);
    db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
});
```

### Test Cases
1. `test_steiner_edge_midpoint` — triangle where only one edge exceeds deflection; midpoint inserted on that edge.
2. `test_steiner_centroid_fallback` — all three edges fail; centroid inserted.
3. `test_steiner_sort_by_deflection` — worst triangles get refined first within the 256-split cap.

---

## 3.3 FixIntersectingWires — Multi-Wire Intersection Repair

### File: `crates/rc3d-io/src/step/brep/heal/intersecting_wires.rs` (new)

### Purpose
A face with both an outer wire and one or more inner wires may have intersecting wires in UV space. This is invalid — inner wires must lie entirely within the outer wire and must not intersect each other. OCC's FixIntersectingWires detects and repairs such intersections.

### Core Function

```rust
/// Detect intersections between outer and inner wires (and between inner wires).
/// Repair strategy:
///   1. If an inner wire intersects the outer wire: trim the inner wire to
///      lie entirely inside. Split at intersection points.
///   2. If two inner wires intersect: merge them into a single inner wire
///      (the union of their enclosed regions).
///   3. If an inner wire lies entirely outside the outer wire: remove it
///      (it's likely a data error).
///
/// Returns updated inner wire list.
pub fn fix_intersecting_wires(
    face_key: FaceKey,
    reg: &mut BRepRegistry,
) -> IntersectingWiresReport
```

### Algorithm

1. **Outer-inner intersection detection**: For each inner wire's UV polygon, check against the outer wire's UV polygon using segment intersection tests (reuse `segments_intersect_2d` from check.rs).
2. **Inner-inner intersection**: Pairwise check between all inner wires.
3. **Repair actions**:
   a. Inner intersects outer but is mostly inside → split inner at intersection points, keep inside portion.
   b. Inner intersects outer and is mostly outside → remove inner wire (log warning).
   c. Two inners intersect → merge into one wire (union boundary).
4. **Point-in-wire test**: Use winding number to determine inside/outside.

### Integration
- Called from `heal_shell()` AFTER `fix_self_intersecting_wire`, BEFORE `fix_missing_seams`

### Test Cases
1. `test_inner_intersects_outer_trim` — inner wire crossing outer boundary; trimmed.
2. `test_inner_outside_outer_removed` — inner wire entirely outside; removed.
3. `test_two_inners_intersect_merged` — intersecting inner wires; merged.
4. `test_no_false_positive_separate_inners` — properly separate inner wires; unchanged.

---

## 3.4 FixPeriodicDegenerated — Periodic Surface Pole Degeneracy

### File: `crates/rc3d-io/src/step/brep/heal/periodic.rs` (new)

### Purpose
On periodic surfaces (cylinder, torus) with a single wire that wraps around the full parameter range, the poles of the parameter domain need degenerated edges. This is the FixPeriodicDegenerated pass from OCC ShapeFix_Face, which must run BEFORE FixMissingSeam.

### Core Function

```rust
/// For a single wire that wraps a periodic surface (U-period or V-period),
/// detect if the wire passes through a parameter singularity (pole) and
/// reconstruct degenerated edges at those poles.
///
/// This must be called BEFORE fix_missing_seams, as seam detection depends
/// on correctly placed degenerated edges.
pub fn fix_periodic_degenerated(
    face_key: FaceKey,
    reg: &mut BRepRegistry,
) -> PeriodicDegeneratedReport
```

### When This Applies
- **Cylinder**: Single wire spanning full U range [0, 2π]. No poles for cylinder — no degeneracy.
- **Torus**: Single wire spanning full U or V range. No poles for torus — no degeneracy.
- **Sphere**: Single wire wrapping the sphere. Poles at u=0 and u=π need degenerated edges.
- **Surface of revolution**: Similar to sphere if the profile touches the axis.

### Algorithm

1. Determine if the outer wire wraps a full parameter period.
2. If yes, check whether the wire passes through u=0 (or u=π for sphere).
3. At each pole crossing, split the PCurve and insert a degenerated edge (v_low = v_high = pole vertex).
4. Update the wire edge list with the split edges and degenerated edges.

### Integration
- Called from `heal_shell()` BEFORE `fix_missing_seams`

### Test Cases
1. `test_periodic_degenerated_sphere_full_wrap` — sphere with single wire wrapping full U; degenerated edges at both poles.
2. `test_periodic_degenerated_no_wrap` — face that does NOT wrap full period; no change.

---

## Phase 3 File Manifest

| File | Action | Lines |
|------|--------|-------|
| `mesh/face_cdt.rs` (degen CDT + Steiner) | Modify (+~180) | — |
| `heal/intersecting_wires.rs` | New | ~160 |
| `heal/periodic.rs` | New | ~130 |
| `heal/mod.rs` (heal_shell) | Modify (+~20) | — |
| **Total** | | **~490** |

## Phase 3 Test Plan

| Test | What it verifies |
|------|------------------|
| `test_cdt_degenerated_sphere` | CDT handles degenerated sphere edges |
| `test_cdt_degenerated_cone` | CDT handles cone apex |
| `test_steiner_edge_midpoint` | Midpoint insertion strategy |
| `test_steiner_sort_by_deflection` | Prioritized refinement |
| `test_intersecting_wires_*` (×4) | Wire intersection repair |
| `test_periodic_degenerated_*` (×2) | Periodic pole degeneracy |
| Quality benchmark vs OCC | Triangle count + chord error comparison |

---

# Phase 4: Automation + Advanced Features

**Goal**: One-click STEP import with automatic heal strategy selection. Advanced features: relative deflection, 2D deflection control, global properties, continuity checking.

**Dependencies**: Phases 1-3 (heal pipeline is feature-complete).

## 4.1 HealPipeline — Iterative Auto-Heal

### File: `crates/rc3d-io/src/step/brep/heal/pipeline.rs` (new)

### Purpose
Replace the fixed-sequence `heal_shell()` with an iterative auto-heal pipeline that applies fixes, checks results, and re-applies fixes as needed until the shape is valid or no more progress is made.

### Types

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealLevel {
    /// FixConnected + FixSmall + GapClose3d + FixOrientation only.
    Basic,
    /// Basic + FixGaps2d + FixShifted + FixEdgeCurves + FixMissingSeams.
    Standard,
    /// Standard + FixSelfIntersection + FixDegenerated + FixIntersectingWires + FixPeriodic.
    Advanced,
}

impl Default for HealLevel {
    fn default() -> Self { HealLevel::Standard }
}

#[derive(Debug)]
pub struct HealPipelineReport {
    pub level: HealLevel,
    pub iterations: usize,
    pub checks_before: CheckReport,
    pub checks_after: CheckReport,
    pub fixes_applied: Vec<String>,  // human-readable fix descriptions
    pub total_fixes: usize,
    pub converged: bool,  // true if last iteration made no new fixes
}
```

### Core Function

```rust
/// Run iterative auto-heal on a shell. Each iteration:
///   1. Run BRepCheck to identify issues.
///   2. Select and apply fixes based on detected issues and HealLevel.
///   3. Re-check. If new issues found (or old ones remain), iterate.
///   4. Stop when converged or max_iterations (default: 5) reached.
///
/// The heal sequence is NOT fixed — it adapts to what BRepCheck finds.
pub fn auto_heal_shell(
    shell_key: ShellKey,
    reg: &mut BRepRegistry,
    level: HealLevel,
    max_iterations: usize,
) -> HealPipelineReport
```

### Fix Selection Logic (per iteration)

```rust
fn select_fixes(check: &CheckReport, level: HealLevel) -> Vec<FixKind> {
    let mut fixes = Vec::new();
    
    // Always run these first (foundational)
    fixes.push(FixKind::Connected);
    fixes.push(FixKind::Small);
    fixes.push(FixKind::Gap3d);
    fixes.push(FixKind::Orientation);
    
    if level >= HealLevel::Standard {
        if check.has_uv_gaps() { fixes.push(FixKind::Gap2d); }
        if check.has_shifted_pcurves() { fixes.push(FixKind::Shifted); }
        if check.has_curve_deviations() { fixes.push(FixKind::EdgeCurves); }
        fixes.push(FixKind::MissingSeams);
    }
    
    if level >= HealLevel::Advanced {
        if check.has_self_intersections() { fixes.push(FixKind::SelfIntersect); }
        if check.has_degeneracies() { fixes.push(FixKind::Degenerated); }
        if check.has_intersecting_wires() { fixes.push(FixKind::IntersectingWires); }
        if check.has_periodic_degeneracies() { fixes.push(FixKind::PeriodicDegenerated); }
    }
    
    fixes
}
```

### Integration
- Replaces the fixed-sequence calls in `heal_shell()` when opted in
- Backward compatible: existing `heal_shell()` remains for tests; `auto_heal_shell()` is new entry point
- `step/mod.rs` uses `auto_heal_shell()` with `HealLevel::Standard` by default

### Test Cases
1. `test_auto_heal_converges` — damaged shell; auto-heal runs 2 iterations then converges.
2. `test_auto_heal_basic_vs_standard` — Basic doesn't fix UV gaps; Standard does.
3. `test_auto_heal_max_iterations` — shell with unfixable issues stops at max_iterations.

---

## 4.2 FixVertexPosition — Vertex Position Correction

### File: `crates/rc3d-io/src/step/brep/heal/vertex_position.rs` (new)

### Purpose
After gap closing and edge curve adjustments, vertex positions may have drifted from their ideal locations on the associated surfaces. This pass projects vertices back onto their owning surfaces/curves.

```rust
/// Project each vertex in the shell onto the surfaces of faces that reference it.
/// If the vertex is farther than `tolerance` from the surface, move it to the
/// closest projection point. If no valid projection exists, leave the vertex as-is.
pub fn fix_vertex_positions(
    shell_key: ShellKey,
    reg: &mut BRepRegistry,
    tolerance: f32,
) -> usize  // vertices adjusted
```

### Test Cases
1. `test_fix_vertex_on_surface` — vertex 0.01 from surface; projected back.
2. `test_fix_vertex_no_projection` — vertex with no valid projection; unchanged.

---

## 4.3 Relative Deflection Mode

### File: Modifies `crates/rc3d-io/src/step/brep/mesh/edge_disc.rs`

### Purpose
OCC's `isRelative` mode computes edge deflection as `edge_length * relative_factor` rather than an absolute value. This produces more consistent mesh density across models of different scales.

### Changes

```rust
#[derive(Debug, Clone)]
pub struct EdgeDiscConfig {
    pub deflection: f32,
    pub angle_deflection: f32,
    pub min_points: usize,
    pub max_points: usize,
    /// When true, deflection = edge_length * deflection (OCC isRelative).
    pub relative_deflection: bool,
}

// In discretize_edge, before calling sample_curve_adaptive:
let effective_deflection = if config.relative_deflection {
    edge.curve.length() * config.deflection
} else {
    config.deflection
};
```

---

## 4.4 Global Properties (BRepGProp)

### File: `crates/rc3d-io/src/step/brep/properties.rs` (new)

### Purpose
Compute volume, surface area, and center of mass from the tessellated mesh. These are useful for validation (compare expected vs actual volume) and for downstream consumers (physics simulation, bounding volume hierarchies).

```rust
#[derive(Debug, Default)]
pub struct MeshProperties {
    pub volume: f64,
    pub surface_area: f64,
    pub center_of_mass: [f64; 3],
}

/// Compute mesh properties using the divergence theorem.
/// Assumes a closed, watertight mesh with consistent orientation.
pub fn compute_mesh_properties(
    vertices: &[Vec3],
    indices: &[i32],  // [i0, i1, i2, -1] format
) -> MeshProperties
```

### Algorithm
- Volume: sum of signed tetrahedra volumes (each triangle + origin)
- Surface area: sum of triangle areas
- Center of mass: weighted average of tetrahedra centroids

### Test Cases
1. `test_unit_cube_properties` — unit cube mesh; volume=1.0, area=6.0.
2. `test_unit_sphere_properties` — sphere mesh; volume ≈ 4π/3, area ≈ 4π.

---

## 4.5 Continuity Check (G0/G1)

### File: `crates/rc3d-io/src/step/brep/heal/continuity.rs` (new)

### Purpose
Detect geometric continuity defects between adjacent faces along shared edges. G0 = positional (gap), G1 = tangential (normal mismatch). These are quality diagnostics, not blocking errors.

```rust
#[derive(Debug)]
pub struct ContinuityDefect {
    pub edge_key: EdgeKey,
    pub face_a: FaceKey,
    pub face_b: FaceKey,
    pub kind: ContinuityKind,
    pub max_deviation: f32,
}

#[derive(Debug)]
pub enum ContinuityKind {
    G0 { max_gap: f32 },
    G1 { max_angle_deg: f32 },
}

/// Check G0/G1 continuity along all shared edges in a shell.
/// Returns defects found (empty = fully continuous).
pub fn check_shell_continuity(
    shell_key: ShellKey,
    reg: &BRepRegistry,
    g0_tolerance: f32,
    g1_angle_tolerance_deg: f32,
) -> Vec<ContinuityDefect>
```

### Test Cases
1. `test_g0_continuous_shell` — properly connected faces; no G0 defects.
2. `test_g1_discontinuity_at_seam` — faces with mismatched normals at edge; G1 defects reported.

---

## Phase 4 File Manifest

| File | Action | Lines |
|------|--------|-------|
| `heal/pipeline.rs` | New | ~150 |
| `heal/vertex_position.rs` | New | ~80 |
| `heal/continuity.rs` | New | ~100 |
| `mesh/edge_disc.rs` | Modify (+~40) | — |
| `brep/properties.rs` | New | ~80 |
| `heal/mod.rs` | Modify (+~20) | — |
| `step/mod.rs` (use auto_heal) | Modify (+~15) | — |
| **Total** | | **~485** |

## Phase 4 Test Plan

| Test | What it verifies |
|------|------------------|
| `test_auto_heal_*` (×3) | Iterative auto-heal pipeline |
| `test_vertex_position_*` (×2) | Vertex projection |
| `test_relative_deflection` | Relative deflection mode |
| `test_mesh_properties_*` (×2) | Volume/area/centroid computation |
| `test_continuity_*` (×2) | G0/G1 defect detection |

---

# Cross-Cutting Concerns

## Backward Compatibility
- All new heal passes are opt-in via `HealConfig` flags (default: true for Phase 1-2 fixes, matching OCC defaults).
- `heal_shell()` signature gains optional parameters but existing callers compile unchanged.
- `BRepMeshConfig::default()` is unchanged; quality improvements are config-driven.

## Performance Budget
- Each new heal pass targets O(n) or O(n log n) in wire edge count.
- Steiner edge-midpoint insertion adds ~10% computation per iteration (edge midpoint evaluation already done for deflection check; only the insertion point changes).
- Degenerated edge CDT integration adds one constraint edge per degeneracy — negligible.

## Error Recovery
- Any heal pass that encounters an unfixable issue logs a warning and continues.
- Faces that fail heal checks are added to `skip_face_keys` — they are excluded from meshing but don't block other faces.
- The auto-heal pipeline has a hard iteration cap (default: 5) to prevent infinite loops.

## Logging Convention
All new heal passes use the `[BRep heal]` prefix for warn/error, `[BRep heal]` for info:
```rust
log::info!("[BRep heal] FixConnected: merged {} vertex pairs", report.merged_vertices);
log::warn!("[BRep heal] FixSelfIntersection: too many intersections, skipping face {:?}", fk);
```

---

# Total Implementation Summary

| Phase | New Files | Modified Files | Net Lines | Cumulative |
|-------|-----------|----------------|-----------|------------|
| 1 — Core Correctness | 4 | 4 | ~620 | 620 |
| 2 — Mesh Correctness | 4 | 3 | ~750 | 1370 |
| 3 — Mesh Quality | 2 | 2 | ~490 | 1860 |
| 4 — Automation + Advanced | 4 | 3 | ~485 | 2345 |
| **Total** | **14** | **12** | **~2345** | |

## Dependency Graph

```
Phase 1 (Foundations)
  ├── FixConnected → FixSmall → FixGaps2d → FixShifted
  ├── BRepCheck expansion
  └── PCURVE pathway
       ↓
Phase 2 (Mesh Correctness)
  ├── FixEdgeCurves → FixLacking
  ├── FixSelfIntersection
  └── FixDegenerated (detection)
       ↓
Phase 3 (Mesh Quality)
  ├── Degenerated CDT integration
  ├── Steiner midpoint improvement
  ├── FixIntersectingWires
  └── FixPeriodicDegenerated
       ↓
Phase 4 (Automation)
  ├── HealPipeline (wraps all above)
  ├── FixVertexPosition
  ├── Relative deflection
  ├── Global properties
  └── Continuity check
```
