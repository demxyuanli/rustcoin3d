# Regression Test Review — 2026-07-08

Post geometry-engine reduction. Workspace: **472 passed, 8 ignored** (53 suites).

---

## 1. Test Inventory by Crate

| Crate | Tests | Status |
|-------|-------|--------|
| rc3d-shape | 124 | ✅ |
| rc3d-io | 0 | ❌ GAP |
| rc3d-core | ~40 | ✅ (math, bvh, spatial) |
| rc3d-render | ~100 | ✅ (shaders, passes) |
| rc3d-scene | ~80 | ✅ (scene graph) |
| rc3d-engine | ~40 | ✅ (engine) |
| Others (small crates) | ~88 | ✅ |

---

## 2. rc3d-shape Test Breakdown (124 tests)

### 2.1 Topology / Store (10 tests) — CRITICAL, KEEP

| Test | Lines | Covers |
|------|-------|--------|
| `test_find_or_add_vertex_dedup` | store | Vertex deduplication |
| `test_find_or_add_vertex_near_duplicate_within_tolerance` | store | Tolerance-based vertex merge |
| `test_find_or_add_vertex_distinct_beyond_tolerance` | store | Non-merge for distant vertices |
| `test_add_edge_with_pcurve_dedup` | store | Edge dedup by vertex pair + curve |
| `test_find_shared_edges` | store | Edge sharing between faces |
| `test_set_pcurve_replace` | store | PCurve replace |
| `test_pcurve_mut` | store | PCurve mutable access |
| `test_set_pcurve_new_face` | store | PCurve for new face |
| `test_vertex_to_edges_index` | store | Vertex→edge inverted index |
| `test_dual_index_consistency` | store | edge_to_faces + vertex_to_edges consistency |
| `test_seam_edge_indices` | store | Seam edge index maintenance |

**Verdict**: Core topology tests. Essential for any BRepStore modification.

### 2.2 Geometry — Curve Evaluation (21 tests) — KEEP

| Area | Tests | Covers |
|------|-------|--------|
| CurveGeom d0/d1/d2 | 12 | Line, Circle, Ellipse, Hyperbola, Parabola, Offset, BSpline, Composite |
| Curve construction | 5 | Hyperbola ctor, Parabola ctor, Ellipse trim, Circle curvature, Sample adaptive |
| BSpline internals | 2 | ndu_idx, d012 degree 3/8 |

**Verdict**: Needed as long as curve_eval.rs exists (store.rs dependency).

### 2.3 Geometry — Surface Evaluation (32 tests) — KEEP

| Area | Tests | Covers |
|------|-------|--------|
| Plane | 6 | d0, d1, normal, project, grid, curvature |
| Cylinder | 5 | d0, d1, normal, project, curvature |
| Cone | 3 | d0 at apex, d0 radius, d1 |
| Sphere | 5 | d0, normal, project, curvature, radius |
| Torus | 4 | d0 inner/top/equator, normal, project |
| Extrusion | 3 | d0, d1, project |
| Revolution | 2 | d0 torus, project |
| BSpline surface | 3 | param_range, project, native_uv roundtrip |
| Offset surface | 6 | d0, d1, project, self-intersection |
| D2 analytical | 1 | d2 vs numerical |

**Verdict**: Core surface evaluation. Needed for any surface/curve interaction.

### 2.4 Geometry — Projection (4 tests) — KEEP

| Test | Covers |
|------|--------|
| `test_project_line_endpoint` | Endpoint projection |
| `test_project_line_midpoint` | Midpoint projection |
| `test_project_circle` | Circle projection |
| `test_project_circle_exact` | Exact circle match |

**Verdict**: `store.rs` calls `find_param_on_curve` → `project_point_on_curve`. Essential dependency.

### 2.5 Geometry — Properties (5 tests) — MARGINAL

| Test | Covers |
|------|--------|
| `cube_area`, `sphere_area` | Face area |
| `solid_volume_gk_cube` | Solid volume |
| `face_area_gk_plane`, `face_area_gk_sphere` | Gauss-Kronrod integration |

**Verdict**: Not called by topology conversion layer. Could be removed but low priority.

### 2.6 Geometry — Curve2D (7 tests) — KEEP

| Test | Covers |
|------|--------|
| `test_line_curve_eval` | 2D line evaluation |
| `test_line_to_bezier` | Line→Bezier conversion |
| `test_bezier_split_midpoint` | Bezier splitting |
| `test_bezier_clip_crossing_lines` | Bezier clipping |
| `test_bezier_clip_separated_no_intersection` | Bezier clip miss |
| `test_intersect_curves_2d_lines` | 2D intersection |
| `simplify_vertical_u_wrap_polyline` | U-wrap polyline→Line simplification |

**Verdict**: Curve2d needed for PCurves in BRepEdge. Keep.

### 2.7 NURBS (19 tests) — MARGINAL

All in `nurbs::tests`. Cover cylinder, cone, sphere, torus NURBS construction + knot insertion + degree elevation.

**Verdict**: Visualization-related (NurbsRenderSurface). Low priority for topology layer. Keep while nurbs.rs exists.

### 2.8 Tolerance (1 test) — KEEP

`from_model_clamps_extremes` — tolerance bounds.

**Verdict**: Core config.

### 2.9 Topo Iterator (8 tests) — KEEP

| Tests | Covers |
|------|--------|
| Face, shell, vertex, edge, wire iteration | Traversal correctness |
| Missing key handling | Graceful failure |
| Dedup | Duplicate handling |

**Verdict**: Essential for any code that traverses BRep topology.

### 2.10 Document (2 tests) — MARGINAL

`world_transform_cache_hit`, `world_transform_parent_chain` — ShapeDocument transform caching.

**Verdict**: Not directly relevant to topology conversion. Low priority.

### 2.11 XDE (2 tests) — MARGINAL

`resolved_color_inherits_from_parent`, `walk_visits_children` — XDE label walking.

**Verdict**: Not relevant to topology conversion.

---

## 3. rc3d-io Test Breakdown (0 tests) — CRITICAL GAP

All 22 previous tests were deleted with the STEP/IGES modules. The remaining files (`brep_diff.rs`, `brep_validate.rs`, `fixtures/`) are **utility scripts**, not regression tests.

**Lost coverage**:
- BREP binary roundtrip (read/write BRepStore)
- STL/OBJ export
- glTF import
- STEP import pipeline (deleted intentionally)

**Remaining risk**: `brep_binary.rs` has NO test coverage. If the binary format changes, no test catches it.

---

## 4. Gap Analysis

### CRITICAL Gaps

| Gap | Impact | Fix |
|-----|--------|-----|
| **BREP binary roundtrip** | `brep_binary.rs` writes BRepStore to binary; no test verifies read-back produces equivalent data. Format change → silent corruption. | Write `test_brep_binary_roundtrip`: create minimal BRepStore, write→read, assert V/E/F counts match. |
| **BREP ASCII writer** | `write_brep()` is the only BREP output path. No test verifies output can be parsed back. | Write `test_brep_ascii_roundtrip`: write→read via brep_binary, assert structural integrity. |
| **STL export** | `write_binary_stl()` / `write_ascii_stl()` — no tests. | Write `test_stl_export_roundtrip`. |
| **glTF import** | `parse_gltf_file()` — no tests. | Test with minimal glTF buffer. |

### HIGH Gaps

| Gap | Fix |
|-----|-----|
| **weld_vertices** | Store test — `test_weld_vertices` missing. Should verify vertex dedup + edge remapping. |
| **add_seam_edge** | Store test — `test_add_seam_edge` missing. Self-loop edge with PCurve. |
| **edge_hash_index consistency** | After weld/merge, edge_hash_index should match edge endpoints. |

### MEDIUM Gaps

| Gap | Fix |
|-----|-----|
| **PCurve type simplification** | `simplify_polyline_to_line` tested but `from_pcurve_3d` flow not tested end-to-end. |
| **BREP writer Locations** | `write_locations` outputs Location matrix. No test verifies format. |
| **OBJ export** | Untested. |

---

## 5. Recommended Actions

1. **Immediate**: Add 3 tests for `brep_binary.rs` (roundtrip, empty store, PCurve roundtrip).
2. **Immediate**: Add `test_weld_vertices` and `test_add_seam_edge` to store tests.
3. **This week**: Add `test_brep_ascii_writer` (write BREP, parse via brep_binary, verify counts).
4. **This week**: Add STL export roundtrip test.
5. **Nice to have**: Add OBJ roundtrip, glTF minimal test.
6. **Low priority**: Remove/document marginal tests (properties, NURBS, XDE, document) if those modules are removed later.

### Quick hit — brep_binary roundtrip

```rust
#[test]
fn brep_binary_roundtrip() {
    let mut store = BRepStore::new();
    let v0 = store.vertices.insert(BRepVertex { position: PVec3::ZERO, tolerance: 1e-4 });
    let v1 = store.vertices.insert(BRepVertex { position: PVec3::X, tolerance: 1e-4 });
    let ek = store.add_edge_with_pcurve(v0, v1,
        CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X },
        1e-4, FaceKey::default(),
        Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) },
        true);
    let data = write_brep_binary(&store);
    let restored = read_brep_binary(&data).unwrap();
    assert_eq!(restored.vertices.len(), 2);
    assert_eq!(restored.edges.len(), 1);
}
```
