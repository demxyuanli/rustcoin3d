# Geometry Engine Scope Review — 2026-07-02

**Decision**: Keep 3D visualization engine as-is. For the geometry engine, retain only
the topology conversion interface layer (BRepStore ↔ BREP format). Remove all
geometry parsing, evaluation, healing, boolean operations, and meshing internals.

---

## 1. Crate Classification

### Visualization Engine (KEEP AS-IS)

| Crate | Lines | Role |
|-------|-------|------|
| rc3d-render | 28,706 | wgpu rendering, shaders, render passes |
| rc3d-scene | 4,547 | Scene graph (NodeData, actions, visitors) |
| rc3d-actions | 4,193 | Action system |
| rc3d-editor | 2,672 | Editor UI |
| rc3d-cli-editor | 2,042 | CLI editor |
| rc3d-scene-api | 1,687 | Scene graph API |
| rc3d-engine | 881 | Engine orchestration |
| rc3d-engine-api | 1,584 | Engine API |
| rc3d-core | 1,613 | Math (Vec3, Mat4, PVec3), BVH, spatial index, utils |
| rc3d-gizmo | 725 | Gizmos |
| rc3d-fields | 383 | Field data |
| rc3d-effects | 368 | Effects graph |
| rc3d-nodes | 4 | Node system stub |
| rc3d-pointcloud | 489 | Point cloud |
| rc3d-pdf | 215 | PDF export |
| rc3d-examples | 196 | Examples |
| rc3d-script | 108 | Scripting |
| rc3d-app | 82 | Application entry |

### Geometry Engine (SCOPE REDUCTION)

| Crate | Lines | Action |
|-------|-------|--------|
| **rc3d-shape** | 53,450 | Keep ~3,000 lines, remove ~50,000 |
| **rc3d-io** | 26,088 | Keep ~3,000 lines, remove ~23,000 |
| **rc3d-mesh** | 1,872 | Remove crate entirely |
| **rc3d-nurbs** | 2,072 | Remove crate entirely |

---

## 2. rc3d-shape — Keep vs Remove

### MUST KEEP — Topology Data Structures (~2,500 lines)

| Module | Lines | Reason |
|--------|-------|--------|
| `topo.rs` | ~80 | BRepVertex, BRepEdge, BRepWire, BRepFace, BRepShell, BRepSolid, key types (VertexKey, EdgeKey, etc.), Orientation |
| `store.rs` | ~300 | BRepStore (SlotMaps, spatial index, edge_hash_index, vertex_locations, locations), find_or_add_vertex, weld_vertices, add_edge_with_pcurve, add_seam_edge |
| `tolerance.rs` | ~80 | ToleranceContext |
| `topo_iter.rs` | ~200 | Topology iterators |

### KEEP — Topology Conversion Interface (~2,000 lines)

| Module | Lines | Reason |
|--------|-------|--------|
| `brep/write.rs` | ~1,000 | BREP ASCII writer (DBRep_DrawableShape format). Write BRepStore → OCC-compatible BREP |
| `geom/curve2d.rs` | ~400 | Curve2d enum (needed for PCurves in BRepEdge) |
| `geom/` mod.rs | ~50 | Type re-exports: CurveGeom, SurfaceGeom enums (type tags needed for BRepStore) |
| `document.rs` | ~150 | BRep document container |
| `error.rs` | ~30 | Error types |
| `lib.rs` | ~50 | Crate root (trim public exports) |

### CAN REMOVE — Geometry Evaluation (~1,700 lines)

| Module | Lines | Notes |
|--------|-------|-------|
| `geom/curve_eval.rs` | 62,800 | Curve d0/d1 evaluation, bspline eval, normalize_edge_curve_to_vertices, curve_param_range_from_vertices, build_ortho_axes, approx_chordal_length. **Only needed by heal/bool/mesh.** |
| `geom/surface_eval.rs` | 66,600 | Surface d0/d1 evaluation, NURBS surface, param_range, project. **Only needed by heal/bool/mesh.** |
| `geom/project.rs` | 11,600 | Point → curve/surface projection. **Only needed by pcurve computation.** |
| `geom/properties.rs` | 16,700 | Geometric properties (volume, area, centroid). |
| `geom/bspline.rs` | 195 | BSpline re-export. |

### CAN REMOVE — Healing Pipeline (~450,000 lines across 31 files)

All of `heal/`:

| File | Lines | Notes |
|------|-------|-------|
| mod.rs | 31,800 | Healing orchestration |
| same_param_fix.rs | 22,700 | Same-parameter edge healing |
| same_param_reparam.rs | 11,500 | Curve reparameterization |
| pcurve_fix.rs | 21,200 | PCurve repair |
| edge_tolerance.rs | 8,100 | Edge tolerance adjustment |
| wire_join.rs | 22,200 | Wire joining |
| wire_ops.rs | 22,300 | Wire operations (reorder, fix tails, remove small edges) |
| intersect_wires.rs | 25,300 | Wire intersection detection |
| face_fix.rs | 20,100 | Face repair |
| face_fold.rs | 4,800 | Face fold detection |
| face_self_intersect.rs | 9,500 | Face self-intersection detection |
| free_bounds.rs | 17,600 | Free boundary detection |
| shell_fix.rs | 17,200 | Shell repair |
| solid_fix.rs | 9,900 | Solid repair |
| compose_shell.rs | 10,100 | Shell composition |
| continuity.rs | 18,200 | Continuity checking |
| degenerated.rs | 19,100 | Degenerated edge processing |
| edge_connect.rs | 6,600 | Edge connectivity |
| geom2d.rs | 2,000 | 2D geometry utilities |
| lacking.rs | 7,300 | Missing edge detection |
| self_intersect.rs | 12,400 | Self-intersection detection |
| topo_diag.rs | 11,000 | Topology diagnostics |
| canonical.rs | 25,900 | Canonical form computation |
| curve_trim.rs | 8,200 | Curve trimming |
| locations.rs | 17,900 | OCC location assignment |
| seam.rs | 20,500 | Seam edge fixing |
| shell_close.rs | 10,600 | Shell closure |
| unify_same_domain.rs | 33,500 | Same-domain unification |
| check/ | — | Diagnostic checks |
| pipeline.rs | 20,400 | Healing pipeline |

### CAN REMOVE — Boolean Operations (~300,000 lines across 17 files)

All of `bool/`:

| File | Lines | Notes |
|------|-------|-------|
| mod.rs | 22,900 | Boolean orchestration |
| intersect.rs | 45,100 | BRep face-face intersection |
| coplanar.rs | 30,900 | Coplanar face handling |
| pave_filler.rs | 34,000 | Pave filler (2D constrained triangulation for intersection) |
| bopds.rs | 28,200 | BOP data structures |
| builder_face.rs | 20,200 | Face building after boolean |
| builder_solid.rs | 16,800 | Solid building after boolean |
| classify.rs | 18,900 | Point-in-solid classification |
| split.rs | 16,600 | Face splitting |
| intersect_edge.rs | 12,600 | Edge-edge intersection |
| marching.rs | 10,900 | Marching surface intersection |
| face_intersector.rs | 10,800 | Face intersector |
| stitch.rs | 9,600 | Face stitching |
| ssi_newton.rs | 9,300 | Surface-surface intersection (Newton) |
| select.rs | 6,700 | Selection logic |
| section.rs | 5,800 | Section curves |
| aabb.rs | 2,800 | AABB for booleans |
| check/ | — | Boolean validity checks |
| shell_pipeline/ | — | Boolean shell pipeline |

### CAN REMOVE — Meshing (~400,000 lines across 30+ files)

All of `mesh/`:

| File | Lines | Notes |
|------|-------|-------|
| face_cdt.rs | 36,300 | Constrained Delaunay triangulation |
| face_fill.rs | 42,200 | Face filling |
| face_uv.rs | 37,400 | UV-based face meshing |
| fill_surface.rs | 23,800 | Surface mesh fill |
| fill_plane.rs | 21,900 | Planar face fill |
| edge_disc.rs | 23,200 | Edge discretization |
| edge_pool.rs | 21,400 | Edge pool management |
| post_process.rs | 24,400 | Mesh post-processing |
| solid_mesh.rs | 11,300 | Solid meshing |
| refiner.rs | 10,900 | Mesh refinement |
| model_preprocessor.rs | 9,800 | Model pre-processing |
| ruled.rs | 22,400 | Ruled surface meshing |
| grid.rs | 15,900 | Grid-based meshing |
| fill_revolution.rs | 14,800 | Revolution surface fill |
| report.rs | 7,700 | Mesh report |
| config.rs | 9,400 | Mesh config |
| diagnostic.rs | 8,900 | Mesh diagnostics |
| boundary.rs | 7,800 | Boundary handling |
| void_subtract.rs | 9,900 | Void subtraction |
| t4_quality.rs | 9,900 | Tetrahedron quality |
| fallback_policy.rs | 3,800 | Fallback policies |
| orient.rs | 2,800 | Orientation |
| param_div.rs | 7,800 | Parameter division |
| optimize.rs | 5,000 | Mesh optimization |
| same_param.rs | 4,400 | Same-parameter handling |
| algo_factory.rs | 1,200 | Algorithm factory |
| face_dispatch.rs | 2,700 | Face dispatch |
| uv_loop_builder.rs | 2,100 | UV loop builder |
| uv_source.rs | 2,600 | UV source |
| shell_impl.rs | 72 | Shell implementation stub |
| shell_mesh.rs | 971 | Shell meshing |
| mod.rs | 5,900 | Module root |
| delaunay2d/ | — | 2D Delaunay triangulation |
| shell_pipeline/ | — | Shell meshing pipeline |

### CAN REMOVE — Other

| Module | Lines | Notes |
|--------|-------|-------|
| `shape.rs` | ~500 | Shape construction utilities |
| `tessellation.rs` | ~200 | Tessellation |
| `emit_plan.rs` | ~500 | Mesh emission plan |
| `nurbs.rs` | ~2,000 | NURBS construction (standalone file) |
| `mesh_result.rs` | ~300 | Mesh result types |
| `mesh_split.rs` | ~150 | Mesh splitting |

---

## 3. rc3d-io — Keep vs Remove

### KEEP — Topology I/O (~1,500 lines)

| Module | Lines | Notes |
|--------|-------|--------|
| `brep_binary.rs` | ~600 | Binary BRepStore read/write for persistence |
| `mesh_export.rs` | ~300 | Generic mesh export interface |
| `stl.rs` (export only) | ~200 | STL binary/ASCII export (for visualization) |
| `obj.rs` (export only) | ~100 | OBJ export (for visualization) |
| `gltf.rs` | ~300 | GLTF export (for visualization) |

### CAN REMOVE — Geometry Parsing (~24,000 lines)

| Module | Lines | Notes |
|--------|-------|--------|
| `step/` | ~15,000 | Entire STEP parser: parser, CAF transfer, entity geometry, topology, adapters, schema, BREP builder, validation. Includes `step/brep/build/` (shell, surface, curve, pcurve resolution). |
| `iges.rs` | ~5,000 | IGES parser |
| `iges_writer.rs` | ~500 | IGES writer |
| `iv.rs` | ~500 | Inventor format |
| `vrml.rs` | ~500 | VRML format |
| `fbx.rs` | ~100 | FBX placeholder |
| `stl.rs` (import) | ~100 | STL ASCII/binary import (keep export, remove import) |
| `obj.rs` (import) | ~100 | OBJ import (keep export, remove import) |

### CAN REMOVE — Standalone Crates

| Crate | Lines | Notes |
|-------|-------|--------|
| `rc3d-mesh` | 1,872 | Dedicated mesh processing |
| `rc3d-nurbs` | 2,072 | NURBS construction library |

---

## 4. Dependency Graph After Reduction

With the geometry engine reduced, `rc3d-io` no longer depends on `step/`, `iges.rs`, etc.
`rc3d-shape` no longer depends on `heal/`, `bool/`, `mesh/`, `geom/curve_eval`, `geom/surface_eval`.

The rendering layer (`rc3d-render`) depends on `rc3d-shape` only for BRepStore
topology types (for reading BREP data for visualization).

**Net reduction: ~73,000 lines removed, 2 crates deleted.**

### What stays connected:

```
rc3d-render → rc3d-shape (topo types + BRepStore)
rc3d-io → rc3d-shape (BRepStore → BREP conversion)
rc3d-io → mesh export (STL/OBJ from mesh data, not from BRep)
```

### What gets disconnected:

```
rc3d-io → step/ (STEP import)
rc3d-io → iges.rs (IGES import)  
rc3d-shape → geom/curve_eval (curve evaluation)
rc3d-shape → geom/surface_eval (surface evaluation)
rc3d-shape → heal/ (healing pipeline)
rc3d-shape → bool/ (booleans)
rc3d-shape → mesh/ (meshing)
```

---

## 5. Risk Assessment

### Safety
- `brep/write.rs` retains `geom/curve2d.rs` (Curve2d enum needed for PCurves in BRepEdge)
- `geom/` mod.rs retains CurveGeom and SurfaceGeom enum definitions (type tags)
- `store.rs` retains all BRepStore methods needed for topology manipulation
- All BREP export tests continue to pass (they only need BRepStore → BREP)

### Known Gaps
- No STEP import (was the primary source of BRepStore data). Need alternative:
  a) Binary BREP files from external tools (OCCT), loaded via `brep_binary.rs`
  b) Programmatic BRepStore construction for simple primitives
  c) Existing pre-built BRepStore data files
- No mesh-from-BRep generation (moved to external preprocessing)

### Files Requiring Dependency Cleanup
- Cargo.toml workspace members list
- `rc3d-shape/src/lib.rs` — remove module declarations
- `rc3d-io/src/lib.rs` — remove module declarations
- Any cross-crate imports referencing removed modules
- `rc3d-shape/src/brep/write.rs` — may reference curve_eval functions that need stubbing
