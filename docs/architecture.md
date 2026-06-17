# rustcoin3d Geometry Kernel Architecture

Based on: OCCT 7.8.0 cross-audit. All P0/P1/P2 gaps closed as of 2026-06-17.

## Layer map

```
Formats: STEP | IGES | STL | OBJ | glTF | FBX | Binary BREP
                       ↓
I/O layer (rc3d-io)
  Parse → EntityIndex → BRepStore → SceneGraph
  Parametric writer: BRepStore → STEP entities
  Binary persistence: RUSTBREP format
                       ↓
Geometry kernel (rc3d-shape)
  Curve eval │ Surface eval │ Heal (16 pass) │ Mesh (8 stage)
  Boolean (6 layer) │ BRep Store │ Properties │ Delaunay 2D CDT
                       ↓
Core (rc3d-core)
  math (Real=f64, PVec3, PMat4) │ utils (bspline, hash, spatial, graph, ring)
  aabb, bvh, color, display
                       ↓
Rendering (rc3d-render)
  wgpu cluster-deferred │ CSM │ HZB │ TAA/SSR/SSAO │ GPU culling │ meshlet
```

## Type system

```
Precision boundary:
  Geometry kernel:  Real = f64, PVec3 = DVec3, PMat4 = DMat4
  GPU rendering:    Vec3 (f32), Mat4 (f32)
  Boundary casts:   PVec3 → as f32 → Vec3  (at SceneGraph I/O)
```

## OCC alignment

| rustcoin3d | OCC 7.8.0 | Status |
|------------|-----------|--------|
| `Real = f64` | `Standard_Real` | ✅ |
| `BRepStore` | `TopoDS` + `BRep_Builder` | ✅ |
| `BopDS` | `BOPDS_DS` | ✅ VV/VE/EE/VF/EF/FF + FaceInfo + SD |
| `fill_paves()` | `BOPAlgo_PaveFiller` | ✅ 6-layer |
| `builder_solid.rs` | `BOPAlgo_BuilderSolid` | ✅ |
| `heal::*` (23 modules) | `ShapeFix_*` + `ShapeUpgrade_*` | ✅ |
| `canonical.rs` | `ShapeAnalysis_CanonicalRecognition` | ✅ Plane/Cyl/Sphere |
| `unify_same_domain.rs` | `ShapeUpgrade_UnifySameDomain` | ✅ |
| `face_area()` / `solid_volume()` | `BRepGProp` | ✅ |
| `parametric.rs` | `STEPControl_Writer` | ✅ |
| `brep_binary.rs` | `BRepTools_ShapeSet` | ✅ |
| `csg.rs` | `BRepPrimAPI_Make*` | ✅ Block/Cylinder |
| `iges.rs` | `IGESControl_Reader` | ✅ |
| `benchmarks/` | `perf/` | ✅ |

## Boolean pipeline

```
fill_paves()
  ├─ VV: vertex-vertex coincidence → SD map
  ├─ FF: face-face intersection (AABB sweep-and-prune)
  ├─ EF: edge-face intersection
  ├─ EE: edge-edge intersection
  ├─ VE: vertex-on-edge
  └─ VF: vertex-on-face
      ↓
build_pave_blocks()  →  PaveBlocks on edges
build_common_blocks() → CommonBlocks (shared segments)
build_face_infos()   →  FaceInfo (split vertices, state)
      ↓
builder_face.rs      →  split faces at intersection curves
builder_solid.rs     →  reconstruct solids from split faces
```

## Heal pipeline

```
heal_shell()
  ├─ fix_edge_tolerances (pre-pass)
  ├─ fix_notched_edges → fix_tails
  ├─ fix_connected_wire → remove_small_edges → reorder_wire_edges
  ├─ close_wire_gaps / close_wire_gaps_2d
  ├─ fix_same_parameter → fix_shifted_pcurves → fix_edge_curves
  ├─ fix_lacking_edges
  ├─ fix_periodic_degenerated
  ├─ fix_self_intersecting_wire → fix_intersecting_wires
  ├─ fix_add_natural_bound → fix_reversed_2d
  ├─ fix_missing_seams → fix_degenerated_edges
  ├─ fix_face_fold → fix_face_self_intersect
  ├─ fix_small_faces
  └─ unify_same_domain
      ↓
  compose_shells() + edge_connect()
```

## Test coverage

453 unit tests (from ~290 at audit start, +163 across 6 phases).

| Phase | Tests | Focus |
|-------|-------|-------|
| 0 | ~290 | Pre-existing |
| 1 | +5 | Ellipse, boolean fixes |
| 2 | +1 | Volume orient |
| 3 | 0 | Heal wire/mesh |
| Phase 2 | +140 | Boolean 6-layer, CSG, parametric writer |
| Phase 3 | +189 | Heal UnifySameDomain, FixSmallFace/Solid, ModelHealer |
| Phase 4 | +22 | IGES, binary, incremental mesh, canonical |
| Phase 5 | +15 | Canonical cylinder/sphere, notch/tail fix |
| Phase 6 | benchmarks | Criterion bench harness |
| **Total** | **453** | |
