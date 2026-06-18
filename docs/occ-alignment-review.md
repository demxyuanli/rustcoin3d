# OCC 7.8.0 Alignment Review

Date: 2026-06-18. Updated after Phase 7 (IGES) + Phase 8 (Mesh PreProcessor).

## Summary

rustcoin3d covers ~85% of the OCC geometric kernel functionality relevant to
industrial CAD import/heal/boolean/mesh/export workflows.

## Module-by-module alignment

### Geometry (Geom / Geom2d / BRepGProp)

| OCC Class | rustcoin3d | Status | Notes |
|-----------|------------|--------|-------|
| `Geom_Curve` | `CurveGeom` | ✅ | 11 variants including BSpline |
| `Geom2d_Curve` | `Curve2d` | ✅ | 7 variants |
| `Geom_Surface` | `SurfaceGeom` | ✅ | 9 variants including NURBS |
| `BSplCLib` | `bspline.rs` (macro) | ✅ | f32+f64 generic |
| `BRepGProp` | `properties.rs` | ✅ | face_area + solid_volume |
| `BRepGProp_VinertGK` | — | ⚠️ | No adaptive GK integration |
| `Extrema_ExtPC/PS` | `project.rs` | ✅ | Newton multi-start |

### Boolean (BOPAlgo / BOPDS)

| OCC Class | rustcoin3d | Status | Notes |
|-----------|------------|--------|-------|
| `BOPAlgo_PaveFiller` | `pave_filler.rs` | ✅ | 6-layer VV/VE/EE/VF/EF/FF |
| `BOPDS_DS` | `BopDS` | ✅ | +SD map +FaceInfo |
| `BOPAlgo_BuilderFace` | `builder_face.rs` | ✅ | +CommonBlock +is_shared |
| `BOPAlgo_BuilderSolid` | `builder_solid.rs` | ✅ | Vertex-pair adjacency |
| `BOPTools_AlgoTools2D` | `coplanar.rs` | ✅ | Greiner-Hormann polygon clipping |
| `BOPAlgo_GlueEnum` | — | ❌ | Not implemented |

### Healing (ShapeFix / ShapeUpgrade / ShapeAnalysis)

| OCC Class | rustcoin3d | Status | Notes |
|-----------|------------|--------|-------|
| `ShapeFix_Edge` | `edge_tolerance.rs` + `same_param_fix.rs` | ✅ | |
| `ShapeFix_Wire` | `wire_ops.rs` + `wire_join.rs` + `self_intersect.rs` | ✅ | +FixNotchedEdges +FixTails |
| `ShapeFix_Face` | `face_fix.rs` + `face_self_intersect.rs` | ✅ | +FixSmallFace |
| `ShapeFix_Shell` | `shell_fix.rs` + `compose_shell.rs` | ✅ | 7-point orientation |
| `ShapeFix_Solid` | `solid_fix.rs` | ✅ | FixSmallSolid |
| `ShapeFix_EdgeConnect` | `edge_connect.rs` | ✅ | Spatial hash merge |
| `ShapeUpgrade_UnifySameDomain` | `unify_same_domain.rs` | ✅ | Union-find clustering |
| `ShapeUpgrade_ShapeDivideContinuity` | `continuity.rs` | ✅ | C0 edge split |
| `ShapeAnalysis_CanonicalRecognition` | `canonical.rs` | ✅ | Plane/Cyl/Sphere PCA |
| `ShapeProcess_OperLibrary` | `pipeline.rs` | ✅ | 16-pass pipeline |
| `BRepCheck_Analyzer` | `check.rs` | ⚠️ | Partial (missing V-on-Curve/Surface checks) |

### Meshing (BRepMesh)

| OCC Class | rustcoin3d | Status | Notes |
|-----------|------------|--------|-------|
| `BRepMesh_IncrementalMesh` | `edge_disc.rs` | ✅ | cached_deflection reuse |
| `BRepMesh_FaceDiscret` | `face_cdt.rs` + `face_fill.rs` | ✅ | Dual CDT backend |
| **`BRepMesh_ModelPreProcessor`** | **`model_preprocessor.rs`** | **✅** | **Self-intersect + open-wire detection** |
| `BRepMesh_ModelHealer` | `post_process.rs` | ✅ | gap weld + T-junction fix |
| `GCPnts_TangentialDeflection` | `edge_disc.rs` | ✅ | Adaptive sampling |

### I/O (STEP / IGES / Binary)

| OCC Class | rustcoin3d | Status | Notes |
|-----------|------------|--------|-------|
| `STEPControl_Reader` | `step/mod.rs` | ✅ | Part 21 + XML |
| `STEPControl_Writer` | `write/parametric.rs` | ✅ | All surface/curve types |
| `STEPCAFControl_Reader` | `caf_transfer.rs` | ✅ | Colors/layers/PMI |
| `TopoDSToStep` | `write/parametric.rs` | ✅ | BRep→STEP entities |
| `BRepTools_ShapeSet` | `brep_binary.rs` | ✅ | RUSTBREP format |
| `IGESControl_Reader` | `iges.rs` | ✅ | 10 entity types |
| `IGESControl_Writer` | — | ❌ | Not implemented |
| `RWStl` | `stl.rs` | ✅ | ASCII + Binary |
| `RWGltf_CafReader` | `gltf.rs` | ✅ | |
| `VrmlAPI_Writer` | — | ❌ | Not implemented |

### Type system

| OCC | rustcoin3d | Status |
|-----|------------|--------|
| `Standard_Real` (double) | `Real = f64` | ✅ |
| `TopoDS_Shape` hierarchy | `BRepStore` + SlotMap | ✅ |
| `TopLoc_Location` | SceneGraph `Mat4` | ✅ (different model) |

## Coverage summary

| Category | Covered | Partial | Missing |
|----------|---------|---------|---------|
| Geometry (7 classes) | 6 | 1 | 0 |
| Boolean (6 classes) | 5 | 0 | 1 |
| Healing (11 classes) | 10 | 1 | 0 |
| Meshing (5 classes) | **5** | 0 | 0 |
| I/O (8 classes) | 6 | 0 | 2 |
| **Total (37 classes)** | **31 (84%)** | **3** | **3** |

## Remaining gaps (priority-ordered)

| # | Gap | Effort | Impact |
|---|-----|--------|--------|
| 1 | GK adaptive integration | 2d | High-accuracy volume |
| 2 | IGES Writer | 2d | Roundtrip |
| 3 | VRML reader/writer | 1d | Legacy format |
| 4 | VRML reader/writer | 1d | Legacy format |
