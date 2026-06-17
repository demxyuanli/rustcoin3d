# Phase 4: Format I/O + Incremental Mesh

Based on: OCCT audit remaining gaps.
Date: 2026-06-17.

## D1: Incremental Mesh Reuse
**OCC**: `BRepMesh_IncrementalMesh`, `IMeshData_Reused/Outdated` flags
**Goal**: Skip re-tessellation of unchanged edges/faces.
**Algorithm**:
1. Store discretization timestamp/deflection per edge
2. Before meshing: check if edge deflection is still "consistent"
3. Mark edges as `Reused` or `Outdated`
4. Only discretize outdated edges
**File**: `mesh/edge_disc.rs` (extend)
**Verify**: Second mesh of same model → 0 new discretizations

## D2: DivideContinuity
**OCC**: `ShapeUpgrade_ShapeDivideContinuity`
**Goal**: Split faces/edges at C0 discontinuities to produce C1 sub-shapes.
**Algorithm**:
1. Sample curve/surface at high density
2. Compute tangent discontinuities (G1 gaps > threshold)
3. Split at discontinuity points
4. Create new edges/faces from sub-segments
**File**: `heal/continuity.rs` (extend)
**Verify**: Face with C0 seam → split into C1 sub-faces

## D3: Canonical Recognition
**OCC**: `ShapeAnalysis_CanonicalRecognition`
**Goal**: Detect when a BSpline is actually a plane/cylinder/sphere for optimization.
**Algorithm**:
1. For each BSpline surface: check control points against known forms
   - Plane: all CPs coplanar
   - Cylinder: CPs form constant-distance extrusion
   - Sphere: CPs lie on sphere
   - Cone: CPs form constant-angle cone
2. Replace recognized BSplines with analytic surfaces
**File**: `heal/canonical.rs` (new)
**Verify**: BSpline plane → recognized and replaced

## D4: Binary Persistence
**OCC**: `BRepTools_ShapeSet`
**Goal**: Serialize BRepStore to compact binary format.
**Algorithm**:
1. Write header (magic, version, counts)
2. Write vertex table (position + tolerance)
3. Write edge table (curve type + parameters + PCurve refs)
4. Write face table (surface type + parameters + wire refs)
5. Write shell/solid/compound tables
6. Read back with validation
**File**: `io/binary_persist.rs` (new in rc3d-io)
**Verify**: Cube BRepStore → write → read → identical geometry

## D5: IGES Format
**OCC**: `IGESControl_Reader`
**Goal**: Import IGES files.
**Algorithm**: Minimal reader for IGES 5.3 entities:
- Entity 100 (Circular Arc), 110 (Line), 108 (Plane), etc.
- Map to CurveGeom/SurfaceGeom
- Build BRepStore from IGES topology
**File**: `io/iges.rs` (new in rc3d-io)
**Verify**: Import simple IGES file → valid BRepStore

## Execution order (independence)

```
D1 (Incremental mesh) ─ independent
D2 (DivideContinuity) ─ independent
D3 (Canonical) ─ independent
D4 (Binary) ─ independent
D5 (IGES) ─ independent
```

All five are independent → parallel execution.

## Success criteria
- [ ] `cargo check --workspace` — 0 errors
- [ ] `cargo test` — unit tests pass (≥630)
- [ ] D1: re-mesh reuses valid edges
- [ ] D2: C0 face split at discontinuity
- [ ] D3: BSpline plane → SurfaceGeom::Plane
- [ ] D4: roundtrip preserves geometry
- [ ] D5: IGES import produces valid BRepStore
